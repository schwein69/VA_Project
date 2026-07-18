import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import pickle
import warnings

warnings.filterwarnings('ignore')

GESTURE_CLASSES = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']
PERCORSO_DATI = "dataset/train"
IMG_SIZE = (224, 224)
VOCAB_SIZE = 100

sift = cv2.SIFT_create()

MODEL_PATHS = {
    "CNN": "model_name_cnn.keras",
    "LANDMARKS": "model_name_landmarks.keras",
    "MOBILENET": "model_name_mobilenet.keras",
    "RANDOM_FOREST": "model_name_random_forest.pkl",
    "BOVW_VOCAB": "model_name_bovw_vocab.pkl",
    "BOVW_SVM": "model_name_bovw_svm.pkl"
}

def normalize_landmarks(landmarks):
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    base_x, base_y, base_z = points[0]
    points[:, 0] -= base_x
    points[:, 1] -= base_y
    points[:, 2] -= base_z
    max_value = np.max(np.abs(points))
    if max_value > 0:
        points /= max_value
    return points.flatten()

def plot_models_confusion_matrices(y_true, preds_cnn, preds_land, preds_mobile, preds_rf, preds_bovw):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Confronto Benchmark: Deep Learning vs Handcrafted", fontsize=18, fontweight='bold', y=0.95)

    # 1. CNN Custom
    cm_cnn = confusion_matrix(y_true, preds_cnn, normalize='true')
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=GESTURE_CLASSES)
    disp1.plot(cmap=plt.cm.Reds, ax=axes[0, 0], xticks_rotation=45)
    axes[0, 0].set_title('CNN Custom (Pixel)')

    # 2. MobileNetV2
    cm_mobile = confusion_matrix(y_true, preds_mobile, normalize='true')
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_mobile, display_labels=GESTURE_CLASSES)
    disp2.plot(cmap=plt.cm.Blues, ax=axes[0, 1], xticks_rotation=45)
    axes[0, 1].set_title('MobileNetV2 (Pixel)')

    # 3. Landmarks MLP (Keras)
    cm_land = confusion_matrix(y_true, preds_land, normalize='true')
    disp3 = ConfusionMatrixDisplay(confusion_matrix=cm_land, display_labels=GESTURE_CLASSES)
    disp3.plot(cmap=plt.cm.Greens, ax=axes[0, 2], xticks_rotation=45)
    axes[0, 2].set_title('Rete Neurale MLP (Landmarks)')

    # 4. Random Forest
    cm_rf = confusion_matrix(y_true, preds_rf, normalize='true')
    disp4 = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=GESTURE_CLASSES)
    disp4.plot(cmap=plt.cm.Oranges, ax=axes[1, 0], xticks_rotation=45)
    axes[1, 0].set_title('Random Forest (Landmarks)')

    # 5. Bag of Visual Words (SVM)
    cm_bovw = confusion_matrix(y_true, preds_bovw, normalize='true')
    disp5 = ConfusionMatrixDisplay(confusion_matrix=cm_bovw, display_labels=GESTURE_CLASSES)
    disp5.plot(cmap=plt.cm.Purples, ax=axes[1, 1], xticks_rotation=45)
    axes[1, 1].set_title('Bag of Visual Words SVM (SIFT)')

    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()

def print_model_reports(y_true, preds_dict):
    print("\n" + "="*60)
    print(" REPORT DI CLASSIFICAZIONE MODELLI")
    print("="*60)
    for name, preds in preds_dict.items():
        print(f"\n--- {name.upper()} ---")
        print(classification_report(y_true, preds, target_names=GESTURE_CLASSES))

if __name__ == "__main__":
    
    modelli = {}
    print("\nCaricamento modelli")
    for nome, path in MODEL_PATHS.items():
        if os.path.exists(path):
            if path.endswith('.keras'):
                modelli[nome] = tf.keras.models.load_model(path)
            elif path.endswith('.pkl'):
                with open(path, "rb") as f:
                    modelli[nome] = pickle.load(f)
            print(f"  -> {nome} caricato con successo")
        else:
            print(f"  -> ERRORE: File {path} non trovato. Esegui prima l'addestramento.")
            exit()

    model_path_mp = 'hand_landmarker.task'
    if not os.path.exists(model_path_mp):
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path_mp)
    
    base_options = python.BaseOptions(model_asset_path=model_path_mp)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    X_cnn_all = []
    X_mobile_all = []
    X_land_all = []
    X_bovw_all = [] 
    y_labels_all = []

    print("\nEstrazione e processamento del dataset per il Test")
    for class_id, class_name in enumerate(GESTURE_CLASSES):
        cartella_classe = os.path.join(PERCORSO_DATI, class_name)
        if not os.path.exists(cartella_classe): continue
        
        for file_name in os.listdir(cartella_classe):
            if not file_name.endswith(('.jpg', '.png')): continue
            
            img_path = os.path.join(cartella_classe, file_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = detector.detect(mp_image)
            
            if results.hand_landmarks:
                landmarks = results.hand_landmarks[0]
                h, w, _ = img.shape
                
                x_coords = [lm.x * w for lm in landmarks]
                y_coords = [lm.y * h for lm in landmarks]
                
                margin = 20
                xmin = max(0, int(min(x_coords)) - margin)
                xmax = min(w, int(max(x_coords)) + margin)
                ymin = max(0, int(min(y_coords)) - margin)
                ymax = min(h, int(max(y_coords)) + margin)
                
                hand_crop = img[ymin:ymax, xmin:xmax]
                
                if hand_crop.size > 0:
                    # --- 1. Estrazione Input Keras (CNN / MobileNet) ---
                    lab = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    hand_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    hand_resized = cv2.resize(hand_processed, IMG_SIZE)
                    
                    cnn_input = hand_resized / 255.0
                    mobile_input = tf.keras.applications.mobilenet_v2.preprocess_input(hand_resized.astype(np.float32))
                    
                    # --- 2. Estrazione Input Classici (MLP / RF) ---
                    punti_normalizzati = normalize_landmarks(landmarks)
                    
                    # --- 3. Estrazione Input BoVW (SIFT -> K-Means) ---
                    # Effettuiamo SIFT sull'immagine intera in scala di grigi
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray_enhanced = clahe.apply(gray)
                    keypoints, descriptors = sift.detectAndCompute(gray_enhanced, None)
                    
                    if descriptors is not None and len(descriptors) > 0:
                        desc_float32 = descriptors.astype(np.float32)
                        visual_words = modelli["BOVW_VOCAB"].predict(desc_float32)
                        istogramma, _ = np.histogram(visual_words, bins=range(VOCAB_SIZE + 1), density=True)
                    else:
                        istogramma = np.zeros(VOCAB_SIZE)
                    
                    X_cnn_all.append(cnn_input)
                    X_mobile_all.append(mobile_input)
                    X_land_all.append(punti_normalizzati)
                    X_bovw_all.append(istogramma)
                    y_labels_all.append(class_id)

    X_cnn_all = np.array(X_cnn_all)
    X_mobile_all = np.array(X_mobile_all)
    X_land_all = np.array(X_land_all)
    X_bovw_all = np.array(X_bovw_all)
    y_labels_all = np.array(y_labels_all)

    print("\nGenerazione predizioni sul Test Set")
    
    # Split perfettamente allineato per tutti i 5 modelli
    _, X_test_cnn, _, y_test = train_test_split(X_cnn_all, y_labels_all, test_size=0.30, random_state=42)
    _, X_test_mobile, _, _ = train_test_split(X_mobile_all, y_labels_all, test_size=0.30, random_state=42)
    _, X_test_land, _, _ = train_test_split(X_land_all, y_labels_all, test_size=0.30, random_state=42)
    _, X_test_bovw, _, _ = train_test_split(X_bovw_all, y_labels_all, test_size=0.30, random_state=42)

    # 1. Predizione CNN
    preds_cnn_prob = modelli["CNN"].predict(X_test_cnn, verbose=0)
    y_pred_cnn = np.argmax(preds_cnn_prob, axis=1)

    # 2. Predizione Landmarks (Keras MLP)
    preds_land_prob = modelli["LANDMARKS"].predict(X_test_land, verbose=0)
    y_pred_land = np.argmax(preds_land_prob, axis=1)

    # 3. Predizione MobileNet
    preds_mobile_prob = modelli["MOBILENET"].predict(X_test_mobile, verbose=0)
    y_pred_mobile = np.argmax(preds_mobile_prob, axis=1)
    
    # 4. Predizione Random Forest
    y_pred_rf = modelli["RANDOM_FOREST"].predict(X_test_land)
    
    # 5. Predizione Bag of Visual Words (SVM)
    y_pred_bovw = modelli["BOVW_SVM"].predict(X_test_bovw)

    # Stampa in console
    print_model_reports(y_test, {
        "CNN Custom": y_pred_cnn,
        "MobileNetV2": y_pred_mobile,
        "Landmarks MLP": y_pred_land,
        "Random Forest": y_pred_rf,
        "Bag of Visual Words": y_pred_bovw
    })
    
    # Disegna grafici a schermo
    plot_models_confusion_matrices(y_test, y_pred_cnn, y_pred_land, y_pred_mobile, y_pred_rf, y_pred_bovw)