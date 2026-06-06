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


GESTURE_CLASSES = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']
PERCORSO_DATI = "dataset/train"  
IMG_SIZE = (224, 224)

MODEL_PATHS = {
    "CNN": "model_name_cnn.keras",
    "LANDMARKS": "model_name_landmarks.keras",
    "MOBILENET": "model_name_mobilenet.keras"
}

def plot_models_confusion_matrices(y_true, preds_cnn, preds_land, preds_mobile):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confronto Modelli di Classificazione CNN vs Landmarks vs MobileNetV2", fontsize=16, fontweight='bold', y=1.05)

    cm_cnn = confusion_matrix(y_true, preds_cnn, normalize='true')
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=GESTURE_CLASSES)
    disp1.plot(cmap=plt.cm.Reds, ax=axes[0], xticks_rotation=45)
    axes[0].set_title('1. CNN Custom')

    cm_land = confusion_matrix(y_true, preds_land, normalize='true')
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_land, display_labels=GESTURE_CLASSES)
    disp2.plot(cmap=plt.cm.Greens, ax=axes[1], xticks_rotation=45)
    axes[1].set_title('2. Rete a Landmarks')

    cm_mobile = confusion_matrix(y_true, preds_mobile, normalize='true')
    disp3 = ConfusionMatrixDisplay(confusion_matrix=cm_mobile, display_labels=GESTURE_CLASSES)
    disp3.plot(cmap=plt.cm.Blues, ax=axes[2], xticks_rotation=45)
    axes[2].set_title('3. MobileNetV2')

    plt.tight_layout()
    plt.show()

def print_model_reports(y_true, preds_dict):
    print("\n" + "="*60)
    print(" REPORT DI CLASSIFICAZIONE MODELLI REALI")
    print("="*60)
    for name, preds in preds_dict.items():
        print(f"\n--- {name.upper()} ---")
        print(classification_report(y_true, preds, target_names=GESTURE_CLASSES))

if __name__ == "__main__":
    
    modelli = {}
    for nome, path in MODEL_PATHS.items():
        if os.path.exists(path):
            modelli[nome] = tf.keras.models.load_model(path)
            print(f"  -> {nome} caricato con successo!")
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
    y_labels_all = []

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
                    
                    
                    lab = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    hand_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    
                    hand_resized = cv2.resize(hand_processed, IMG_SIZE)
                    
                   
                    # Input CNN: normalizzazione standard 0-1
                    cnn_input = hand_resized / 255.0
                    
                    # Input MobileNet: normalizzazione specifica di Keras
                    mobile_input = tf.keras.applications.mobilenet_v2.preprocess_input(hand_resized.astype(np.float32))
                    
                    # Estrazione Landmarks (63 punti) per il modello MLP
                    punti = []
                    for lm in landmarks:
                        punti.extend([lm.x, lm.y, lm.z])
                    
                    # Salvataggio nei dataset
                    X_cnn_all.append(cnn_input)
                    X_mobile_all.append(mobile_input)
                    X_land_all.append(punti)
                    y_labels_all.append(class_id)

    X_cnn_all = np.array(X_cnn_all)
    X_mobile_all = np.array(X_mobile_all)
    X_land_all = np.array(X_land_all)
    y_labels_all = np.array(y_labels_all)

   

    _, X_test_cnn, _, y_test = train_test_split(X_cnn_all, y_labels_all, test_size=0.30, random_state=42)
    _, X_test_mobile, _, _ = train_test_split(X_mobile_all, y_labels_all, test_size=0.30, random_state=42)
    _, X_test_land, _, _ = train_test_split(X_land_all, y_labels_all, test_size=0.30, random_state=42)

    preds_cnn_prob = modelli["CNN"].predict(X_test_cnn, verbose=0)
    y_pred_cnn = np.argmax(preds_cnn_prob, axis=1)

    preds_land_prob = modelli["LANDMARKS"].predict(X_test_land, verbose=0)
    y_pred_land = np.argmax(preds_land_prob, axis=1)

    preds_mobile_prob = modelli["MOBILENET"].predict(X_test_mobile, verbose=0)
    y_pred_mobile = np.argmax(preds_mobile_prob, axis=1)

    print_model_reports(y_test, {
        "CNN Custom (Immagini Processate e Ritagliate)": y_pred_cnn,
        "Landmarks MLP": y_pred_land,
        "MobileNetV2 (Immagini Processate e Ritagliate)": y_pred_mobile
    })
    
    plot_models_confusion_matrices(y_test, y_pred_cnn, y_pred_land, y_pred_mobile)