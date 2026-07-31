import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import random
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# FUNZIONE DI NORMALIZZAZIONE LANDMARKS
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

#"CNN", "MOBILENET", "LANDMARKS", "RANDOM_FOREST", "BOVW_SVM"
TIPO_ESPERIMENTO = "BOVW_SVM" 

percorso_dati = "dataset/train"
classi = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']
img_size = (224, 224) 
VOCAB_SIZE = 100 

X_data = []
y_labels = []
original_images = []  
roi_images = []  
max_examples_to_save = 3 

# Variabili specifiche per BoVW
sift = cv2.SIFT_create()
tutti_i_descrittori = []
descrittori_per_immagine = []

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

print(f"\nAVVIO ESTRAZIONE DATI PER: {TIPO_ESPERIMENTO}")

for class_id, class_name in enumerate(classi):
    cartella_classe = os.path.join(percorso_dati, class_name)
    if not os.path.exists(cartella_classe): continue
    class_examples_saved = 0  
        
    for file_name in os.listdir(cartella_classe):
        if not file_name.endswith(('.jpg', '.png')): continue
            
        img_path = os.path.join(cartella_classe, file_name)

        if TIPO_ESPERIMENTO in ["CNN", "MOBILENET"]:
            img = cv2.imread(img_path)
            if img is None: continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = detector.detect(mp_image)
            
            if results.hand_landmarks:
                h, w, _ = img.shape
                x_coords = [lm.x * w for lm in results.hand_landmarks[0]]
                y_coords = [lm.y * h for lm in results.hand_landmarks[0]]
                min_x, max_x = int(min(x_coords)), int(max(x_coords))
                min_y, max_y = int(min(y_coords)), int(max(y_coords))
                
                padding = 20
                min_x, max_x = max(0, min_x - padding), min(w, max_x + padding)
                min_y, max_y = max(0, min_y - padding), min(h, max_y + padding)
                
                hand_img = img[min_y:max_y, min_x:max_x]
                if hand_img.size == 0: continue
                
                lab = cv2.cvtColor(hand_img, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                hand_img_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                if class_examples_saved < max_examples_to_save:
                    original_images.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), class_name))
                    roi_images.append((cv2.cvtColor(hand_img_processed, cv2.COLOR_BGR2RGB), class_name))
                    class_examples_saved += 1

                hand_resized = cv2.resize(hand_img_processed, img_size)
                if TIPO_ESPERIMENTO == "CNN":
                    X_data.append(hand_resized / 255.0)
                elif TIPO_ESPERIMENTO == "MOBILENET":
                    X_data.append(tf.keras.applications.mobilenet_v2.preprocess_input(hand_resized.astype(np.float32)))
                y_labels.append(class_id)

        elif TIPO_ESPERIMENTO in ["LANDMARKS", "RANDOM_FOREST"]:
            try:
                image = mp.Image.create_from_file(img_path)
                detection_result = detector.detect(image)
                
                if detection_result.hand_landmarks:
                    normalized_points = normalize_landmarks(detection_result.hand_landmarks[0])
                    X_data.append(normalized_points)
                    y_labels.append(class_id)
            except Exception as e:
                pass

        elif TIPO_ESPERIMENTO == "BOVW_SVM":
            img = cv2.imread(img_path)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_enhanced = clahe.apply(gray)
            
            keypoints, descriptors = sift.detectAndCompute(gray_enhanced, None)
            if descriptors is not None and len(descriptors) > 0:
                descrittori_per_immagine.append(descriptors)
                tutti_i_descrittori.extend(descriptors)
                y_labels.append(class_id)

# Elaborazione Extra per BOVW
if TIPO_ESPERIMENTO == "BOVW_SVM":
    print("\n--- CREAZIONE VOCABOLARIO E ISTOGRAMMI (BoVW) ---")
    kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, batch_size=1000, random_state=42)
    
    matrice_descrittori = np.vstack(tutti_i_descrittori).astype(np.float32)
    kmeans.fit(matrice_descrittori)
    
    for desc in descrittori_per_immagine:
        desc_float32 = desc.astype(np.float32)
        visual_words = kmeans.predict(desc_float32)
        istogramma, _ = np.histogram(visual_words, bins=range(VOCAB_SIZE + 1), density=True)
        X_data.append(istogramma)
        
    with open("model_name_bovw_vocab.pkl", "wb") as f:
        pickle.dump(kmeans, f)

X_data = np.array(X_data)
y_labels = np.array(y_labels)

print(f"Dati pronti: {len(X_data)} campioni trovati.")
X_train_full, X_test, y_train_full, y_test = train_test_split(X_data, y_labels, test_size=0.20, random_state=42)

if TIPO_ESPERIMENTO in ["RANDOM_FOREST", "BOVW_SVM"]:
    print(f"\n--- ADDESTRAMENTO {TIPO_ESPERIMENTO.replace('_', ' ')} ---")
    
    if TIPO_ESPERIMENTO == "RANDOM_FOREST":
        model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    elif TIPO_ESPERIMENTO == "BOVW_SVM":
        model = SVC(kernel='rbf', C=10.0, probability=True, random_state=42) 
        
    model.fit(X_train_full, y_train_full)
    
    test_acc = model.score(X_test, y_test)
    print(f"Accuratezza Finale {TIPO_ESPERIMENTO} sul Test: {test_acc * 100:.2f}%")
    
    model_name = f"model_name_{TIPO_ESPERIMENTO.lower()}.pkl"
    with open(model_name, "wb") as f:
        pickle.dump(model, f)
    print(f"Modello salvato come '{model_name}'")

else:
    print(f"\n ADDESTRAMENTO RETE NEURALE ({TIPO_ESPERIMENTO}) ")
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.30, random_state=42)
    
    if TIPO_ESPERIMENTO == "CNN":
        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)), 
            layers.Conv2D(256, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'), layers.Dropout(0.5),
            layers.Dense(128, activation='relu'), layers.Dropout(0.5),
            layers.Dense(len(classi), activation='softmax')
        ])
        epoche = 30

    elif TIPO_ESPERIMENTO == "LANDMARKS":
        model = models.Sequential([
            layers.Input(shape=(63,)), 
            layers.Dense(128, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.3),
            layers.Dense(64, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.3),
            layers.Dense(len(classi), activation='softmax')
        ])
        epoche = 50 

    elif TIPO_ESPERIMENTO == "MOBILENET":
        base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_model.trainable = False 
        model = models.Sequential([
            base_model, layers.GlobalAveragePooling2D(), 
            layers.Dense(128, activation='relu'), layers.Dropout(0.5),
            layers.Dense(len(classi), activation='softmax')
        ])
        epoche = 20

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    storia = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epoche, batch_size=50, callbacks=[early_stop])

    plt.plot(range(1, len(storia.history['loss']) + 1), storia.history['loss'], 'b', label='Training Loss')
    plt.plot(range(1, len(storia.history['val_loss']) + 1), storia.history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuratezza Finale {TIPO_ESPERIMENTO} sul Test: {test_acc * 100:.2f}%")

    model_name = f"model_name_{TIPO_ESPERIMENTO.lower()}.keras"
    model.save(model_name) 

print("\n  TEST ")
numero_immagini = 5
indici = random.sample(range(len(X_test)), min(numero_immagini, len(X_test)))

fig, axes = plt.subplots(1, numero_immagini, figsize=(15, 4))
fig.suptitle(f"Risultati Test: {TIPO_ESPERIMENTO}", fontsize=16)

for i, idx in enumerate(indici):
    ax = axes[i]
    dati_input = X_test[idx]
    vera_classe_id = y_test[idx]
    
    input_rete = np.expand_dims(dati_input, axis=0)
    
    if TIPO_ESPERIMENTO in ["RANDOM_FOREST", "BOVW_SVM"]:
        predizioni = model.predict_proba(input_rete)
    else:
        predizioni = model.predict(input_rete, verbose=0)
        
    pred_class_id = np.argmax(predizioni[0])
    confidenza = predizioni[0][pred_class_id]
    
    vera_label = classi[vera_classe_id]
    pred_label = classi[pred_class_id]
    
    if TIPO_ESPERIMENTO == "CNN":
        ax.imshow(cv2.cvtColor((dati_input * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax.axis('off')
    elif TIPO_ESPERIMENTO == "MOBILENET":
        ax.imshow(cv2.cvtColor(((dati_input + 1.0) / 2.0).astype(np.float32), cv2.COLOR_BGR2RGB))
        ax.axis('off')
    elif TIPO_ESPERIMENTO in ["LANDMARKS", "RANDOM_FOREST"]:
        img_da_mostrare = np.ones((224, 224, 3), dtype=np.uint8) * 255
        for j in range(0, 63, 3):
            x = int(dati_input[j] * 100 + 112)
            y = int(dati_input[j+1] * 100 + 112)
            if 0 <= x < 224 and 0 <= y < 224:
                cv2.circle(img_da_mostrare, (x, y), radius=4, color=(255, 0, 0), thickness=-1)
        ax.imshow(img_da_mostrare)
        ax.axis('off')
    elif TIPO_ESPERIMENTO == "BOVW_SVM":
        ax.bar(range(VOCAB_SIZE), dati_input, color='orange')
        ax.set_ylim(0, max(dati_input) + 0.05)
        ax.set_xticks([])
        ax.set_yticks([])
    
    colore = 'green' if pred_label == vera_label else 'red'
    ax.set_title(f"Pred: {pred_label}\n({confidenza*100:.0f}%)\n\nVera: {vera_label}", color=colore, fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()

if TIPO_ESPERIMENTO in ["CNN", "MOBILENET"] and len(original_images) > 0:
    num_examples = len(original_images)
    fig, axes = plt.subplots(2, num_examples, figsize=(15, 6))
    fig.suptitle("Confronto: Immagine Originale vs Dopo ROI", fontsize=16, fontweight='bold')
    
    for i, (orig_img, class_name) in enumerate(original_images):
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Originale - {class_name}", fontsize=10)
        axes[0, i].axis('off')
        
        roi_img, _ = roi_images[i]
        axes[1, i].imshow(roi_img)
        axes[1, i].set_title(f"Dopo ROI - {class_name}", fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()