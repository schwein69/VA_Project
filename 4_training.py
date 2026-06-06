import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import random
import matplotlib.pyplot as plt

TIPO_ESPERIMENTO = "MOBILENET" 

percorso_dati = "dataset/train"
classi = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']
img_size = (224, 224) 

X_data = []
y_labels = []
original_images = []  # Per visualizzazione
roi_images = []  # Per visualizzazione
max_examples_to_save = 3 # Numero di esempi per classe da visualizzare

# Inizializza il detector di MediaPipe per entrambe le modalità
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

for class_id, class_name in enumerate(classi):
    cartella_classe = os.path.join(percorso_dati, class_name)
    if not os.path.exists(cartella_classe):
        continue
    class_examples_saved = 0  # Counter per esempi salvati di questa classe
        
    for file_name in os.listdir(cartella_classe):
        if not file_name.endswith(('.jpg', '.png')):
            continue
            
        img_path = os.path.join(cartella_classe, file_name)

        if TIPO_ESPERIMENTO in ["CNN","MOBILENET"]:
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Converti in RGB per MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = detector.detect(mp_image)
            
            if results.hand_landmarks:
                # Calcola bounding box dalla mano rilevata
                h, w, _ = img.shape
                x_coords = [lm.x * w for lm in results.hand_landmarks[0]]
                y_coords = [lm.y * h for lm in results.hand_landmarks[0]]
                min_x = int(min(x_coords))
                max_x = int(max(x_coords))
                min_y = int(min(y_coords)) 
                max_y = int(max(y_coords))
                
                # Aggiungi padding
                padding = 20
                min_x = max(0, min_x - padding)
                max_x = min(w, max_x + padding)
                min_y = max(0, min_y - padding)
                max_y = min(h, max_y + padding)
                
                # Ritaglia la regione della mano
                hand_img = img[min_y:max_y, min_x:max_x]
                if hand_img.size == 0: continue
                
                # Preprocessing: CLAHE per migliorare il contrasto
                lab = cv2.cvtColor(hand_img, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                hand_img_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                # Salva immagini originali e dopo ROI per visualizzazione
                if class_examples_saved < max_examples_to_save:
                    # Converti a RGB per matplotlib
                    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    roi_rgb = cv2.cvtColor(hand_img_processed, cv2.COLOR_BGR2RGB)
                    original_images.append((original_rgb, class_name))
                    roi_images.append((roi_rgb, class_name))
                    class_examples_saved += 1

                # Ridimensiona
                hand_resized = cv2.resize(hand_img_processed, img_size)
                if TIPO_ESPERIMENTO == "CNN":
                    img_normalized = hand_resized / 255.0
                elif TIPO_ESPERIMENTO == "MOBILENET":
                    img_normalized = tf.keras.applications.mobilenet_v2.preprocess_input(hand_resized.astype(np.float32))
                
                # Normalizza
                X_data.append(img_normalized)
                y_labels.append(class_id)

        elif TIPO_ESPERIMENTO == "LANDMARKS":
            try:
                image = mp.Image.create_from_file(img_path)

                detection_result = detector.detect(image)
                
                if detection_result.hand_landmarks:
                    for hand_landmarks in detection_result.hand_landmarks:
                        # 63 punti (21 * 3) in un array piatto
                        punti = []
                        for lm in hand_landmarks:
                            punti.extend([lm.x, lm.y, lm.z])
                        X_data.append(punti)
                        y_labels.append(class_id)
            except Exception as e:
                print(f"Errore nel processare l'immagine {file_name}: {e}")
            

X_data = np.array(X_data)
y_labels = np.array(y_labels)

print(f"Dati pronti: {len(X_data)} campioni trovati.")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_data, y_labels, test_size=0.20, random_state=42
)
# Dal train rimasto estraiamo la validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.30, random_state=42
)

print(f"Split in: Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


if TIPO_ESPERIMENTO == "CNN":
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)), 

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(classi), activation='softmax')
    ])
    epoche_addestramento = 30

elif TIPO_ESPERIMENTO == "LANDMARKS":
    model = models.Sequential([
        layers.Input(shape=(63,)), 
        
        # Normalizzazione diretta delle coordinate in ingresso
        layers.BatchNormalization(),
        
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(len(classi), activation='softmax')
    ])
    epoche_addestramento = 50 

elif TIPO_ESPERIMENTO == "MOBILENET":
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False 
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), # Converte le features 2D in un vettore piatto 1D
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(classi), activation='softmax')
    ])
    epoche_addestramento = 20

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

storia = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epoche_addestramento,
    batch_size=50,
    callbacks=[early_stop]
)

# Grafico della perdita durante l'addestramento
training_loss = storia.history['loss']
validation_loss = storia.history['val_loss']

# Plotting the loss graph
epochs = range(1, len(training_loss) + 1)

plt.plot(epochs, training_loss, 'b', label='Training Loss')
plt.plot(epochs, validation_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Valutazione finale sul test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuratezza Finale sul Test: {test_acc * 100:.2f}%")


model_name = f"model_name_{TIPO_ESPERIMENTO.lower()}.keras"
model.save(model_name) 


print("\n Test")
numero_immagini_da_testare = 5

indici_casuali = random.sample(range(len(X_test)), min(numero_immagini_da_testare, len(X_test)))

fig, axes = plt.subplots(1, numero_immagini_da_testare, figsize=(15, 4))
fig.suptitle(f"Risultati Test Modello: {TIPO_ESPERIMENTO}", fontsize=16)

for i, idx in enumerate(indici_casuali):
    ax = axes[i]
    
    dati_input = X_test[idx]
    vera_classe_id = y_test[idx]
    
    input_rete = np.expand_dims(dati_input, axis=0)
    
    # Facciamo la predizione
    predizioni = model.predict(input_rete, verbose=0)
    pred_class_id = np.argmax(predizioni[0])
    confidenza = predizioni[0][pred_class_id]
    
    vera_label = classi[vera_classe_id]
    pred_label = classi[pred_class_id]
    
    if TIPO_ESPERIMENTO == "CNN":
        img_da_mostrare = (dati_input * 255).astype(np.uint8)
        img_da_mostrare = cv2.cvtColor(img_da_mostrare, cv2.COLOR_BGR2RGB)

    elif TIPO_ESPERIMENTO == "MOBILENET":
        img_da_mostrare = (dati_input + 1.0) / 2.0
        img_da_mostrare = cv2.cvtColor(img_da_mostrare, cv2.COLOR_BGR2RGB)
    
    elif TIPO_ESPERIMENTO == "LANDMARKS":
        img_da_mostrare = np.ones((224, 224, 3), dtype=np.uint8) * 255
        
        for j in range(0, 63, 3):
            x = int(dati_input[j] * 224)
            y = int(dati_input[j+1] * 224)
            if 0 <= x < 224 and 0 <= y < 224:
                cv2.circle(img_da_mostrare, (x, y), radius=3, color=(255, 0, 0), thickness=-1)

    ax.imshow(img_da_mostrare)
    ax.axis('off')
    
    # Verde se indovina, Rosso se sbaglia
    colore = 'green' if pred_label == vera_label else 'red'
    titolo = f"Pred: {pred_label}\n({confidenza*100:.0f}%)\n\nVera: {vera_label}"
    ax.set_title(titolo, color=colore, fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()


# Visualizzazione confronto tra immagini originali e dopo ROI
if TIPO_ESPERIMENTO in ["CNN","MOBILENET"] and len(original_images) > 0:
    print(f"\n=== VISUALIZZAZIONE: Immagini Originali vs Dopo ROI ===")
    num_examples = len(original_images)
    
    fig, axes = plt.subplots(2, num_examples, figsize=(15, 6))
    fig.suptitle("Confronto: Immagine Originale vs Dopo ROI (Region of Interest)", fontsize=16, fontweight='bold')
    
    for i, (orig_img, class_name) in enumerate(original_images):
        # Immagine originale
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Originale - {class_name}", fontsize=10)
        axes[0, i].axis('off')
        
        # Immagine dopo ROI
        roi_img, _ = roi_images[i]
        axes[1, i].imshow(roi_img)
        axes[1, i].set_title(f"Dopo ROI - {class_name}", fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"Visualizzati {num_examples} esempi di preprocessing.")
