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

TIPO_ESPERIMENTO = "LANDMARKS" 

percorso_dati = "dataset/train"
classi = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']
img_size = (224, 224) 

X_data = []
y_labels = []

if TIPO_ESPERIMENTO == "LANDMARKS":
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
        
    for file_name in os.listdir(cartella_classe):
        if not file_name.endswith(('.jpg', '.png')):
            continue
            
        img_path = os.path.join(cartella_classe, file_name)

        if TIPO_ESPERIMENTO == "CNN":
            img = cv2.imread(img_path)
            if img is None: continue
            img_resized = cv2.resize(img, img_size)
            img_normalized = img_resized / 255.0
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

X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_labels, test_size=0.30, random_state=42)

# 15% Validation, 15% Test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Split in: Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


if TIPO_ESPERIMENTO == "CNN":
    modello = models.Sequential([
        layers.Input(shape=(img_size[0], img_size[1], 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(classi), activation='softmax')
    ])
    epoche_addestramento = 30

elif TIPO_ESPERIMENTO == "LANDMARKS":
    modello = models.Sequential([
        layers.Input(shape=(63,)), 
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(len(classi), activation='softmax')
    ])
    epoche_addestramento = 50 

modello.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

modello.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

storia = modello.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epoche_addestramento,
    batch_size=40,
    callbacks=[early_stop]
)

test_loss, test_acc = modello.evaluate(X_test, y_test, verbose=0)
print(f"Accuratezza Finale sul Test: {test_acc * 100:.2f}%")

""" 
nome_salvataggio = f"modello_finale_{TIPO_ESPERIMENTO.lower()}.keras"
modello.save(nome_salvataggio) """


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
    predizioni = modello.predict(input_rete, verbose=0)
    pred_class_id = np.argmax(predizioni[0])
    confidenza = predizioni[0][pred_class_id]
    
    vera_label = classi[vera_classe_id]
    pred_label = classi[pred_class_id]
    
    if TIPO_ESPERIMENTO == "CNN":
        img_da_mostrare = (dati_input * 255).astype(np.uint8)
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
