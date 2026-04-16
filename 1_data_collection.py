import cv2
import os
import random
import shutil
import time
# Configurazione
classes = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']  
num_images_per_class = 50 

root_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(root_dir, 'dataset')

for cls in classes:
    path = os.path.join(dataset_dir, cls)
    os.makedirs(path, exist_ok=True)

def capture_images_for_class(class_name, num_images):
    cap = cv2.VideoCapture(0)
    print("Premi 'q' per passare alla classe successiva.\n")

    saved_images = 0
    last_capture_time = 0
    while saved_images < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        current_time = time.time()
        if current_time - last_capture_time >= 1:
            save_path = os.path.join(temp_dir, f"{class_name}_{saved_images}.jpg")
            cv2.imwrite(save_path, frame)
            saved_images += 1
            last_capture_time = current_time
            print(f"Salvata immagine {saved_images}/{num_images} per {class_name}")

        cv2.putText(frame, f"Classe: {class_name} ({saved_images}/{num_images})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Dataset Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

    cap.release()
    cv2.destroyAllWindows()

for cls in classes:
    capture_images_for_class(cls, num_images_per_class)


print("\nDataset creato con successo!")
print(f"Cartelle create in '{dataset_dir}/train', '{dataset_dir}/val', '{dataset_dir}/test'")
