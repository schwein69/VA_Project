import cv2
import os
import random
import shutil

# ----------------------------
# Configurazione
# ----------------------------
classes = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']  
num_images_per_class = 80 
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

dataset_dir = 'dataset'

# Crea cartelle vuote
for split in ['train', 'val', 'test']:
    for cls in classes:
        path = os.path.join(dataset_dir, split, cls)
        os.makedirs(path, exist_ok=True)


def capture_images_for_class(class_name, num_images):
    cap = cv2.VideoCapture(0)
    print(f"\nPremi 's' per salvare un'immagine per la classe: {class_name}")
    print("Premi 'q' per passare alla classe successiva.\n")

    saved_images = 0
    while saved_images < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f"Classe: {class_name} ({saved_images}/{num_images})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Dataset Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            temp_path = os.path.join('temp_images', f"{class_name}_{saved_images}.jpg")
            os.makedirs('temp_images', exist_ok=True)
            cv2.imwrite(temp_path, frame)
            saved_images += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# Acquisizione immagini per tutte le classi
# ----------------------------
for cls in classes:
    capture_images_for_class(cls, num_images_per_class)

# ----------------------------
# Suddivisione train/val/test
# ----------------------------
for cls in classes:
    class_images = [os.path.join('temp_images', f) for f in os.listdir('temp_images') if cls in f]
    random.shuffle(class_images)

    n_train = int(len(class_images) * train_ratio)
    n_val = int(len(class_images) * val_ratio)

    splits = {
        'train': class_images[:n_train],
        'val': class_images[n_train:n_train+n_val],
        'test': class_images[n_train+n_val:]
    }

    for split, images in splits.items():
        for img_path in images:
            shutil.move(img_path, os.path.join(dataset_dir, split, cls))

shutil.rmtree('temp_images', ignore_errors=True)

print("\nDataset creato con successo!")
print(f"Cartelle create in '{dataset_dir}/train', '{dataset_dir}/val', '{dataset_dir}/test'")
