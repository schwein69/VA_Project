import cv2
import os
import random
import numpy as np

classes = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']

root_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(root_dir, 'dataset', 'train')
num_aug_per_image = 5  

def random_flip(img):
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    return img

def random_rotation(img):
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def random_brightness_contrast(img):
    brightness = random.randint(-50, 50)
    contrast = random.uniform(0.8, 1.2)
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

def random_crop_zoom(img):
    h, w = img.shape[:2]
    scale = random.uniform(0.9, 1.1)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(img, (nw, nh))
    if scale < 1:
        pad_h, pad_w = (h - nh) // 2, (w - nw) // 2
        canvas = np.zeros_like(img)
        canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized
        return canvas
    else:
        start_h, start_w = (nh - h) // 2, (nw - w) // 2
        return resized[start_h:start_h+h, start_w:start_w+w]

def augment_image(img):
    img_copy = img.copy()
    img_copy = random_flip(img_copy)
    img_copy = random_rotation(img_copy)
    img_copy = random_brightness_contrast(img_copy)
    img_copy = random_crop_zoom(img_copy)
    return img_copy


for cls in classes:
    input_dir = os.path.join(dataset_dir, cls)
    if not os.path.exists(input_dir):
        print(f"Attenzione: cartella non trovata per la classe {cls}")
        continue

    all_images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png')) and '_aug' not in f]
    
    for file_name in all_images:
        img_path = os.path.join(input_dir, file_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        for i in range(num_aug_per_image):
            aug_img = augment_image(img)
            base_name = os.path.splitext(file_name)[0]
            save_path = os.path.join(input_dir, f"{base_name}_aug{i}.jpg")
            cv2.imwrite(save_path, aug_img)
            
    print(f"Classe '{cls}': generate {len(all_images) * num_aug_per_image} nuove immagini.")
