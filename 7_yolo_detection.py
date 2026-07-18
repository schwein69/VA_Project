import os
import cv2
import shutil
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import glob
import random

GESTURE_CLASSES = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']
SOURCE_DIR = "dataset/train"
YOLO_DIR = "yolo_dataset"
EPOCHS = 30  
PATIENCE = 5

def extract_history(model):
  df = pd.read_csv(model.trainer.csv)

  box_w=model.trainer.args.box
  cls_w=model.trainer.args.cls
  dfl_w=model.trainer.args.dfl

  history=  {
              "train_loss" : box_w*df["train/box_loss"] + cls_w*df["train/cls_loss"] + dfl_w*df["train/dfl_loss"],
              "val_loss" : box_w*df["val/box_loss"] + cls_w*df["val/cls_loss"] + dfl_w*df["val/dfl_loss"],
              "val_precision" : df["metrics/precision(B)"],
              "val_recall" : df["metrics/recall(B)"],
              "val_mAP50" : df["metrics/mAP50(B)"],
              "val_mAP50-95" : df["metrics/mAP50-95(B)"],
            }

  return history

def plot_history(history):
  fig, ax1 = plt.subplots(figsize=(10, 8))

  epoch_count=len(history["train_loss"])

  line1,=ax1.plot(range(1,epoch_count+1),history["train_loss"],label="train_loss",color="orange")
  ax1.plot(range(1,epoch_count+1),history["val_loss"],label="val_loss",color = line1.get_color(), linestyle = '--')
  ax1.set_xlim([1,epoch_count])
  ax1.set_ylim([0, max(max(history["train_loss"]),max(history["val_loss"]))])
  ax1.set_ylabel("loss",color = line1.get_color())
  ax1.tick_params(axis="y", labelcolor=line1.get_color())
  ax1.set_xlabel("Epochs")
  _=ax1.legend(loc="lower left")

  ax2 = ax1.twinx()
  line2,=ax2.plot(range(1,epoch_count+1),history["val_mAP50"],label="val_mAP50")
  ax2.set_ylim([0, 1])
  ax2.set_ylabel("metrics",color=line2.get_color())
  ax2.tick_params(axis="y", labelcolor=line2.get_color())
  _=ax2.legend(loc="upper right")

os.makedirs(f"{YOLO_DIR}/images/train", exist_ok=True)
os.makedirs(f"{YOLO_DIR}/labels/train", exist_ok=True)

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Scaricamento modello MediaPipe...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path)

options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path), num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

immagini_salvate = 0

for class_id, class_name in enumerate(GESTURE_CLASSES):
    class_dir = os.path.join(SOURCE_DIR, class_name)
    if not os.path.exists(class_dir): continue
    
    print(f"Annotazione classe '{class_name}'...")
    for file_name in os.listdir(class_dir):
        if not file_name.endswith(('.jpg', '.png')): continue
        
        img_path = os.path.join(class_dir, file_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)
        
        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            
            x_center = (max(x_coords) + min(x_coords)) / 2.0
            y_center = (max(y_coords) + min(y_coords)) / 2.0
            
            box_w = min(max(x_coords) - min(x_coords) + 0.05, 1.0)
            box_h = min(max(y_coords) - min(y_coords) + 0.05, 1.0)
            
            new_img_name = f"{class_name}_{file_name}"
            shutil.copy(img_path, f"{YOLO_DIR}/images/train/{new_img_name}")
            
            label_name = new_img_name.rsplit('.', 1)[0] + ".txt"
            with open(f"{YOLO_DIR}/labels/train/{label_name}", "w") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                
            immagini_salvate += 1

print(f"{immagini_salvate} immagini convertite e annotate in formato YOLO.")

percorso_assoluto = os.path.abspath(YOLO_DIR).replace("\\", "/")
yaml_content = f"""
path: {percorso_assoluto}
train: images/train
val: images/train

names:
  0: open_palm
  1: fist
  2: index
  3: two_fingers
  4: pinch
"""
yaml_path = "gestures.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content.strip())
print(f"File '{yaml_path}' generato correttamente")

yolo_detector = YOLO("yolo11n.pt") 

yolo_detector.info(detailed=True)



results = yolo_detector.train(
    data=yaml_path, 
    epochs=EPOCHS, 
    patience=PATIENCE,
    device="cpu",   
    cache=True,
    verbose=False,
    workers=4      
)

history=extract_history(yolo_detector)

plot_history(history)

yolo_detector.eval()
tutte_le_immagini = glob.glob(f"{YOLO_DIR}/images/train/*.*")
results=yolo_detector.predict(source="416x416_aug/test/images")

immagini_di_test = random.sample(tutte_le_immagini, min(5, len(tutte_le_immagini)))
results = yolo_detector.predict(source=immagini_di_test, save=True, show=True)