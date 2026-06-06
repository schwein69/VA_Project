import os
import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import time
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading

class KalmanHandTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32) * 0.03
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.5
        self.initialized = False

    def update(self, x, y):
        measured = np.array([[x], [y]], dtype=np.float32)
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
            
        self.kf.predict()
        estimated = self.kf.correct(measured)
        return int(estimated[0, 0]), int(estimated[1, 0])

TRACKER_MODE = "CAMSHIFT"
MODELLO_SCELTO = "LANDMARKS" # Ora puoi usare "LANDMARKS", "CNN" o "MOBILENET"

GESTURE_CLASSES = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']
IMG_SIZE = (224, 224)

MODEL_PATHS = {
    "CNN": "model_name_cnn.keras",
    "MOBILENET": "model_name_mobilenet.keras",
    "LANDMARKS": "model_name_landmarks.keras"
}

pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

global_crop = None            
global_landmarks = None       
global_gesture = None         
global_confidence = 0.0


model_path_mp = 'hand_landmarker.task'
if not os.path.exists(model_path_mp):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path_mp)

base_options = python.BaseOptions(model_asset_path=model_path_mp)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)


def ai_worker_thread(model):
    global global_crop, global_landmarks, global_gesture, global_confidence
    
    while True:
        model_input = None
        
        # A. Preparazione dati per modello LANDMARKS
        if MODELLO_SCELTO == "LANDMARKS" and global_landmarks is not None:
            punti = []
            for lm in global_landmarks:
                punti.extend([lm.x, lm.y, lm.z])
            model_input = np.expand_dims(np.array(punti), axis=0)
            global_landmarks = None # Svuota la coda
            
        # B. Preparazione dati per modello CNN / MOBILENET
        elif MODELLO_SCELTO in ["CNN", "MOBILENET"] and global_crop is not None:
            crop_to_process = global_crop.copy()
            global_crop = None # Svuota la coda
            try:
                lab = cv2.cvtColor(crop_to_process, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                hand_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                hand_resized = cv2.resize(hand_processed, IMG_SIZE)
                
                if MODELLO_SCELTO == "CNN":
                    model_input = np.expand_dims(hand_resized / 255.0, axis=0)
                else:
                    model_input = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(hand_resized.astype(np.float32), axis=0))
            except:
                pass
                
        # Esecuzione Inferenza
        if model_input is not None:
            try:
                preds = model.predict(model_input, verbose=0)
                class_id = np.argmax(preds[0])
                global_confidence = preds[0][class_id]
                global_gesture = GESTURE_CLASSES[class_id]
            except Exception as e:
                pass
        
        time.sleep(0.01)


if __name__ == "__main__":
    path_modello = MODEL_PATHS.get(MODELLO_SCELTO)
    if not os.path.exists(path_modello):
        print(f"ERRORE: Modello {path_modello} non trovato.")
        exit()
        
    print(f"\n[Sistema] Caricamento modello {MODELLO_SCELTO}...")
    model = tf.keras.models.load_model(path_modello)

    ai_thread = threading.Thread(target=ai_worker_thread, args=(model,), daemon=True)
    ai_thread.start()

    cap = cv2.VideoCapture(0)
    mouse_tracker = KalmanHandTracker()
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    roi_hist = None
    track_window = None
    
    prev_target_y = 0
    last_click_time = 0
    click_cooldown = 0.5
    frames_senza_mano = 0
    
    prev_cs_cx = 0
    prev_cs_cy = 0

    print("[Sistema] Avvio... Inquadra la mano per l'auto-calibrazione!")

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        cam_h, cam_w, _ = frame.shape
        display_frame = frame.copy()

        # --- A. MEDIAPIPE (Auto-Calibrazione & Estrazione Landmarks) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            frames_senza_mano = 0
            landmarks = results.hand_landmarks[0]
            
            # Inviamo i landmarks al Thread AI se richiesto
            if MODELLO_SCELTO == "LANDMARKS" and global_landmarks is None:
                global_landmarks = landmarks
                
            # AUTO-CALIBRAZIONE CAMSHIFT (Eseguita solo 1 volta all'inizio o dopo aver perso la mano)
            if roi_hist is None:
                mp_cx = int(landmarks[9].x * cam_w)
                mp_cy = int(landmarks[9].y * cam_h)
                box = 15
                skin_crop = frame[max(0, mp_cy-box):min(cam_h, mp_cy+box), max(0, mp_cx-box):min(cam_w, mp_cx+box)]
                
                if skin_crop.size > 0:
                    hsv_roi = cv2.cvtColor(skin_crop, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv_roi, np.array((0., 10., 10.)), np.array((25., 255., 255.)))
                    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                    track_window = (mp_cx - 80, mp_cy - 80, 160, 160)
                    prev_cs_cx, prev_cs_cy = mp_cx, mp_cy
                    print("[Auto-Calibrazione] Colore campionato con successo!")
        else:
            frames_senza_mano += 1

        if frames_senza_mano > 15:
            roi_hist = None
            track_window = None
            global_gesture = None
            mouse_tracker.initialized = False
            cv2.putText(display_frame, "Inquadra la mano...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- B. TRACKING CAMSHIFT & KALMAN ---
        if roi_hist is not None and track_window is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            
            if TRACKER_MODE == "CAMSHIFT":
                ret_val, track_window = cv2.CamShift(dst, track_window, term_crit)
                x, y, w, h = track_window
                pts = cv2.boxPoints(ret_val)
                pts = np.int32(pts) 
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
            else:
                ret_val, track_window = cv2.meanShift(dst, track_window, term_crit)
                x, y, w, h = track_window
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            x, y = max(0, x), max(0, y)
            w, h = min(cam_w - x, w), min(cam_h - y, h)

            if w > 20 and h > 20:
                raw_cx, raw_cy = x + (w // 2), y + (h // 2)
                
                # Applichiamo Kalman
                smooth_cx, smooth_cy = mouse_tracker.update(raw_cx, raw_cy)
                
                # Disegno Indicatori
                cv2.circle(display_frame, (int(raw_cx), int(raw_cy)), 4, (0, 0, 255), -1)
                cv2.circle(display_frame, (int(smooth_cx), int(smooth_cy)), 8, (0, 255, 0), 2)
                
                # Mappatura sul monitor PC
                margin = 50
                target_mouse_x = np.interp(smooth_cx, (margin, cam_w - margin), (0, SCREEN_W))
                target_mouse_y = np.interp(smooth_cy, (margin, cam_h - margin), (0, SCREEN_H))
                delta_y = target_mouse_y - prev_target_y
                
                # Invio immagine al Thread se usiamo CNN/MobileNet
                if MODELLO_SCELTO in ["CNN", "MOBILENET"] and global_crop is None:
                    hand_crop = frame[y:y+h, x:x+w]
                    if hand_crop.size > 0:
                        global_crop = hand_crop 
                
                # --- C. AZIONI OS MOUSE ---
                if global_gesture and global_confidence > 0.85:
                    if global_gesture == "index":
                        pyautogui.moveTo(int(target_mouse_x), int(target_mouse_y))
                    
                    elif global_gesture == "open_palm" and (time.time() - last_click_time) > click_cooldown:
                        pyautogui.click()
                        print(f"[{MODELLO_SCELTO}] Click Sinistro")
                        last_click_time = time.time()
                    
                    elif global_gesture == "pinch":
                        if abs(delta_y) > 5:
                            pyautogui.scroll(int(delta_y * -0.5)) 
                            
                    elif global_gesture == "two_fingers" and (time.time() - last_click_time) > click_cooldown:
                        pyautogui.rightClick()
                        print(f"[{MODELLO_SCELTO}] Click Destro")
                        last_click_time = time.time()
                        
                    elif global_gesture == "fist" and (time.time() - last_click_time) > click_cooldown:
                        pyautogui.doubleClick()
                        print(f"[{MODELLO_SCELTO}] Doppio Click")
                        last_click_time = time.time()
                
                prev_target_y = target_mouse_y

                if global_gesture:
                    text_y = max(20, y - 10)
                    cv2.putText(display_frame, f"{global_gesture} ({global_confidence*100:.0f}%)", (x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            dst_resized = cv2.resize(dst, (160, 120))
            dst_colored = cv2.cvtColor(dst_resized, cv2.COLOR_GRAY2BGR)
            display_frame[cam_h-120:cam_h, cam_w-160:cam_w] = dst_colored

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(display_frame, f"FPS: {int(fps)} | Modello: {MODELLO_SCELTO}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Virtual Mouse", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()