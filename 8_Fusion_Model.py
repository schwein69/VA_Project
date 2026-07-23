"""
virtual_mouse.py
==========================================

  FRAME -> YOLO (thread) -> Kalman (stabilizza bbox) -> hand_crop
                                                              |
                        +-------------------------------------+-------------------------------------+
                        |                                                                             |
                        v                                                                             v
              FLOW RGB (thread separato)                                            FLOW LANDMARKS (sincrono)
              CNN + MobileNet + BoVW(SIFT)/SVM                                      MediaPipe -> normalize_landmarks()
                        |                                                           -> Landmarks(keras) + RandomForest
                        v                                                                             |
                  Score_RGB                                                                     Score_LM
                        +-------------------------------------+-------------------------------------+
                                                              |
                                       Score_Finale = ALPHA*Score_RGB + BETA*Score_LM
                                                              |
                                                     classe finale -> comando mouse

Il MOVIMENTO del cursore utilizza il NaturalMouseController (Active Area).
Questo mappa una zona centrale della webcam all'intero schermo del computer,
permettendo di raggiungere comodamente tutti i bordi senza stancare il braccio.
Il movimento è azionato in via esclusiva dal classificatore AI fuso
quando riconosce la classe "index".

Controlli: Q = esci
"""

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
from ultralytics import YOLO
import threading
import queue
import pickle

# ============================================================
# CONFIGURAZIONE
# ============================================================
GESTURE_CLASSES = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']
IMG_SIZE        = (224, 224)

# ---- Flow RGB (CNN + MobileNet + BoVW/SIFT+SVM) ----
MODEL_PATHS = {
    "CNN"      : "model_name_cnn.keras",
    "MOBILENET": "model_name_mobilenet.keras",
}
BOVW_VOCAB_PATH = "model_name_bovw_vocab.pkl"   
BOVW_SVM_PATH   = "model_name_bovw_svm.pkl"
SIFT_CLAHE_CLIP_LIMIT = 2.0
SIFT_CLAHE_TILE_GRID  = (8, 8)

WEIGHT_CNN    = 0.40
WEIGHT_MOBILE = 0.30
WEIGHT_BOVW   = 0.30   

# ---- Flow Landmarks (Landmarks-keras + RandomForest) ----
LANDMARKS_MODEL_PATH = "model_name_landmarks.keras"
RF_MODEL_PATH         = "model_name_random_forest.pkl"

WEIGHT_LANDMARKS_KERAS = 0.50
WEIGHT_RANDOMFOREST    = 0.50

# ---- YOLO (detection mano) ----
YOLO_MODEL_PATH = "runs/detect/train/weights/best.pt"
YOLO_IMGSZ = 384

# ---- Score-Level Fusion finale ----
WEIGHT_FLOW_RGB = 0.40
WEIGHT_FLOW_LM  = 0.60

GESTURE_CONFIDENCE_THRESHOLD = 0.60

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0
SCREEN_W, SCREEN_H = pyautogui.size()


# ============================================================
# LANDMARK: normalizzazione
# ============================================================
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


# ============================================================
# KALMAN BOX TRACKER — stabilizza il bounding box di YOLO
# ============================================================
class KalmanBoxTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.hstack(
            [np.eye(4, dtype=np.float32), np.zeros((4, 4), dtype=np.float32)]
        )
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = 1.0
        self.kf.transitionMatrix    = F
        self.kf.processNoiseCov     = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 4.0
        self.kf.errorCovPost        = np.eye(8, dtype=np.float32)
        self.initialized = False
        self.last = np.zeros(4, dtype=np.float32)

    def update(self, cx, cy, w, h):
        meas = np.array([cx, cy, w, h], dtype=np.float32)
        if self.initialized and np.abs(meas[:2] - self.last[:2]).sum() > 200:
            self.initialized = False
        if not self.initialized:
            state0 = np.concatenate([meas, np.zeros(4, dtype=np.float32)])
            self.kf.statePre  = state0.reshape(8, 1)
            self.kf.statePost = state0.reshape(8, 1)
            self.initialized  = True
        self.kf.predict()
        est = self.kf.correct(meas.reshape(4, 1))
        cx_s, cy_s, w_s, h_s = est[0, 0], est[1, 0], est[2, 0], est[3, 0]
        self.last = np.array([cx_s, cy_s, w_s, h_s], dtype=np.float32)
        return float(cx_s), float(cy_s), float(max(w_s, 1)), float(max(h_s, 1))

    def reset(self):
        self.initialized = False


# ============================================================
# MOUSE Controller
# ============================================================
class JoystickMouseController:
    """
    Controller a 'Joystick Virtuale' a 8 direzioni.
    Area di controllo grande, centrata in alto a destra.
    """
    def __init__(self, screen_w, screen_h, cam_w, cam_h, speed=20, deadzone=70):
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        # Centro del joystick: a destra (75% larghezza), in alto (20% altezza)
        self.anchor_x = int(cam_w * 0.75)
        self.anchor_y = int(cam_h * 0.20)
        
        self.speed = speed       # Velocità del cursore
        self.deadzone = deadzone # Zona morta ampia
        
        self.curr_x = screen_w / 2.0
        self.curr_y = screen_h / 2.0

    def update(self, cam_x, cam_y):
        dx = cam_x - self.anchor_x
        dy = cam_y - self.anchor_y
        dist = np.sqrt(dx**2 + dy**2)

        if dist < self.deadzone:
            return self.curr_x, self.curr_y

        # Calcola angolo e arrotonda a multipli di 45 gradi (8 direzioni)
        angle = np.arctan2(dy, dx)
        move_x = np.cos(np.round(angle / (np.pi/4)) * (np.pi/4))
        move_y = np.sin(np.round(angle / (np.pi/4)) * (np.pi/4))

        # Applica velocità
        self.curr_x += move_x * self.speed
        self.curr_y += move_y * self.speed

        self.curr_x = np.clip(self.curr_x, 0, self.screen_w)
        self.curr_y = np.clip(self.curr_y, 0, self.screen_h)

        return self.curr_x, self.curr_y
        
    def draw_active_area(self, frame):
        color = (255, 0, 255) # Magenta acceso
        thick = 4 # Spessore linea alto
        
        # Disegna la zona morta (cerchio ampio)
        cv2.circle(frame, (self.anchor_x, self.anchor_y), self.deadzone, color, thick)
        
        # Disegna la croce grande (40px per lato)
        cv2.line(frame, (self.anchor_x - 40, self.anchor_y), (self.anchor_x + 40, self.anchor_y), color, thick)
        cv2.line(frame, (self.anchor_x, self.anchor_y - 40), (self.anchor_x, self.anchor_y + 40), color, thick)
        
        # Etichetta
        cv2.putText(frame, "JOYSTICK", (self.anchor_x - 45, self.anchor_y - self.deadzone - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def reset(self):
        pass

# ============================================================
# WRAPPER MODELLI CLASSICI (RandomForest / SVM) -> vettore prob.
# ============================================================
def _proba_or_softmax(clf, X_row, n_classes):
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_row.reshape(1, -1))[0]
    else:
        dec = np.atleast_1d(clf.decision_function(X_row.reshape(1, -1))[0])
        exp = np.exp(dec - np.max(dec))
        proba = exp / exp.sum()
    full = np.zeros(n_classes, dtype=np.float32)
    classes_ = getattr(clf, "classes_", np.arange(n_classes))
    for cls_label, p in zip(classes_, proba):
        full[int(cls_label)] = p
    return full

class RandomForestBranch:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.clf = pickle.load(f)

    def predict(self, norm_landmarks_vec):
        return _proba_or_softmax(self.clf, norm_landmarks_vec, len(GESTURE_CLASSES))

class BoVWBranch:
    def __init__(self, vocab_path, svm_path):
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        with open(svm_path, "rb") as f:
            self.svm = pickle.load(f)
        self.vocab_size = (self.vocab.cluster_centers_.shape[0]
                            if hasattr(self.vocab, "cluster_centers_")
                            else self.vocab.n_clusters)
        self.sift  = cv2.SIFT_create()
        self.clahe = cv2.createCLAHE(clipLimit=SIFT_CLAHE_CLIP_LIMIT,
                                      tileGridSize=SIFT_CLAHE_TILE_GRID)

    def predict(self, crop_bgr):
        gray          = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray_enhanced = self.clahe.apply(gray)
        _, descriptors = self.sift.detectAndCompute(gray_enhanced, None)
        if descriptors is None or len(descriptors) == 0:
            return None
        visual_words = self.vocab.predict(descriptors.astype(np.float32))
        hist, _ = np.histogram(visual_words, bins=range(self.vocab_size + 1), density=True)
        return _proba_or_softmax(self.svm, hist.astype(np.float32), len(GESTURE_CLASSES))


# ============================================================
# THREAD YOLO
# ============================================================
yolo_lock        = threading.Lock()
yolo_frame_queue = queue.Queue(maxsize=1)
global_yolo_box  = None   

def yolo_worker_thread(yolo_model):
    global global_yolo_box
    while True:
        try:
            frame_to_process = yolo_frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        try:
            results = yolo_model.predict(frame_to_process, verbose=False, imgsz=YOLO_IMGSZ)
            boxes = results[0].boxes
            dominant_box, max_area = None, 0
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    dominant_box = box
            with yolo_lock:
                if dominant_box is not None:
                    x1, y1, x2, y2 = dominant_box.xyxy[0].tolist()
                    global_yolo_box = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                else:
                    global_yolo_box = None
        except Exception:
            pass


# ============================================================
# THREAD FLOW RGB 
# ============================================================
rgb_frame_queue = queue.Queue(maxsize=1)
score_lock      = threading.Lock()
global_score_rgb = None
clahe_processor  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def flow_rgb_worker_thread(models_dict, bovw_branch):
    global global_score_rgb
    while True:
        try:
            crop = rgb_frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        try:
            lab          = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe_processor.apply(lab[:, :, 0])
            hand_proc    = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            hand_r       = cv2.resize(hand_proc, IMG_SIZE)

            input_cnn    = np.expand_dims(hand_r / 255.0, axis=0)
            input_mobile = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.expand_dims(hand_r.astype(np.float32), axis=0)
            )

            prob_cnn    = models_dict["CNN"].predict_on_batch(input_cnn)[0]
            prob_mobile = models_dict["MOBILENET"].predict_on_batch(input_mobile)[0]
            prob_bovw   = bovw_branch.predict(crop)

            if prob_bovw is not None:
                score_rgb = (WEIGHT_CNN * prob_cnn +
                             WEIGHT_MOBILE * prob_mobile +
                             WEIGHT_BOVW * prob_bovw)
            else:
                denom = WEIGHT_CNN + WEIGHT_MOBILE
                score_rgb = (WEIGHT_CNN * prob_cnn + WEIGHT_MOBILE * prob_mobile) / denom

            with score_lock:
                global_score_rgb = score_rgb
        except Exception:
            pass


def fuse_scores(score_rgb, score_lm):
    if score_rgb is None and score_lm is None:
        return None, 0.0, None
    if score_rgb is None:
        fused = score_lm
    elif score_lm is None:
        fused = score_rgb
    else:
        fused = WEIGHT_FLOW_RGB * score_rgb + WEIGHT_FLOW_LM * score_lm
    cls_id = int(np.argmax(fused))
    return cls_id, float(fused[cls_id]), fused


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\nInizializzazione Modelli RGB...")
    loaded_models = {}
    for nome, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"ERRORE: modello mancante -> {path}"); exit()
        loaded_models[nome] = tf.keras.models.load_model(path)

    for path in [BOVW_VOCAB_PATH, BOVW_SVM_PATH]:
        if not os.path.exists(path):
            print(f"ERRORE: file mancante -> {path}"); exit()
            
    bovw_branch = BoVWBranch(BOVW_VOCAB_PATH, BOVW_SVM_PATH)
    threading.Thread(target=flow_rgb_worker_thread, args=(loaded_models, bovw_branch), daemon=True).start()

    print("Inizializzazione Modelli Landmarks...")
    for path in [LANDMARKS_MODEL_PATH, RF_MODEL_PATH]:
        if not os.path.exists(path):
            print(f"ERRORE: file mancante -> {path}"); exit()
            
    landmarks_model = tf.keras.models.load_model(LANDMARKS_MODEL_PATH)
    rf_branch = RandomForestBranch(RF_MODEL_PATH)

    print("Inizializzazione YOLO...")
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_tracker = YOLO(YOLO_MODEL_PATH)
    else:
        yolo_tracker = YOLO("yolov8n.pt")
        
    threading.Thread(target=yolo_worker_thread, args=(yolo_tracker,), daemon=True).start()

    model_path_mp = 'hand_landmarker.task'
    if not os.path.exists(model_path_mp):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            model_path_mp
        )
    base_options = python.BaseOptions(model_asset_path=model_path_mp)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        running_mode=vision.RunningMode.VIDEO
    )
    mp_detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    box_kalman = KalmanBoxTracker()
    mouse_ctrl = JoystickMouseController(SCREEN_W, SCREEN_H, cam_w, cam_h,speed=20, deadzone=40)

    last_click_time = 0
    click_cooldown  = 1.0
    scroll_accumulator = 0
    last_timestamp_ms  = 0
    prev_ty = None

    cur_mouse_x = float(SCREEN_W // 2)
    cur_mouse_y = float(SCREEN_H // 2)
    pyautogui.moveTo(int(cur_mouse_x), int(cur_mouse_y))

    print("\n" + "=" * 60)
    print("  VIRTUAL MOUSE PRONTO")
    print("  Premi 'Q' sulla finestra video per uscire")
    print("=" * 60 + "\n")

    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        frame_timestamp_ms = int(time.time() * 1000)
        if frame_timestamp_ms <= last_timestamp_ms:
            frame_timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = frame_timestamp_ms

        cur_gesture, cur_confidence, fused = None, 0.0, None

        # ---- YOLO async + Kalman: ROI stabilizzata ----
        try:
            yolo_frame_queue.put_nowait(frame)
        except queue.Full:
            pass
            
        with yolo_lock:
            yolo_box = global_yolo_box

        if yolo_box is not None:
            cx_raw, cy_raw, w_raw, h_raw = yolo_box
            cx, cy, w, h = box_kalman.update(cx_raw, cy_raw, w_raw, h_raw)

            pad, pad_bottom = 30, 70
            xmin = max(0, int(cx - w / 2 - pad))
            ymin = max(0, int(cy - h / 2 - pad))
            xmax = min(cam_w, int(cx + w / 2 + pad))
            ymax = min(cam_h, int(cy + h / 2 + pad + pad_bottom))
            
            cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

            hand_crop = frame[ymin:ymax, xmin:xmax]
            crop_h, crop_w = hand_crop.shape[:2]

            if crop_h > 0 and crop_w > 0:
                try:
                    rgb_frame_queue.put_nowait(hand_crop.copy())
                except queue.Full:
                    pass

                # ---- Flow Landmarks ----
                rgb_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                results_mp = mp_detector.detect_for_video(mp_image, frame_timestamp_ms)

                if results_mp.hand_landmarks:
                    landmarks = results_mp.hand_landmarks[0]
                    norm_vec  = normalize_landmarks(landmarks)

                    prob_land = landmarks_model.predict_on_batch(norm_vec.reshape(1, -1))[0]
                    prob_rf   = rf_branch.predict(norm_vec)
                    score_lm  = WEIGHT_LANDMARKS_KERAS * prob_land + WEIGHT_RANDOMFOREST * prob_rf

                    tip_lm = landmarks[8]
                    tx = int(xmin + tip_lm.x * crop_w)
                    ty = int(ymin + tip_lm.y * crop_h)

                    with score_lock:
                        score_rgb = global_score_rgb

                    cls_id, cur_confidence, fused = fuse_scores(score_rgb, score_lm)
                    cur_gesture = GESTURE_CLASSES[cls_id] if cls_id is not None else None
                    color = (0, 255, 0) if cur_gesture == "index" else (200, 200, 0)

                    # -- Movimento: controllato dal classificatore fuso --
                    if cur_gesture == "index" and cur_confidence > GESTURE_CONFIDENCE_THRESHOLD:
                        cur_mouse_x, cur_mouse_y = mouse_ctrl.update(tx, ty)
                        pyautogui.moveTo(int(cur_mouse_x), int(cur_mouse_y))
                    else:
                        mouse_ctrl.update(tx, ty)

                    # -- Azioni --
                    if cur_gesture and cur_confidence > GESTURE_CONFIDENCE_THRESHOLD:
                        if cur_gesture == "open_palm":
                            if (time.time() - last_click_time) > click_cooldown:
                                pyautogui.click()
                                last_click_time = time.time()
                                
                        elif cur_gesture == "pinch":
                            if prev_ty is not None:
                                scroll_accumulator += (ty - prev_ty)
                            if scroll_accumulator < -8:
                                pyautogui.scroll(100)
                                scroll_accumulator = 0
                            elif scroll_accumulator > 8:
                                pyautogui.scroll(-10)
                                scroll_accumulator = 0
                            cv2.putText(display_frame, "SCROLLING...", (xmin, ymin - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
                        elif cur_gesture == "fist":
                            if (time.time() - last_click_time) > click_cooldown:
                                pyautogui.rightClick()
                                last_click_time = time.time()
                                
                        elif cur_gesture == "two_fingers":
                            if (time.time() - last_click_time) > click_cooldown:
                                pyautogui.click(clicks=2, interval=0.1)
                                last_click_time = time.time()
                                
                    if cur_gesture != "pinch":
                        scroll_accumulator = 0
                        
                    prev_ty = ty

                    cv2.circle(display_frame, (tx, ty), 7, (0, 220, 255), -1)

                    if fused is not None:
                        for ci, cls in enumerate(GESTURE_CLASSES):
                            bar_w = int(fused[ci] * 100)
                            cv2.rectangle(display_frame, (cam_w - 115, 40 + ci * 22), (cam_w - 115 + bar_w, 56 + ci * 22), (0, 200, 100), -1)
                            cv2.putText(display_frame, cls[:6], (cam_w - 115, 54 + ci * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

                    cv2.putText(display_frame, f"{cur_gesture}  {cur_confidence * 100:.0f}%", (xmin, max(20, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        else:
            box_kalman.reset()
            mouse_ctrl.reset()
            prev_ty = None
            scroll_accumulator = 0

        fps = 1.0 / (time.time() - t0 + 1e-9)
        with score_lock:
            sr_dbg = global_score_rgb
            
        sr_str = f"RGB:{sr_dbg[np.argmax(sr_dbg)] * 100:.0f}%" if sr_dbg is not None else "RGB:--"
        cv2.putText(display_frame, f"FPS:{int(fps)}  {sr_str}  Fused:{cur_gesture or '---'} {cur_confidence * 100:.0f}%  Q=esci", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        legend = [
            ("index", "Muovi cursore", (0, 220, 255)),
            ("open_palm", "Click sin.", (0, 255, 0)),
            ("two_fingers", "Doppio click", (100, 255, 100)),
            ("fist", "Click des.", (0, 140, 255)),
            ("pinch", "Scroll", (255, 200, 0)),
        ]
        
        for i, (g, desc, col) in enumerate(legend):
            active = (cur_gesture == g)
            cv2.putText(display_frame, f"  {g}: {desc}", (8, cam_h - 14 - i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col if active else (80, 80, 80), 2 if active else 1)

        mouse_ctrl.draw_active_area(display_frame)
        cv2.imshow("Virtual Mouse", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
