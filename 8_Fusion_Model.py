"""
==============================
Pipeline real-time a DUE FLUSSI, con tutti i modelli già addestrati e
caricati da disco (nessun training a runtime).

  FRAME
    |
    v
  YOLO  ------------------------------------------------------> bbox mano
    |
    v
  Kalman Box Tracker (stabilizza cx,cy,w,h del box YOLO)
    |
    v
  hand_crop  (ROI stabile, ~80% di calcolo risparmiato sul resto)
    |
    +----------------------------------+----------------------------------+
    |                                  |
    v                                  v
  FLOW RGB (thread separato)        FLOW LANDMARKS (sincrono, leggero)
  --------------------------        -----------------------------------
  hand_crop -> CNN(keras)           hand_crop -> MediaPipe -> 21 landmark
  hand_crop -> MobileNet(keras)          |
  hand_crop -> gray+CLAHE -> SIFT        v
       -> istogramma BoVW        normalize_landmarks() (x,y,z normalizzati
       -> SVM(svm.pkl)            rispetto al polso e scalati)
       |                                 |
       v                    +------------+------------+
  Score_RGB =                |                         |
   w1*CNN + w2*MobileNet     v                         v
   + w3*BoVW-SVM      Landmarks(keras)            RandomForest
                              |                         |
                              +---- Score_LM = a*Land + b*RF
    |                                  |
    +----------------------------------+
                    |
                    v
     Score_Finale = ALPHA*Score_RGB + BETA*Score_LM   (Score-Level Fusion)
                    |
                    v
              classe finale -> comando mouse
       (cursore = punta indice MediaPipe, ancorata al box
        stabilizzato YOLO+Kalman)

NOTA sul ramo BoVW: nel training, il vocabolario (MiniBatchKMeans,
128-dim = descrittori SIFT) e l'istogramma sono costruiti PER SINGOLA
IMMAGINE (non su una finestra temporale di più frame):
  gray -> CLAHE(2.0, 8x8) -> SIFT.detectAndCompute -> descrittori (N,128)
  -> kmeans.predict(descrittori) -> np.histogram(..., density=True)
Qui in real-time viene rifatto esattamente lo stesso identico procedimento
sul crop del frame corrente.

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
from collections import deque

# ============================================================
# CONFIGURAZIONE — aggiusta i percorsi ai tuoi file reali
# ============================================================
GESTURE_CLASSES = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']
IMG_SIZE        = (224, 224)

# ---- Flow RGB (deep + classico, sul crop) ----------------------
MODEL_PATHS = {
    "CNN"      : "model_name_cnn.keras",
    "MOBILENET": "model_name_mobilenet.keras",
}
BOVW_VOCAB_PATH = "model_name_bovw_vocab.pkl"   # MiniBatchKMeans (128-dim, descrittori SIFT)
BOVW_SVM_PATH   = "model_name_svm.pkl"                     # SVM addestrata sugli istogrammi BoVW
SIFT_CLAHE_CLIP_LIMIT = 2.0
SIFT_CLAHE_TILE_GRID  = (8, 8)

WEIGHT_CNN    = 0.40
WEIGHT_MOBILE = 0.30
WEIGHT_BOVW   = 0.30  

# ---- Flow Landmarks (deep + classico, sui 21 keypoint) ---------
LANDMARKS_MODEL_PATH = "model_name_landmarks.keras"
RF_MODEL_PATH         = "model_name_random_forest.pkl"

WEIGHT_LANDMARKS_KERAS = 0.50
WEIGHT_RANDOMFOREST    = 0.50

# ---- YOLO (detection mano) -------------------------------------
YOLO_MODEL_PATH = "runs/detect/train/weights/best.pt"

# ---- Score-Level Fusion finale -----------------------------------
WEIGHT_FLOW_RGB = 0.40
WEIGHT_FLOW_LM  = 0.60

GESTURE_CONFIDENCE_THRESHOLD = 0.60
REID_INTERVAL = 30

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0
SCREEN_W, SCREEN_H = pyautogui.size()


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


class RelativeVelocityMouse:
    def __init__(self, sensitivity=2.5, smoothing=0.35, dead_zone_px=2, max_delta_px=40):
        self.sensitivity  = sensitivity
        self.smoothing    = smoothing
        self.dead_zone_px = dead_zone_px
        self.max_delta_px = max_delta_px
        self.prev_x = self.prev_y = None
        self.smooth_dx = self.smooth_dy = 0.0

    def update(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return 0.0, 0.0
        raw_dx = x - self.prev_x
        raw_dy = y - self.prev_y
        self.prev_x, self.prev_y = x, y
        if abs(raw_dx) > self.max_delta_px or abs(raw_dy) > self.max_delta_px:
            self.smooth_dx = self.smooth_dy = 0.0
            return 0.0, 0.0
        if (raw_dx**2 + raw_dy**2)**0.5 < self.dead_zone_px:
            self.smooth_dx *= (1 - self.smoothing)
            self.smooth_dy *= (1 - self.smoothing)
            if abs(self.smooth_dx) < 0.1 and abs(self.smooth_dy) < 0.1:
                return 0.0, 0.0
            return self.smooth_dx * self.sensitivity, self.smooth_dy * self.sensitivity
        self.smooth_dx = self.smoothing * raw_dx + (1 - self.smoothing) * self.smooth_dx
        self.smooth_dy = self.smoothing * raw_dy + (1 - self.smoothing) * self.smooth_dy
        return self.smooth_dx * self.sensitivity, self.smooth_dy * self.sensitivity

    def reset(self):
        self.prev_x = self.prev_y = None
        self.smooth_dx = self.smooth_dy = 0.0


def _proba_or_softmax(clf, X_row, n_classes):
    """Restituisce un vettore di probabilità lungo n_classes, sia che
    il classificatore supporti predict_proba, sia che esponga solo
    decision_function (es. SVC senza probability=True)."""
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
        print(f"[FlowLM] RandomForest caricato da {path}")

    def predict(self, norm_landmarks_vec):
        return _proba_or_softmax(self.clf, norm_landmarks_vec, len(GESTURE_CLASSES))


class BoVWBranch:
    """
    Riproduce ESATTAMENTE la pipeline di training:
        gray = cvtColor(img, BGR2GRAY)
        gray_enhanced = CLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        keypoints, descriptors = sift.detectAndCompute(gray_enhanced, None)
        visual_words = kmeans.predict(descriptors)              # (N,) int
        istogramma   = np.histogram(visual_words,
                                     bins=range(VOCAB_SIZE+1),
                                     density=True)[0]
    L'istogramma è calcolato per singolo frame (non su una finestra
    temporale), esattamente come una singola immagine nel training set.
    """
    def __init__(self, vocab_path, svm_path):
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)   # MiniBatchKMeans, 128-dim (SIFT)
        with open(svm_path, "rb") as f:
            self.svm = pickle.load(f)
        self.vocab_size = (self.vocab.cluster_centers_.shape[0]
                            if hasattr(self.vocab, "cluster_centers_")
                            else self.vocab.n_clusters)
        self.sift  = cv2.SIFT_create()
        self.clahe = cv2.createCLAHE(clipLimit=SIFT_CLAHE_CLIP_LIMIT,
                                      tileGridSize=SIFT_CLAHE_TILE_GRID)
        print(f"[FlowRGB] Vocabolario BoVW ({self.vocab_size} parole, SIFT 128-dim) "
              f"e SVM caricati.")

    def predict(self, crop_bgr):
        """Ritorna il vettore di probabilità, oppure None se SIFT non
        trova nessun keypoint nel crop (es. mano molto uniforme/sfocata)."""
        gray          = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray_enhanced = self.clahe.apply(gray)
        _, descriptors = self.sift.detectAndCompute(gray_enhanced, None)

        if descriptors is None or len(descriptors) == 0:
            return None

        visual_words = self.vocab.predict(descriptors.astype(np.float32))
        hist, _ = np.histogram(visual_words, bins=range(self.vocab_size + 1),
                                density=True)
        return _proba_or_softmax(self.svm, hist.astype(np.float32), len(GESTURE_CLASSES))


# ============================================================
# STATO CONDIVISO tra main thread e worker Flow RGB
# ============================================================
frame_queue  = queue.Queue(maxsize=1)
gesture_lock = threading.Lock()

global_score_rgb        = None
global_valid_identity    = True
hand_features_template   = None

clahe_processor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def extract_appearance_features(crop_img, feature_extractor_model):
    try:
        img_resized = cv2.resize(crop_img, IMG_SIZE)
        img_ready   = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.expand_dims(img_resized.astype(np.float32), axis=0)
        )
        features = feature_extractor_model.predict_on_batch(img_ready)[0]
        return features / (np.linalg.norm(features) + 1e-8)
    except Exception:
        return None


# ============================================================
# WORKER THREAD — FLOW RGB (CNN + MobileNet + BoVW/SIFT+SVM)
# ============================================================
def flow_rgb_worker_thread(models_dict, bovw_branch):
    global global_score_rgb, global_valid_identity, hand_features_template

    try:
        feature_extractor = tf.keras.Model(
            inputs  = models_dict["MOBILENET"].input,
            outputs = models_dict["MOBILENET"].layers[-2].output
        )
    except Exception:
        feature_extractor = models_dict["MOBILENET"]

    frame_counter = 0

    while True:
        try:
            crop_to_process = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        try:
            frame_counter += 1

            # ── Re-ID periodico ──
            if frame_counter % REID_INTERVAL == 0:
                current_features = extract_appearance_features(crop_to_process, feature_extractor)
                if current_features is not None:
                    if hand_features_template is None:
                        hand_features_template = current_features
                    else:
                        cosine_sim = np.dot(current_features, hand_features_template)
                        with gesture_lock:
                            global_valid_identity = cosine_sim >= 0.50
                        if cosine_sim < 0.50:
                            continue

            lab          = cv2.cvtColor(crop_to_process, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe_processor.apply(lab[:, :, 0])
            hand_proc    = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            hand_r       = cv2.resize(hand_proc, IMG_SIZE)

            input_cnn    = np.expand_dims(hand_r / 255.0, axis=0)
            input_mobile = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.expand_dims(hand_r.astype(np.float32), axis=0)
            )

            prob_cnn    = models_dict["CNN"].predict_on_batch(input_cnn)[0]
            prob_mobile = models_dict["MOBILENET"].predict_on_batch(input_mobile)[0]

            # BoVW/SIFT lavora sul crop originale (non sul preprocessing CLAHE-LAB
            # usato per CNN/MobileNet), replicando esattamente il training.
            prob_bovw = bovw_branch.predict(crop_to_process)

            if prob_bovw is not None:
                score_rgb = (WEIGHT_CNN * prob_cnn +
                             WEIGHT_MOBILE * prob_mobile +
                             WEIGHT_BOVW * prob_bovw)
            else:
                # SIFT non ha trovato keypoint: ridistribuisco il peso su CNN/MobileNet
                denom = WEIGHT_CNN + WEIGHT_MOBILE
                score_rgb = (WEIGHT_CNN * prob_cnn + WEIGHT_MOBILE * prob_mobile) / denom

            with gesture_lock:
                global_score_rgb = score_rgb

        except Exception:
            pass


# ============================================================
# SCORE-LEVEL FUSION
# ============================================================
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

    # ---- Flow RGB: CNN + MobileNet + BoVW/SIFT+SVM ----
    print("\n[Flow RGB] Caricamento modelli...")
    loaded_models = {}
    for nome, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"ERRORE: modello mancante -> {path}")
            exit()
        loaded_models[nome] = tf.keras.models.load_model(path)
        print(f"  -> {nome} caricato.")

    for path in [BOVW_VOCAB_PATH, BOVW_SVM_PATH]:
        if not os.path.exists(path):
            print(f"ERRORE: file mancante -> {path}")
            exit()
    bovw_branch = BoVWBranch(BOVW_VOCAB_PATH, BOVW_SVM_PATH)

    ai_thread = threading.Thread(
        target=flow_rgb_worker_thread,
        args=(loaded_models, bovw_branch),
        daemon=True
    )
    ai_thread.start()

    # ---- Flow Landmarks: keras + RandomForest ----
    print("\n[Flow Landmarks] Caricamento modelli...")
    for path in [LANDMARKS_MODEL_PATH, RF_MODEL_PATH]:
        if not os.path.exists(path):
            print(f"ERRORE: file mancante -> {path}")
            exit()

    landmarks_model = tf.keras.models.load_model(LANDMARKS_MODEL_PATH)
    print(f"  -> Landmarks (keras) caricato.")
    rf_branch = RandomForestBranch(RF_MODEL_PATH)

    # ---- YOLO ----
    print("\n[YOLO] Inizializzazione detector mano...")
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_tracker = YOLO(YOLO_MODEL_PATH)
        print(f"  -> YOLO caricato ({YOLO_MODEL_PATH})")
    else:
        yolo_tracker = YOLO("yolov8n.pt")
        print(f"  -> ATTENZIONE: {YOLO_MODEL_PATH} non trovato, uso pesi YOLO standard!")

    # ---- MediaPipe (solo estrazione landmark sul crop) ----
    model_path_mp = 'hand_landmarker.task'
    if not os.path.exists(model_path_mp):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            model_path_mp
        )
    base_options = python.BaseOptions(model_asset_path=model_path_mp)
    options = vision.HandLandmarkerOptions(
        base_options                  = base_options,
        num_hands                     = 1,
        min_hand_detection_confidence = 0.4,
        min_hand_presence_confidence  = 0.4,
        min_tracking_confidence       = 0.4,
        running_mode = vision.RunningMode.VIDEO
    )
    mp_detector = vision.HandLandmarker.create_from_options(options)

    # ---- Webcam ----
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    box_kalman = KalmanBoxTracker()
    velocity   = RelativeVelocityMouse(sensitivity=2.5, smoothing=0.35,
                                        dead_zone_px=2, max_delta_px=40)

    last_click_time    = 0
    click_cooldown     = 0.5
    scroll_accumulator = 0
    last_timestamp_ms  = 0
    prev_smooth_x = prev_smooth_y = None

    cur_mouse_x = float(SCREEN_W // 2)
    cur_mouse_y = float(SCREEN_H // 2)
    pyautogui.moveTo(int(cur_mouse_x), int(cur_mouse_y))

    print("\n" + "=" * 60)
    print("  VIRTUAL MOUSE PRONTO — Two-Flow (RGB deep | Landmarks classici)")
    print("  Q = esci")
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

        # ========================================================
        # YOLO + KALMAN — ROI stabilizzata
        # ========================================================
        results_yolo = yolo_tracker.predict(frame, verbose=False)
        boxes = results_yolo[0].boxes

        dominant_box, max_area = None, 0
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                dominant_box = box

        if dominant_box is not None:
            x1, y1, x2, y2 = dominant_box.xyxy[0].tolist()
            cx_raw, cy_raw = (x1 + x2) / 2, (y1 + y2) / 2
            w_raw,  h_raw  = (x2 - x1), (y2 - y1)

            cx, cy, w, h = box_kalman.update(cx_raw, cy_raw, w_raw, h_raw)

            pad = 30
            xmin = max(0, int(cx - w / 2 - pad))
            ymin = max(0, int(cy - h / 2 - pad))
            xmax = min(cam_w, int(cx + w / 2 + pad))
            ymax = min(cam_h, int(cy + h / 2 + pad))

            cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

            hand_crop = frame[ymin:ymax, xmin:xmax]
            crop_h, crop_w = hand_crop.shape[:2]

            if crop_h > 0 and crop_w > 0:
                # ---- FLOW RGB: invia il crop al worker in background ----
                try:
                    frame_queue.put_nowait(hand_crop.copy())
                except queue.Full:
                    pass

                # ========================================================
                # FLOW LANDMARKS — MediaPipe + Landmarks(keras) + RF + BoVW/SVM
                # ========================================================
                rgb_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                results_mp = mp_detector.detect_for_video(mp_image, frame_timestamp_ms)

                score_lm = None
                if results_mp.hand_landmarks:
                    landmarks = results_mp.hand_landmarks[0]
                    norm_vec  = normalize_landmarks(landmarks)

                    prob_land = landmarks_model.predict_on_batch(
                        norm_vec.reshape(1, -1)
                    )[0]
                    prob_rf   = rf_branch.predict(norm_vec)

                    score_lm = (WEIGHT_LANDMARKS_KERAS * prob_land +
                                WEIGHT_RANDOMFOREST    * prob_rf)

                    tip_lm = landmarks[8]
                    tx = int(xmin + tip_lm.x * crop_w)
                    ty = int(ymin + tip_lm.y * crop_h)

                    with gesture_lock:
                        score_rgb = global_score_rgb
                        cur_valid = global_valid_identity

                    cls_id, cur_confidence, fused = fuse_scores(score_rgb, score_lm)
                    cur_gesture = GESTURE_CLASSES[cls_id] if cls_id is not None else None

                    if cur_valid:
                        color = (0, 255, 0) if cur_gesture == "index" else (200, 200, 0)

                        if cur_gesture == "index" and cur_confidence > GESTURE_CONFIDENCE_THRESHOLD:
                            dx, dy = velocity.update(tx, ty)
                            cur_mouse_x = float(np.clip(cur_mouse_x + dx, 0, SCREEN_W - 1))
                            cur_mouse_y = float(np.clip(cur_mouse_y + dy, 0, SCREEN_H - 1))
                            pyautogui.moveTo(int(cur_mouse_x), int(cur_mouse_y))
                        else:
                            velocity.update(tx, ty)

                        if cur_gesture and cur_confidence > GESTURE_CONFIDENCE_THRESHOLD:
                            if cur_gesture == "open_palm":
                                if (time.time() - last_click_time) > click_cooldown:
                                    pyautogui.click()
                                    last_click_time = time.time()
                            elif cur_gesture == "pinch":
                                if prev_smooth_y is not None:
                                    scroll_accumulator += (ty - prev_smooth_y)
                                if scroll_accumulator < -15:
                                    pyautogui.scroll(10);  scroll_accumulator = 0
                                elif scroll_accumulator > 15:
                                    pyautogui.scroll(-10); scroll_accumulator = 0
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
                        prev_smooth_x, prev_smooth_y = tx, ty

                        cv2.circle(display_frame, (tx, ty), 7, (0, 220, 255), -1)

                        if fused is not None:
                            for ci, cls in enumerate(GESTURE_CLASSES):
                                bar_w = int(fused[ci] * 100)
                                cv2.rectangle(display_frame,
                                              (cam_w - 115, 40 + ci * 22),
                                              (cam_w - 115 + bar_w, 56 + ci * 22),
                                              (0, 200, 100), -1)
                                cv2.putText(display_frame, cls[:6],
                                            (cam_w - 115, 54 + ci * 22),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                            (255, 255, 255), 1)

                        if cur_gesture:
                            cv2.putText(display_frame,
                                        f"{cur_gesture}  {cur_confidence * 100:.0f}%",
                                        (xmin, max(20, ymin - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                    else:
                        cv2.putText(display_frame, "RE-ID BLOCCATO",
                                    (xmin, max(20, ymin - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            box_kalman.reset()
            velocity.reset()
            prev_smooth_x = prev_smooth_y = None
            scroll_accumulator = 0

        fps = 1.0 / (time.time() - t0 + 1e-9)
        with gesture_lock:
            sr_dbg = global_score_rgb
        sr_str = f"RGB:{sr_dbg[np.argmax(sr_dbg)] * 100:.0f}%" if sr_dbg is not None else "RGB:--"
        cv2.putText(display_frame,
                    f"FPS:{int(fps)}  {sr_str}  "
                    f"Fused:{cur_gesture or '---'} {cur_confidence * 100:.0f}%  Q=esci",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        legend = [
            ("index",       "Muovi cursore", (0, 220, 255)),
            ("open_palm",   "Click sin.",    (0, 255, 0)),
            ("two_fingers", "Doppio click",  (100, 255, 100)),
            ("fist",        "Click des.",    (0, 140, 255)),
            ("pinch",       "Scroll",        (255, 200, 0)),
        ]
        for i, (g, desc, col) in enumerate(legend):
            active = (cur_gesture == g)
            cv2.putText(display_frame, f"  {g}: {desc}",
                        (8, cam_h - 14 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        col if active else (80, 80, 80),
                        2 if active else 1)

        cv2.imshow("Virtual Mouse v9 — Two-Flow (RGB + Landmarks)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()