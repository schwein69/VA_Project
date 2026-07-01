"""
virtual_mouse_v7_tig_hog_bow.py
================================
Implementazione della pipeline "Classificazione delle azioni" (slide):

  ┌─────────────────────────────────────────────────────────┐
  │  RGB branch        │  Skeleton branch                   │
  │                    │                                     │
  │  Frame buffer      │  MediaPipe landmarks               │
  │       ↓            │       ↓                            │
  │  TIG (Temporal     │  BOW encoding                      │
  │  Image Gradient)   │  (landmark → vocabolario)          │
  │       ↓            │       ↓                            │
  │  HOG encoding      │  Ensemble CNN+MobileNet+Landmarks  │
  │       ↓            │  (i tuoi modelli)                  │
  │  SVM lineare       │       ↓                            │
  │  (per classe)      │  score S_SK per classe             │
  │       ↓            │                                     │
  │  score S_F         │                                     │
  └────────┬───────────┴──────────────┬──────────────────── ┘
           │    Score-level fusion    │
           └──────────┬───────────────┘
                      ↓
               classe finale

Differenze rispetto alla v5:
  - Branch RGB aggiunto: TIG su buffer di N frame → HOG → SVM multiclasse
  - Branch Skeleton: landmarks MediaPipe → BOW → ensemble (come v5)
  - Fusione: weighted sum di S_F e S_SK (configurabile)
  - SVM viene addestrato online al primo avvio su campioni bootstrap,
    oppure caricato da file se già addestrato (SVM_PATH)

Controlli:
  Q → esci
  T → toggle: mostra/nascondi TIG frame in overlay
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
import threading
import queue
import pickle
from collections import deque
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# CONFIGURAZIONE
# ============================================================
GESTURE_CLASSES = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']
IMG_SIZE        = (224, 224)

MODEL_PATHS = {
    "CNN"      : "model_name_cnn.keras",
    "MOBILENET": "model_name_mobilenet.keras",
    "LANDMARKS": "model_name_landmarks.keras"
}

WEIGHT_CNN       = 0.30
WEIGHT_MOBILE    = 0.30
WEIGHT_LANDMARKS = 0.40

# Pesi fusione branch finale (slide: somma S_F + S_SK)
WEIGHT_RGB_BRANCH = 0.40   # peso branch RGB (TIG+HOG+SVM)
WEIGHT_SKE_BRANCH = 0.60   # peso branch Skeleton (MediaPipe+ensemble)

GESTURE_CONFIDENCE_THRESHOLD = 0.55
REID_INTERVAL = 30

# ── TIG (Temporal Image Gradient) ──────────────────────────
# La slide usa M TIG su una sequenza di frame.
# TIG_i = |frame_t - frame_{t-1}| normalizzato.
# Qui usiamo un buffer di TIG_BUFFER_SIZE frame per costruire
# la sequenza TIG da dare a HOG.
TIG_BUFFER_SIZE = 8     # numero di frame nella finestra temporale
TIG_RESIZE      = (64, 64)  # dimensione su cui calcolare TIG+HOG

# ── HOG ────────────────────────────────────────────────────
HOG_WIN_SIZE    = (64, 64)
HOG_BLOCK_SIZE  = (16, 16)
HOG_BLOCK_STRIDE= (8, 8)
HOG_CELL_SIZE   = (8, 8)
HOG_NBINS       = 9

# ── BOW (Bag of Words sui landmark) ────────────────────────
# Il vocabolario BOW è costruito quantizzando i vettori landmark
# in BOW_VOCAB_SIZE cluster. Ogni frame landmark diventa un
# istogramma di frequenze sul vocabolario.
BOW_VOCAB_SIZE  = 50   # numero di "visual words" per landmark

# ── SVM ────────────────────────────────────────────────────
SVM_PATH = "svm_rgb_branch.pkl"   # salva/carica SVM addestrato

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# ============================================================
# 0. KALMAN FILTER
# ============================================================
class KalmanHandTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.last_x = self.last_y = 0

    def update(self, x, y):
        if self.initialized:
            if abs(x - self.last_x) + abs(y - self.last_y) > 150:
                self.initialized = False
        if not self.initialized:
            self.kf.statePre  = np.array([[x],[y],[0],[0]], np.float32)
            self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
            self.initialized  = True
        self.kf.predict()
        est = self.kf.correct(np.array([[np.float32(x)],[np.float32(y)]]))
        sx, sy = int(est[0,0]), int(est[1,0])
        self.last_x, self.last_y = sx, sy
        return sx, sy

    def reset(self):
        self.initialized = False


# ============================================================
# 1. VELOCITY CONTROLLER (identico v5)
# ============================================================
class RelativeVelocityMouse:
    def __init__(self, sensitivity=2.5, smoothing=0.35,
                 dead_zone_px=2, max_delta_px=40):
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


# ============================================================
# 2. RGB BRANCH — TIG + HOG + SVM
# ============================================================
class TIGHOGExtractor:
    """
    Temporal Image Gradient (TIG) + HOG encoding.

    Come nella slide:
      - Mantieni un buffer di M frame grigi ridimensionati
      - TIG_i = |frame_i - frame_{i-1}|   per i = 1..M-1
      - Ogni TIG_i viene descritto con HOG
      - I descrittori HOG vengono concatenati → feature vector fisso

    Lunghezza del vettore finale:
      hog_len * (TIG_BUFFER_SIZE - 1)
    """
    def __init__(self):
        self.hog = cv2.HOGDescriptor(
            HOG_WIN_SIZE,
            HOG_BLOCK_SIZE,
            HOG_BLOCK_STRIDE,
            HOG_CELL_SIZE,
            HOG_NBINS
        )
        self.frame_buffer = deque(maxlen=TIG_BUFFER_SIZE)
        # Lunghezza descrittore HOG per un singolo TIG
        dummy = np.zeros(HOG_WIN_SIZE, dtype=np.uint8)
        self._hog_len = len(self.hog.compute(dummy))

    @property
    def feature_size(self):
        return self._hog_len * (TIG_BUFFER_SIZE - 1)

    def push_frame(self, frame_bgr):
        """Aggiunge un frame al buffer (convertito in grigio e ridimensionato)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, TIG_RESIZE)
        self.frame_buffer.append(gray.astype(np.float32))

    def compute(self):
        """
        Calcola il vettore TIG+HOG dal buffer corrente.
        Ritorna None se il buffer non è ancora pieno.
        """
        if len(self.frame_buffer) < TIG_BUFFER_SIZE:
            return None

        buf = list(self.frame_buffer)
        hog_features = []

        for i in range(1, len(buf)):
            # TIG: gradiente temporale = differenza assoluta tra frame consecutivi
            tig = np.abs(buf[i] - buf[i - 1]).astype(np.uint8)
            # HOG sul TIG
            h = self.hog.compute(tig).flatten()
            hog_features.append(h)

        # Concatenazione di tutti i descrittori HOG → vettore fisso
        return np.concatenate(hog_features)

    def get_last_tig(self):
        """Ritorna l'ultimo TIG (per visualizzazione)."""
        if len(self.frame_buffer) < 2:
            return None
        buf = list(self.frame_buffer)
        tig = np.abs(buf[-1] - buf[-2]).astype(np.uint8)
        return tig

    def reset(self):
        self.frame_buffer.clear()


class SVMClassifier:
    """
    LinearSVC addestrato sui descrittori TIG+HOG.

    Come nella slide: un SVM per ogni porzione del volume 3D
    della sequenza (qui semplificato: un SVM multiclasse unico
    con score-level fusion implicita nei decision_function scores).

    Se il file SVM_PATH esiste, lo carica.
    Altrimenti viene addestrato online durante la sessione
    (richiede che l'utente etichetti qualche frame con i tasti 1-5).
    """
    def __init__(self, n_classes, feature_size, svm_path=SVM_PATH):
        self.n_classes    = n_classes
        self.feature_size = feature_size
        self.svm_path     = svm_path
        self.pipeline     = None
        self.trained      = False

        # Buffer per raccolta campioni online
        self._X_buffer = []
        self._y_buffer = []

        if os.path.exists(svm_path):
            self._load()

    def _load(self):
        with open(self.svm_path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.trained = True
        print(f"[SVM] Caricato da {self.svm_path}")

    def save(self):
        with open(self.svm_path, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"[SVM] Salvato in {self.svm_path}")

    def add_sample(self, feature_vec, label_idx):
        """Aggiunge un campione al buffer di training online."""
        self._X_buffer.append(feature_vec)
        self._y_buffer.append(label_idx)
        print(f"[SVM] Campione aggiunto: {GESTURE_CLASSES[label_idx]} "
              f"(totale: {len(self._y_buffer)})")

    def fit(self):
        """Addestra l'SVM sui campioni raccolti."""
        if len(self._y_buffer) < self.n_classes:
            print(f"[SVM] Servono almeno {self.n_classes} campioni "
                  f"(hai {len(self._y_buffer)})")
            return False
        X = np.array(self._X_buffer)
        y = np.array(self._y_buffer)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    LinearSVC(max_iter=2000, C=1.0))
        ])
        self.pipeline.fit(X, y)
        self.trained = True
        self.save()
        print(f"[SVM] Addestrato su {len(y)} campioni.")
        return True

    def predict_scores(self, feature_vec):
        """
        Ritorna uno score per ogni classe (decision_function di LinearSVC).
        I valori sono softmax-normalizzati per essere comparabili con
        gli score del branch Skeleton.
        """
        if not self.trained or self.pipeline is None:
            return np.ones(self.n_classes) / self.n_classes  # uniforme

        feat = feature_vec.reshape(1, -1)
        raw  = self.pipeline.decision_function(feat)[0]

        # Softmax per portare gli score in [0,1] sommando a 1
        raw_shifted = raw - np.max(raw)
        exp_raw     = np.exp(raw_shifted)
        scores      = exp_raw / (np.sum(exp_raw) + 1e-10)
        return scores


# ============================================================
# 3. SKELETON BRANCH — BOW + Ensemble (i tuoi modelli)
# ============================================================
class BOWLandmarkEncoder:
    """
    Bag of Words encoding sui landmark MediaPipe.

    Come nella slide (BOW dictionary sul lato Skeleton):
      - Ogni frame landmark è un vettore di 63 valori (21 punti × 3)
      - Il vocabolario BOW è costruito con K-Means su vettori landmark
      - Ogni frame landmark viene quantizzato → indice della visual word più vicina
      - Su una finestra di frame → istogramma di frequenze → descrittore BOW

    In questo contesto real-time usiamo una finestra scorrevole di
    BOW_BUFFER_SIZE landmark per costruire l'istogramma.
    """
    def __init__(self, vocab_size=BOW_VOCAB_SIZE, buffer_size=8):
        self.vocab_size  = vocab_size
        self.buffer_size = buffer_size
        self.vocabulary  = None   # centri K-Means (vocab_size × 63)
        self.lm_buffer   = deque(maxlen=buffer_size)
        self._collect    = []     # campioni per costruire vocabolario

    def push_landmarks(self, landmarks):
        """Aggiunge un vettore landmark al buffer."""
        vec = []
        for lm in landmarks:
            vec.extend([lm.x, lm.y, lm.z])
        arr = np.array(vec, dtype=np.float32)
        self.lm_buffer.append(arr)
        self._collect.append(arr)

    def build_vocabulary(self):
        """Costruisce il vocabolario BOW con K-Means sui landmark raccolti."""
        if len(self._collect) < self.vocab_size:
            print(f"[BOW] Servono più campioni landmark "
                  f"({len(self._collect)}/{self.vocab_size})")
            return False
        data = np.array(self._collect, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, _, centers = cv2.kmeans(
            data, self.vocab_size, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        self.vocabulary = centers
        print(f"[BOW] Vocabolario costruito: {self.vocab_size} visual words")
        return True

    def compute_histogram(self):
        """
        Calcola l'istogramma BOW dal buffer corrente.
        Ritorna None se il vocabolario non è ancora pronto.
        """
        if self.vocabulary is None or len(self.lm_buffer) == 0:
            return None

        hist = np.zeros(self.vocab_size, dtype=np.float32)
        for vec in self.lm_buffer:
            # Distanza dal vettore landmark a ogni visual word
            dists = np.linalg.norm(self.vocabulary - vec, axis=1)
            nearest = np.argmin(dists)
            hist[nearest] += 1.0

        # Normalizza L1
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def reset(self):
        self.lm_buffer.clear()


# ============================================================
# 4. STATO CONDIVISO
# ============================================================
frame_queue  = queue.Queue(maxsize=1)
gesture_lock = threading.Lock()

# Score di entrambi i branch (array per classe)
global_score_skeleton = None   # S_SK: score branch skeleton
global_score_rgb      = None   # S_F:  score branch RGB
global_gesture        = None
global_confidence     = 0.0
global_valid_identity = True
hand_features_template = None

clahe_processor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ============================================================
# 5. DEEPSORT FEATURE EXTRACTOR
# ============================================================
def extract_appearance_features(crop_img, feature_extractor_model):
    try:
        img_resized = cv2.resize(crop_img, IMG_SIZE)
        img_ready   = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.expand_dims(img_resized.astype(np.float32), axis=0)
        )
        features = feature_extractor_model.predict_on_batch(img_ready)[0]
        return features / (np.linalg.norm(features) + 1e-8)
    except:
        return None


# ============================================================
# 6. WORKER THREAD — branch Skeleton (ensemble modelli)
# ============================================================
def ensemble_ai_worker_thread(models_dict):
    """
    Branch Skeleton della slide:
      landmark → BOW encoding → ensemble (CNN+MobileNet+Landmarks) → S_SK

    La queue riceve (crop, landmarks).
    Produce global_score_skeleton per ogni classe.
    """
    global global_score_skeleton, global_gesture, global_confidence
    global hand_features_template, global_valid_identity

    try:
        feature_extractor = tf.keras.Model(
            inputs  = models_dict["MOBILENET"].input,
            outputs = models_dict["MOBILENET"].layers[-2].output
        )
    except:
        feature_extractor = models_dict["MOBILENET"]

    frame_counter = 0

    while True:
        try:
            crop_to_process, landmarks_to_process = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        try:
            frame_counter += 1

            # Re-ID DeepSORT
            if frame_counter % REID_INTERVAL == 0:
                current_features = extract_appearance_features(
                    crop_to_process, feature_extractor
                )
                if current_features is not None:
                    if hand_features_template is None:
                        hand_features_template = current_features
                        print("[DeepSORT Re-ID] Profilo salvato.")
                    else:
                        cosine_sim = np.dot(current_features, hand_features_template)
                        with gesture_lock:
                            if cosine_sim < 0.50:
                                global_valid_identity = False
                                global_gesture        = "Sconosciuto (ID Errato)"
                                continue
                            else:
                                global_valid_identity = True

            # Landmarks → input MLP
            punti = []
            for lm in landmarks_to_process:
                punti.extend([lm.x, lm.y, lm.z])
            input_land = np.expand_dims(np.array(punti, dtype=np.float32), axis=0)

            # Preprocessing immagine
            lab          = cv2.cvtColor(crop_to_process, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe_processor.apply(lab[:, :, 0])
            hand_proc    = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            hand_r       = cv2.resize(hand_proc, IMG_SIZE)

            input_cnn    = np.expand_dims(hand_r / 255.0, axis=0)
            input_mobile = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.expand_dims(hand_r.astype(np.float32), axis=0)
            )

            # Inference ensemble
            prob_cnn    = models_dict["CNN"].predict_on_batch(input_cnn)[0]
            prob_mobile = models_dict["MOBILENET"].predict_on_batch(input_mobile)[0]
            prob_land   = models_dict["LANDMARKS"].predict_on_batch(input_land)[0]

            # S_SK = score branch skeleton (weighted fusion interna)
            s_sk = (WEIGHT_CNN       * prob_cnn    +
                    WEIGHT_MOBILE    * prob_mobile +
                    WEIGHT_LANDMARKS * prob_land)

            with gesture_lock:
                global_score_skeleton = s_sk

        except Exception:
            pass


# ============================================================
# 7. FUSIONE FINALE (slide: Score-level fusion S_F + S_SK → S)
# ============================================================
def fuse_scores(s_f, s_sk):
    """
    Score-level fusion finale come da slide:
      S = WEIGHT_RGB_BRANCH * S_F + WEIGHT_SKE_BRANCH * S_SK

    s_f  : array (n_classes,) — score branch RGB
    s_sk : array (n_classes,) — score branch Skeleton
    Ritorna: (classe_idx, confidence, score_fuso)
    """
    if s_f is None and s_sk is None:
        return None, 0.0, None
    if s_f is None:
        fused = s_sk
    elif s_sk is None:
        fused = s_f
    else:
        fused = WEIGHT_RGB_BRANCH * s_f + WEIGHT_SKE_BRANCH * s_sk

    cls_id = int(np.argmax(fused))
    conf   = float(fused[cls_id])
    return cls_id, conf, fused


# ============================================================
# 8. MAIN
# ============================================================
if __name__ == "__main__":

    # ── Carica modelli ensemble ──────────────────────────────
    loaded_models = {}
    print("\n[Ensemble] Caricamento modelli...")
    for nome, path in MODEL_PATHS.items():
        if os.path.exists(path):
            loaded_models[nome] = tf.keras.models.load_model(path)
            print(f"  -> {nome} caricato.")
        else:
            print(f"ERRORE: modello mancante → {path}")
            exit()

    # ── Avvia worker skeleton ────────────────────────────────
    ai_thread = threading.Thread(
        target=ensemble_ai_worker_thread,
        args=(loaded_models,),
        daemon=True
    )
    ai_thread.start()

    # ── Init MediaPipe ───────────────────────────────────────
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

    # ── Init componenti RGB branch ───────────────────────────
    tig_hog   = TIGHOGExtractor()
    svm_clf   = SVMClassifier(
        n_classes    = len(GESTURE_CLASSES),
        feature_size = tig_hog.feature_size,
        svm_path     = SVM_PATH
    )
    bow_enc   = BOWLandmarkEncoder(
        vocab_size  = BOW_VOCAB_SIZE,
        buffer_size = TIG_BUFFER_SIZE
    )

    # ── Webcam ───────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    kalman   = KalmanHandTracker()
    velocity = RelativeVelocityMouse(sensitivity=2.5, smoothing=0.35,
                                      dead_zone_px=2, max_delta_px=40)

    last_click_time    = 0
    click_cooldown     = 0.5
    scroll_accumulator = 0
    last_timestamp_ms  = 0
    prev_smooth_x      = None
    prev_smooth_y      = None
    show_tig           = False   # toggle con T

    cur_mouse_x = float(SCREEN_W // 2)
    cur_mouse_y = float(SCREEN_H // 2)
    pyautogui.moveTo(int(cur_mouse_x), int(cur_mouse_y))

    # Istruzioni SVM training online
    print("\n" + "="*60)
    print("  TRAINING SVM ONLINE (branch RGB)")
    if svm_clf.trained:
        print("  SVM già addestrato e caricato — pronto all'uso.")
    else:
        print("  SVM NON addestrato. Raccogli campioni con i tasti:")
        for i, cls in enumerate(GESTURE_CLASSES):
            print(f"    {i+1} → {cls}")
        print("  Poi premi INVIO per addestrare l'SVM.")
        print("  Raccomandato: almeno 10 campioni per classe.")
    print("  T = toggle TIG overlay  |  Q = esci")
    print("="*60 + "\n")

    while cap.isOpened():
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ── Branch RGB: aggiorna buffer TIG ─────────────────
        tig_hog.push_frame(frame)
        tig_feat = tig_hog.compute()   # None se buffer non pieno

        # Score branch RGB
        s_f = None
        if tig_feat is not None:
            s_f = svm_clf.predict_scores(tig_feat)

        # ── MediaPipe ────────────────────────────────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms = int(time.time() * 1000)
        if frame_timestamp_ms <= last_timestamp_ms:
            frame_timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = frame_timestamp_ms

        results = mp_detector.detect_for_video(mp_image, frame_timestamp_ms)

        # Leggi stato condiviso
        with gesture_lock:
            s_sk       = global_score_skeleton
            cur_valid  = global_valid_identity

        # ── Fusione branch RGB + Skeleton ────────────────────
        cls_id, cur_confidence, fused = fuse_scores(s_f, s_sk)
        cur_gesture = GESTURE_CLASSES[cls_id] if cls_id is not None else None

        display_frame = frame.copy()

        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]

            # Punto di controllo: punta indice (L8)
            tip_lm = landmarks[8]
            tx_raw = int(tip_lm.x * cam_w)
            ty_raw = int(tip_lm.y * cam_h)
            tx, ty = kalman.update(tx_raw, ty_raw)

            # ROI mano
            x_coords = [lm.x * cam_w for lm in landmarks]
            y_coords = [lm.y * cam_h for lm in landmarks]
            pad  = 20
            xmin = max(0,     int(min(x_coords)) - pad)
            xmax = min(cam_w, int(max(x_coords)) + pad)
            ymin = max(0,     int(min(y_coords)) - pad)
            ymax = min(cam_h, int(max(y_coords)) + pad)

            # Invia al worker skeleton
            hand_crop = frame[ymin:ymax, xmin:xmax]
            if hand_crop.size > 0:
                try:
                    frame_queue.put_nowait((hand_crop.copy(), landmarks))
                except queue.Full:
                    pass

            # BOW: aggiorna buffer landmark
            bow_enc.push_landmarks(landmarks)
            if bow_enc.vocabulary is None and len(bow_enc._collect) >= BOW_VOCAB_SIZE:
                bow_enc.build_vocabulary()

            if cur_valid:
                # ── Movimento cursore ────────────────────────
                if (cur_gesture == "index" and
                        cur_confidence > GESTURE_CONFIDENCE_THRESHOLD):
                    dx, dy = velocity.update(tx, ty)
                    cur_mouse_x = float(np.clip(cur_mouse_x + dx, 0, SCREEN_W-1))
                    cur_mouse_y = float(np.clip(cur_mouse_y + dy, 0, SCREEN_H-1))
                    pyautogui.moveTo(int(cur_mouse_x), int(cur_mouse_y))
                else:
                    velocity.update(tx, ty)

                # ── Azioni mouse ─────────────────────────────
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
                prev_smooth_x = tx
                prev_smooth_y = ty

                # ── Feedback visivo ──────────────────────────
                color = (0, 255, 0) if cur_gesture == "index" else (200, 200, 0)
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.circle(display_frame, (tx_raw, ty_raw), 5, (100,100,255), 1)
                cv2.circle(display_frame, (tx, ty),         7, (0,220,255), -1)

                if prev_smooth_x and cur_gesture == "index":
                    ddx = int((tx - prev_smooth_x) * 5)
                    ddy = int((ty - prev_smooth_y) * 5)
                    if abs(ddx) + abs(ddy) > 1:
                        ex = int(np.clip(tx + ddx, 0, cam_w-1))
                        ey = int(np.clip(ty + ddy, 0, cam_h-1))
                        cv2.arrowedLine(display_frame, (tx, ty), (ex, ey),
                                        (0,100,255), 2, tipLength=0.4)

                # Score breakdown visivo
                if fused is not None:
                    for ci, cls in enumerate(GESTURE_CLASSES):
                        bar_w = int(fused[ci] * 100)
                        cv2.rectangle(display_frame,
                                      (cam_w - 115, 40 + ci*22),
                                      (cam_w - 115 + bar_w, 56 + ci*22),
                                      (0, 200, 100), -1)
                        cv2.putText(display_frame, cls[:6],
                                    (cam_w - 115, 54 + ci*22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    (255,255,255), 1)

                if cur_gesture:
                    cv2.putText(display_frame,
                                f"{cur_gesture}  {cur_confidence*100:.0f}%",
                                (xmin, max(20, ymin-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            else:
                cv2.rectangle(display_frame,(xmin,ymin),(xmax,ymax),(0,0,255),2)
                cv2.putText(display_frame, "RE-ID BLOCCATO",
                            (xmin, max(20, ymin-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        else:
            kalman.reset()
            velocity.reset()
            bow_enc.reset()
            prev_smooth_x = prev_smooth_y = None
            with gesture_lock:
                global_gesture = None
            scroll_accumulator = 0

        # ── TIG overlay (tasto T) ────────────────────────────
        if show_tig:
            tig_img = tig_hog.get_last_tig()
            if tig_img is not None:
                tig_show = cv2.resize(tig_img, (120, 90))
                tig_bgr  = cv2.cvtColor(tig_show, cv2.COLOR_GRAY2BGR)
                display_frame[0:90, 0:120] = tig_bgr
                cv2.putText(display_frame, "TIG", (4, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,220,255), 1)

        # ── SVM status overlay ───────────────────────────────
        svm_status = "SVM: pronto" if svm_clf.trained else \
                     f"SVM: raccogli ({len(svm_clf._y_buffer)} camp.)"
        cv2.putText(display_frame, svm_status,
                    (10, cam_h - 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0,255,128) if svm_clf.trained else (0,180,255), 1)

        if not svm_clf.trained:
            for i, cls in enumerate(GESTURE_CLASSES):
                cv2.putText(display_frame,
                            f"  [{i+1}] raccogli {cls}",
                            (10, cam_h - 112 + i*16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)

        # ── Legenda ──────────────────────────────────────────
        legend = [
            ("index",       "Muovi cursore", (0,220,255)),
            ("open_palm",   "Click sin.",    (0,255,0)),
            ("two_fingers", "Doppio click",  (100,255,100)),
            ("fist",        "Click des.",    (0,140,255)),
            ("pinch",       "Scroll",        (255,200,0)),
        ]
        for i, (g, desc, col) in enumerate(legend):
            active = cur_gesture == g
            cv2.putText(display_frame, f"  {g}: {desc}",
                        (8, cam_h - 14 - i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        col if active else (80,80,80),
                        2 if active else 1)

        fps = 1.0 / (time.time() - t0 + 1e-9)
        sf_str  = f"SF:{s_f[np.argmax(s_f)]*100:.0f}%" if s_f  is not None else "SF:--"
        ssk_str = f"SSK:{s_sk[np.argmax(s_sk)]*100:.0f}%" if s_sk is not None else "SSK:--"
        cv2.putText(display_frame,
                    f"FPS:{int(fps)}  {sf_str}  {ssk_str}  "
                    f"Fused:{cur_gesture or '---'} {cur_confidence*100:.0f}%  T=TIG Q=esci",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

        cv2.imshow("Virtual Mouse v7 — TIG+HOG+SVM | MediaPipe+Ensemble", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            show_tig = not show_tig
        elif key == 13:   # INVIO → addestra SVM
            if not svm_clf.trained:
                svm_clf.fit()
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            # Raccolta campione SVM: tasto = indice classe
            label_idx = key - ord('1')
            if tig_feat is not None:
                svm_clf.add_sample(tig_feat, label_idx)
            else:
                print("[SVM] Buffer TIG non ancora pieno, aspetta un momento.")

    cap.release()
    cv2.destroyAllWindows()