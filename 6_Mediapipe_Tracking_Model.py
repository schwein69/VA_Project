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

# KALMAN FILTER
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

# VELOCITY CONTROLLER 
class RelativeVelocityMouse:
    """
    Muove il cursore in base al DELTA frame→frame della punta dell'indice

    Funzionamento:
      - Ogni frame misura (x_corrente - x_precedente) della punta (landmark 8)
      - Quel delta, amplificato da un fattore di sensibilità, viene sommato
        alla posizione corrente del cursore
      - Mano ferma  → delta = 0 → cursore fermo
      - Mano veloce → delta grande → cursore veloce

    Vantaggi rispetto alla posizione assoluta:
      - Nessun limite di area: la mano può stare sempre al centro della camera
      - La camera piccola non è più un problema
      - Lo schermo intero è raggiungibile con micro-movimenti

    Parametri:
      sensitivity   : pixel schermo per pixel camera. Default 2.5.
                      Aumenta se il cursore si muove troppo poco.
                      Diminuisci se è difficile da controllare.
      smoothing     : EMA sul delta (0=reattivo, 1=lento).
                      Default 0.35: buon compromesso tra fluidità e risposta.
      dead_zone_px  : movimento minimo in pixel camera da ignorare.
                      Filtra il movimento vibrante della mano.
                      Default 2px. Aumentare a un valore più alto se la mano trema molto.
      max_delta_px  : clamp sul delta grezzo. Ignora salti bruschi
                      (es. mano persa e rilevata in posizione diversa).
                      Default 40px camera → corrisponde a ~100px schermo.
    """

    def __init__(self,
                 sensitivity=2.5,
                 smoothing=0.35,
                 dead_zone_px=2,
                 max_delta_px=40):

        self.sensitivity   = sensitivity
        self.smoothing     = smoothing
        self.dead_zone_px  = dead_zone_px
        self.max_delta_px  = max_delta_px

        self.prev_x = None
        self.prev_y = None

        # EMA del delta (velocità smoothed)
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0

    def update(self, x, y):
        """
        Riceve la posizione corrente (già smoothed dal Kalman) della punta.
        Ritorna (delta_x, delta_y) da applicare al cursore schermo.
        Ritorna (0, 0) al primo frame o se nella dead zone.
        """
        if self.prev_x is None:
            # Primo frame: nessun delta disponibile
            self.prev_x = x
            self.prev_y = y
            return 0.0, 0.0

        raw_dx = x - self.prev_x
        raw_dy = y - self.prev_y

        self.prev_x = x
        self.prev_y = y

        # Clamp: ignora salti troppo grandi (mano persa/ritrovata)
        if abs(raw_dx) > self.max_delta_px or abs(raw_dy) > self.max_delta_px:
            self.smooth_dx = 0.0
            self.smooth_dy = 0.0
            return 0.0, 0.0

        # Dead zone: ignora micro-tremori
        magnitude = (raw_dx**2 + raw_dy**2) ** 0.5
        if magnitude < self.dead_zone_px:
            # Lascia decadere la velocità smoothed verso zero
            self.smooth_dx *= (1 - self.smoothing)
            self.smooth_dy *= (1 - self.smoothing)
            if abs(self.smooth_dx) < 0.1 and abs(self.smooth_dy) < 0.1:
                return 0.0, 0.0
            return self.smooth_dx * self.sensitivity, self.smooth_dy * self.sensitivity

        # EMA: mescola il delta attuale con la "velocità" precedente
        self.smooth_dx = self.smoothing * raw_dx + (1 - self.smoothing) * self.smooth_dx
        self.smooth_dy = self.smoothing * raw_dy + (1 - self.smoothing) * self.smooth_dy

        return self.smooth_dx * self.sensitivity, self.smooth_dy * self.sensitivity

    def reset(self):
        """Chiamato quando la mano esce dal frame."""
        self.prev_x    = None
        self.prev_y    = None
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0

# ==========================================
# 2. CONFIGURAZIONE
# ==========================================
GESTURE_CLASSES = ['open_palm', 'fist', 'index', 'two_fingers', 'pinch']
IMG_SIZE = (224, 224)

MODEL_PATHS = {
    "CNN":       "model_name_cnn.keras",
    "MOBILENET": "model_name_mobilenet.keras",
    "LANDMARKS": "model_name_landmarks.keras"
}

WEIGHT_CNN       = 0.30
WEIGHT_MOBILE    = 0.30
WEIGHT_LANDMARKS = 0.40

GESTURE_CONFIDENCE_THRESHOLD = 0.60
REID_INTERVAL = 30

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# ==========================================
# STATO CONDIVISO
# ==========================================
frame_queue  = queue.Queue(maxsize=1)
gesture_lock = threading.Lock()

global_gesture         = None
global_confidence      = 0.0
global_valid_identity  = True
hand_features_template = None

clahe_processor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ==========================================
# 3. DEEPSORT FEATURE EXTRACTOR
# ==========================================
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

# ==========================================
# 4. WORKER THREAD
# ==========================================
def ensemble_ai_worker_thread(models_dict):
    global global_gesture, global_confidence, hand_features_template, global_valid_identity

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

            if frame_counter % REID_INTERVAL == 0:
                current_features = extract_appearance_features(crop_to_process, feature_extractor)
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

            punti = []
            for lm in landmarks_to_process:
                punti.extend([lm.x, lm.y, lm.z])
            input_land = np.expand_dims(np.array(punti, dtype=np.float32), axis=0)

            lab = cv2.cvtColor(crop_to_process, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe_processor.apply(lab[:, :, 0])
            hand_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            hand_resized   = cv2.resize(hand_processed, IMG_SIZE)

            input_cnn    = np.expand_dims(hand_resized / 255.0, axis=0)
            input_mobile = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.expand_dims(hand_resized.astype(np.float32), axis=0)
            )

            prob_cnn    = models_dict["CNN"].predict_on_batch(input_cnn)[0]
            prob_mobile = models_dict["MOBILENET"].predict_on_batch(input_mobile)[0]
            prob_land   = models_dict["LANDMARKS"].predict_on_batch(input_land)[0]

            fused  = (WEIGHT_CNN * prob_cnn) + (WEIGHT_MOBILE * prob_mobile) + (WEIGHT_LANDMARKS * prob_land)
            cls_id = np.argmax(fused)

            with gesture_lock:
                global_confidence = fused[cls_id]
                global_gesture    = GESTURE_CLASSES[cls_id]

        except Exception:
            pass

# ==========================================
# 5. MAIN THREAD
# ==========================================
if __name__ == "__main__":

    loaded_models = {}
    print("\n[Ensemble] Caricamento modelli...")
    for nome, path in MODEL_PATHS.items():
        if os.path.exists(path):
            loaded_models[nome] = tf.keras.models.load_model(path)
            print(f"  -> {nome} caricato.")
        else:
            print(f"ERRORE: modello mancante → {path}")
            exit()

    ai_thread = threading.Thread(
        target=ensemble_ai_worker_thread,
        args=(loaded_models,),
        daemon=True
    )
    ai_thread.start()

    model_path_mp = 'hand_landmarker.task'
    if not os.path.exists(model_path_mp):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
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
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    kalman   = KalmanHandTracker()
    velocity = RelativeVelocityMouse(
        sensitivity=2.5,
        smoothing=0.35,
        dead_zone_px=2,
        max_delta_px=40
    )

    last_click_time    = 0
    click_cooldown     = 0.5
    scroll_accumulator = 0
    last_timestamp_ms  = 0

    # Posizione corrente cursore (float per precisione sub-pixel)
    cur_mouse_x = float(SCREEN_W  // 2)
    cur_mouse_y = float(SCREEN_H // 2)
    pyautogui.moveTo(int(cur_mouse_x), int(cur_mouse_y))

    # Per il trail visivo del delta
    prev_smooth_x = None
    prev_smooth_y = None

    while cap.isOpened():
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms = int(time.time() * 1000)
        if frame_timestamp_ms <= last_timestamp_ms:
            frame_timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = frame_timestamp_ms

        results = detector.detect_for_video(mp_image, frame_timestamp_ms)

        with gesture_lock:
            cur_gesture    = global_gesture
            cur_confidence = global_confidence
            cur_valid_id   = global_valid_identity

        display_frame = frame.copy()

        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]

            # Punta indice (landmark 8) — punto di controllo
            tip_lm = landmarks[8]
            tx_raw = int(tip_lm.x * cam_w)
            ty_raw = int(tip_lm.y * cam_h)

            # Kalman sulla punta: stabilizza il punto prima di calcolare il delta
            tx, ty = kalman.update(tx_raw, ty_raw)

            # ROI per il worker AI
            x_coords = [lm.x * cam_w for lm in landmarks]
            y_coords = [lm.y * cam_h for lm in landmarks]
            padding  = 20
            xmin = max(0,     int(min(x_coords)) - padding)
            xmax = min(cam_w, int(max(x_coords)) + padding)
            ymin = max(0,     int(min(y_coords)) - padding)
            ymax = min(cam_h, int(max(y_coords)) + padding)

            hand_crop = frame[ymin:ymax, xmin:xmax]
            if hand_crop.size > 0:
                try:
                    frame_queue.put_nowait((hand_crop.copy(), landmarks))
                except queue.Full:
                    pass

            if cur_valid_id:

                # ─────────────────────────────────────────
                # METODO B: VELOCITÀ RELATIVA
                # Il cursore si muove SOLO con gesto "index"
                # ─────────────────────────────────────────
                if (cur_gesture == "index" and
                        cur_confidence > GESTURE_CONFIDENCE_THRESHOLD):

                    dx, dy = velocity.update(tx, ty)

                    cur_mouse_x = float(np.clip(cur_mouse_x + dx, 0, SCREEN_W - 1))
                    cur_mouse_y = float(np.clip(cur_mouse_y + dy, 0, SCREEN_H - 1))
                    pyautogui.moveTo(int(cur_mouse_x), int(cur_mouse_y))

                else:
                    # Con qualsiasi altro gesto, aggiorna la posizione "precedente"
                    # senza muovere il cursore, così non c'è salto al ritorno su index.
                    velocity.update(tx, ty)

                # ─────────────────────────────────────────
                # ALTRE AZIONI
                # ─────────────────────────────────────────
                if cur_gesture and cur_confidence > GESTURE_CONFIDENCE_THRESHOLD:

                    if cur_gesture == "open_palm":
                        if (time.time() - last_click_time) > click_cooldown:
                            pyautogui.click()
                            last_click_time = time.time()

                    elif cur_gesture == "pinch":
                        # Scroll: usa il delta verticale della punta
                        if prev_smooth_y is not None:
                            scroll_accumulator += (ty - prev_smooth_y)
                        if scroll_accumulator < -15:
                            pyautogui.scroll(10)
                            scroll_accumulator = 0
                        elif scroll_accumulator > 15:
                            pyautogui.scroll(-10)
                            scroll_accumulator = 0

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

                # ─────────────────────────────────────────
                # FEEDBACK VISIVO
                # ─────────────────────────────────────────
                color_box = (0, 255, 0) if cur_gesture == "index" else (200, 200, 0)
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color_box, 2)

                # Cerchio punta indice (raw vs smoothed)
                cv2.circle(display_frame, (tx_raw, ty_raw), 5, (100, 100, 255), 1)
                cv2.circle(display_frame, (tx, ty),         7, (0, 220, 255),   -1)

                # Freccia delta (amplificata x5 per renderla visibile)
                if prev_smooth_x is not None and cur_gesture == "index":
                    ddx = int((tx - prev_smooth_x) * 5)
                    ddy = int((ty - prev_smooth_y) * 5)
                    if abs(ddx) + abs(ddy) > 1:
                        ex = int(np.clip(tx + ddx, 0, cam_w - 1))
                        ey = int(np.clip(ty + ddy, 0, cam_h - 1))
                        cv2.arrowedLine(display_frame, (tx, ty), (ex, ey),
                                        (0, 100, 255), 2, tipLength=0.4)

                if cur_gesture:
                    cv2.putText(display_frame,
                                f"{cur_gesture}  {cur_confidence*100:.0f}%",
                                (xmin, max(20, ymin - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_box, 2)

            else:
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(display_frame, "RE-ID BLOCCATO",
                            (xmin, max(20, ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        else:
            kalman.reset()
            velocity.reset()
            prev_smooth_x = None
            prev_smooth_y = None
            with gesture_lock:
                global_gesture = None
            scroll_accumulator = 0

        # ─────────────────────────────────────────
        # LEGENDA + HUD
        # ─────────────────────────────────────────
        legend = [
            ("index",       "Muovi cursore (muovi la mano)", (0, 220, 255)),
            ("open_palm",   "Click sinistro",                (0, 255, 0)),
            ("two_fingers", "Doppio click",                  (100, 255, 100)),
            ("fist",        "Click destro",                  (0, 140, 255)),
            ("pinch",       "Scroll",                        (255, 200, 0)),
        ]
        for i, (g, desc, col) in enumerate(legend):
            active = cur_gesture == g
            cv2.putText(display_frame, f"  {g}: {desc}",
                        (8, cam_h - 14 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        col if active else (90, 90, 90),
                        2 if active else 1)

        fps = 1.0 / (time.time() - t0 + 1e-9)
        cv2.putText(display_frame,
                    f"FPS:{int(fps)}  Gesto:{cur_gesture or '---'}  "
                    f"Conf:{cur_confidence*100:.0f}%  Mouse:({int(cur_mouse_x)},{int(cur_mouse_y)})",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Virtual Mouse v5 - Relative Velocity", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()