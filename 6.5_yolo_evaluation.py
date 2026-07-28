import os
import time
import glob
import random
import yaml as pyyaml
from ultralytics import YOLO


def evaluate_yolo_model(model_path="runs/detect/train/weights/best.pt",
                         data_yaml="gestures.yaml",
                         imgsz=384, conf=0.25, iou=0.6,
                         n_speed_test=30):
    """
    Valuta un modello YOLO per la detection della mano su due fronti:
      1. Metriche standard (precision, recall, mAP50, mAP50-95) via model.val()
      2. Velocità di inferenza media (ms/immagine, FPS stimati)
    """
    model = YOLO(model_path)

    # Metriche di detection 
    print("=" * 60)
    print("METRICHE DI DETECTION")
    print("=" * 60)
    metrics = model.val(data=data_yaml, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

    print(f"Precision  : {metrics.box.mp:.3f}")
    print(f"Recall     : {metrics.box.mr:.3f}")
    print(f"mAP50      : {metrics.box.map50:.3f}")
    print(f"mAP50-95   : {metrics.box.map:.3f}")
    print(f"\n(grafici dettagliati salvati automaticamente in runs/detect/val*/ "
          f"- confusion matrix, curva PR, ecc.)")

    # Velocità di inferenza reale 
    print("\n" + "=" * 60)
    print(f"VELOCITÀ DI INFERENZA (media su {n_speed_test} immagini)")
    print("=" * 60)

    with open(data_yaml) as f:
        data_cfg = pyyaml.safe_load(f)
    val_dir = os.path.join(data_cfg["path"], data_cfg["val"])
    test_images = glob.glob(f"{val_dir}/*.*")

    if not test_images:
        print("Nessuna immagine trovata per il test di velocità.")
        return metrics

    sample = random.sample(test_images, min(n_speed_test, len(test_images)))

    model.predict(sample[0], imgsz=imgsz, verbose=False)  # warm-up

    times = []
    for img_path in sample:
        t0 = time.time()
        model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)
        times.append(time.time() - t0)

    avg_time = sum(times) / len(times)
    print(f"Tempo medio per immagine : {avg_time * 1000:.1f} ms")
    print(f"FPS stimati              : {1 / avg_time:.1f}")

    return metrics


metrics = evaluate_yolo_model()