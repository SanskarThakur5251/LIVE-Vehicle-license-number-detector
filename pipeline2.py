import os
import cv2
import torch
import datetime
from collections import Counter
from ultralytics import YOLO
import easyocr

from crnn_model import CRNN  


# =========================
# CONFIG (TUNABLE)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_MODEL_PATH = "models/yolo.pt"
CRNN_MODEL_PATH = "models/RCNN.pth"

CONF_THRESHOLD = 0.45
PADDING = 5

FRAME_SKIP = 5                  # 🔹 process every Nth frame
OCR_CONF_THRESHOLD = 0.6        # 🔹 ignore weak OCR
MIN_PLATE_LENGTH = 6            # 🔹 ignore junk predictions

OUTPUT_DIR = "output"
VIDEO_LOG = os.path.join(OUTPUT_DIR, "video_results.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD MODELS
# =========================
print("[INFO] Loading YOLO...")
yolo = YOLO(YOLO_MODEL_PATH)

print("[INFO] Loading RCNN ...")
try:
    crnn = CRNN(imgH=32, nc=1, nclass=37, nh=256)
    state_dict = torch.load(CRNN_MODEL_PATH, map_location=DEVICE)
    crnn.load_state_dict(state_dict, strict=False)
    crnn.to(DEVICE).eval()
except:
    pass

reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"))

print("[INFO] Models loaded")


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (int(w * 2.5), int(h * 2.5)))
    clahe = cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(gray)


# =========================
# VIDEO PIPELINE
# =========================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3))
    height = int(cap.get(4))

    base = os.path.basename(video_path).split(".")[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_path = os.path.join(
        OUTPUT_DIR, f"{base}_{timestamp}_processed.mp4"
    )

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_id = 0
    plate_buffer = []
    conf_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # 🔹 FRAME SKIP
        if frame_id % FRAME_SKIP != 0:
            writer.write(frame)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo(rgb)[0]

        best_box, best_conf = None, 0.0
        for box in results.boxes:
            conf = float(box.conf)
            if conf > best_conf and conf >= CONF_THRESHOLD:
                best_conf = conf
                best_box = box.xyxy[0].cpu().numpy()

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            h, w, _ = frame.shape
            x1, y1 = max(0, x1-PADDING), max(0, y1-PADDING)
            x2, y2 = min(w, x2+PADDING), min(h, y2+PADDING)

            crop = rgb[y1:y2, x1:x2]
            processed = preprocess(crop)

            ocr = reader.readtext(
                processed,
                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                paragraph=False
            )

            if ocr:
                text, conf = max(ocr, key=lambda x: x[2])[1:]
                text = text.replace(" ", "").upper()

                if conf >= OCR_CONF_THRESHOLD and len(text) >= MIN_PLATE_LENGTH:
                    plate_buffer.append(text)
                    conf_buffer.append(conf)

                    # draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(
                        frame, f"{text} ({conf:.2f})",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2
                    )

        writer.write(frame)

    cap.release()
    writer.release()

    # =========================
    # CONSOLIDATE RESULTS
    # =========================
    if plate_buffer:
        final_plate = Counter(plate_buffer).most_common(1)[0][0]
        avg_conf = sum(conf_buffer) / len(conf_buffer)
    else:
        final_plate = "NOT_DETECTED"
        avg_conf = 0.0

    with open(VIDEO_LOG, "a") as f:
        f.write(
            f"{timestamp} | {base} | "
            f"FINAL_PLATE={final_plate} | "
            f"AVG_OCR_CONF={avg_conf:.3f}\n"
        )

    print("[INFO] Video processed")
    print("[INFO] Final Plate:", final_plate)
    print("[INFO] Avg RCNN Conf:", round(avg_conf, 3))


# =========================
# RUN
# =========================
if __name__ == "__main__":
    process_video("test/video2.mp4")
