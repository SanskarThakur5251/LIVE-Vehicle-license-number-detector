import os
import cv2
import torch
import datetime
from ultralytics import YOLO
import easyocr

from crnn_model import CRNN   # CRNN KEPT (NOT USED FOR OCR)


# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_MODEL_PATH = "models/yolo.pt"
CRNN_MODEL_PATH = "models/RCNN.pth"

CONF_THRESHOLD = 0.45
PADDING = 5

OUTPUT_DIR = "output"
RESULT_FILE = os.path.join(OUTPUT_DIR, "results.txt")

# =========================
# CREATE OUTPUT DIR
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD YOLO
# =========================
print("[INFO] Loading YOLO model...")
yolo = YOLO(YOLO_MODEL_PATH)

# =========================
# LOAD RCNN
# =========================
print("[INFO] Loading RCNN model ")
try:
    crnn = CRNN(imgH=32, nc=1, nclass=37, nh=256)
    state_dict = torch.load(CRNN_MODEL_PATH, map_location=DEVICE)

    clean_state = {}
    model_state = crnn.state_dict()

    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if k in model_state and model_state[k].shape == v.shape:
            clean_state[k] = v

    crnn.load_state_dict(clean_state, strict=False)
    crnn.to(DEVICE)
    crnn.eval()
    print("[INFO] RCNN loaded successfully")
except Exception as e:
    print("[WARN] RCNN could not be loaded:", e)
reader = easyocr.Reader(
    ['en'],
    gpu=True if DEVICE == "cuda" else False)
print("[INFO] Models loaded successfully")


def preprocess_for_easyocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    gray = cv2.resize(gray, (int(w * 2.5), int(h * 2.5)))

    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)

    return enhanced

# =========================
# MAIN PIPELINE
# =========================
def recognize_license_plate(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(image_path)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ---------- YOLO DETECTION ----------
    results = yolo(image_rgb)[0]

    best_box = None
    best_conf = 0.0

    for box in results.boxes:
        conf = float(box.conf)
        if conf >= CONF_THRESHOLD and conf > best_conf:
            best_conf = conf
            best_box = box.xyxy[0].cpu().numpy()

    if best_box is None:
        return None

    x1, y1, x2, y2 = map(int, best_box)

    # ---------- ADD PADDING ----------
    h, w, _ = image_rgb.shape
    x1 = max(0, x1 - PADDING)
    y1 = max(0, y1 - PADDING)
    x2 = min(w, x2 + PADDING)
    y2 = min(h, y2 + PADDING)

    cropped = image_rgb[y1:y2, x1:x2]

    # ---------- EASY OCR ----------
    processed = preprocess_for_easyocr(cropped)

    ocr_results = reader.readtext(
        processed,
        allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        paragraph=False
    )

    if len(ocr_results) == 0:
        plate_text = ""
        ocr_conf = 0.0
    else:
        best_ocr = max(ocr_results, key=lambda x: x[2])
        plate_text = best_ocr[1].replace(" ", "").upper()
        ocr_conf = float(best_ocr[2])

    # ---------- DRAW RESULT ----------
    output_img = image_bgr.copy()
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"{plate_text} ({ocr_conf:.2f})"
    cv2.putText(
        output_img,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ---------- SAVE IMAGE (NO OVERWRITE) ----------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(image_path).split(".")[0]
    out_img_path = os.path.join(
        OUTPUT_DIR, f"{base}_{timestamp}.jpg"
    )
    cv2.imwrite(out_img_path, output_img)

    # ---------- APPEND TEXT RESULT ----------
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{timestamp} | {base} | "
            f"PLATE={plate_text} | "
            f"YOLO={best_conf:.3f} | "
            f"OCR={ocr_conf:.3f}\n"
        )

    return {
        "plate_text": plate_text,
        "yolo_confidence": round(best_conf, 3),
        "ocr_confidence": round(ocr_conf, 3),
        "output_image": out_img_path
    }

# =========================
# RUN
# =========================
if __name__ == "__main__":
    img_path = "test/carImage4.jpeg"

    result = recognize_license_plate(img_path)

    print("\n===== RESULT =====")
    if result is None:
        print("❌ No license plate detected")
    else:
        print("Plate Text      :", result["plate_text"])
        print("YOLO Confidence :", result["yolo_confidence"])
        print("OCR Confidence  :", result["ocr_confidence"])
        print("Saved Image     :", result["output_image"])
