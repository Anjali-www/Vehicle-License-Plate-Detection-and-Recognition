import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import re

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "license_plate_detector.pt"
IMAGE_FOLDER = "Images"
OUTPUT_CSV = "t6_1.acc.csv"

model = YOLO(MODEL_PATH)

# ---------------------------
# VALID STATES
# ---------------------------
VALID_PREFIXES = {
    "AN","AP","AR","AS","BH","BR","CG","CH","DD","DL","DN","GA","GJ",
    "HP","HR","JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ",
    "NL","OD","PB","PY","RJ","SK","TN","TR","TS","UK","UP","WB",
}

COMMON_STATES = ["UP", "MH", "DL", "KA", "TN"]

# ---------------------------
# FIX MAPS
# ---------------------------
LETTER_FIX = {"0":"O","1":"I","2":"Z","5":"S","6":"G","8":"B"}
DIGIT_FIX = {
    "O":"0","Q":"0","D":"0",
    "I":"1","L":"1",
    "Z":"2","S":"5",
    "B":"8","G":"6"
}

# ---------------------------
# SMART FIX (IMPROVED)
# ---------------------------
def smart_fix(text):
    text = list(text)

    for i in range(len(text)):
        ch = text[i]

        # LETTER POSITIONS
        if i in [0, 1, 4, 5]:
            if ch in LETTER_FIX:
                text[i] = LETTER_FIX[ch]

            # extra OCR confusion fixes
            if ch == 'T': text[i] = 'I'
            if ch == '7': text[i] = 'T'

        # DIGIT POSITIONS
        elif i in [2, 3, 6, 7, 8, 9]:
            if ch in DIGIT_FIX:
                text[i] = DIGIT_FIX[ch]

            # extra OCR confusion fixes
            if ch == 'T': text[i] = '1'
            if ch == 'B': text[i] = '8'

    return "".join(text)

# ---------------------------
# PREFIX FIX
# ---------------------------
def fix_prefix(p):
    if len(p) < 2:
        return p

    prefix = p[:2]

    if prefix in VALID_PREFIXES:
        return p

    best = prefix
    min_diff = 10

    for valid in VALID_PREFIXES:
        diff = sum(1 for a, b in zip(prefix, valid) if a != b)

        if diff < min_diff or (diff == min_diff and valid in COMMON_STATES):
            min_diff = diff
            best = valid

    return best + p[2:]

# ---------------------------
# FORMAT (IMPROVED)
# ---------------------------
def format_plate(text):
    text = re.sub('[^A-Z0-9]', '', text)

    pattern = r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}'

    # 🔥 sliding window (very important)
    for i in range(len(text) - 9):
        chunk = text[i:i+10]
        fixed = smart_fix(chunk)

        if re.fullmatch(pattern, fixed):
            return fixed

    # fallback
    if len(text) >= 10:
        fixed = smart_fix(text[:10])
        if re.fullmatch(pattern, fixed):
            return fixed

    return text

# ---------------------------
# VALID CHECK
# ---------------------------
def is_valid_plate(p):
    return len(p) == 10 and p[:2] in VALID_PREFIXES

# ---------------------------
# DESKEW
# ---------------------------
def deskew(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)

    if lines is None:
        return plate

    angles = []
    for l in lines[:20]:
        rho, theta = l[0]
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return plate

    angle = np.median(angles)
    h, w = plate.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)

    return cv2.warpAffine(plate, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

# ---------------------------
# PREPROCESS
# ---------------------------
def preprocess(plate):
    plate = cv2.resize(plate, None, fx=3, fy=3)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 2
    )

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh

# ---------------------------
# CHAR BOXES
# ---------------------------
def get_char_boxes(thresh):
    h, w = thresh.shape
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = cw / float(ch)

        if area > 50 and 0.15 < ar < 1.2 and ch > h * 0.4:
            boxes.append((x,y,cw,ch))

    return sorted(boxes, key=lambda b: b[0])

# ---------------------------
# OCR CHAR
# ---------------------------
def ocr_char(img):
    img = 255 - img
    img = cv2.resize(img, (60, 90))
    img = cv2.copyMakeBorder(img, 10,10,10,10, cv2.BORDER_CONSTANT, value=255)

    text = pytesseract.image_to_string(
        img,
        config='--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    text = re.sub(r'[^A-Z0-9]', '', text)
    return text[0] if text else ""

# ---------------------------
# SEGMENT OCR
# ---------------------------
def segment_ocr(thresh, boxes):
    text = ""
    for (x,y,w,h) in boxes:
        text += ocr_char(thresh[y:y+h, x:x+w])
    return text

# ---------------------------
# FULL OCR
# ---------------------------
def full_ocr(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(
        th,
        config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    return re.sub(r'[^A-Z0-9]', '', text)

def clean(text):
    return text.upper().replace(" ", "")

# ---------------------------
# ACCURACY
# ---------------------------
def char_accuracy(gt, pred):
    matches = sum(1 for a, b in zip(gt, pred) if a == b)
    return matches / max(len(gt), len(pred)) if max(len(gt), len(pred)) > 0 else 0

# ---------------------------
# MAIN LOOP
# ---------------------------
rows = []
exact_matches = 0
total_chars = 0
matched_chars = 0
total = 1

for file in os.listdir(IMAGE_FOLDER):
    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    actual = os.path.splitext(file)[0].upper()
    img_path = os.path.join(IMAGE_FOLDER, file)

    img = cv2.imread(img_path)
    res = model(img)

    predicted = ""

    for r in res:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            plate = img[y1:y2, x1:x2]

            plate = deskew(plate)
            thresh = preprocess(plate)
            boxes = get_char_boxes(thresh)

            seg_text = segment_ocr(thresh, boxes)
            full_text = full_ocr(plate)

            seg_formatted = format_plate(clean(seg_text))
            full_formatted = format_plate(clean(full_text))

            total += 1

            if is_valid_plate(seg_formatted):
                predicted = seg_formatted

            elif is_valid_plate(full_formatted):
                predicted = full_formatted

            else:
                fallback = seg_formatted if len(seg_text) >= len(full_text) else full_formatted
                predicted = fix_prefix(smart_fix(fallback))  # 🔥 FINAL FIX

    exact = int(actual == predicted)
    char_acc = char_accuracy(actual, predicted)

    if exact:
        exact_matches += 1

    matched_chars += sum(1 for a, b in zip(actual, predicted) if a == b)
    total_chars += max(len(actual), len(predicted))

    rows.append([file, actual, predicted, exact, round(char_acc, 3)])

# ---------------------------
# SAVE CSV
# ---------------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "actual", "predicted", "exact_match", "char_accuracy"])
    writer.writerows(rows)

# ---------------------------
# RESULTS
# ---------------------------
print("\n===== RESULTS =====")
print("Total :", total)
print("Exact Accuracy:", (exact_matches / len(rows))*100 if rows else 0)
print("Character Accuracy:", (matched_chars / total_chars)*100 if total_chars > 0 else 0)
print("CSV saved as:", OUTPUT_CSV)