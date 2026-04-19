from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
import re
import os 

# ---------------------------
# CONFIG
# ---------------------------
IMAGE_PATH = "Images/AP29AN0074.png"
MODEL_PATH = "license_plate_detector.pt"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

model = YOLO(MODEL_PATH)

def format_plate(text):
    text = re.sub('[^A-Z0-9]', '', text)

    # Fix common OCR mistakes
    text = text.replace('O', '0')
    text = text.replace('I', '1')
    text = text.replace('Z', '2')

    # Extract valid Indian format
    pattern = r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}'
    match = re.search(pattern, text)

    if match:
        return match.group()

    # fallback: last 10 chars
    if len(text) > 10:
        return text[-10:]

    return text
# ---------------------------
# DESKEW (ROBUST)
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

    if len(angles) == 0:
        return plate

    angle = np.median(angles)

    h, w = plate.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)

    return cv2.warpAffine(
        plate, M, (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )

# ---------------------------
# PREPROCESS (STRONG VERSION)
# ---------------------------
def preprocess(plate):
    plate = cv2.resize(plate, None, fx=3, fy=3)

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # 🔥 contrast boost (VERY IMPORTANT)
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
# CHARACTER SEGMENTATION
# ---------------------------
def get_char_boxes(thresh):
    h, w = thresh.shape
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = cw / float(ch)

        if (
            area > 50 and
            0.15 < ar < 1.2 and
            ch > h * 0.4
        ):
            boxes.append((x,y,cw,ch))

    return sorted(boxes, key=lambda b: b[0])

# ---------------------------
# OCR SINGLE CHAR
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
        char = thresh[y:y+h, x:x+w]
        text += ocr_char(char)

    return text

# ---------------------------
# FULL OCR (FALLBACK)
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

# ---------------------------
# FORMAT CLEANING
# ---------------------------
def clean(text):
    text = text.upper().replace(" ", "")
    return text

# ---------------------------
# MAIN PIPELINE
# ---------------------------
img = cv2.imread(IMAGE_PATH)
results = model(img)

for r in results:
    for box in r.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])

        plate = img[y1:y2, x1:x2]

        plate = deskew(plate)
        thresh = preprocess(plate)

        boxes = get_char_boxes(thresh)

        seg_text = segment_ocr(thresh, boxes)
        full_text = full_ocr(plate)

        # 🔥 SMART FUSION LOGIC
        if len(seg_text) >= 6:
            final = seg_text
        else:
            final = full_text

        final = clean(final)

        print("Segment:", seg_text)
        print("Full:", full_text)
        print("FINAL:", final)

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img, final, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imwrite("final_output.jpg", img)
print("DONE")