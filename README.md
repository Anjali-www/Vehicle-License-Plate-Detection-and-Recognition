# 🚗 Vehicle License Plate Detection & Recognition (ANPR)

An advanced **Automatic Number Plate Recognition (ANPR)** system that detects and recognizes Indian vehicle license plates from images with high accuracy. This project combines **YOLO (Ultralytics)** for detection and **Tesseract OCR** with intelligent post-processing to handle real-world challenges like noise, skew, blur, and OCR misclassification.

---

## 📌 Features

- 🔍 License Plate Detection using YOLO  
- 🔠 Dual OCR System (Segmented + Full Plate Recognition)  
- 🧠 Smart OCR Correction (O↔0, B↔8, T↔1, etc.)  
- 📐 Automatic Deskewing using Hough Transform  
- 🧼 Advanced Image Preprocessing (CLAHE, Thresholding, Filtering)  
- ✅ Indian License Plate Format Validation  
- 📊 Accuracy Evaluation (Exact Match + Character Accuracy)  
- 📁 CSV Output Generation  

---

## 📖 Overview

License plate recognition in real-world scenarios is challenging due to:

- Poor lighting conditions  
- Skewed or angled plates  
- Motion blur  
- OCR misinterpretations (`O ↔ 0`, `B ↔ 8`, `T ↔ 1`)  
- Noise and irregular spacing  


## 🧠 Key Techniques Used

### 🔍 License Plate Detection
- Model: YOLO (Ultralytics)
- Detects bounding boxes of license plates in images  

### 🧼 Image Preprocessing
- Resizing (improves OCR accuracy)  
- CLAHE (contrast enhancement)  
- Bilateral filtering (noise reduction)  
- Adaptive thresholding  
- Morphological operations  

### 📐 Deskewing
- Uses Hough Transform to detect and correct plate rotation  

### 🔠 OCR Strategy
- **Segmented OCR**: character-by-character recognition  
- **Full OCR**: entire plate recognition  
- Best result selected dynamically  

### 🛠 Smart Correction System
Handles OCR errors using:
- Character correction maps:
  - `O → 0`, `B → 8`, `T → 1`, etc.  
- Position-based correction (letters vs digits)  
- Sliding window alignment for noisy OCR output  

### ✅ Format Validation
- Enforces Indian license plate format:
- Validates state codes (UP, MH, DL, RJ, etc.)  

---

## 📂 Project Structure

---

## 📊 Output Format

The system generates a CSV file:

| filename | actual | predicted | exact_match | char_accuracy |
|----------|--------|----------|-------------|---------------|
| UP70DZ0080.png | UP70DZ0080 | UP70DZ0080 | 1 | 1.0 |

---

## 📈 Evaluation Metrics

- **Exact Match Accuracy**  
  Percentage of perfectly predicted plates  

- **Character Accuracy**  
  Measures per-character correctness  

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/anpr-project.git
cd anpr-project

pip install -r requirements.txt

sudo apt install tesseract-ocr

