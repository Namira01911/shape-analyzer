import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import pytesseract

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title("📐 Smart Shape Analyzer")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(image, caption="Uploaded Image")

    # =========================
    # 🔷 SHAPE DETECTION (FILTERED)
    # =========================

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smooth + edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    data = []
    img_copy = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # ❌ Ignore small contours (text)
        if area < 2000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        sides = len(approx)

        x, y, w, h = cv2.boundingRect(cnt)

        # ❌ Ignore small width/height (text)
        if w < 40 or h < 40:
            continue

        aspect_ratio = round(w / h, 2)

        # Shape classification
        if sides == 3:
            shape = "Triangle"

        elif sides == 4:
            if 0.9 <= aspect_ratio <= 1.1:
                shape = "Square"
            else:
                shape = "Rectangle"

        elif sides > 4:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)

            if circularity > 0.75:
                shape = "Circle"
            else:
                shape = "Polygon"

        else:
            shape = "Unknown"

        # Draw box
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img_copy,
            shape,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Store data
        data.append({
            "Shape": shape,
            "Area (pixels²)": round(area, 2),
            "Perimeter (pixels)": round(perimeter, 2),
            "Aspect Ratio (W/H)": aspect_ratio
        })

    st.image(img_copy, caption="Detected Shapes")

    # Show results table
    if len(data) > 0:
        df = pd.DataFrame(data)
        st.subheader("📊 Results Table")
        st.dataframe(df)
    else:
        st.warning("No valid shapes detected (text is ignored).")

    # =========================
    # 📝 OCR (CLEAR OUTPUT)
    # =========================

    st.subheader("📝 Extracted Text")

    gray_ocr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh_ocr = cv2.adaptiveThreshold(
        gray_ocr,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    st.image(thresh_ocr, caption="OCR Processed Image")

    text = pytesseract.image_to_string(thresh_ocr)

    if text.strip() == "":
        st.warning("No text detected.")
    else:
        st.success("Detected Text:")
        st.code(text)
