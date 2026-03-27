import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import easyocr  # OCR without Tesseract

st.title("📐 Smart Shape Analyzer + OCR")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(image, caption="Uploaded Image")

    # =========================
    # 🔷 SHAPE DETECTION
    # =========================

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    data = []
    img_copy = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        sides = len(approx)
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 40:
            continue

        aspect_ratio = round(w / h, 2)

        # Shape classification
        if sides == 3:
            shape = "Triangle"
        elif sides == 4:
            shape = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        elif sides > 4:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            shape = "Circle" if circularity > 0.75 else "Polygon"
        else:
            shape = "Unknown"

        # Draw box
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_copy, shape, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
    # 📝 OCR using EasyOCR
    # =========================

    st.subheader("📝 Extracted Text")
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)

    text = " ".join([res[1] for res in result])

    if text.strip() == "":
        st.warning("No text detected.")
    else:
        st.success("Detected Text:")
        st.code(text)
