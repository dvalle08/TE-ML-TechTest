from fastapi import FastAPI, File, UploadFile, Form  
from typing import List, Dict  
import pytesseract  
from fuzzywuzzy import fuzz  
import cv2  
import numpy as np  
from transformers import pipeline  
import re  
  
app = FastAPI()  
  
@app.post("/process-document/")  
async def process_document(file: UploadFile = File(...), name: str = Form(...)):  
    # Leer la imagen desde el archivo subido  
    contents = await file.read()  
    nparr = np.frombuffer(contents, np.uint8)  
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
  
    # Preprocesar la imagen y extraer texto  
    extracted_text, processed_img = extract_text_from_image(image)  
  
    # Limpiar y dividir el texto  
    cleaned_text = clean_and_split_text(extracted_text)  
  
    # Reconocer nombres de personas  
    person_names = person_name_entity_recognition(cleaned_text)  
  
    # Extraer datos OCR  
    ocr_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)  
  
    results = []  

    bounding_boxes = find_bounding_boxes(person_names[0].split(), ocr_data)  

    # Depuración: Imprimir el nombre recibido y los nombres reconocidos  
    print(f"Nombre proporcionado: {name}")  
    print(f"Nombres reconocidos: {person_names}")  

    fuzzy_similarity, similar_name = fuzzy_match_names(person_names[0], name)  
    results.append({  
        "extracted_name": person_names[0],  
        "bounding_box": bounding_boxes,  
        "fuzzy_similarity": fuzzy_similarity,  
        "similar_name": similar_name  
    })  

    return {"results": results}  
  
def extract_text_from_image(image):  
    rotated_image = rotate_image(image)  
    crop_img = cropping_image(rotated_image)  
    text = pytesseract.image_to_string(crop_img)  
    return text, crop_img  
  
def rotate_image(image):  
    rect = get_document_contour(image)  
    angle = rect[-1]  
    (width, height) = rect[1]  
    if width < height:  
        angle = angle - 90  
    else:  
        angle = -angle  
    (h, w) = image.shape[:2]  
    center = (w // 2, h // 2)  
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)  
    return rotated  
  
def cropping_image(rotated_image):  
    rect = get_document_contour(rotated_image)  
    box = cv2.boxPoints(rect)  
    box = np.int0(box)  
    x, y, w, h = cv2.boundingRect(box)  
    crop_img = rotated_image[y:y+h, x:x+w]  
    return crop_img  
  
def get_document_contour(image):  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  
    edged = cv2.Canny(blurred, 50, 150)  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))  
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)  
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    largest_contour = max(contours, key=cv2.contourArea)  
    rect = cv2.minAreaRect(largest_contour)  
    return rect  
  
def clean_and_split_text(text):  
    def clean_text(text):  
        text = re.sub(r'\d', '', text)  
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  
        text = re.sub(r'\s+', ' ', text)  
        return text.strip()  
    blocks = text.split('\n')  
    blocks = [block.strip() for block in blocks if block.strip()]  
    clean_blocks = [clean_text(bk) for bk in blocks]  
    clean_blocks = [c_bk.lower() for c_bk in clean_blocks if c_bk not in ['', ' ']]  
    return clean_blocks  
  
def person_name_entity_recognition(cleaned_text):  
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  
    labels = ["Person Name"]  
    person_names = []  
    for split_t in cleaned_text:  
        results = classifier(split_t, candidate_labels=labels)  
        if results['scores'][0] > 0.95:  
            person_names.append(results['sequence'].upper())  
    return person_names  
  
def find_bounding_boxes(detected_names, ocr_data):  
    bounding_boxes = []  
    for name in detected_names:  
        for i, word in enumerate(ocr_data['text']):  
            if name.lower() in word.lower() and int(ocr_data['conf'][i]) > 60:  
                bbox = {  
                    'name': name,  
                    'left': ocr_data['left'][i],  
                    'top': ocr_data['top'][i],  
                    'width': ocr_data['width'][i],  
                    'height': ocr_data['height'][i]  
                }  
                bounding_boxes.append(bbox)  
    return bounding_boxes  
  
def fuzzy_match_names(extracted_name, provided_name, threshold=90):  
    fuzzy_similarity = fuzz.ratio(extracted_name, provided_name)  
    similar_name = fuzzy_similarity >= threshold  
    return fuzzy_similarity, similar_name  
