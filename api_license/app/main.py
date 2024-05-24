# app/services/image_processing_service.py  
import cv2  
import numpy as np  
import pytesseract
from transformers import pipeline 
import re  
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz  


def show_image(imagen):  
    # Mostrar la imagen en una ventana  
    cv2.imshow('Imagen', imagen)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

def show_image_with_bounding_box(rect):
    box = cv2.boxPoints(rect)  
    box = np.int0(box)  
    image_with_box = cv2.drawContours(image.copy(), [box], 0, (0, 255, 0), 2)  
    show_image(image_with_box)  

def rotate_and_crop_image(image):  
    # Convertir la imagen a escala de grises  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #show_image(gray)  
    # Aplicar el umbral para obtener una imagen binaria  
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   
    #show_image(binary)  

    # Encontrar los contornos  
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    # Obtener el contorno más grande  
    largest_contour = max(contours, key=cv2.contourArea)  

    #image_contours = cv2.drawContours(image.copy(), [largest_contour], -1, (0, 255, 0), 3)  
    #show_image(image_contours)  

    # Obtener el rectángulo delimitador con el ángulo de rotación  
    rect = cv2.minAreaRect(largest_contour)  
    angle = rect[-1]  
  
    if angle < -45:  
        angle = -(90 + angle)  
    else:  
        angle = -angle  
  
    # Rotar la imagen  
    (h, w) = image.shape[:2]  
    center = (w // 2, h // 2)  
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)  
  
    return rotated  

def get_document_contour(image):
    # Convertir la imagen a escala de grises  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    #show_image(gray)

    # Aplicar un desenfoque gaussiano  
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  
    #show_image(blurred)

    # Utilizar la detección de bordes de Canny  
    edged = cv2.Canny(blurred, 50, 150)  
    #show_image(edged)

    # Aplicar una transformación morfológica para cerrar pequeños huecos en los bordes  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))  
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)    
    #show_image(closed)

    # Encontrar los contornos  
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
  
    # Obtener el contorno más grande  
    largest_contour = max(contours, key=cv2.contourArea)  
    
    #image_contours = cv2.drawContours(image.copy(), [largest_contour], -1, (0, 255, 0), 3)  
    #show_image(image_contours)   

    # Obtener el rectángulo delimitador con el ángulo de rotación  
    rect = cv2.minAreaRect(largest_contour)  
    return rect

def rotate_image(image):  

    rect = get_document_contour(image)

    # Dibujar el rectángulo en la imagen  
    #?show_image_with_bounding_box(rect)

    # Obtener el ángulo de rotación  
    angle = rect[-1]  
    (width, height) = rect[1]  
  
    if width < height:  
        #angle = 90+ angle 
        angle = angle -90
    else:  
        angle = -angle  

    # Rotar la imagen  
    (h, w) = image.shape[:2]  
    center = (w // 2, h // 2)  
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)  

    return rotated  

def cropping_image(rotated_image):
    rect = get_document_contour(rotated_image)

    #?show_image_with_bounding_box(rect)
    box = cv2.boxPoints(rect)
    box = np.int0(box)  

    x, y, w, h = cv2.boundingRect(box)  
    # Recortar la imagen  
    crop_img = rotated_image[y:y+h, x:x+w]  
    #show_image(crop_img)
    return crop_img  


#! Limitaciones, la función se encarga de rotar el documento para que quede horizontal, pero es posible que quede al revés 
#! para eso se necesitaría aplica un OCR para ver si hay palabras legibles de no haber se giraría 180 grados.

#ToDo: Implement tesseract
def extract_text_from_image(image):  

    # Enderezar la imagen  
    rotated_image = rotate_image(image) 
    crop_img = cropping_image(rotated_image)
    # Aplicar OCR  
    text = pytesseract.image_to_string(crop_img)  
    return text, crop_img  

image = cv2.imread('data/rotated_30 (1).png') 
#image = cv2.imread('data/rotated_360 (1).png') 
show_image(image) 
extracted_text, processed_img = extract_text_from_image(image)  
show_image(processed_img) 


#ToDo: Implement entity recognition Model

def clean_and_split_text(text):  
    def clean_text(text):  
        # Eliminar caracteres numéricos  
        text = re.sub(r'\d', '', text)  
        # Remover caracteres no alfanuméricos excepto espacios  
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  
        # Reemplazar múltiples espacios por un solo espacio  
        text = re.sub(r'\s+', ' ', text)  
        return text.strip()  

    text = extracted_text
    blocks = text.split('\n')  
    # Limpiar los bloques y eliminar bloques vacíos  
    blocks = [block.strip() for block in blocks if block.strip()]
    clean_blocks =[clean_text(bk) for bk in blocks]
    clean_blocks = [c_bk.lower() for c_bk in clean_blocks if c_bk not in ['', ' ']]

    return clean_blocks  
def person_name_entity_recognition(cleaned_text):
    # Inicializar el pipeline de clasificación de cero disparos  
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  

    # Etiquetas para la clasificación  
    labels = ["Person Name"]  
    
    # Realizar la clasificación de cero disparos  
    Person_name=[]
    for split_t in cleaned_text: 
        results = classifier(split_t, candidate_labels=labels)  
        if results['scores'][0]>0.95:
            Person_name.append(results['sequence'].upper())
    return Person_name


cleaned_text = clean_and_split_text(extracted_text)  
print("Texto limpio:")  
print(cleaned_text)  

person_name = person_name_entity_recognition(cleaned_text)


# ToDo: Parse bounding Boxes from image:
# Dibujar las cajas delimitadoras  
def drawing_bounding_boxes(processed_img, bounding_boxes):
    for bbox in bounding_boxes:  
        left = bbox['left']  
        top = bbox['top']  
        right = left + bbox['width']  
        bottom = top + bbox['height']  
        # Dibujar el rectángulo  
        cv2.rectangle(processed_img, (left, top), (right, bottom), (0, 0, 255), 2)  
        # Añadir el texto  
        cv2.putText(processed_img, bbox['name'], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)  
        #cv2.putText(processed_img, bbox['name'], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  
    
    # Convertir la imagen de BGR (formato de OpenCV) a RGB (formato de Matplotlib)  
    image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)  
    
    # Mostrar la imagen con las cajas delimitadoras  
    plt.figure(figsize=(10, 10))  
    plt.imshow(image_rgb)  
    plt.axis('off')  
    plt.show()  

# Función para encontrar las coordenadas de las cajas delimitadoras  
def find_bounding_boxes(detected_names, ocr_data):  
    #detected_names, ocr_data = data['text'], data
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
  
# Extraer texto limpio con coordenadas  
data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)  

# Encontrar las cajas delimitadoras para los nombres identificados  
bounding_boxes = find_bounding_boxes(person_name[0].split(), data)  
#drawing_bounding_boxes(processed_img.copy(),bounding_boxes)


#ToDo: Return the extracted names, their bounding box coordinates, and the fuzzy matching results as a JSON response.
  
def fuzzy_match_names(extracted_name, provided_name, threshold=90):  
    fuzzy_similarity = fuzz.ratio(extracted_name, provided_name) 
    if fuzzy_similarity >= threshold: similar_name = True
    else: similar_name = False
    return fuzzy_similarity, similar_name

fuzzy_similarity, similar_name = fuzzy_match_names(person_name[0], 'DEREK T.')  







# Formatear el resultado como JSON  
import json  
output = {"person_names": detected_person_names}  
  
# Imprimir el resultado  
print(json.dumps(output, indent=2))  