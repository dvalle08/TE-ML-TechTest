# app/services/image_processing_service.py  
import cv2  
import numpy as np  
   
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


#image = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)  
image = cv2.imread('data/rotated_30 (1).png') 
show_image(image)
# Enderezar la imagen  
rotated_image = rotate_image(image) 
show_image(rotated_image)

#! Limitaciones, la función se encarga de rotar el documento para que quede horizontal, pero es posible que quede al revés 
#! para eso se necesitaría aplica un OCR para ver si hay palabras legibles de no haber se giraría 180 grados.

def extract_text_from_image(file):  
    # Leer la imagen  
    image = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)  
    # Enderezar la imagen  
    rotated_image = rotate_image(image)  
    # Aplicar OCR  
    text = pytesseract.image_to_string(rotated_image)  
    return text  
