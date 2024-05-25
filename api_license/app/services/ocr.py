import pytesseract  
import cv2  
import numpy as np  
  
def extract_text_from_image(image):  
    rotated_image = rotate_image(image)  
    crop_img = cropping_image(rotated_image)  
    text = pytesseract.image_to_string(crop_img)  
    return text, crop_img  
  
def rotate_image(image):  
    rect = get_document_contour(image)  
    #?show_image_with_bounding_box(rect, image.copy())  
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
    #?show_image_with_bounding_box(rect, image.copy())  
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
