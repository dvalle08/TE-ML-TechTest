from fastapi import APIRouter, File, UploadFile, Form  
from app.services.ocr import extract_text_from_image  
from app.services.nlp import person_name_entity_recognition  
from app.services.utils import clean_and_split_text, find_bounding_boxes, fuzzy_match_names  
import pytesseract  
import cv2  
import numpy as np  
  
router = APIRouter()  
  
@router.post("/")  
async def process_document(file: UploadFile = File(...), name: str = Form(...)):  
    """  
    Processes an uploaded image document, extracts text, and recognizes person names.  
  
    Args:  
        file (UploadFile): The image file uploaded by the user.  
        name (str): The name provided by the user for comparison.  
  
    Returns:  
        dict: A dictionary with the processing results, including the extracted name,  
              bounding box of the name, fuzzy similarity score, and the similar name found.  
    """  
    contents = await file.read()  
    nparr = np.frombuffer(contents, np.uint8)  
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
      
    extracted_text, processed_img = extract_text_from_image(image)  
    cleaned_text = clean_and_split_text(extracted_text)  
    person_names = person_name_entity_recognition(cleaned_text)  
    ocr_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)  
      
    results = []  
    bounding_boxes = find_bounding_boxes(person_names[0].split(), ocr_data)  
  
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
