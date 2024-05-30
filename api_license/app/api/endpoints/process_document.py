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
    ## COMMENT AV: You can use Pydantic models to define the request and response schemas.
    ## COMMENT AV: This is a good practice to document the API and validate the input data.

    # class ProcessDocumentRequest(BaseModel):
    #     file: UploadFile
    #     name: str
    
    # class Results(BaseModel):
    #     extracted_name: str
    #     bounding_box: List[int]
    #     fuzzy_similarity: float
    #     similar_name: str
    
    # class ProcessDocumentResponse(BaseModel):
    #     results: List[Results]

    # async def process_document(request: ProcessDocumentRequest) -> ProcessDocumentResponse:

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
    ## COMMENT AV: You need to be careful when mixing async and sync code. OCR is a blocking operation, that will block the server from accepting new requests, so it's better to run it in a separate thread or process.
    ## COMMENT AV: You can use the `concurrent.futures` module to run blocking operations asynchronously with run_in_executor.
    ## COMMENT AV: This will prevent blocking the event loop and improve the performance of your application.
    
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
    ## COMMENT AV: Separate the business logic from the API logic to make your code more modular and testable.
    ## COMMENT AV: You can create a separate module for the business logic and import it in the API endpoint.
    ## COMMENT AV: Like DocumentHandler.process_document(file, name) -> Dict[str, Any] in a separate module. api_license/app/api/handlers/document_handler.py
    return {"results": results}  
