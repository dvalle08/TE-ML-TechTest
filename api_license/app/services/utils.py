import re  
from fuzzywuzzy import fuzz  
  
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
#drawing_bounding_boxes(processed_img.copy(),bounding_boxes)

def fuzzy_match_names(extracted_name, provided_name, threshold=90):  
    fuzzy_similarity = fuzz.ratio(extracted_name, provided_name)  
    similar_name = fuzzy_similarity >= threshold  
    return fuzzy_similarity, similar_name  

