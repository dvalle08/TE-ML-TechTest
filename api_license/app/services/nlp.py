from transformers import pipeline  
  
def person_name_entity_recognition(cleaned_text):  
    """  
    Recognizes person names in the cleaned text using a zero-shot classification model.  
  
    Args:  
        cleaned_text (list of str): A list of cleaned text strings to be analyzed for person names.  
  
    Returns:  
        list of str: A list of recognized person names in uppercase.  
    """  
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  
    labels = ["Person Name"]  
    person_names = []  
    for split_t in cleaned_text:  
        results = classifier(split_t, candidate_labels=labels)  
        if results['scores'][0] > 0.95:  
            person_names.append(results['sequence'].upper())  
    return person_names  
