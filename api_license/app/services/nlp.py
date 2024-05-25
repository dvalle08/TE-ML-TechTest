from transformers import pipeline  
  
def person_name_entity_recognition(cleaned_text):  
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  
    labels = ["Person Name"]  
    person_names = []  
    for split_t in cleaned_text:  
        results = classifier(split_t, candidate_labels=labels)  
        if results['scores'][0] > 0.95:  
            person_names.append(results['sequence'].upper())  
    return person_names  
