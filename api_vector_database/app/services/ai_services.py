import google.generativeai as genai  
  
def generate_answer_gemini(question, snippets, model_name='gemini-1.5-flash'):  
    model = genai.GenerativeModel(model_name)  
    context = " ".join(snippets)  
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"  
    response = model.generate_content(prompt)  
    return response  

