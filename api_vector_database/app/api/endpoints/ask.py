from fastapi import APIRouter, Form, Request  
from app.services.db_manager import retrieve_relevant_snippets  
from app.services.ai_services import generate_answer_gemini  
 
router = APIRouter()  
  
@router.post("/ask")  
async def ask_question(request: Request, question: str = Form(...)):  
    """  
    Handles a POST request to answer a question by retrieving relevant snippets from the database  
    and generating an answer using a language model.  
  
    Args:  
        request (Request): The FastAPI request object, which contains the application state.  
        question (str): The question provided by the user.  
  
    Returns:  
        dict: A dictionary containing the generated answer.  
    """ 
    db = request.app.state.db  # Acceder a db desde el objeto request  
    if db is None:  
        raise ValueError("Database has not been initialized.") 
    
    retrieved_snippets = retrieve_relevant_snippets(question, db, top_k=5)  
    answer = generate_answer_gemini(question, retrieved_snippets, model_name='gemini-pro')  
    return {"answer": answer.text}  