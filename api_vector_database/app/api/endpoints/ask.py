from app.services.ai_services import generate_answer_gemini
from app.services.db_manager import retrieve_relevant_snippets
from fastapi import APIRouter, Form, Request

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
    ## COMMENT AV: You can use Pydantic models to define the request and response schemas as I showed you for the other endpoint.
    ## My issue here with app.state is that I do not know what happens with concurrency. If the state is shared between requests, what happens if two requests are requesting the same resource at the same time?.
    ## COMMENT AV: Same as in the ohter endpoint, try to separate the bussiness logic from the request handling logic.
    db = request.app.state.db  # Acceder a db desde el objeto request
    if db is None:
        raise ValueError("Database has not been initialized.")

    retrieved_snippets = retrieve_relevant_snippets(question, db, top_k=5)
    answer = generate_answer_gemini(
        question, retrieved_snippets, model_name="gemini-pro"
    )
    ## COMMENT AV: If you are using a async endpoint, you should use the async version of the function, I do not know if google generativeai has an async version.
    ## if not, you can use run_in_executor to run the blocking operation in a separate thread or process. LLM calls are time consuming, so it's better to run them in a separate thread or async to avoid blocking the event loop.
    return {"answer": answer.text}
