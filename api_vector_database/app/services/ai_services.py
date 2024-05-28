import google.generativeai as genai


def generate_answer_gemini(question, snippets, model_name="gemini-1.5-flash"):
    """
    Generates an answer to a question using a specified generative language model and a context of relevant snippets.

    Args:
        question (str): The question that needs to be answered.
        snippets (list of str): A list of relevant snippets to be used as context for generating the answer.
        model_name (str, optional): The name of the generative model to use. Defaults to 'gemini-1.5-flash'.

    Returns:
        response: The generated response from the language model.
    """
    model = genai.GenerativeModel(model_name)
    context = " ".join(snippets)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = model.generate_content(prompt)
    ## COMMENT AV: You can use a Pydantic model to define the response schema.
    ## Here you need to check for blocking operations, if the model.generate_content is a blocking operation, you should run it in a separate thread or process to avoid blocking the event loop.
    return response
