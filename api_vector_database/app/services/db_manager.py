import pandas as pd  
import chromadb  
from chromadb.utils import embedding_functions  
import google.generativeai as genai  
from app.services.pdf_extractor import extract_text_by_paragraph  
  
def initialize_db(pdf_path, api_key):  
    genai.configure(api_key=api_key)  
  
    extracted_paragraphs = extract_text_by_paragraph(pdf_path)  
    df_paragraphs = pd.DataFrame(extracted_paragraphs)  
    df_paragraphs = df_paragraphs.reset_index().rename(columns={'index': 'ids'})  
  
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')  
    chroma_client = chromadb.Client()  
    db = chroma_client.create_collection(name='Residential_Contract',embedding_function=sentence_transformer_ef)
    db.add(  
        ids=df_paragraphs['ids'].astype(str).tolist(),  
        documents=df_paragraphs['text'].tolist(),  
        metadatas=df_paragraphs.drop(['ids', 'text'], axis=1).to_dict('records')  
    )  
    return db  
  
def retrieve_relevant_snippets(question, db, top_k=3):  
    results = db.query(query_texts=[question], n_results=top_k)  
    snippets = [result for result in results['documents'][0]]  
    return snippets  
