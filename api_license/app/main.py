from fastapi import FastAPI  
from app.api.endpoints import process_document  
  
app = FastAPI()  
  
app.include_router(process_document.router, prefix="/process-document")  
