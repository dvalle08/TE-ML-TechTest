import os  
from fastapi import FastAPI  
from app.api.endpoints import ask  
from app.services.db_manager import initialize_db  
from contextlib import asynccontextmanager  

  
@asynccontextmanager  
async def lifespan(app: FastAPI):  
    # Verificar que la variable de entorno GOOGLE_API_KEY esté configurada  
    api_key = os.getenv('GOOGLE_API_KEY')  
    if not api_key:  
        raise EnvironmentError("GOOGLE_API_KEY is not set. Please set the environment variable with your Google API Key.")  
  
    pdf_path = "data/AS-IS-Residential-Contract-for-Sale-And-Purchase (1).pdf"  

    if not os.path.exists(pdf_path):  
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")  
    else:  
        print(f"The file {pdf_path} was found.") 

    app.state.db = initialize_db(pdf_path, api_key)  # Adjuntar db al objeto app 

    yield  # Esto permite que la aplicación continúe y acepte solicitudes.  
  
app = FastAPI(lifespan=lifespan)  
  
# Include endpoints  
app.include_router(ask.router)  
