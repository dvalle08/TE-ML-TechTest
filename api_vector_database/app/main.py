import os  
from fastapi import FastAPI  
from app.api.endpoints import ask  
from app.services.db_manager import initialize_db  
from contextlib import asynccontextmanager  

  
@asynccontextmanager  
async def lifespan(app: FastAPI):  
    """  
    Manages the lifespan of the FastAPI application, ensuring necessary environment variables and files are set up,  
    and initializes the database.  
  
    Args:  
        app (FastAPI): The FastAPI application instance.  
  
    Yields:  
        None: This context manager yields control back to the application to continue and accept requests.  
    """  
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
