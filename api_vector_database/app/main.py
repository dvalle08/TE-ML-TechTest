import os
from contextlib import asynccontextmanager

from app.api.endpoints import ask
from app.services.db_manager import initialize_db
from fastapi import FastAPI


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
    api_key = os.getenv("GOOGLE_API_KEY")
    ## For this type of checkings I would suggest to use pydantic-settings, It will make your life easier.
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Please set the environment variable with your Google API Key."
        )

    pdf_path = "data/AS-IS-Residential-Contract-for-Sale-And-Purchase (1).pdf"

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    else:
        print(f"The file {pdf_path} was found.")

    app.state.db = initialize_db(pdf_path, api_key)  # Adjuntar db al objeto app
    ## I do not know if the chroma client requieres to be closed, i think it would be better to have an independent handler outside the context manager to manage the db connection.

    yield  # Esto permite que la aplicación continúe y acepte solicitudes.


app = FastAPI(lifespan=lifespan)

# Include endpoints
app.include_router(ask.router)
