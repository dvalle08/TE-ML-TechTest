FROM python:3.10-slim  
  
# Set the working directory  
WORKDIR /app  
  
# Install system dependencies  
RUN apt-get update && apt-get install -y \  
    tesseract-ocr \  
    libtesseract-dev \  
    libleptonica-dev \  
    && apt-get clean \  
    && rm -rf /var/lib/apt/lists/*  
  
# Copy the requirements file  
COPY requirements.txt .  
  
# Install Python dependencies   
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu  

  
# Copy the application code  
COPY . .  

# Descargar el modelo de HuggingFace durante la construcción del contenedor  
RUN python -c "from transformers import pipeline; classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"  
## COMMENT AV: You can also download the model in the run.sh script or in the server startup script, so the container is smaller to reduce the loading/downloading times from/to the container repo. 
# Ensure the run.sh script is executable  
RUN chmod +x run.sh  
  
# Expose the port the app runs on  
EXPOSE 8000  
  
# Run the run.sh script  
CMD ["./run.sh"]  
