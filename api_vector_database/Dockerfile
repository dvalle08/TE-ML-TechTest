FROM python:3.10-slim  
  
# Set the working directory  
WORKDIR /app  
  
# Copy the requirements file  
COPY requirements.txt .  
  
# Install the dependencies  
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu  
  
# Copy the rest of the application code  
COPY . .  
  
# Descargar el modelo de HuggingFace durante la construcción del contenedor  
RUN python -c "from chromadb.utils import embedding_functions; sentence_transformer = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')"  

# Ensure the run.sh script is executable  
RUN chmod +x run.sh  
  
# Expose the port the app runs on  
EXPOSE 8000  
  
# Run the run.sh script  
CMD ["./run.sh"]  

