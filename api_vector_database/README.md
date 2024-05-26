# API API Vector Database  
  
## Description  
This API provides functionality to store extracted texts from PDF documents in a vector database and perform searches using the RAG (Retrieval-Augmented Generation) strategy, leveraging the LLM Gemini from Google.  

## Project Structure 
```
├── app
│ ├─api
│ │ ├── endpoints
│ │ │ ├── ask.py
│ ├── main.py
│ └─services
│ ├── ai_services.py
│ ├── db_manager.py
│ ├── pdf_extractor.py
├── data
│ └── AS-IS-Residential-Contract-for-Sale-And-Purchase (1).pdf
├── Dockerfile
├── README.md
├── requirements.txt
└── run.sh
```
# Configuration
You need to configure a `.env` file with your Google API key for the LLM Gemini.
```
GOOGLE_API_KEY=your_google_api_key
```

# Running
## Build the Docker image: 
```
docker build -t vectordb_api .   
```
## Run the container:
```
sudo docker run -d --name vectordb_api -p 8000:8000 --env-file .env vectordb_api
```

# Usage Endpoint
***POST /ask***

### Parameters 
* **question**: Question in text format.

## Example
Using curl:
```
curl -X POST "http://127.0.0.1:8000/ask" -F "question=What are the terms of the residential contract?" 
```
### Response
```json
{  
  "answer": "Generated answer based on the RAG strategy"  
}  
```
