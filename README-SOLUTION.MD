# Document Name Extraction and PDF Search Project

## Description  
This project provides two APIs:  
1. `api_license`: To extract names and last names from scanned PDF documents and perform fuzzy matching.  
2. `api_vector_database`: To store extracted texts in a vector database and perform searches using the RAG (Retrieval-Augmented Generation) strategy.  
  
## Project Structure 
```
├── api_license
│ ├─app
│ │ ├── api
│ │ │ ├── endpoints
│ │ │ │ ├── process_document.py
│ ├── main.py
│ └─services
│ ├── nlp.py
│ ├── ocr.py
│ └── utils.py
│ ├── Dockerfile
│ ├── README.md
│ ├── requirements.txt
│ └── run.sh
├── api_vector_database
│ ├─app
│ │ ├── api
│ │ │ ├── endpoints
│ │ │ │ ├── ask.py
│ ├── main.py
│ └─services
│ ├── ai_services.py
│ ├── db_manager.py
│ ├── pdf_extractor.py
│ ├── data
│ │ └── AS-IS-Residential-Contract-for-Sale-And-Purchase (1).pdf
│ ├── Dockerfile
│ ├── README.md
│ ├── requirements.txt
│ └── run.sh
├── data_license
│ ├── rotated_30 (1).png
│ └── rotated_360 (1).png
├── docker-compose.yml
```
  
# Running
## Using Docker Compose: 
1. Build and run the containers
```
docker-compose up --build   
```

# Usage Endpoint
* `api_license`: Available at http://localhost:8001.

* `api_vector_database`: Available at http://localhost:8002.

# API License
## Description
This API extracts names and last names from scanned PDF documents and performs fuzzy matching between the extracted names and the names provided in the request.

## Endpoint
POST /process-document/
### Parameters
* `file`: Scanned PDF document.

* `name`: Name to perform fuzzy matching against the extracted names.

## Example
Using curl:
```
curl -X POST "http://localhost:8000/process-document/" -F "file=@path_to_your_pdf_file" -F "name=DECKER T" 
```
### Response
```json
{  
  "results": [  
    {  
      "extracted_name": "DEREK THOMAS",  
      "bounding_box": [x1, y1, x2, y2],  
      "fuzzy_similarity": 95,  
      "similar_name": "DECKER T"  
    }  
  ]  
}  

```

# API Vector Database  
  
## Description  
This API provides functionality to store extracted texts from PDF documents in a vector database and perform searches using the RAG (Retrieval-Augmented Generation) strategy, leveraging the LLM Gemini from Google.  

# Configuration
You need to configure a `.env` file with your Google API key for the LLM Gemini.
```
GOOGLE_API_KEY=your_google_api_key
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