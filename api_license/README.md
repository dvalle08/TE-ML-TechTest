# API License  
  
## Description  
This API is responsible for extracting names and last names from scanned PDF documents, identifying their bounding box coordinates, and providing an endpoint to perform fuzzy matching between the extracted names and the names provided in the request.  

## Project Structure 
```
├── app
│ ├──api
│ │ ├── endpoints
│ │ │ ├── process_document.py
│ ├── main.py
│ └─services
│ ├── nlp.py
│ ├── ocr.py
│ └── utils.py
├── Dockerfile
├── README.md
├── requirements.txt
└── run.sh
```
# Running
## Build the Docker image: 
```
docker build -t api_license .  
```
## Run the container:
```
docker run -d -p 8001:8000 --name api_license_container api_license
```

# Usage Endpoint
***POST /process_document***

### Parameters 
* **file**: Scanned PDF document.
* **name_lastname_pairs**: List of name-last name pairs for fuzzy matching.

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

