import fitz  # PyMuPDF  
import pandas as pd  
import chromadb
from chromadb.utils import embedding_functions

def extract_text_by_paragraph(pdf_path):  
    document = fitz.open(pdf_path)  
    data = []  
    for page_num in range(len(document)):  
        page = document.load_page(page_num)  
        blocks = page.get_text("blocks")[2:] #Avoid the footnote
        #blocks[0][4]
        #blocks[1][4]
        #print('\n')
        blocks_len = str(len(blocks))
        for No, block in enumerate(blocks):
            if block[4].strip():  # Ignorar bloques vacíos  
                data.append({  
                    'page': page_num + 1,
                    'block_no': str(No)+' de '+blocks_len,  
                    'text': block[4].strip()  
                })  
    return data 

pdf_path = "data/AS-IS-Residential-Contract-for-Sale-And-Purchase (1).pdf"  
extracted_paragraphs = extract_text_by_paragraph(pdf_path)  

df_paragraphs = pd.DataFrame(extracted_paragraphs)  
df_paragraphs = df_paragraphs.reset_index().rename(columns={'index':'ids'})

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = 'all-MiniLM-L6-v2')

chroma_client = chromadb.Client()
db = chroma_client.create_collection(name='Residential_Contract', embedding_function=sentence_transformer_ef)

db.add(
    ids=df_paragraphs['ids'].astype(str).tolist(),
    documents=df_paragraphs['text'].tolist(),
    metadatas= df_paragraphs.drop(['ids','text'],axis=1).to_dict('records')
)

#!----------------------------------------------------------------
#Queries
db.peek(3)
result = db.query(
    query_texts=['PROPERTY INSPECTIONS AND RIGHT TO CANCEL'],
    n_results=2
)

result.keys()
result['metadatas']
#!------------------------------------------------------------


