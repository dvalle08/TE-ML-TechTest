import fitz  # PyMuPDF  
import pandas as pd  
  
def extract_text_by_paragraph(pdf_path):  
    document = fitz.open(pdf_path)  
    data = []  
    for page_num in range(len(document)):  
        page = document.load_page(page_num)  
        blocks = page.get_text("blocks")[2:] #Avoid the footnote
        #blocks[0][4]
        #blocks[1][4]
        #print('\n')
    
        for block in blocks:
            if block[4].strip():  # Ignorar bloques vacíos  
                data.append({  
                    'page': page_num + 1,  
                    'text': block[4].strip()  
                })  
    return data 

pdf_path = "data/AS-IS-Residential-Contract-for-Sale-And-Purchase (1).pdf"  
extracted_paragraphs = extract_text_by_paragraph(pdf_path)  
print(extracted_paragraphs)
# Crear un DataFrame con el texto de cada párrafo  
df_paragraphs = pd.DataFrame(extracted_paragraphs)  
  
# Mostrar el DataFrame  
print(df_paragraphs)  




