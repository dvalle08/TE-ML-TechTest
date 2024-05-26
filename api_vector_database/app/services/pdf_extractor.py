import fitz  # PyMuPDF  
  
def extract_text_by_paragraph(pdf_path):  
    document = fitz.open(pdf_path)  
    data = []  
    for page_num in range(len(document)):  
        page = document.load_page(page_num)  
        blocks = page.get_text("blocks")[2:]  # Avoid the footnote  
        blocks_len = str(len(blocks))  
        for No, block in enumerate(blocks):  
            if block[4].strip():  # Ignore empty blocks  
                data.append({  
                    'page': page_num + 1,  
                    'block_no': str(No) + ' de ' + blocks_len,  
                    'text': block[4].strip()  
                })  
    return data  
