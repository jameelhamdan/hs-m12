import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf_document = fitz.open(stream=f.read(), filetype="pdf")
        full_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            full_text += page.get_text()
    return full_text

