import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        return ""
    doc = fitz.open(pdf_file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
