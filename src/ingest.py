import pdfplumber


def extract_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    with open("data/parsed/ingest.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return text
