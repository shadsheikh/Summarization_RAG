import os
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, Form
from transformers import pipeline

# Initialize once (can be large, takes some RAM)
summarizer = pipeline("summarization", model="t5-small")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = embedder.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)
chunks_db = []

app = FastAPI()


def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def chunk_text(text, max_len=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_len
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_len - overlap
    return chunks


@app.post("/upload")
async def upload(file: UploadFile):
    temp_path = "data/raw/Shobha_G_Bagwe_vs_Air_India_Ltd_And_Anr_on_25_August_2025.PDF"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    text = extract_text(temp_path)
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index.add(embeddings)
    chunks_db.extend(chunks)
    return {"status": "ok", "num_chunks": len(chunks)}


@app.post("/upload_local")
async def upload_local():
    # Directly use the PDF path
    temp_path = "data/raw/Shobha_G_Bagwe_vs_Air_India_Ltd_And_Anr_on_25_August_2025.PDF"

    # Extract text from PDF
    text = extract_text(temp_path)

    # Chunk the text
    chunks = chunk_text(text)

    # Generate embeddings
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    # Add embeddings to FAISS index
    index.add(embeddings)

    # Store chunks in memory
    chunks_db.extend(chunks)

    return {"status": "ok", "num_chunks": len(chunks)}


@app.post("/summarize")
async def summarize(document_id: str, style: str = "brief"):
    # Get most relevant chunks from FAISS
    query_embedding = embedder.encode([document_id], convert_to_numpy=True)
    D, I = index.search(query_embedding, 3)
    selected_chunks = [chunks_db[i] for i in I[0] if i < len(chunks_db)]

    # Summarize each chunk separately
    partial_summaries = []
    for chunk in selected_chunks:
        result = summarizer(
            chunk,
            max_length=100 if style == "brief" else 300,
            min_length=30,
            do_sample=False
        )
        partial_summaries.append(result[0]["summary_text"])

    # Combine partial summaries
    combined_text = " ".join(partial_summaries)

    # Do a final summarization pass
    final_summary = summarizer(
        combined_text,
        max_length=150 if style == "brief" else 350,
        min_length=50,
        do_sample=False
    )[0]["summary_text"]

    return {"summary": final_summary, "citations": selected_chunks}
