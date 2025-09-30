from fastapi import FastAPI, UploadFile, File
from ingest import extract_text
from chunker import chunk_text
from embedder import embedder
from rag import summarizer
from transformers import pipeline
import sys
import os
import chromadb

# Initialize client (local, persisted in ./chroma)
chroma_client = chromadb.PersistentClient(path="data/chroma_db")

# Create collection
collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=None  # we’ll provide embeddings manually
)

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

app = FastAPI()

qa_pipeline = pipeline("question-answering",
                       model="deepset/roberta-base-squad2")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_path = f"data/raw/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Extract text from PDF
    text = extract_text(temp_path)

    # Chunk the text
    chunks = chunk_text(text)

    # Generate embeddings
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=[f"{file.filename}_{i}" for i in range(len(chunks))]
    )

    return {"status": "ok", "num_chunks": len(chunks), "filename": file.filename}


@app.post("/summarize")
async def summarize(document_id: str, style: str = "brief"):
    # Get most relevant chunks from ChromaDB
    query_embedding = embedder.encode(
        [document_id], convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    selected_chunks = results["documents"][0]

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

    return {"summary": final_summary}


@app.post("/ask")
async def ask(question: str):
    # Embed the question
    query_embedding = embedder.encode(
        [question], convert_to_numpy=True).tolist()

    # Search top 3 relevant chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    selected_chunks = results["documents"][0]

    # Build context
    context = " ".join(selected_chunks)

    # Get answer from QA pipeline
    result = qa_pipeline({"question": question, "context": context})
    answer = result["answer"]

    return {"question": question, "answer": answer}
