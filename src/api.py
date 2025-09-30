from fastapi import FastAPI, UploadFile, File
import os
import shutil
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from transformers import pipeline

# =========================================================
# Configuration
# =========================================================
DATA_DIR = "data/raw"
DB_DIR = "data/chroma_db"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Embeddings (MiniLM for vector search)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Chroma vector DB (persistent)
vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

# LLM for Q&A (Flan-T5)
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
qa_llm = HuggingFacePipeline(pipeline=qa_model)

# LLM for Summarization (DistilBART)
sum_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
sum_llm = HuggingFacePipeline(pipeline=sum_model)

app = FastAPI()


# =========================================================
# Upload Endpoint
# =========================================================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIR, file.filename)

    # Save file locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Generate unique user_id
    user_id = str(uuid.uuid4())

    # Load PDF with LangChain
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Inject metadata into each chunk
    for d in docs:
        d.metadata["user_id"] = user_id
        d.metadata["filename"] = file.filename

    # Store in Chroma DB
    vectorstore.add_documents(docs)

    return {"status": "ok", "user_id": user_id, "filename": file.filename, "num_chunks": len(docs)}


# =========================================================
# Ask Endpoint (Q&A over document)
# =========================================================
@app.post("/ask")
async def ask(question: str, filename: str, user_id: str):
    try:
        # Create a retriever scoped to user_id + filename
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": {
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"filename": {"$eq": filename}}
                    ]
                }
            }
        )

        qa_chain = RetrievalQA.from_chain_type(llm=qa_llm, retriever=retriever)
        response = qa_chain.invoke({"query": question})

        if isinstance(response, dict):
            answer = response.get("result", response)
        else:
            answer = str(response)

        return {"question": question, "answer": answer, "filename": filename, "user_id": user_id}

    except Exception as e:
        return {"error": str(e)}


# =========================================================
# Summarize Endpoint
# =========================================================
@app.post("/summarize")
async def summarize(filename: str, user_id: str):
    try:
        # Get all relevant docs for this user + file
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": {
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"filename": {"$eq": filename}}
                    ]
                }
            }
        )
        docs = retriever.invoke("Summarize this document")

        if not docs:
            return {"error": "No documents found for this file/user. Please upload first."}

        chain = load_summarize_chain(sum_llm, chain_type="map_reduce")
        result = chain.invoke(docs)
        summary_text = result["output_text"]

        return {"filename": filename, "user_id": user_id, "summary": summary_text}

    except Exception as e:
        return {"error": str(e)}
