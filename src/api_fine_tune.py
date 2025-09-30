from fastapi import FastAPI, UploadFile, File
import os
import shutil
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFacePipeline

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset

# =========================================================
# Configuration
# =========================================================
DATA_DIR = "data/raw"
DB_DIR = "data/chroma_db"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Base model and tokenizer (Flan-T5)
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# Initialize Chroma vector DB
embeddings = HuggingFacePipeline(pipeline=pipeline(
    "feature-extraction", model=MODEL_NAME, tokenizer=tokenizer, device=0 if device == "cuda" else -1))
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

app = FastAPI()

# =========================================================
# Helper: LoRA Fine-tuning
# =========================================================


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def fine_tune_lora(documents: List[str], base_model, tokenizer, epochs=1, lr=1e-4):
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    model.train()
    model.to(device)

    # Prepare dataset
    dataset = TextDataset(documents, tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} loss: {loss.item()}")

    return model

# =========================================================
# Upload Endpoint
# =========================================================


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load PDF and split
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Store in Chroma DB
    vectorstore.add_documents(docs)

    # Fine-tune LoRA on the document chunks (text only)
    text_chunks = [doc.page_content for doc in docs]
    lora_model = fine_tune_lora(text_chunks, base_model, tokenizer, epochs=1)

    # Save LoRA model adapter for reuse
    adapter_dir = os.path.join("data/lora_adapters", file.filename)
    os.makedirs(adapter_dir, exist_ok=True)
    lora_model.save_pretrained(adapter_dir)

    return {"status": "ok", "filename": file.filename, "num_chunks": len(docs)}

# =========================================================
# Ask Endpoint
# =========================================================


@app.post("/ask")
async def ask(question: str, filename: str):
    try:
        # Load LoRA adapter for this file
        adapter_dir = os.path.join("data/lora_adapters", filename)
        if not os.path.exists(adapter_dir):
            return {"error": "LoRA adapter not found. Upload first."}

        lora_model = get_peft_model(
            base_model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM))
        lora_model.load_adapter(adapter_dir)
        lora_model.eval()
        lora_model.to(device)

        qa_pipeline = pipeline("text2text-generation", model=lora_model,
                               tokenizer=tokenizer, device=0 if device == "cuda" else -1)
        qa_llm = HuggingFacePipeline(pipeline=qa_pipeline)
        qa_chain = RetrievalQA.from_chain_type(
            llm=qa_llm, retriever=retriever, chain_type="stuff")

        response = qa_chain.invoke({"query": question})
        answer = response.get("result", str(response)) if isinstance(
            response, dict) else str(response)

        return {"question": question, "answer": answer}
    except Exception as e:
        return {"error": str(e)}

# =========================================================
# Summarize Endpoint
# =========================================================


@app.post("/summarize")
async def summarize(filename: str):
    try:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            return {"error": "File not found. Please upload first."}

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        text_chunks = [doc.page_content for doc in docs]

        # Load LoRA adapter
        adapter_dir = os.path.join("data/lora_adapters", filename)
        if not os.path.exists(adapter_dir):
            return {"error": "LoRA adapter not found. Upload first."}

        lora_model = get_peft_model(
            base_model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM))
        lora_model.load_adapter(adapter_dir)
        lora_model.eval()
        lora_model.to(device)

        sum_pipeline = pipeline("summarization", model=lora_model,
                                tokenizer=tokenizer, device=0 if device == "cuda" else -1)
        sum_llm = HuggingFacePipeline(pipeline=sum_pipeline)
        chain = load_summarize_chain(sum_llm, chain_type="map_reduce")
        result = chain.invoke(docs)
        summary_text = result["output_text"]

        return {"filename": filename, "summary": summary_text}
    except Exception as e:
        return {"error": str(e)}
