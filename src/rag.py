from transformers import pipeline

# Use a lightweight summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
