from transformers import pipeline

# Use a lightweight summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
