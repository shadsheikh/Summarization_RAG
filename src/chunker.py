def chunk_text(text: str, max_len: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_len
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_len - overlap

    return chunks
