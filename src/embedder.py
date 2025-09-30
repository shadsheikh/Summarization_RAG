from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = embedder.get_sentence_embedding_dimension()
