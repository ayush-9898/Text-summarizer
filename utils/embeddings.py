
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Load the embedding model ────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
print(f"⏳ Loading embedding model: {MODEL_NAME} ...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("✅ Embedding model loaded!")


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        texts: List of strings to embed

    Returns:
        numpy array of shape (num_texts, 384)
        Each row is a 384-dimensional vector for one text.

    Example:
        texts = ["Hello world", "AI is great"]
        → array([[0.12, -0.34, ...],   # vector for "Hello world"
                 [0.56,  0.78, ...]])  # vector for "AI is great"
    """
    # show_progress_bar=True shows a progress bar for large batches
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


def get_single_embedding(text: str) -> np.ndarray:
    """
    Convert a single text string into an embedding vector.
    Useful for embedding the user's query.

    Returns:
        1D numpy array of shape (384,)
    """
    return embedding_model.encode(text, convert_to_numpy=True)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Measure how similar two vectors are (range: -1 to 1).
    1.0  = identical meaning
    0.0  = unrelated
    -1.0 = opposite meaning

    This is what FAISS does internally when searching.
    """
    dot_product = np.dot(vec1, vec2)
    norm        = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm == 0:
        return 0.0
    return dot_product / norm



