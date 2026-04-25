# ============================================================
# utils/embeddings.py
# PURPOSE: Convert text into numerical vectors (embeddings)
#          so we can compare meaning/similarity between chunks.
# ============================================================

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Load the embedding model ────────────────────────────────
# 'all-MiniLM-L6-v2' is a small, fast, accurate model (80MB).
# It converts any sentence into a 384-dimensional vector.
# First run: downloads the model automatically from HuggingFace.

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


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    sentences = [
        "Artificial intelligence is transforming the world.",
        "Machine learning is a part of AI.",
        "I love eating pizza on weekends.",
    ]

    vecs = get_embeddings(sentences)
    print(f"\n📐 Embedding shape: {vecs.shape}")  # (3, 384)

    # Compare similarity
    sim_ai_ml = cosine_similarity(vecs[0], vecs[1])
    sim_ai_pizza = cosine_similarity(vecs[0], vecs[2])
    print(f"\n🔗 AI ↔ ML similarity    : {sim_ai_ml:.3f}")   # should be HIGH
    print(f"🔗 AI ↔ Pizza similarity : {sim_ai_pizza:.3f}") # should be LOW
