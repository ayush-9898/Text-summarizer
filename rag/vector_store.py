
import faiss
import numpy as np
from utils.embeddings import get_embeddings, get_single_embedding


class VectorStore:
    """
    A wrapper around FAISS that stores text chunks and their vectors.

    How it works:
    1. You add text chunks → they get converted to vectors → stored in FAISS
    2. You search with a query → query is vectorized → FAISS finds similar chunks
    3. You get back the most relevant text chunks!
    """

    def __init__(self):
        self.index  = None    # FAISS index (the search engine)
        self.chunks = []      # Original text chunks (we keep these separately)
        self.dim    = 384     # Vector size from all-MiniLM-L6-v2

    def build_index(self, chunks: list[str]) -> None:
        """
        Convert chunks to vectors and build the FAISS search index.

        Args:
            chunks: List of text strings to index
        """
        print(f"🔨 Building FAISS index for {len(chunks)} chunks...")

        # Store original text for later retrieval
        self.chunks = chunks

        # Convert all chunks to vectors
        embeddings = get_embeddings(chunks)

        # FAISS requires float32 data type
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize vectors (makes cosine similarity = dot product, faster)
        faiss.normalize_L2(embeddings)

        # Create a flat (exact) L2/cosine index
        # IndexFlatIP = Inner Product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.dim)

        # Add all vectors to the index
        self.index.add(embeddings)

        print(f"✅ Index built! Total vectors: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Find the top-k most relevant chunks for a given query.

        Args:
            query : The user's question or topic
            top_k : How many chunks to retrieve (default 3)

        Returns:
            List of dicts, each with:
            - 'chunk'  : the matching text
            - 'score'  : similarity score (higher = more similar)
            - 'index'  : position in original chunk list
        """
        if self.index is None:
            raise ValueError("❌ Index not built yet! Call build_index() first.")

        # Convert query to vector
        query_vec = get_single_embedding(query)
        query_vec = np.array([query_vec], dtype=np.float32)

        # Normalize query vector
        faiss.normalize_L2(query_vec)

        # Search! Returns (scores, indices) arrays
        scores, indices = self.index.search(query_vec, top_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1 means no result found
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "index": int(idx),
                })

        return results

    def get_chunk_count(self) -> int:
        """Return how many chunks are stored."""
        return len(self.chunks)


