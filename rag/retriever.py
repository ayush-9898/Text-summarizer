
from rag.vector_store import VectorStore
from utils.chunking  import chunk_by_sentences


class Retriever:
    """
    Orchestrates the RAG retrieval pipeline:
    1. Takes a full document
    2. Chunks it
    3. Builds a FAISS vector index
    4. Retrieves relevant chunks for any query
    """

    def __init__(self, chunk_size: int = 5, overlap: int = 1):
        """
        Args:
            chunk_size: Sentences per chunk (tune this!)
            overlap   : Shared sentences between consecutive chunks
        """
        self.vector_store = VectorStore()
        self.chunk_size   = chunk_size
        self.overlap      = overlap

    def ingest(self, text: str) -> int:
        """
        Process a document: chunk it → embed → store in FAISS.

        Args:
            text: Full document text

        Returns:
            Number of chunks created
        """
        print("📥 Ingesting document...")

        # Step 1: Split into chunks
        chunks = chunk_by_sentences(text, self.chunk_size, self.overlap)
        print(f"   📦 Created {len(chunks)} chunks")

        # Step 2: Build FAISS index
        self.vector_store.build_index(chunks)

        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """
        Retrieve the top-k most relevant text chunks for a query.

        Args:
            query : What you're looking for (or a summary topic)
            top_k : How many chunks to get back

        Returns:
            List of relevant text strings
        """
        results = self.vector_store.search(query, top_k=top_k)

        # Sort by chunk position so the context reads naturally
        results.sort(key=lambda r: r["index"])

        return [r["chunk"] for r in results]

    def retrieve_all_for_summary(self, top_k: int = 5) -> str:
        """
        A special method for summarization:
        Uses a generic "summarize" query to get top chunks,
        then joins them as context for the LLM.

        Returns:
            Single string with all retrieved chunks joined
        """
        # We use a generic query — FAISS returns the most "central" chunks
        query = "main topic key points important information summary"
        chunks = self.retrieve(query, top_k=top_k)
        return "\n\n".join(chunks)



