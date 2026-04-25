
from transformers import pipeline, Pipeline

# ── Load the BART summarization model ──────────────────────

MODEL_NAME = "facebook/bart-large-cnn"


def load_summarizer() -> Pipeline:
    """Load and return the HuggingFace summarization pipeline."""
    print(f"⏳ Loading summarization model: {MODEL_NAME} ...")
    summarizer = pipeline("summarization", model=MODEL_NAME)
    print("✅ Summarization model loaded!")
    return summarizer


class Summarizer:
    """
    Wrapper around the BART model.
    Handles both:
    - Simple summarization (direct text input)
    - RAG summarization (retrieved context + text)
    """

    def __init__(self):
        self.model = load_summarizer()

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50,
    ) -> str:
        """
        Generate a summary for a given text.

        Args:
            text       : Input text to summarize
            max_length : Maximum tokens in summary (default 150)
            min_length : Minimum tokens in summary (default 50)

        Returns:
            Summary string
        """
        # BART has a 1024 token limit — truncate if needed
        # (rough rule: 1 token ≈ 4 characters)
        max_input_chars = 3000
        if len(text) > max_input_chars:
            text = text[:max_input_chars]

        result = self.model(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,     # deterministic output (no randomness)
        )

        return result[0]["summary_text"]

    def rag_summarize(
        self,
        retrieved_context: str,
        original_text: str,
        max_length: int = 200,
        min_length: int = 50,
    ) -> str:
        """
        RAG-enhanced summarization:
        Combines retrieved context + original text for a better summary.

        The key insight: by providing RELEVANT chunks as context,
        the LLM focuses on the most important parts, reducing hallucination.

        Args:
            retrieved_context : Top-k relevant chunks from FAISS
            original_text     : The beginning of the original document
            max_length        : Max summary length
            min_length        : Min summary length

        Returns:
            Enhanced summary string
        """
        # Combine context + original text
        # We give the model a clear structure to follow
        combined = (
            f"Key information:\n{retrieved_context}\n\n"
            f"Document excerpt:\n{original_text[:1000]}"  # first 1000 chars
        )

        return self.summarize(combined, max_length, min_length)



