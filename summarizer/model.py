# ============================================================
# summarizer/model.py
# PURPOSE: The LLM that generates the final summary.
#          Uses Facebook's BART model (pre-trained on CNN/DailyMail).
#
# BART = Bidirectional and Auto-Regressive Transformers
# It's great at text summarization out of the box!
# ============================================================

from transformers import pipeline, Pipeline

# ── Load the BART summarization model ──────────────────────
# "facebook/bart-large-cnn" is trained specifically for summarization.
# First run: downloads ~1.6GB model from HuggingFace (cached locally).
#
# Alternative lighter models (less accurate but faster):
#   - "sshleifer/distilbart-cnn-12-6"   (~300MB, faster)
#   - "facebook/bart-large-xsum"         (~1.6GB, more abstractive)
#   - "t5-small"                         (~60MB,  very lightweight)

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


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    text = """
    Artificial intelligence (AI) has rapidly evolved over the past decade,
    transforming industries from healthcare to transportation. In healthcare,
    AI systems can now detect cancer in medical images with accuracy comparable
    to expert radiologists. Self-driving vehicles use deep learning and computer
    vision to navigate complex road environments. In natural language processing,
    large language models like GPT and BART can understand and generate human-like
    text. These advances are driven by improvements in computing power, larger
    datasets, and innovative model architectures. However, concerns about AI safety,
    bias in algorithms, and job displacement remain significant challenges that
    researchers and policymakers are working to address.
    """

    summarizer = Summarizer()
    summary = summarizer.summarize(text, max_length=80, min_length=30)
    print("\n📝 Summary:")
    print(summary)
