# ============================================================
# utils/chunking.py
# PURPOSE: Split long text into smaller overlapping chunks
#          so the embedding model can process them easily.
# ============================================================

from utils.preprocessing import tokenize_sentences


def chunk_by_sentences(text: str, chunk_size: int = 5, overlap: int = 1) -> list[str]:
    """
    Split text into chunks of N sentences each, with overlap.

    WHY OVERLAP? So that important context at chunk boundaries
    is not lost — each chunk shares 1 sentence with the next.

    Args:
        text       : The full document text
        chunk_size : How many sentences per chunk (default 5)
        overlap    : How many sentences to share between chunks (default 1)

    Returns:
        List of text chunks (strings)

    Example (chunk_size=3, overlap=1):
        sentences = [S1, S2, S3, S4, S5]
        Chunk 1   = [S1, S2, S3]
        Chunk 2   = [S3, S4, S5]  ← S3 is shared (overlap)
    """
    sentences = tokenize_sentences(text)

    if not sentences:
        return []

    chunks = []
    step   = chunk_size - overlap          # how far to move the window each time
    i      = 0

    while i < len(sentences):
        # Take a window of sentences
        window = sentences[i : i + chunk_size]
        chunk  = " ".join(window)          # join sentences into one string
        chunks.append(chunk)
        i += step                          # slide the window forward

    return chunks


def chunk_by_characters(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into chunks of N characters each, with overlap.

    Useful when you want consistent chunk sizes regardless of sentence length.

    Args:
        text       : The full document text
        chunk_size : Max characters per chunk (default 500)
        overlap    : Characters to share between chunks (default 50)
    """
    chunks = []
    step   = chunk_size - overlap
    i      = 0

    while i < len(text):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += step

    return chunks


def print_chunks(chunks: list[str]) -> None:
    """Helper to display chunks nicely during development."""
    print(f"\n📦 Total chunks: {len(chunks)}")
    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {idx + 1} ---")
        print(chunk[:200], "..." if len(chunk) > 200 else "")


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Artificial intelligence is intelligence demonstrated by machines.
    It is used in many industries around the world.
    Machine learning is a subset of AI.
    Deep learning is a subset of machine learning.
    Neural networks are the backbone of deep learning.
    Natural language processing helps computers understand text.
    Computer vision helps machines interpret images.
    """
    chunks = chunk_by_sentences(sample, chunk_size=3, overlap=1)
    print_chunks(chunks)
