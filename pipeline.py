
import fitz  # PyMuPDF — for reading PDF files
from utils.preprocessing import preprocess_text
from rag.retriever       import Retriever
from summarizer.model    import Summarizer


def read_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file_path: Path to the .pdf file

    Returns:
        Extracted text as a single string
    """
    doc  = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    print(f"✅ PDF loaded: {len(text)} characters from {len(doc)} pages")
    return text


class RAGSummarizerPipeline:
    """
    The complete RAG summarization pipeline.

    Usage:
        pipeline = RAGSummarizerPipeline()
        summary  = pipeline.run(my_text, summary_length="medium")

    Or with a PDF:
        text    = read_pdf("my_document.pdf")
        summary = pipeline.run(text)
    """

    # Map human-readable lengths to token counts
    LENGTH_MAP = {
        "short":  (50,  80),
        "medium": (100, 150),
        "long":   (150, 250),
    }

    def __init__(self, chunk_size: int = 5, top_k: int = 4):
        """
        Args:
            chunk_size: Sentences per chunk
            top_k     : Number of chunks to retrieve for context
        """
        self.retriever  = Retriever(chunk_size=chunk_size, overlap=1)
        self.summarizer = Summarizer()
        self.top_k      = top_k

    def run(
        self,
        text: str,
        summary_length: str = "medium",
        use_rag: bool = True,
    ) -> dict:
        """
        Run the full pipeline.

        Args:
            text           : Input document text
            summary_length : "short", "medium", or "long"
            use_rag        : If True, use RAG; if False, direct summarization

        Returns:
            dict with:
            - summary         : Final summary text
            - num_chunks      : How many chunks were created
            - retrieved_chunks: The chunks used as context (RAG mode)
            - word_count_in   : Word count of input
            - word_count_out  : Word count of summary
        """
        print("\n" + "="*50)
        print("🚀 Starting RAG Summarization Pipeline")
        print("="*50)

        # ── Preprocess ──────────────────────────────
        print("\n📝 Step 1: Preprocessing text...")
        processed = preprocess_text(text)
        clean_text = processed["cleaned_text"]
        print(f"   Words: {len(processed['tokens'])} | "
              f"Sentences: {len(processed['sentences'])}")

        # ── Ingest into FAISS ───────────────────────
        print("\n🗄️  Step 2: Chunking & building vector index...")
        num_chunks = self.retriever.ingest(clean_text)

        # ── Retrieve relevant chunks ────────────────
        retrieved_chunks = []
        if use_rag:
            print(f"\n🔍 Step 3: Retrieving top-{self.top_k} relevant chunks...")
            context = self.retriever.retrieve_all_for_summary(top_k=self.top_k)
            retrieved_chunks = context.split("\n\n")
            print(f"   Retrieved {len(retrieved_chunks)} chunks")
        else:
            context = clean_text
            print("\n⏭️  Step 3: Skipping retrieval (RAG disabled)")

        # ── Summarize ───────────────────────────────
        print("\n✍️  Step 4: Generating summary with BART...")
        min_len, max_len = self.LENGTH_MAP.get(summary_length, (100, 150))

        if use_rag:
            summary = self.summarizer.rag_summarize(
                retrieved_context=context,
                original_text=clean_text,
                max_length=max_len,
                min_length=min_len,
            )
        else:
            summary = self.summarizer.summarize(
                text=clean_text,
                max_length=max_len,
                min_length=min_len,
            )

        # ── Return results ──────────────────────────
        print("\n✅ Pipeline complete!")

        return {
            "summary":          summary,
            "num_chunks":       num_chunks,
            "retrieved_chunks": retrieved_chunks,
            "word_count_in":    len(clean_text.split()),
            "word_count_out":   len(summary.split()),
        }


# ── CLI usage ───────────────────────────────────────────────
if __name__ == "__main__":
    sample_text = """
    Climate change refers to long-term shifts in temperatures and weather patterns.
    These shifts may be natural, such as through variations in the solar cycle.
    But since the 1800s, human activities have been the main driver of climate change,
    primarily due to burning fossil fuels like coal, oil and gas.
    Burning fossil fuels generates greenhouse gas emissions that act like a blanket
    wrapped around the Earth, trapping the sun's heat and raising temperatures.
    Examples of greenhouse gas emissions that are causing climate change include
    carbon dioxide and methane. These come from using gasoline for driving a car or
    coal for heating a building, for example. Clearing land and forests can also
    release carbon dioxide. Landfills for garbage are a major source of methane
    emissions. Energy, industry, transport, buildings, agriculture and land use
    are among the main emitters. Temperatures are already 1.1 degrees Celsius warmer
    than they were in the late 1800s, and emissions continue to rise.
    As a result, the Earth is now about 1.1 degrees Celsius warmer than it was
    at the end of the 19th century. The last decade (2011-2020) was the warmest
    on record. Many people think climate change mainly means warmer temperatures.
    But temperature rise is only the beginning of the story. Because the Earth is
    a system, where everything is connected, changes in one area can influence
    changes in all others. The consequences of climate change now include, among
    others, intense droughts, water scarcity, severe fires, rising sea levels,
    flooding, melting polar ice, catastrophic storms and declining biodiversity.
    """

    # Run pipeline
    pipe   = RAGSummarizerPipeline()
    result = pipe.run(sample_text, summary_length="medium", use_rag=True)

    print("\n" + "="*50)
    print("📄 FINAL SUMMARY:")
    print("="*50)
    print(result["summary"])
    print(f"\n📊 Compression: {result['word_count_in']} → "
          f"{result['word_count_out']} words")
