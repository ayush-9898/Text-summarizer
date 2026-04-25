# ============================================================
# app.py
# PURPOSE: Streamlit web interface for the RAG Summarizer
#
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import fitz  # PyMuPDF for PDF reading

# ── Page config (must be first Streamlit call) ──────────────
st.set_page_config(
    page_title="RAG Text Summarizer",
    page_icon="📝",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; }
    .sub-header  { font-size: 1.1rem; color: #888; margin-bottom: 2rem; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Cache the pipeline to avoid reloading models ────────────
@st.cache_resource
def load_pipeline():
    from pipeline import RAGSummarizerPipeline
    return RAGSummarizerPipeline()


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from an uploaded PDF file."""
    pdf_bytes = uploaded_file.read()
    doc       = fitz.open(stream=pdf_bytes, filetype="pdf")
    text      = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-header">📝 RAG Text Summarizer</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Summarize long documents with '
    'NLP + Retrieval-Augmented Generation (RAG)</div>',
    unsafe_allow_html=True
)

# ── Sidebar — settings ───────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    summary_length = st.select_slider(
        "Summary Length",
        options=["short", "medium", "long"],
        value="medium",
        help="Short: ~60 words | Medium: ~120 words | Long: ~200 words"
    )

    use_rag = st.toggle(
        "Enable RAG",
        value=True,
        help="RAG retrieves the most relevant parts of your document "
             "to improve summary quality and reduce hallucination."
    )

    chunk_size = st.slider(
        "Chunk Size (sentences)",
        min_value=2, max_value=10, value=5,
        help="How many sentences per chunk. "
             "Smaller = more precise retrieval. "
             "Larger = more context per chunk."
    )

    top_k = st.slider(
        "Top-K Chunks",
        min_value=1, max_value=8, value=4,
        help="How many chunks to retrieve for context."
    )

    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    **NLP Steps:**
    1. Text cleaning
    2. Sentence tokenization
    3. Stopword removal
    4. Lemmatization

    **RAG Steps:**
    1. Chunk text
    2. Embed chunks (MiniLM)
    3. Store in FAISS
    4. Retrieve top-K
    5. Summarize with BART
    """)

# ── Input section ────────────────────────────────────────────
st.subheader("📥 Input")

input_mode = st.radio(
    "Choose input method:",
    ["📄 Paste Text", "📂 Upload PDF"],
    horizontal=True
)

text = ""

if input_mode == "📄 Paste Text":
    text = st.text_area(
        "Paste your text here:",
        height=250,
        placeholder="Paste any article, research paper, or long text here...",
    )

else:  # PDF upload
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="The text will be extracted automatically."
    )
    if uploaded_file:
        with st.spinner("📖 Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        st.success(f"✅ PDF loaded: {len(text):,} characters extracted")
        with st.expander("Preview extracted text"):
            st.text(text[:1000] + ("..." if len(text) > 1000 else ""))

# ── Summarize button ─────────────────────────────────────────
col1, col2 = st.columns([1, 4])
with col1:
    summarize_btn = st.button("✨ Summarize", type="primary", use_container_width=True)

if summarize_btn:
    if not text.strip():
        st.warning("⚠️ Please enter some text or upload a PDF first!")
    elif len(text.split()) < 30:
        st.warning("⚠️ Text is too short. Please provide at least 30 words.")
    else:
        with st.spinner("⏳ Loading AI models (first run may take a minute)..."):
            pipeline = load_pipeline()
            pipeline.retriever.chunk_size = chunk_size
            pipeline.top_k = top_k

        with st.spinner("🔄 Running RAG pipeline..."):
            result = pipeline.run(
                text=text,
                summary_length=summary_length,
                use_rag=use_rag,
            )

        # ── Save to session state so results persist ──────────
        st.session_state["result"] = result

# ── Results (shown persistently, survives any button click) ──
if "result" in st.session_state:
    result = st.session_state["result"]

    st.markdown("---")
    st.subheader("📋 Summary")
    st.info(result["summary"])

    # ── Statistics ────────────────────────────────────────────
    st.markdown("### 📊 Statistics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Input Words",    result["word_count_in"])
    with m2:
        st.metric("Output Words",   result["word_count_out"])
    with m3:
        compression = round(
            (1 - result["word_count_out"] / max(result["word_count_in"], 1)) * 100
        )
        st.metric("Compression",    f"{compression}%")
    with m4:
        st.metric("Chunks Created", result["num_chunks"])

    st.success("✅ Done!")