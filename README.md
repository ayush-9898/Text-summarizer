#  RAG Text Summarizer

An intelligent document summarization system built with **NLP + Retrieval-Augmented Generation (RAG)**. Paste any article, research paper, or long text and get a clean, accurate summary powered by BART and FAISS.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat&logo=huggingface)
![FAISS](https://img.shields.io/badge/FAISS-CPU-green?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

---

##  Demo

> Paste text → Click Summarize → Get summary with statistics

![App Screenshot](screenshot.png)

---

##  Features

- 📄 **Text & PDF input** — paste text directly or upload a PDF file
- 🧠 **RAG pipeline** — retrieves the most relevant chunks before summarizing
- ⚡ **FAISS vector search** — fast semantic similarity search
- 🤖 **BART summarizer** — abstractive summarization using `facebook/bart-large-cnn`
- 📊 **Statistics** — input words, output words, compression ratio, chunks created
- ⚙️ **Adjustable settings** — summary length, chunk size, top-K retrieval, RAG toggle
- 🌗 **Dark & light theme** — works with Streamlit's built-in theming

---

##  How It Works

```
Your Text / PDF
      ↓
 Preprocessing (NLP)
 Clean → Tokenize → Remove Stopwords → Lemmatize
      ↓
 Chunking
 Split into overlapping sentence windows
      ↓
 Embedding
 Each chunk → 384-dim vector (all-MiniLM-L6-v2)
      ↓
 FAISS Index
 Store all vectors for fast similarity search
      ↓
 Retrieval
 Find top-K most relevant chunks
      ↓
 BART Summarizer
 Generate abstractive summary from retrieved context
      ↓
 Final Summary ✅
```

---

##  Project Structure

```
text_summarizer/
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py   # Clean, tokenize, remove stopwords, lemmatize
│   ├── chunking.py        # Split text into overlapping sentence windows
│   └── embeddings.py      # Convert text → vectors using MiniLM
│
├── rag/
│   ├── __init__.py
│   ├── vector_store.py    # FAISS index — build and search vectors
│   └── retriever.py       # Orchestrate chunking + retrieval
│
├── summarizer/
│   ├── __init__.py
│   └── model.py           # BART summarization model
│
├── pipeline.py            # Full RAG pipeline (ties everything together)
├── evaluate.py            # ROUGE score evaluation
├── app.py                 # Streamlit web interface
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-text-summarizer.git
cd rag-text-summarizer
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install FAISS (separate step)

```bash
# CPU (recommended for most users)
pip install faiss-cpu

# GPU (only if you have an NVIDIA GPU)
pip install faiss-gpu

# Conda users
conda install -c conda-forge faiss-cpu
```

### 5. Install spaCy language model

```bash
python -m spacy download en_core_web_sm

# If the above fails, use the direct link:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

### 6. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 7. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

##  Model Downloads (automatic on first run)

These download automatically from HuggingFace when you first run the app:

| Model | Size | Purpose |
|-------|------|---------|
| `all-MiniLM-L6-v2` | ~80 MB | Text embeddings |
| `facebook/bart-large-cnn` | ~1.6 GB | Summarization |

Models are cached in `~/.cache/huggingface/` and never downloaded again.

### Want a lighter model?

Edit `summarizer/model.py`:

```python
# Default (best quality, 1.6 GB)
MODEL_NAME = "facebook/bart-large-cnn"

# Lighter option (300 MB, still good)
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Lightest option (60 MB, basic quality)
MODEL_NAME = "t5-small"
```

---

##  Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Summary Length | medium | short (~60 words) / medium (~120) / long (~200) |
| Enable RAG | on | Toggle RAG retrieval on or off |
| Chunk Size | 5 | Sentences per chunk |
| Top-K Chunks | 4 | Number of chunks retrieved as context |

---

##  Evaluation

Use ROUGE scores to measure summary quality against a reference:

```python
from evaluate import evaluate_summary, print_evaluation

scores = evaluate_summary(generated_summary, reference_summary)
print_evaluation(scores)

# Output:
#   ROUGE1  F1: 0.4234
#   ROUGE2  F1: 0.2156
#   ROUGEL  F1: 0.3891
```

| Score | Range | Meaning |
|-------|-------|---------|
| 0.0 – 0.2 | Poor | Misses most key points |
| 0.2 – 0.4 | Fair | Captures some information |
| 0.4 – 0.6 | Good | Solid overlap with reference |
| 0.6 – 1.0 | Excellent | Closely matches reference |

---

##  Use in Python directly

```python
from pipeline import RAGSummarizerPipeline

pipeline = RAGSummarizerPipeline()

result = pipeline.run(
    text="Your long document text here...",
    summary_length="medium",  # short / medium / long
    use_rag=True,
)

print(result["summary"])
print(f"Compressed {result['word_count_in']} → {result['word_count_out']} words")
```

---

##  Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` |
| `ModuleNotFoundError: fitz` | `pip install PyMuPDF` |
| `Can't load en_core_web_sm` | `python -m spacy download en_core_web_sm` |
| `404 error on spacy download` | Use the direct `.whl` link in Step 5 above |
| `torch not found` | `pip install torch` |
| Blank summary box in dark theme | Already fixed in latest `app.py` |
| Results disappear on button click | Already fixed with `st.session_state` |
| Slow on first run | Normal — models are downloading, wait it out |
| pip install fails on Windows | Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| NLP | NLTK, spaCy |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | FAISS |
| Summarization | HuggingFace Transformers (BART) |
| PDF Reading | PyMuPDF (fitz) |
| Evaluation | ROUGE Score |
| Web UI | Streamlit |

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

##  Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/) — BART model and pipeline
- [Sentence Transformers](https://www.sbert.net/) — MiniLM embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
- [Streamlit](https://streamlit.io/) — Web interface
- [spaCy](https://spacy.io/) — NLP pipeline
