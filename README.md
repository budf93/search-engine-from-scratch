# Search Engine From Scratch 🚀

Search Engine From Scratch is a highly optimized, fully featured custom Information Retrieval (IR) system extending classic inverted indexing techniques with state-of-the-art Neural Search algorithms. 

It includes both a command-line pipeline for evaluating standard benchmark queries and a sleek, interactive glassmorphic web dashboard to visually experience latency scaling, compare model metrics (NDCG, MAP, DCG), and conduct real-time searches.

## ✨ Features

### 1. Indexing & Compression
- **BSBI (Blocked Sort-Based Indexing):** Disk-based index construction.
- **SPIMI (Single Pass In-Memory Indexing):** Speedy, memory-based index generation framework. 
- **Postings Compression:** Supports Standard integer arrays, **Variable-Byte (V-Byte)**, and **Elias-Gamma** compression to shrink index footprint drastically.

### 2. Traditional Sparse Retrieval
- **TF-IDF:** Baselines weighting.
- **BM25:** State-of-the-art probabilistic scoring.
- **WAND (Weak AND):** Massive latency optimization loop allowing Top-K approximations without exhaustively scoring every document.

### 3. Neural Search & Hybrid Expansion
- **SPLADE (Sparse Lexical and Expansion Model):** Uses a BERT Masked Language Model (`naver/splade-cocondenser-ensembledistil`) to hallucinate highly relevant terminology related to your query, dynamically weighting and expanding it across the existing sparse inverted index.
- **ColBERT Re-ranking:** Uses a Bi-Encoder Late-Interaction Transformer (`sentence-transformers/msmarco-distilbert-base-tas-b`) to re-score the top 100 BM25-retrieved candidates with extreme precision.

### 4. Evaluation Suite
Automated execution of predefined queries against `qrels.txt` to calculate:
- **RBP** (Rank-Biased Precision)
- **DCG & NDCG** (Discounted Cumulative Gain)
- **AP / MAP** (Mean Average Precision)

---

## 💻 Getting Started (Installation)

### Prerequisites
Make sure you have Python 3 + installed on your machine.

1. **Install Dependencies:**
The neural models require PyTorch and HuggingFace Transformers, alongside Flask for the web UI.
```bash
pip install -r requirements.txt
```

*(Note: Downloading the SPLADE and ColBERT transformers upon first run will require an active internet connection and will store roughly ~500MB of weights locally).*

---

## 🌐 Running the Interactive Web App (Recommended)

The easiest way to experience the features is to run the local Flask server. It comes with a modern frontend to test queries and benchmark the models against each other.

```bash
python app.py
```

Once the server says `* Running on http://127.0.0.1:5000`:
1. Open **http://127.0.0.1:5000** in your web browser.
2. Select your Indexing strategy, Compression type, and Scoring method.
3. Use the **Build/Rebuild Index** button to dynamically construct the necessary inverted index on the fly directly from the frontend. (The system securely segregates and isolates binary data by compression format to prevent memory corruption, e.g. V-Byte vs Elias-Gamma).
4. Try querying the engine to view actual response latency and ranked documents.
5. Click **"Benchmark All Models"** to run the test suite against the engine and watch the animated metric comparisons (NDCG, MAP) render in real-time.

---

## 🛠 Running via Command Line (Pipeline)

If you prefer to operate via the terminal or run large scale evaluations automatically, you can use the unified pipeline interface: `main.py`. Note that `main.py` intelligently detects existing index formats on disk and will securely bypass the build phase if you've already generated them via the web app!

```bash
# General Usage
python main.py --index [bsbi|spimi] --encoding [vbyte|eliasgamma|standard] --scoring [tfidf|bm25|wand|colbert|splade]

# Example: Run SPIMI with V-Byte Compression using ColBERT Re-ranking
python main.py --index spimi --encoding vbyte --scoring colbert

# Example: Run BSBI with Elias-Gamma heavily optimized using SPLADE expansion
python main.py --index bsbi --encoding eliasgamma --scoring splade
```

### Options List:
* `--index`: `bsbi` or `spimi` *(default: spimi)*
* `--encoding`: `vbyte`, `eliasgamma`, or `standard` *(default: vbyte)*
* `--scoring`: `tfidf`, `bm25`, `wand`, `colbert`, `splade` *(default: colbert)*
* `--colbert-candidates`: Integer. Determines how many BM25 documents ColBERT evaluates in its second stage *(default: 100)*.
