import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class ColBERTReranker:
    """
    ColBERT-style late interaction re-ranker.

    How it works:
    -------------
    1. Encode the query into a matrix of per-token embeddings  [Q_len x dim]
    2. Encode each candidate document into a matrix            [D_len x dim]
    3. Score = sum over every query token of its MAX cosine
               similarity against any document token  (MaxSim)
    4. Re-rank the BM25 candidates by this ColBERT score

    This is "late interaction": query and document are encoded
    independently (fast, like a bi-encoder), but matching happens
    at the token level (expressive, like a cross-encoder).

    Model
    -----
    We use 'sentence-transformers/msmarco-distilbert-base-tas-b' by default.
    For better medical domain performance swap in:
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    """

    def __init__(self,
                 model_name: str = "sentence-transformers/msmarco-distilbert-base-tas-b",
                 device: str = None,
                 max_query_len: int = 32,
                 max_doc_len: int = 256,
                 batch_size: int = 32):
        """
        Parameters
        ----------
        model_name   : HuggingFace model identifier
        device       : 'cuda' or 'cpu' (auto-detected if None)
        max_query_len: max tokens for query (ColBERT paper uses 32)
        max_doc_len  : max tokens per document
        batch_size   : documents encoded per forward pass
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.batch_size = batch_size

        print(f"[ColBERT] Loading model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        print("[ColBERT] Model loaded.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str], max_len: int) -> torch.Tensor:
        """
        Encode a list of texts → [N x seq_len x hidden_dim] tensor.
        Padding tokens are zeroed out so they don't pollute MaxSim.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)

        # Take the last hidden state as token embeddings
        token_embeddings = output.last_hidden_state  # [N, seq_len, dim]

        # L2-normalise each token vector (required for cosine MaxSim)
        token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)

        # Zero out padding positions so they can't win MaxSim
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()  # [N, seq_len, 1]
        token_embeddings = token_embeddings * attention_mask

        return token_embeddings  # [N, seq_len, dim]

    def _maxsim_score(self,
                      query_emb: torch.Tensor,
                      doc_emb: torch.Tensor) -> float:
        """
        ColBERT MaxSim:
            score = sum_i  max_j  cos_sim(q_i, d_j)

        query_emb : [q_len, dim]
        doc_emb   : [d_len, dim]
        """
        # similarity matrix [q_len, d_len]
        sim_matrix = torch.matmul(query_emb, doc_emb.T)

        # for each query token, take max similarity over all doc tokens
        max_sim = sim_matrix.max(dim=-1).values  # [q_len]

        return max_sim.sum().item()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a single query → [q_len x dim].
        Call once per query, then reuse across all candidates.
        """
        emb = self._encode([query], self.max_query_len)  # [1, q_len, dim]
        return emb[0]  # [q_len, dim]

    def rerank(self,
               query: str,
               candidates: list[tuple[float, str]],
               collection_dir: str = "collection",
               top_k: int = 10) -> list[tuple[float, str]]:
        """
        Re-rank BM25 candidates using ColBERT MaxSim scoring.

        Parameters
        ----------
        query         : raw query string (NOT preprocessed — BERT handles that)
        candidates    : list of (bm25_score, doc_path) from retrieve_bm25
        collection_dir: root folder containing the .txt files
        top_k         : how many results to return

        Returns
        -------
        List of (colbert_score, doc_path) sorted descending
        """
        if not candidates:
            return []

        # 1. Encode query once
        query_emb = self.encode_query(query)  # [q_len, dim]

        # 2. Read document texts
        doc_paths = [doc_path for (_, doc_path) in candidates]
        doc_texts = []
        for path in doc_paths:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    doc_texts.append(f.read()[:2000])  # truncate for speed
            except FileNotFoundError:
                # try stripping leading './'
                clean = path.lstrip("./").lstrip("/")
                try:
                    with open(clean, "r", encoding="utf-8", errors="ignore") as f:
                        doc_texts.append(f.read()[:2000])
                except FileNotFoundError:
                    doc_texts.append("")  # empty fallback

        # 3. Encode documents in batches and score them
        scored = []
        for i in range(0, len(doc_texts), self.batch_size):
            batch_texts = doc_texts[i: i + self.batch_size]
            batch_paths = doc_paths[i: i + self.batch_size]
            # replace empty strings to avoid tokenizer issues
            batch_texts = [t if t else "[PAD]" for t in batch_texts]
            
            batch_emb = self._encode(batch_texts, self.max_doc_len)  # [B, d_len, dim]
            
            # 4. Score each document in this batch with MaxSim
            for j in range(len(batch_paths)):
                score = self._maxsim_score(query_emb, batch_emb[j])
                scored.append((score, batch_paths[j]))

        # 5. Sort descending and return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]