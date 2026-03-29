from bsbi import BSBIIndex
from spimi import SPIMIIndex
from compression import VBEPostings, EliasGammaPostings
from colbert_reranker import ColBERTReranker

def test_search(index_instance, scoring_method="tfidf", colbert_candidates=100):
    print(f"--- Menjalankan Search ({scoring_method}) ---\n")
    
    # Inisialisasi ColBERT hanya jika dibutuhkan
    colbert = ColBERTReranker() if scoring_method == "colbert" else None
    
    queries = ["alkylated with radioactive iodoacetate",
               "psychodrama for disturbed children",
               "lipid metabolism in toxemia and normal pregnancy"]
    
    for query in queries:
        print("Query  : ", query)
    
        if scoring_method == "tfidf":
            results = index_instance.retrieve_tfidf(query, k=10)
        elif scoring_method == "bm25":
            results = index_instance.retrieve_bm25(query, k=10)
        elif scoring_method == "wand":
            results = index_instance.retrieve_wand(query, k=10)
        elif scoring_method == "colbert":
            candidates = index_instance.retrieve_bm25(query, k=colbert_candidates)
            results = colbert.rerank(query, candidates, top_k=10)
        else:
            results = []
    
        print("Results:")
        for (score, doc) in results:
            print(f"{doc:30} {score:>.3f}")
        print()

if __name__ == '__main__':
    # --- KONFIGURASI PILIHAN ---
    INDEX_CLASS = SPIMIIndex
    POSTINGS_ENCODING = VBEPostings
    SCORING_METHOD = "colbert"
    COLBERT_CANDIDATES = 100
    # ---------------------------

    index_instance = INDEX_CLASS(data_dir='collection',
                                 postings_encoding=POSTINGS_ENCODING,
                                 output_dir=f'index_{INDEX_CLASS.__name__.lower()}')
    test_search(index_instance, scoring_method=SCORING_METHOD, colbert_candidates=COLBERT_CANDIDATES)