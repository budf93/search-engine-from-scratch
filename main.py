import argparse
import os
import time

from bsbi import BSBIIndex
from spimi import SPIMIIndex
from compression import VBEPostings, EliasGammaPostings, StandardPostings
from search import test_search
from evaluation import load_qrels, eval as evaluate_index

def run_pipeline():
    parser = argparse.ArgumentParser(description="Jalankan pipeline lengkap: Indexing -> Search -> Evaluasi")
    parser.add_argument('--index', choices=['bsbi', 'spimi'], default='spimi', help="Metode indexing (bsbi atau spimi)")
    parser.add_argument('--encoding', choices=['vbyte', 'eliasgamma', 'standard'], default='vbyte', help="Metode kompresi postings")
    parser.add_argument('--scoring', choices=['tfidf', 'bm25', 'wand', 'colbert', 'splade'], default='bm25', help="Metode scoring untuk search dan evaluasi")
    parser.add_argument('--colbert-candidates', type=int, default=100, help="Jumlah kandidat BM25 sebelum rerank ColBERT")

    args = parser.parse_args()

    # Mapping args ke class
    INDEX_CLASS = BSBIIndex if args.index == 'bsbi' else SPIMIIndex

    if args.encoding == 'vbyte':
        POSTINGS_ENCODING = VBEPostings
    elif args.encoding == 'eliasgamma':
        POSTINGS_ENCODING = EliasGammaPostings
    else:
        POSTINGS_ENCODING = StandardPostings

    SCORING_METHOD = args.scoring
    COLBERT_CANDIDATES = args.colbert_candidates

    # Consistent output_dir used everywhere
    output_dir = f'index_{INDEX_CLASS.__name__.lower()}_{POSTINGS_ENCODING.__name__.lower()}'

    print("="*60)
    print("OPSI YANG DIPILIH UNTUK PIPELINE:")
    print(f"  Indexer                : {INDEX_CLASS.__name__}")
    print(f"  Postings Encoding      : {POSTINGS_ENCODING.__name__}")
    print(f"  Scoring Method         : {SCORING_METHOD.upper()}")
    print(f"  Output Dir             : {output_dir}")
    if SCORING_METHOD == "colbert":
        print(f"  ColBERT BM25 Candidates: {COLBERT_CANDIDATES}")
    print("="*60 + "\n")

    # 1. Inisialisasi Indexer
    index_instance = INDEX_CLASS(data_dir='collection',
                                 postings_encoding=POSTINGS_ENCODING,
                                 output_dir=output_dir)

    # 2. Tahap Indexing
    print(">>> TAHAP 1: INDEXING")
    start_time = time.time()
    index_file = os.path.join(output_dir, 'main_index.index')
    if os.path.exists(index_file):
        print(f"Index sudah ada di '{output_dir}', melewati indexing dan memuat dari disk.")
        index_instance.load()
    else:
        os.makedirs(output_dir, exist_ok=True)
        index_instance.index()
    print(f"Waktu Indexing: {time.time() - start_time:.2f} detik\n")

    # 3. Tahap Search
    print(">>> TAHAP 2: SEARCHING")
    start_time = time.time()
    test_search(index_instance, scoring_method=SCORING_METHOD, colbert_candidates=COLBERT_CANDIDATES)
    print(f"Waktu Searching: {time.time() - start_time:.2f} detik\n")

    # 4. Tahap Evaluasi
    print(">>> TAHAP 3: EVALUATION")
    start_time = time.time()
    qrels = load_qrels()
    evaluate_index(qrels, index_class=INDEX_CLASS, encoding=POSTINGS_ENCODING,
                   scoring=SCORING_METHOD, colbert_candidates=COLBERT_CANDIDATES)
    print(f"\nWaktu Evaluation: {time.time() - start_time:.2f} detik\n")

    print("="*60)
    print("PIPELINE SELESAI")
    print("="*60)

if __name__ == '__main__':
    run_pipeline()