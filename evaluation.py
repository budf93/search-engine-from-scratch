import re
import math
from bsbi import BSBIIndex
from spimi import SPIMIIndex
from compression import VBEPostings, EliasGammaPostings
from colbert_reranker import ColBERTReranker

######## >>>>> Metrik Evaluasi: Rank Biased Precision (RBP)

def rbp(ranking, p = 0.8):
  """ 
  Menghitung skor RBP. RBP mengasumsikan pengguna akan terus membaca ke peringkat
  berikutnya dengan probabilitas 'p'.
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    # Kontribusi dokumen di rank i adalah: relevansi * p^(i-1)
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


######## >>>>> Metrik Evaluasi: DCG, NDCG, dan AP

def dcg(ranking):
    """
    Discounted Cumulative Gain (DCG).
    Mengurangi bobot dokumen relevan jika muncul di peringkat bawah menggunakan logaritma.
    """
    if not ranking: return 0.0
    
    # Dokumen pertama tidak didiskon
    score = float(ranking[0])
    
    # Dokumen rank 2 dst didiskon dengan log2(rank)
    for i in range(2, len(ranking) + 1):
        rel = ranking[i-1]
        score += rel / math.log2(i)
    return score

def ndcg(ranking):
    """
    Normalized DCG.
    Membagi DCG aktual dengan DCG ideal (jika semua dokumen relevan ada di atas).
    Menghasilkan skor antara 0.0 sampai 1.0.
    """
    actual_dcg = dcg(ranking)
    # IDCG didapat dengan mengurutkan ranking secara descending
    ideal_ranking = sorted(ranking, reverse=True)
    ideal_dcg = dcg(ideal_ranking)
    if ideal_dcg == 0: return 0.0
    return actual_dcg / ideal_dcg

def ap(ranking):
    """
    Average Precision (AP).
    Rata-rata presisi pada setiap posisi dokumen relevan ditemukan.
    """
    relevant_count = 0
    precision_sum = 0.0
    for i in range(1, len(ranking) + 1):
        if ranking[i-1] == 1: # Jika relevan
            relevant_count += 1
            # Precision@i = jumlah relevan sejauh ini / posisi saat ini
            precision_sum += relevant_count / i
            
    if relevant_count == 0: return 0.0
    return precision_sum / relevant_count


######## >>>>> Manajemen Qrels & Evaluasi

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ Memuat ground truth relevansi dari file qrels.txt """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1 # Set dokumen did relevan untuk query qid
  return qrels

def eval(qrels, query_file = "queries.txt", k = 1000,
         index_class = BSBIIndex, encoding = VBEPostings, scoring = "tfidf",
         colbert_candidates = 100, splade_instance = None):
  """ Menghitung rata-rata skor evaluasi untuk semua query """
  # Use separate directories per index class and encoding protocol
  output_dir = f'index_{index_class.__name__.lower()}_{encoding.__name__.lower()}'

  index_instance = index_class(data_dir='collection',
                                postings_encoding=encoding,
                                output_dir=output_dir)

  # Inisialisasi ColBERT reranker sekali saja jika diperlukan
  colbert = None
  if scoring == "colbert":
      from colbert_reranker import ColBERTReranker
      colbert = ColBERTReranker()

  with open(query_file) as file:
    all_scores = {"RBP": [], "DCG": [], "NDCG": [], "AP": []}

    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # Jalankan pencarian sesuai metode pilihan
      if scoring == "tfidf":
          results = index_instance.retrieve_tfidf(query, k = k)
      elif scoring == "bm25":
          results = index_instance.retrieve_bm25(query, k = k)
      elif scoring == "wand":
          results = index_instance.retrieve_wand(query, k = k)
      elif scoring == "splade":
          results = index_instance.retrieve_splade(query, splade_instance, k=k)
      elif scoring == "colbert":
          # Step 1: BM25 retrieve top candidates
          candidates = index_instance.retrieve_bm25(query, k = colbert_candidates)
          # Step 2: ColBERT re-rank using raw query (BERT handles its own tokenization)
          results = colbert.rerank(query, candidates, top_k = k)
      else:
          results = []

      ranking = []
      for (score, doc) in results:
          # Ekstrak ID dokumen dari path file menggunakan regex
          # Format path bisa berbeda di Windows/Linux, handle keduanya
          match = re.search(r'[\\/](.*)\.txt', doc)
          if match:
              did = int(match.group(1).split('/')[-1].split('\\')[-1])
              ranking.append(qrels.get(qid, {}).get(did, 0))
      
      # Simpan skor untuk query ini
      all_scores["RBP"].append(rbp(ranking))
      all_scores["DCG"].append(dcg(ranking))
      all_scores["NDCG"].append(ndcg(ranking))
      all_scores["AP"].append(ap(ranking))

  # Tampilkan Mean (rata-rata) untuk setiap metrik
  means = {}
  print(f"--- Hasil Evaluasi ({index_class.__name__} + {scoring}) ---")
  for metric, scores in all_scores.items():
      means[metric] = sum(scores)/len(scores) if len(scores) > 0 else 0
      print(f"Mean {metric}: {means[metric]:.4f}")
      
  return means

if __name__ == '__main__':
  # --- KONFIGURASI PILIHAN ---
  # 1. Pilih Kelas Index: BSBIIndex atau SPIMIIndex
  INDEX_CLASS = SPIMIIndex
#   INDEX_CLASS = BSBIIndex

  # 2. Pilih Encoding: VBEPostings atau EliasGammaPostings
  POSTINGS_ENCODING = VBEPostings

  # 3. Pilih Metode Scoring: "tfidf", "bm25", "wand", atau "colbert"
  SCORING_METHOD = "colbert"

  # 4. (Hanya untuk colbert) Jumlah kandidat BM25 sebelum di-rerank
  COLBERT_CANDIDATES = 100
  # ---------------------------

  qrels = load_qrels()
  eval(qrels, index_class=INDEX_CLASS, encoding=POSTINGS_ENCODING,
       scoring=SCORING_METHOD, colbert_candidates=COLBERT_CANDIDATES)