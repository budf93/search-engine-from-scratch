import os
import pickle
import contextlib
import heapq
import math

# Mengambil tools yang sudah ada di project
from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import VBEPostings, EliasGammaPostings
from preprocess import preprocess
from tqdm import tqdm

class SPIMIIndex:
    """
    Scaffolding untuk algoritma Single Pass In-Memory Indexing (SPIMI).
    Tujuan: Membangun Inverted Index tanpa melakukan sorting besar di awal.
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        os.makedirs(self.output_dir, exist_ok=True)
        self.intermediate_indices = []

    def save(self):
        """ Menyimpan kamus ID (terms & docs) ke disk """
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """ Memuat kamus ID dari disk """
        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    
    def spimi_invert(self, block_dir_relative):
        """
        Implementasi algoritma SPIMI.
        """
        # 1. Tentukan path lengkap ke direktori block (misal: 'collection/1')
        block_full_path = os.path.join(self.data_dir, block_dir_relative)

        # 2. Inisialisasi dictionary index di memori
        memory_index = {}

        # 3. Iterasi setiap file dalam folder tersebut
        for filename in os.listdir(block_full_path):
            if not filename.endswith(".txt"):
                continue

            doc_path = os.path.join(self.data_dir, block_dir_relative, filename).replace("\\", "/")
            doc_id = self.doc_id_map[doc_path]

            # 4. Baca isi file dan Tokenisasi
            with open(os.path.join(block_full_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                tokens = preprocess(content)

                for token in tokens:
                    term_id = self.term_id_map[token]

                    if term_id not in memory_index:
                        memory_index[term_id] = {}

                    if doc_id not in memory_index[term_id]:
                        memory_index[term_id][doc_id] = 0

                    memory_index[term_id][doc_id] += 1

        # 6. Menulis hasil ke disk sebagai index sementara
        index_id = 'intermediate_spimi_' + block_dir_relative

        with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as writer:
            for term_id in sorted(memory_index.keys()):
                postings = sorted(memory_index[term_id].keys())
                tf_list = [memory_index[term_id][doc_id] for doc_id in postings]
                writer.append(term_id, postings, tf_list)

        memory_index = None
        return index_id
                        
    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.
        """
        for reader in indices:
            merged_index.doc_length.update(reader.doc_length)
        
        N = len(merged_index.doc_length)
        avgdl = sum(merged_index.doc_length.values()) / N if N > 0 else 0
        k1, b = 1.2, 0.75

        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        
        try:
            curr, postings, tf_list = next(merged_iter)
        except StopIteration:
            return

        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                df = len(postings)
                idf = math.log(N / df) if df > 0 else 0
                ub = 0.0
                for i in range(df):
                    tf = tf_list[i]
                    doc_len = merged_index.doc_length[postings[i]]
                    score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                    if score > ub:
                        ub = score
                
                merged_index.append(curr, postings, tf_list, upper_bound=ub)
                curr, postings, tf_list = t, postings_, tf_list_
        
        df = len(postings)
        idf = math.log(N / df) if df > 0 else 0
        ub = 0.0
        for i in range(df):
            tf = tf_list[i]
            doc_len = merged_index.doc_length[postings[i]]
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
            if score > ub:
                ub = score
        merged_index.append(curr, postings, tf_list, upper_bound=ub)

    def retrieve_splade(self, query, splade_expander, k=10, k1=1.2, b=0.75):
        """ Ranked Retrieval dengan algoritma SPLADE Zero-Shot Query Expansion """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # 1. Expand the query terms with Neural SPLADE weights
        expanded_term_weights = splade_expander.expand_query(query)

        # 2. Filter expanded terms that actually exist in our collection vocabulary
        valid_terms = {}
        for term_word, weight in expanded_term_weights.items():
            if term_word in self.term_id_map:
                valid_terms[self.term_id_map[term_word]] = weight

        if not valid_terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0: return []
            avgdl = sum(merged_index.doc_length.values()) / N

            scores = {}
            for term_id, query_weight in valid_terms.items():
                if term_id in merged_index.postings_dict:
                    df = merged_index.postings_dict[term_id][1]
                    idf = math.log(N / df)

                    postings, tf_list = merged_index.get_postings_list(term_id)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        doc_len = merged_index.doc_length[doc_id]

                        # Modified BM25 score incorporating SPLADE Neural Vector weights
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                        tf_component = numerator / denominator

                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += query_weight * idf * tf_component

            # Convert to list of (score, doc_name)
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def index(self):
        """ Driver untuk menjalankan proses indexing SPIMI secara utuh """
        for block_dir in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            index_id = self.spimi_invert(block_dir)
            self.intermediate_indices.append(index_id)
        
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(idx_id, self.postings_encoding, directory=self.output_dir))
                               for idx_id in self.intermediate_indices]
                self.merge(indices, merged_index)

    def retrieve_tfidf(self, query, k = 10):
        """ Ranked Retrieval dengan skema TF-IDF """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in preprocess(query) if word in self.term_id_map]
        if not terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k = 10, k1 = 1.2, b = 0.75):
        """ Ranked Retrieval dengan skema scoring BM25 """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in preprocess(query) if word in self.term_id_map]
        if not terms:
            return []
            
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return []
                
            avgdl = sum(merged_index.doc_length.values()) / N
            
            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    idf = math.log(N / df)
                    
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        doc_len = merged_index.doc_length[doc_id]
                        
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                        
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += idf * (numerator / denominator)

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_wand(self, query, k = 10):
        """ Ranked Retrieval dengan algoritma WAND """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [word for word in preprocess(query) if word in self.term_id_map]
        if not terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0: return []
            avgdl = sum(merged_index.doc_length.values()) / N
            k1, b = 1.2, 0.75

            iterators = []
            for word in terms:
                tid = self.term_id_map[word]
                if tid in merged_index.postings_dict:
                    plist, tflist = merged_index.get_postings_list(tid)
                    ub = merged_index.postings_dict[tid][4] 
                    iterators.append([plist[0], plist, tflist, 0, ub])

            if not iterators:
                return []

            top_k = [] 
            threshold = -1.0 

            while True:
                iterators.sort(key=lambda x: x[0])
                
                score_ub_sum = 0
                pivot_idx = -1
                for i in range(len(iterators)):
                    score_ub_sum += iterators[i][4]
                    if score_ub_sum > threshold:
                        pivot_idx = i
                        break
                
                if pivot_idx == -1:
                    break
                
                pivot_doc_id = iterators[pivot_idx][0]
                if pivot_doc_id == float('inf'):
                    break

                if iterators[0][0] == pivot_doc_id:
                    current_doc_id = pivot_doc_id
                    full_score = 0.0
                    
                    for it in iterators:
                        if it[0] == current_doc_id:
                            plist, tflist, idx = it[1], it[2], it[3]
                            tf = tflist[idx]
                            df = len(plist)
                            idf = math.log(N / df)
                            doc_len = merged_index.doc_length[current_doc_id]
                            full_score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                            
                            it[3] += 1
                            if it[3] < len(it[1]):
                                it[0] = it[1][it[3]]
                            else:
                                it[0] = float('inf')
                    
                    if len(top_k) < k:
                        heapq.heappush(top_k, (full_score, self.doc_id_map[current_doc_id]))
                        if len(top_k) == k:
                            threshold = top_k[0][0]
                    elif full_score > threshold:
                        heapq.heapreplace(top_k, (full_score, self.doc_id_map[current_doc_id]))
                        threshold = top_k[0][0]
                else:
                    for i in range(pivot_idx):
                        it = iterators[i]
                        plist = it[1]
                        while it[3] < len(plist) and plist[it[3]] < pivot_doc_id:
                            it[3] += 1
                        
                        if it[3] < len(plist):
                            it[0] = plist[it[3]]
                        else:
                            it[0] = float('inf')
            
            return sorted(top_k, key=lambda x: x[0], reverse=True)

if __name__ == "__main__":
    from compression import VBEPostings, EliasGammaPostings

    # --- KONFIGURASI PILIHAN ---
    # 1. Pilih Encoding: VBEPostings atau EliasGammaPostings
    POSTINGS_ENCODING = VBEPostings 
    
    # 2. Pilih Metode Scoring: "tfidf", "bm25", atau "wand" (BM25 optimized)
    SCORING_METHOD = "tfidf" 
    # ---------------------------

    spimi_instance = SPIMIIndex(data_dir = 'collection', 
                                postings_encoding = POSTINGS_ENCODING, 
                                output_dir=f'index_{SPIMIIndex.__name__.lower()}')
    
    # Jalankan Indexing
    print(f"--- Memulai SPIMI Indexing dengan {POSTINGS_ENCODING.__name__} ---")
    spimi_instance.index()
    print("Indexing selesai.\n")

    # Jalankan Retrieval
    query = "universitas indonesia"
    print(f"--- Mencari: '{query}' dengan metode {SCORING_METHOD} ---")
    
    if SCORING_METHOD == "tfidf":
        results = spimi_instance.retrieve_tfidf(query, k=10)
    elif SCORING_METHOD == "bm25":
        results = spimi_instance.retrieve_bm25(query, k=10)
    elif SCORING_METHOD == "wand":
        results = spimi_instance.retrieve_wand(query, k=10)
    else:
        print("Metode scoring tidak dikenal.")
        results = []

    # Tampilkan Hasil
    for score, doc in results:
        print(f"{score:.4f} - {doc}")