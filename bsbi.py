import os
import pickle
import contextlib
import heapq
import time
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from preprocess import preprocess
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        os.makedirs(self.output_dir, exist_ok=True)

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for token in preprocess(f.read()):
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT
        """
        # --- Update untuk WAND & BM25 ---
        # 1. Gabungkan semua doc_length untuk menghitung N dan avgdl
        for reader in indices:
            merged_index.doc_length.update(reader.doc_length)
        
        N = len(merged_index.doc_length)
        avgdl = sum(merged_index.doc_length.values()) / N if N > 0 else 0
        
        # Parameter BM25 standar
        k1 = 1.2
        b = 0.75

        # 2. Merge postings dan hitung Upper Bound (UB) untuk setiap term
        # kode berikut mengasumsikan minimal ada 1 term
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
                # Hitung UB untuk term 'curr' sebelum di-append
                df = len(postings)
                idf = math.log(N / df) if df > 0 else 0
                ub = 0.0
                for i in range(df):
                    tf = tf_list[i]
                    doc_len = merged_index.doc_length[postings[i]]
                    # Kontribusi BM25 sebagai dasar Upper Bound
                    score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                    if score > ub:
                        ub = score
                
                merged_index.append(curr, postings, tf_list, upper_bound=ub)
                curr, postings, tf_list = t, postings_, tf_list_
        
        # Term terakhir
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

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
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

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k = 10, k1 = 1.2, b = 0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25.
        Method akan mengembalikan top-K retrieval results.

        Rumus BM25:
        Score(D, Q) = sum_{t in Q} IDF(t) * [ (tf(t, D) * (k1 + 1)) / (tf(t, D) + k1 * (1 - b + b * (|D| / avgdl))) ]
        dimana IDF(t) = log(N / df(t))

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi
        k: int
            Jumlah dokumen teratas yang ingin dikembalikan
        k1: float
            Parameter k1 (default 1.2)
        b: float
            Parameter b (default 0.75)

        Result
        ------
        List[(float, str)]
            List of tuple (score, doc_name) terurut mengecil berdasarkan skor.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Tokenisasi query (abaikan kata yang tidak ada di dictionary)
        terms = [self.term_id_map[word] for word in preprocess(query) if word in self.term_id_map]
        if not terms:
            return []
            
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return []
                
            # Hitung Average Document Length (avgdl)
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
                        
                        # Rumus BM25: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                        
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += idf * (numerator / denominator)

            # Convert doc_id ke doc_name dan urutkan
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_wand(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan algoritma WAND (Weak AND).
        Hanya dokumen yang berpotensi masuk Top-K yang akan dihitung skor BM25-nya.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Tokenisasi query
        terms = [word for word in preprocess(query) if word in self.term_id_map]
        if not terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0: return []
            avgdl = sum(merged_index.doc_length.values()) / N
            k1, b = 1.2, 0.75

            # 1. Inisialisasi list of iterators untuk setiap query term
            # Setiap elemen: [current_doc_id, postings_list, tf_list, pointer, upper_bound]
            iterators = []
            for word in terms:
                tid = self.term_id_map[word]
                if tid in merged_index.postings_dict:
                    plist, tflist = merged_index.get_postings_list(tid)
                    ub = merged_index.postings_dict[tid][4] # element ke-5 adalah upper_bound
                    iterators.append([plist[0], plist, tflist, 0, ub])

            if not iterators:
                return []

            top_k = [] # Min-heap untuk menyimpan top-K (score, doc_name)
            threshold = -1.0 # Skor terendah di top_k heap

            while True:
                # a. Sort iterators berdasarkan current_doc_id terkecil
                iterators.sort(key=lambda x: x[0])
                
                # b. Cari pivot term: term ke-p sedemikian sehingga sum(UB_0...UB_p) > threshold
                score_ub_sum = 0
                pivot_idx = -1
                for i in range(len(iterators)):
                    score_ub_sum += iterators[i][4]
                    if score_ub_sum > threshold:
                        pivot_idx = i
                        break
                
                # Jika tidak ada pivot, semua sisa dokumen tidak mungkin masuk Top-K
                if pivot_idx == -1:
                    break
                
                pivot_doc_id = iterators[pivot_idx][0]
                if pivot_doc_id == float('inf'):
                    break

                if iterators[0][0] == pivot_doc_id:
                    # c. Candidate found! Hitung skor asli untuk pivot_doc_id
                    current_doc_id = pivot_doc_id
                    full_score = 0.0
                    
                    for it in iterators:
                        if it[0] == current_doc_id:
                            # Hitung kontribusi BM25 asli
                            plist, tflist, idx = it[1], it[2], it[3]
                            tf = tflist[idx]
                            df = len(plist)
                            idf = math.log(N / df)
                            doc_len = merged_index.doc_length[current_doc_id]
                            full_score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                            
                            # Majukan iterator
                            it[3] += 1
                            if it[3] < len(it[1]):
                                it[0] = it[1][it[3]]
                            else:
                                it[0] = float('inf')
                    
                    # d. Update Top-K heap
                    if len(top_k) < k:
                        heapq.heappush(top_k, (full_score, self.doc_id_map[current_doc_id]))
                        if len(top_k) == k:
                            threshold = top_k[0][0]
                    elif full_score > threshold:
                        heapq.heapreplace(top_k, (full_score, self.doc_id_map[current_doc_id]))
                        threshold = top_k[0][0]
                else:
                    # e. Lompatkan (skip) iterator ke pivot_doc_id atau setelahnya
                    for i in range(pivot_idx):
                        it = iterators[i]
                        plist = it[1]
                        # Gunakan linear skip (bisa dioptimasi dengan binary search)
                        while it[3] < len(plist) and plist[it[3]] < pivot_doc_id:
                            it[3] += 1
                        
                        if it[3] < len(plist):
                            it[0] = plist[it[3]]
                        else:
                            it[0] = float('inf')
            
            return sorted(top_k, key=lambda x: x[0], reverse=True)

    def retrieve_splade(self, query, splade_expander, k=10, k1=1.2, b=0.75):
        """
        Ranked Retrieval dengan algoritma SPLADE Zero-Shot Query Expansion.
        Alih-alih hanya menggunakan tokenize dari query asli, fungsi ini mengeksploitasi 
        hallusinasi Neural SPLADE untuk mengexpand query ke subword bert, lalu
        melakukan Retrieval berbasis BM25 dengan weights yang dihasilkan SPLADE!
        """
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
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":
    from compression import VBEPostings, EliasGammaPostings

    # --- KONFIGURASI PILIHAN ---
    # 1. Pilih Encoding: VBEPostings atau EliasGammaPostings
    POSTINGS_ENCODING = VBEPostings 
    
    # 2. Pilih Metode Scoring: "tfidf", "bm25", atau "wand" (BM25 optimized)
    SCORING_METHOD = "bm25" 
    # ---------------------------

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = POSTINGS_ENCODING, \
                              output_dir=f'index_{BSBIIndex.__name__.lower()}')
    
    # Jalankan Indexing
    print(f"--- Memulai Indexing dengan {POSTINGS_ENCODING.__name__} ---")
    BSBI_instance.index()
    print("Indexing selesai.\n")

    # Jalankan Retrieval
    query = "universitas indonesia"
    print(f"--- Mencari: '{query}' dengan metode {SCORING_METHOD} ---")
    
    if SCORING_METHOD == "tfidf":
        results = BSBI_instance.retrieve_tfidf(query, k=10)
    elif SCORING_METHOD == "bm25":
        results = BSBI_instance.retrieve_bm25(query, k=10)
    elif SCORING_METHOD == "wand":
        results = BSBI_instance.retrieve_wand(query, k=10)
    else:
        print("Metode scoring tidak dikenal.")
        results = []

    # Tampilkan Hasil
    for score, doc in results:
        print(f"{score:.4f} - {doc}")