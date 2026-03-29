import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from bsbi import BSBIIndex
from spimi import SPIMIIndex
from compression import VBEPostings, EliasGammaPostings, StandardPostings

# Import ColBERT re-ranker
try:
    from colbert_reranker import ColBERTReranker
    COLBERT_AVAILABLE = True
except ImportError:
    COLBERT_AVAILABLE = False

# Import SPLADE Expander
try:
    from splade_expander import SpladeExpander
    SPLADE_AVAILABLE = True
except ImportError:
    SPLADE_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Global cache for Index instances and Neural Models
index_cache = {}
colbert_instance = None
splade_instance = None

if COLBERT_AVAILABLE:
    print("Pre-loading ColBERT Re-ranker into memory on startup...")
    try:
        colbert_instance = ColBERTReranker()
    except Exception as e:
        print(f"Failed to load ColBERT: {e}")
        COLBERT_AVAILABLE = False

if SPLADE_AVAILABLE:
    print("Pre-loading SPLADE Expander into memory on startup...")
    try:
        splade_instance = SpladeExpander()
    except Exception as e:
        print(f"Failed to load SPLADE: {e}")
        SPLADE_AVAILABLE = False

def get_postings_class(encoding_str):
    if encoding_str == 'vbyte':
        return VBEPostings
    elif encoding_str == 'eliasgamma':
        return EliasGammaPostings
    return StandardPostings

def get_index_instance(index_type, encoding):
    cache_key = f"{index_type}-{encoding}"
    if cache_key in index_cache:
        return index_cache[cache_key]
    
    idx_class = BSBIIndex if index_type == 'bsbi' else SPIMIIndex
    enc_class = get_postings_class(encoding)
    output_dir = f'index_{idx_class.__name__.lower()}_{enc_class.__name__.lower()}'
    
    instance = idx_class(data_dir='collection', postings_encoding=enc_class, output_dir=output_dir)
    
    try:
        instance.load()
    except Exception as e:
        print(f"Warning: Could not pre-load index {cache_key}. It might not exist yet. {e}")
        
    index_cache[cache_key] = instance
    return instance

def get_colbert():
    return colbert_instance

def get_splade():
    return splade_instance

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search_api():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
        
    query = data['query']
    index_type = data.get('index_type', 'spimi')
    encoding = data.get('encoding', 'vbyte')
    scoring = data.get('scoring', 'bm25')
    top_k = int(data.get('top_k', 10))
    colbert_candidates = int(data.get('colbert_candidates', 100))
    
    start_time = time.time()
    
    try:
        idx_instance = get_index_instance(index_type, encoding)
        
        if len(idx_instance.term_id_map) == 0:
            return jsonify({'error': f"Indexing with {index_type.upper()} and compression method '{encoding}' hasn't been done yet. Please build it first!"}), 400
            
        if scoring == 'tfidf':
            results = idx_instance.retrieve_tfidf(query, k=top_k)
        elif scoring == 'bm25':
            results = idx_instance.retrieve_bm25(query, k=top_k)
        elif scoring == 'wand':
            results = idx_instance.retrieve_wand(query, k=top_k)
        elif scoring == 'splade':
            splade = get_splade()
            if not splade:
                return jsonify({'error': 'SPLADE not available (dependencies missing).'}), 500
            
            # SPLADE + BM25 retrieve
            results = idx_instance.retrieve_splade(query, splade, k=top_k)
        elif scoring == 'colbert':
            colbert = get_colbert()
            if not colbert:
                return jsonify({'error': 'ColBERT not available (dependencies missing).'}), 500
            
            candidates = idx_instance.retrieve_bm25(query, k=colbert_candidates)
            results = colbert.rerank(query, candidates, top_k=top_k)
        else:
            return jsonify({'error': f'Scoring method {scoring} not supported.'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    latency = time.time() - start_time
    
    formatted_results = []
    
    for score, doc_path in results:
        snippet = "No content available."
        try:
            clean_path = doc_path.lstrip("./").lstrip("\\.")
            with open(clean_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                snippet = content[:200] + ("..." if len(content) > 200 else "")
        except FileNotFoundError:
            pass
            
        formatted_results.append({
            'doc_id': doc_path,
            'score': round(score, 4),
            'snippet': snippet
        })
        
    return jsonify({
        'query': query,
        'results': formatted_results,
        'count': len(formatted_results),
        'latency_seconds': round(latency, 4),
        'params': {
            'index_type': index_type,
            'encoding': encoding,
            'scoring': scoring
        }
    })

@app.route('/compare')
def compare_page():
    return render_template('compare.html')

from evaluation import load_qrels, eval as evaluate_index

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    data = request.json
    index_type = data.get('index_type', 'spimi')
    encoding = data.get('encoding', 'vbyte')
    scoring = data.get('scoring', 'bm25')
    colbert_candidates = int(data.get('colbert_candidates', 100))

    try:
        idx_class = BSBIIndex if index_type == 'bsbi' else SPIMIIndex
        enc_class = get_postings_class(encoding)
        
        splade_inst = get_splade() if scoring == 'splade' else None
        
        qrels = load_qrels()
        start = time.time()
        metrics = evaluate_index(qrels=qrels, 
                                 query_file="queries.txt", 
                                 k=1000, 
                                 index_class=idx_class, 
                                 encoding=enc_class, 
                                 scoring=scoring, 
                                 colbert_candidates=colbert_candidates,
                                 splade_instance=splade_inst)
        latency = time.time() - start
        
        return jsonify({
            'metrics': metrics,
            'latency_seconds': round(latency, 4)
        })
    except FileNotFoundError:
        return jsonify({'error': f"Indexing with {index_type.upper()} and compression method '{encoding}' hasn't been done yet. Please build it first!"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/build_index', methods=['POST'])
def api_build_index():
    data = request.json
    index_type = data.get('index_type', 'spimi')
    encoding = data.get('encoding', 'vbyte')
    
    idx_class = BSBIIndex if index_type == 'bsbi' else SPIMIIndex
    enc_class = get_postings_class(encoding)
    output_dir = f'index_{idx_class.__name__.lower()}_{enc_class.__name__.lower()}'
    
    start_time = time.time()
    try:
        instance = idx_class(data_dir='collection', postings_encoding=enc_class, output_dir=output_dir)
        instance.index()
        
        # Update Cache
        cache_key = f"{index_type}-{encoding}"
        index_cache[cache_key] = instance
        
        return jsonify({
            'message': f'Successfully built {index_type.upper()} index using {encoding} compression!',
            'time_seconds': round(time.time() - start_time, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=False)
