import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from preprocess import preprocess

class SpladeExpander:
    """
    SPLADE (Sparse Lexical and Expansion) model for Zero-Shot Query Expansion.
    It takes a short query and uses a Masked Language Model to hallucinate
    highly relevant terms across the entire vocabulary, assigning them weights.
    We then map those terms to our existing inverted index to achieve
    neural-level search using standard sparse indexes!
    """
    def __init__(self, model_name="naver/splade-cocondenser-ensembledistil", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SPLADE] Loading model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        print("[SPLADE] Model loaded.")

    def expand_query(self, query: str, top_k_expansion: int = 50) -> dict:
        """
        Passes the query through SPLADE and extracts the top K most
        important vocabulary expansions.
        Returns a dictionary mapping: { 'word': weight }
        """
        # Tokenize query
        tokens = self.tokenizer(query, return_tensors="pt").to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(**tokens)
            
        # SPLADE MaxPooling over sequence
        # We max over the sequence dimension (dim=1) to get the max activation for each vocabulary token
        vec = torch.max(
            torch.log(1 + torch.relu(output.logits)) * tokens["attention_mask"].unsqueeze(-1),
            dim=1
        )[0].squeeze()
        
        # Find non-zero activations
        non_zero_indices = vec.nonzero().squeeze(-1)
        if non_zero_indices.dim() == 0:
            non_zero_indices = non_zero_indices.unsqueeze(0)
            
        weights = vec[non_zero_indices]
        
        # Sort by weight to only take the top K biggest expansions
        # (This keeps the query fast by avoiding searching the entire vocabulary)
        sorted_val, sorted_idx = torch.sort(weights, descending=True)
        top_indices = sorted_idx[:top_k_expansion]
        
        expanded_terms = {}
        for idx in top_indices:
            token_id = non_zero_indices[idx].item()
            weight = weights[idx].item()
            word = self.tokenizer.decode([token_id]).strip()
            
            # Skip weird BERT artifacts (like ##ing, [PAD], [CLS])
            if not word or word.startswith("##") or word.startswith("["):
                continue
                
            # Preprocess the word exactly like we do in SPIMI/BSBI
            cleaned_tokens = preprocess(word)
            for ct in cleaned_tokens:
                # Apply max weight if multiple BERT subwords map to the same stem
                if ct not in expanded_terms:
                    expanded_terms[ct] = weight
                else:
                    expanded_terms[ct] = max(expanded_terms[ct], weight)
                    
        return expanded_terms
