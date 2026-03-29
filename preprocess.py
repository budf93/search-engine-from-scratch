import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data (only runs once)
nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))

def preprocess(text):
    """
    Lowercasing + punctuation removal + stopword removal + stemming.
    
    Parameters
    ----------
    text : str
        Raw text to preprocess
    
    Returns
    -------
    List[str]
        List of processed tokens
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenize by whitespace
    tokens = text.split()
    
    # 4. Remove stopwords and stem
    tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS]
    
    return tokens