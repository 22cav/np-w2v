import numpy as np
from collections import Counter

class Tokenizer:
    def __init__(self, vocab_size=10000):
        """
        Initialize the Tokenizer. The vocab_size parameter determines the maximum number of unique
        tokens that can be stored in the vocabulary. If the number of unique tokens exceeds this limit,
        the least frequently used tokens will be removed from the vocabulary.
        """
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_counts = {}

    def fit(self, text):
        """
        Fit the Tokenizer on a given piece of text. A number of self.vocab_size tokens will be retained 
        (the more common) and the remaining words will be mapped to the <unk> parameter.
        """
        tokens = text.split()
        self.token_counts = Counter(tokens)
        common_tokens = [('<unk>', 0)] + self.token_counts.most_common(self.vocab_size-1)
        self.token_to_id = {token: idx for idx, (token, _) in enumerate(common_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def transform(self, text):
        """
        This method takes the input text and converts it into a sequence of token IDs based on the token_to_id
        mapping created in the fit method. If a token is not found, an <unk> token will be used.        
        """
        tokens = text.split() 
        return [self.token_to_id.get(token, self.token_to_id['<unk>']) for token in tokens]
    
    def inverse_transform(self, token_ids):
        """
        This method takes a sequence of token IDs and converts it back into the original text using the id_to_token
        mapping. It will return a string of tokens joined by spaces.
        """
        tokens = [self.id_to_token.get(token_id, '<unk>') for token_id in token_ids]
        return ' '.join(tokens)
    
    def get_vocab_size(self):
        """
        This method returns the current size of the vocabulary, which is the number of unique tokens that have been
        stored in the token_to_id mapping.
        """
        return len(self.token_to_id)