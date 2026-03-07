import numpy as np
from tokenizer import Tokenizer
from utils import one_hot_encode


class DataLoader:

    def __init__(self, text, vocab_size=10000, context_window=2):
        
        self.tokenizer = Tokenizer(vocab_size=vocab_size)
        self.tokenizer.fit(text)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.context_window = context_window

        # Build word frequency stats
        counts = np.zeros(self.vocab_size)
        for token, idx in self.tokenizer.token_to_id.items():
            counts[idx] = self.tokenizer.token_counts.get(token, 0)

        # Subsampling: discard frequent words with probability 1 - sqrt(t / f(w))
        freqs = counts / (counts.sum() + 1e-15)
        self.discard_prob = np.clip(1 - np.sqrt(1e-5 / (freqs + 1e-15)), 0, 1)

        # Noise distribution for negative sampling (unigram^0.75)
        counts_ns = counts ** 0.75
        self.noise_dist = counts_ns / counts_ns.sum()

    def sample_negatives(self, k):
        """Sample k negative word indices from the noise distribution."""
        return np.random.choice(self.vocab_size, size=k, p=self.noise_dist)

    def create_skipgram_samples(self, text):
        tokens = self.tokenizer.transform(text)
        tokens = [t for t in tokens if np.random.random() > self.discard_prob[t]]
        samples = []
        half = self.context_window // 2

        for i in range(half, len(tokens) - half):
            center_word = tokens[i]
            context_words = tokens[i-half:i] + tokens[i+1:i+half+1]
            for context_word in context_words:
                samples.append((center_word, context_word))

        return samples

    def iter_skipgram_samples(self, text):
        tokens = self.tokenizer.transform(text)
        tokens = [t for t in tokens if np.random.random() > self.discard_prob[t]]
        half = self.context_window // 2

        for i in range(half, len(tokens) - half):
            center_word = tokens[i]
            context_words = tokens[i-half:i] + tokens[i+1:i+half+1]
            for context_word in context_words:
                yield (center_word, context_word)
    
    def create_cbow_samples(self, text):
        
        tokens = self.tokenizer.transform(text)
        tokens = [t for t in tokens if np.random.random() > self.discard_prob[t]]
        samples = []
        half = self.context_window // 2

        for i in range(half, len(tokens) - half):
            center_word = tokens[i]
            context_words = tokens[i-half:i] + tokens[i+1:i+half+1]
            samples.append((context_words, center_word))

        return samples

    def iter_cbow_samples(self, text):
        """Yield CBOW pairs one at a time as (list_of_context_ids, center_id) — zero extra memory."""
        tokens = self.tokenizer.transform(text)
        tokens = [t for t in tokens if np.random.random() > self.discard_prob[t]]
        half = self.context_window // 2

        for i in range(half, len(tokens) - half):
            center_word = tokens[i]
            context_words = tokens[i-half:i] + tokens[i+1:i+half+1]
            yield (context_words, center_word)