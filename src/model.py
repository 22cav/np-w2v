import numpy as np
from utils import softmax, cross_entropy_loss, sigmoid


class SkipGramW2V:

    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the W2V model with the given vocabulary size and embedding dimension.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01  # Input to hidden layer weights
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01  # Hidden to output layer weights

        # Placeholders for intermediate values during forward pass
        self.h = None  # Hidden layer activations
        self.u = None  # Output layer activations
        self.y_pred = None  # Predicted probabilities for context words


    def forward(self, center_idx):
        """
        Perform the forward pass of the Skip-gram model.
        center_idx: integer index of the center word.
        """
        self.center_idx = center_idx
        self.h = self.W1[center_idx]              # Lookup embedding directly (no dot product)
        self.u = np.dot(self.h, self.W2)           # Output layer activations
        self.y_pred = softmax(self.u)              # Predicted probabilities for context words
        return self.y_pred
    
    def backward(self, center_idx, context_idx):
        
        target = np.zeros(self.vocab_size)
        target[context_idx] = 1.0
        loss = cross_entropy_loss(self.y_pred, target)

        # Gradient of the loss with respect to the output layer activations
        dL_du = self.y_pred - target
        
        # Gradient of the loss with respect to W2
        dL_dW2 = np.outer(self.h, dL_du)

        # Gradient of the loss with respect to the hidden layer activations
        dL_dh = np.dot(self.W2, dL_du)

        # Sparse gradient: only the row for center_idx is nonzero
        dL_dW1 = np.zeros_like(self.W1)
        dL_dW1[center_idx] = dL_dh

        return loss, dL_dW1, dL_dW2
    
    def update_w(self, dL_dW1, dL_dW2, learning_rate):
        """
        Update the weights of the model using the computed gradients and the specified learning rate.
        """
        self.W1 -= learning_rate * dL_dW1
        self.W2 -= learning_rate * dL_dW2

    def train_step_ns(self, center_idx, pos_idx, neg_idxs, learning_rate):
        """
        Single training step with negative sampling.
        Only touches k+1 columns of W2 instead of the full vocabulary.
        """
        h = self.W1[center_idx]  # (d,)

        # Scores for positive and negative words
        pos_score = sigmoid(np.dot(self.W2[:, pos_idx], h))
        neg_scores = sigmoid(np.dot(self.W2[:, neg_idxs].T, h))  # (k,)

        loss = -np.log(pos_score + 1e-15) - np.sum(np.log(1 - neg_scores + 1e-15))

        # Gradients
        pos_grad = pos_score - 1       # scalar
        neg_grad = neg_scores           # (k,)

        # Gradient w.r.t. hidden layer
        dL_dh = pos_grad * self.W2[:, pos_idx] + np.dot(self.W2[:, neg_idxs], neg_grad)

        # Sparse W2 updates (only k+1 columns)
        self.W2[:, pos_idx] -= learning_rate * pos_grad * h
        self.W2[:, neg_idxs] -= learning_rate * np.outer(h, neg_grad)

        # Sparse W1 update (only 1 row)
        self.W1[center_idx] -= learning_rate * dL_dh

        return loss

    def get_embedding(self, word_index):
        """
        Get the embedding vector for a given word index.
        """
        return self.W1[word_index]
    
    def save_model(self, file_path):
        """
        Save the model weights to a file.
        """
        np.savez(file_path, W1=self.W1, W2=self.W2)

    def load_model(self, file_path):
        """
        Load the model weights from a file.
        """
        data = np.load(file_path)
        self.W1 = data['W1']
        self.W2 = data['W2']



class CBOWW2V:

    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the CBOW W2V model with the given vocabulary size and embedding dimension.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01  # Input to hidden layer weights
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01  # Hidden to output layer weights

        # Placeholders for intermediate values during forward pass
        self.h = None  # Hidden layer activations
        self.u = None  # Output layer activations
        self.y_pred = None  # Predicted probabilities for the center word

    def forward(self, context_idxs):
        """
        Perform the forward pass of the CBOW model.
        context_idxs: list of integer indices for the context words.
        """
        self.context_idxs = context_idxs
        self.h = np.mean(self.W1[context_idxs], axis=0)  # Average of context word embeddings
        self.u = np.dot(self.h, self.W2)                  # Output layer activations
        self.y_pred = softmax(self.u)                      # Predicted probabilities for the center word
        return self.y_pred
    
    def backward(self, context_idxs, center_idx):
        
        target = np.zeros(self.vocab_size)
        target[center_idx] = 1.0
        loss = cross_entropy_loss(self.y_pred, target)

        # Gradient of the loss with respect to the output layer activations
        dL_du = self.y_pred - target
        
        # Gradient of the loss with respect to W2
        dL_dW2 = np.outer(self.h, dL_du)

        # Gradient of the loss with respect to the hidden layer activations
        dL_dh = np.dot(self.W2, dL_du)

        # Sparse gradient: only context word rows are nonzero
        dL_dW1 = np.zeros_like(self.W1)
        n_context = len(context_idxs)
        for idx in context_idxs:
            dL_dW1[idx] += dL_dh / n_context

        return loss, dL_dW1, dL_dW2
    
    def update_w(self, dL_dW1, dL_dW2, learning_rate):
        """
        Update the weights of the model using the computed gradients and the specified learning rate.
        """
        self.W1 -= learning_rate * dL_dW1
        self.W2 -= learning_rate * dL_dW2

    def train_step_ns(self, center_idx, context_idxs, neg_idxs, learning_rate):
        """
        Single training step with negative sampling for CBOW.
        center_idx is the target word, context_idxs are averaged to form h.
        """
        h = np.mean(self.W1[context_idxs], axis=0)  # (d,)

        # Scores for positive and negative words
        pos_score = sigmoid(np.dot(self.W2[:, center_idx], h))
        neg_scores = sigmoid(np.dot(self.W2[:, neg_idxs].T, h))  # (k,)

        loss = -np.log(pos_score + 1e-15) - np.sum(np.log(1 - neg_scores + 1e-15))

        # Gradients
        pos_grad = pos_score - 1
        neg_grad = neg_scores

        # Gradient w.r.t. hidden layer
        dL_dh = pos_grad * self.W2[:, center_idx] + np.dot(self.W2[:, neg_idxs], neg_grad)

        # Sparse W2 updates
        self.W2[:, center_idx] -= learning_rate * pos_grad * h
        self.W2[:, neg_idxs] -= learning_rate * np.outer(h, neg_grad)

        # Sparse W1 update (distribute equally to all context words)
        n = len(context_idxs)
        for idx in context_idxs:
            self.W1[idx] -= learning_rate * dL_dh / n

        return loss

    def get_embedding(self, word_index):
        """
        Get the embedding vector for a given word index.
        """
        return self.W1[word_index]
    
    def save_model(self, file_path):
        """
        Save the model weights to a file.
        """
        np.savez(file_path, W1=self.W1, W2=self.W2)

    def load_model(self, file_path):
        """
        Load the model weights from a file.
        """
        data = np.load(file_path)
        self.W1 = data['W1']
        self.W2 = data['W2']