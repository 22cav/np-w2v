import random
from model import SkipGramW2V, CBOWW2V


def train_skipgram(model: SkipGramW2V, data, epochs, learning_rate, print_every=10):

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        for center_word, context_word in data:
            model.forward(center_word)
            loss, dL_dW1, dL_dW2 = model.backward(center_word, context_word)
            model.update_w(dL_dW1, dL_dW2, learning_rate)
            total_loss += loss
        if epoch%print_every==0: print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(data)}")


def train_cbow(model: CBOWW2V, data, epochs, learning_rate, print_every=10):
    
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        for context_words, center_word in data:
            model.forward(context_words)
            loss, dL_dW1, dL_dW2 = model.backward(context_words, center_word)
            model.update_w(dL_dW1, dL_dW2, learning_rate)
            total_loss += loss
        if epoch%print_every==0: print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(data)}")


def train_skipgram_ns(model, dataloader, texts, epochs, learning_rate, k=5, print_every=10):
    
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for text in texts:
            for center_word, context_word in dataloader.iter_skipgram_samples(text):
                neg_idxs = dataloader.sample_negatives(k)
                loss = model.train_step_ns(center_word, context_word, neg_idxs, learning_rate)
                total_loss += loss
                n += 1
        if epoch%print_every==0: print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/n:.4f}")


def train_cbow_ns(model, dataloader, texts, epochs, learning_rate, k=5, print_every=10):

    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for text in texts:
            for context_words, center_word in dataloader.iter_cbow_samples(text):
                neg_idxs = dataloader.sample_negatives(k)
                loss = model.train_step_ns(center_word, context_words, neg_idxs, learning_rate)
                total_loss += loss
                n += 1
        if epoch%print_every==0: print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/n:.4f}")