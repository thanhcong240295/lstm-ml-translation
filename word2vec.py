import numpy as np
from gensim.models import Word2Vec


class Word2VecEmbedding:
    def __init__(self, sg=0):
        self.sg = sg

    def train(self, vocab, documents, embed_dim=128):
        all_sentences = []
        for token_lists in documents.values():
            all_sentences.extend(token_lists)

        model = Word2Vec(sentences=all_sentences, vector_size=embed_dim, window=5, min_count=1, workers=4, sg=self.sg)

        W = self.build_embedding_matrix(vocab, model, embed_dim)
        return {"model": model, "W": W}

    def build_embedding_matrix(self, vocab, w2v_model, embed_dim):
        W = np.zeros((len(vocab.word2idx), embed_dim), dtype=np.float32)

        for word, idx in vocab.word2idx.items():
            if word in w2v_model.wv:
                W[idx] = w2v_model.wv[word]
            else:
                W[idx] = np.random.randn(embed_dim) * 0.01

        return W

    def save_model(self, model, path):
        model.save(path)

    def load_model(self, path):
        return Word2Vec.load(path)
