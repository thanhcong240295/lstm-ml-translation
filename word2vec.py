import numpy as np
from gensim.models import Word2Vec


class Word2VecEmbedding:
    def __init__(self, sg=0):
        self.sg = sg

    def train(self, vocab: object, documents: dict, embed_dim=100) -> dict[str, dict]:
        result = {}

        for filename, token_lists in documents.items():
            model = self._train(token_lists, vector_size=embed_dim)
            W = np.zeros((vocab.size, embed_dim))
            for word, idx in vocab.word2idx.items():
                if word in model.wv:
                    W[idx] = model.wv[word]
                else:
                    W[idx] = np.random.randn(embed_dim) * 0.1

            result[filename] = {"model": model, "W": W}

        return result

    def save_model(self, model: Word2Vec, path: str):
        model.save(path)

    def load_model(self, path: str) -> Word2Vec:
        model = Word2Vec.load(path)
        return model
    
    def build_embedding_matrix(self, vocab, w2v_model, embed_dim):
        W = np.zeros((len(vocab.word2idx), embed_dim))
        for word, idx in vocab.word2idx.items():
            if word in w2v_model.wv:
                W[idx] = w2v_model.wv[word]
            else:
                W[idx] = np.random.randn(embed_dim) * 0.1
        return W

    def _train(self, sentences: list[list[str]], vector_size=100, window=5, min_count=1, workers=4):
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=self.sg,
        )

        return model
