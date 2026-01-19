import os

import numpy as np

from attention import Attention
from bi_lstm import BiLSTM
from visualization import Visualization


class LanguageTranslation:
    def __init__(self, input_size=128, hidden_size=256, use_attention=True):
        self.bi_lstm = BiLSTM(input_size=input_size, hidden_size=hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.attention = Attention(2 * hidden_size) if use_attention else None
        self.Wy = None
        self.by = None
        self.losses = []
        self.visualization = Visualization()

    def train(self, epochs, learning_rate, vocab, dict_data: dict, word2vec_models: dict[str, dict]) -> object:
        X_train, y_train, word2vec_model = self._prepare_training_data(vocab, dict_data, word2vec_models)
        return self._train_model(epochs, learning_rate, X_train, y_train, word2vec_model, vocab)

    def _prepare_training_data(self, vocab, dict_data, word2vec_models):
        X_train = []
        y_train = []
        word2vec_model = None

        for key in dict_data.keys():
            if word2vec_model is None:
                word2vec_model = word2vec_models[key]

            for tokens in dict_data[key]:
                token_ids = [vocab.word2idx.get(token, 1) for token in tokens]
                input_ids = token_ids[:-1]
                target_ids = token_ids[1:]
                X_train.append(input_ids)
                y_train.append(target_ids)

        return X_train, y_train, word2vec_model

    def _train_model(self, epochs, learning_rate, X_train, y_train, word2vec_model, vocab):
        vocab_len = len(vocab.word2idx)
        hidden2 = 2 * self.bi_lstm.hidden_size
        pad_id = vocab.word2idx.get("<PAD>", 0)

        if self.use_attention:
            output_dim = hidden2 * 2
        else:
            output_dim = hidden2

        self.Wy = np.random.randn(vocab_len, output_dim) * 0.1
        self.by = np.zeros((vocab_len, 1))
        self.losses = []

        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}...")
            total_loss = 0.0
            token_count = 0

            for input_ids, target_ids in zip(X_train, y_train):
                X_seq, Y_seq = self._prepare_sequence(input_ids, target_ids, word2vec_model, pad_id)

                if len(X_seq) == 0:
                    continue

                h_concat = self.bi_lstm.forward(np.array(X_seq).squeeze(2))
                loss, count = self._backward_pass(h_concat, Y_seq, learning_rate)
                total_loss += loss
                token_count += count

            avg_loss = total_loss / max(token_count, 1)
            self.losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        self.visualization.plot_loss(self.losses, title="Training Loss", filename="training_loss.png")
        self.visualization.plot_loss_statistics(self.losses, filename="loss_statistics.png")

        return self

    def _prepare_sequence(self, input_ids, target_ids, word2vec_model, pad_id):
        X_seq = []
        Y_seq = []

        for idx, tgt in zip(input_ids, target_ids):
            if idx == pad_id or tgt == pad_id:
                continue
            x_t = word2vec_model["W"][idx].reshape(-1, 1)
            X_seq.append(x_t)
            Y_seq.append(tgt)

        return X_seq, Y_seq

    def _backward_pass(self, h_concat, Y_seq, learning_rate):
        total_loss = 0.0

        for t in range(len(Y_seq)):
            h_t = h_concat[t]
            target_id = Y_seq[t]

            if self.use_attention:
                context, _ = self.attention.forward(h_t, h_concat)
                h_t_with_context = np.vstack((h_t, context))
            else:
                h_t_with_context = h_t

            logits = self.Wy @ h_t_with_context + self.by
            probs = self._softmax(logits)

            loss = -np.log(probs[target_id, 0] + 1e-9)
            total_loss += loss

            dlogits = probs.copy()
            dlogits[target_id] -= 1

            self.Wy -= learning_rate * (dlogits @ h_t_with_context.T)
            self.by -= learning_rate * dlogits

            if self.use_attention:
                self.attention.backward(h_t, h_concat, dlogits, learning_rate)

        return total_loss, len(Y_seq)

    def save_model(self, file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            model_data = {
                "Wy": self.Wy,
                "by": self.by,
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "losses": self.losses,
            }
            np.save(file_path, model_data)
            print(f"Model saved to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, file_path):
        try:
            model_data = np.load(file_path, allow_pickle=True).item()
            self.Wy = model_data["Wy"]
            self.by = model_data["by"]
            self.losses = model_data.get("losses", [])
            print(f"Model loaded from {file_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def _softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
