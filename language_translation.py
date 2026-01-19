import os

import numpy as np

from attention import Attention
from bi_lstm import BiLSTM
from visualization import Visualization


class LanguageTranslation:
    def __init__(self, input_size=256, hidden_size=256, use_attention=False):
        self.bi_lstm = BiLSTM(input_size=input_size, hidden_size=hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.attention = Attention(2 * hidden_size) if use_attention else None
        self.Wy = None
        self.by = None
        self.losses = []
        self.visualization = Visualization()

    def train(
        self,
        epochs,
        learning_rate,
        vocab_source,
        vocab_target,
        X_ids,
        Y_ids,
        embedding_matrix,
    ) -> object:
        X_train, Y_train = self._prepare_training_data(
            X_ids=X_ids,
            Y_ids=Y_ids,
            pad_src=vocab_source.word2idx.get("<PAD>", 0),
            pad_tgt=vocab_target.word2idx.get("<PAD>", 0),
        )

        return self._train_model(
            epochs=epochs,
            learning_rate=learning_rate,
            X_train=X_train,
            Y_train=Y_train,
            embedding_matrix=embedding_matrix,
            vocab_target=vocab_target,
        )

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
            self.input_size = model_data.get("input_size", self.input_size)
            self.hidden_size = model_data.get("hidden_size", self.hidden_size)
            self.use_attention = model_data.get("use_attention", False)
            self.losses = model_data.get("losses", [])
        except Exception as e:
            print(f"Error loading model: {e}")

    def translate(self,
              translate_text_tokenized,
              vocab_src,
              vocab_tgt,
              embedding_matrix,
              max_len=100):

        unk_id = vocab_src.word2idx.get("<UNK>", 1)
        pad_id = vocab_src.word2idx.get("<PAD>", 0)
        bos_id = vocab_tgt.word2idx.get("<BOS>", None)
        eos_id = vocab_tgt.word2idx.get("<EOS>", None)

        # --- Encoder ---
        token_ids = [vocab_src.word2idx.get(tok, unk_id)
                    for tok in translate_text_tokenized]

        X_seq = [embedding_matrix[i].reshape(-1, 1)
                for i in token_ids if i != pad_id]

        if not X_seq:
            return ""

        X_mat = np.array(X_seq).squeeze(2)
        h_concat = self.bi_lstm.forward(X_mat)

        # --- Decoder (autoregressive) ---
        dest_tokens = []
        prev_id = bos_id if bos_id is not None else None

        for t in range(min(len(h_concat), max_len)):
            h_t = h_concat[t].reshape(-1, 1)

            if self.use_attention and self.attention is not None:
                context, _ = self.attention.forward(h_t, h_concat)
                h_t = np.vstack((h_t, context))

            logits = self.Wy @ h_t + self.by
            probs = self._softmax(logits)

            predicted_id = self._sample(probs, temperature=0.9, top_k=5)

            if predicted_id == eos_id or predicted_id == pad_id:
                break

            token = vocab_tgt.idx2word.get(predicted_id, "<UNK>")
            if token not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
                dest_tokens.append(token)

        return " ".join(dest_tokens)
    
    def _sample(self, probs, temperature=0.9, top_k=5):
        p = probs.ravel()
        p = np.log(p + 1e-9) / temperature
        p = np.exp(p)
        p /= p.sum()

        top_idx = np.argsort(p)[-top_k:]
        top_p = p[top_idx]
        top_p /= top_p.sum()

        return int(np.random.choice(top_idx, p=top_p))



    def _prepare_training_data(self, X_ids, Y_ids, pad_src=0, pad_tgt=0):
        X_train = []
        Y_train = []

        for src_ids, tgt_ids in zip(X_ids, Y_ids):
            # bỏ PAD ở cuối (nếu bạn đã pad max_len)
            src = [int(i) for i in src_ids if int(i) != pad_src]
            tgt = [int(i) for i in tgt_ids if int(i) != pad_tgt]

            if len(src) == 0 or len(tgt) == 0:
                continue

            T = min(len(src), len(tgt))  # align theo timestep
            X_train.append(src[:T])
            Y_train.append(tgt[:T])

        return X_train, Y_train

    def _train_model(self, epochs, learning_rate, X_train, Y_train, embedding_matrix, vocab_target):
        vocab_len = len(vocab_target.word2idx)

        hidden2 = 2 * self.bi_lstm.hidden_size  # BiLSTM output per timestep = 2H
        pad_tgt = vocab_target.word2idx.get("<PAD>", 0)

        if self.use_attention:
            output_dim = hidden2 * 2
        else:
            output_dim = hidden2

        if self.Wy is None:
            self.Wy = np.random.randn(vocab_len, output_dim) * 0.1
            self.by = np.zeros((vocab_len, 1))

        self.losses = []

        for epoch in range(epochs):
            total_loss = 0.0
            token_count = 0

            for src_ids, tgt_ids in zip(X_train, Y_train):
                X_seq, Y_seq = self._prepare_sequence(
                    src_ids=src_ids,
                    tgt_ids=tgt_ids,
                    embedding_matrix=embedding_matrix,
                    pad_tgt=pad_tgt,
                )
                if not X_seq:
                    continue

                # X_seq: list[(E,1)] -> (T,E)
                X_mat = np.array(X_seq).squeeze(2)  # (T, embed_dim)

                h_concat = self.bi_lstm.forward(X_mat)  # (T, 2H, 1) hoặc (T,2H) tùy bạn
                loss, count = self._backward_pass(h_concat, Y_seq, learning_rate)
                total_loss += loss
                token_count += count

            avg_loss = total_loss / max(token_count, 1)
            self.losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return self

    def _prepare_sequence(self, src_ids, tgt_ids, embedding_matrix, pad_tgt=0):
        X_seq = []
        Y_seq = []

        T = min(len(src_ids), len(tgt_ids))
        for i in range(T):
            src_id = int(src_ids[i])
            tgt_id = int(tgt_ids[i])

            if tgt_id == pad_tgt:
                continue

            if 0 <= src_id < len(embedding_matrix):
                X_seq.append(embedding_matrix[src_id].reshape(-1, 1))
                Y_seq.append(tgt_id)

        return X_seq, Y_seq

    def _backward_pass(self, h_concat, Y_seq, learning_rate):
        total_loss = 0.0

        for t, target_id in enumerate(Y_seq):
            h_t = h_concat[t]
            if h_t.ndim == 1:
                h_t = h_t.reshape(-1, 1)

            if self.use_attention and self.attention is not None:
                context, _ = self.attention.forward(h_t, h_concat)
                h_t_with_context = np.vstack((h_t, context))
            else:
                h_t_with_context = h_t

            logits = self.Wy @ h_t_with_context + self.by
            probs = self._softmax(logits)

            loss = -np.log(probs[target_id, 0] + 1e-9)
            total_loss += float(loss)

            dlogits = probs.copy()
            dlogits[target_id, 0] -= 1.0

            self.Wy -= learning_rate * (dlogits @ h_t_with_context.T)
            self.by -= learning_rate * dlogits

            if self.use_attention:
                self.attention.backward(h_t, h_concat, dlogits, learning_rate)

        return total_loss, len(Y_seq)

    def _softmax(self, x):
        x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
