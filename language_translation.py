import os

import numpy as np

from bi_lstm import BiLSTM
from visualization import Visualization


class LanguageTranslation:
    def __init__(self, input_size=256, hidden_size=256):
        self.bi_lstm = BiLSTM(input_size=input_size, hidden_size=hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wy = None
        self.by = None
        self.losses = []
        self.val_losses = []
        self.visualization = Visualization()

    def train(
        self,
        epochs,
        learning_rate,
        vocab_source,
        vocab_target,
        X_ids_train,
        Y_ids_train,
        X_ids_val,
        Y_ids_val,
        embedding_matrix,
    ):
        """Train seq2seq model with teacher forcing."""

        return self._train_model(
            epochs=epochs,
            learning_rate=learning_rate,
            X_train=X_ids_train,
            Y_train=Y_ids_train,
            X_val=X_ids_val,
            Y_val=Y_ids_val,
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
                "val_losses": self.val_losses,
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
            self.losses = model_data.get("losses", [])
            self.val_losses = model_data.get("val_losses", [])
        except Exception as e:
            print(f"Error loading model: {e}")

    def translate(self, translate_text_tokenized, vocab_src, vocab_tgt, embedding_matrix, max_len=100):
        """Translate English to Vietnamese using greedy decoding."""

        unk_id = vocab_src.word2idx.get("<UNK>", 1)
        pad_id = vocab_src.word2idx.get("<PAD>", 0)
        bos_id = vocab_tgt.word2idx.get("<BOS>", 2)
        eos_id = vocab_tgt.word2idx.get("<EOS>", 3)

        # Encode source
        token_ids = [vocab_src.word2idx.get(tok, unk_id) for tok in translate_text_tokenized]
        X_seq = [embedding_matrix[i].reshape(-1, 1) for i in token_ids if i != pad_id]

        if not X_seq:
            return ""

        X_mat = np.array(X_seq).squeeze(2)
        h_concat = self.bi_lstm.forward(X_mat)
        encoder_context = np.mean(h_concat, axis=0, keepdims=True).T

        # Decode with greedy selection
        dest_tokens = []
        prev_token_id = bos_id if bos_id is not None else None

        for step in range(max_len):
            logits = self.Wy @ encoder_context + self.by
            probs = self._softmax(logits)
            predicted_id = int(np.argmax(probs))

            if predicted_id == eos_id or predicted_id == pad_id:
                break

            token = vocab_tgt.idx2word.get(predicted_id, "<UNK>")
            if token not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
                dest_tokens.append(token)

            # Stop if generating same token 3 times
            if predicted_id == prev_token_id:
                if len(dest_tokens) > 0 and dest_tokens.count(token) >= 3:
                    break

            prev_token_id = predicted_id

        return " ".join(dest_tokens)

    def _train_model(self, epochs, learning_rate, X_train, Y_train, X_val, Y_val, embedding_matrix, vocab_target):
        """Train with teacher forcing."""

        vocab_len = len(vocab_target.word2idx)
        vocab_tgt_idx = vocab_target.word2idx

        hidden2 = 2 * self.bi_lstm.hidden_size
        pad_tgt = vocab_tgt_idx.get("<PAD>", 0)
        bos_tgt = vocab_tgt_idx.get("<BOS>", 2)
        output_dim = hidden2

        if self.Wy is None:
            self.Wy = np.random.randn(vocab_len, output_dim) * 0.01
            self.by = np.zeros((vocab_len, 1))

        self.losses = []
        self.val_losses = []

        for epoch in range(epochs):
            total_loss = 0.0
            token_count = 0

            for src_ids, tgt_ids in zip(X_train, Y_train):
                # Encode source
                X_seq = []
                src_ids_list = [int(s) for s in src_ids if int(s) > 0]

                for src_id in src_ids_list:
                    if 0 <= src_id < len(embedding_matrix):
                        X_seq.append(embedding_matrix[src_id].reshape(-1, 1))

                if not X_seq:
                    continue

                X_mat = np.array(X_seq).squeeze(2)
                h_concat = self.bi_lstm.forward(X_mat)
                encoder_context = np.mean(h_concat, axis=0, keepdims=True).T

                # Extract target tokens (remove PAD and BOS)
                Y_seq = [int(y) for y in tgt_ids if int(y) != pad_tgt and int(y) != bos_tgt]

                if not Y_seq:
                    continue

                # Teacher forcing: use true target tokens
                batch_loss = 0.0
                for target_id in Y_seq:
                    logits = self.Wy @ encoder_context + self.by
                    probs = self._softmax(logits)

                    loss = -np.log(np.maximum(probs[target_id, 0], 1e-10))
                    batch_loss += float(loss)

                    # Update weights
                    dlogits = probs.copy()
                    dlogits[target_id, 0] -= 1.0
                    dWy = dlogits @ encoder_context.T

                    self.Wy -= learning_rate * dWy
                    self.by -= learning_rate * dlogits

                total_loss += batch_loss
                token_count += len(Y_seq)

            avg_loss = total_loss / max(token_count, 1)
            self.losses.append(avg_loss)

            # Validation
            val_total_loss = 0.0
            val_token_count = 0

            for src_ids, tgt_ids in zip(X_val, Y_val):
                X_seq = []
                src_ids_list = [int(s) for s in src_ids if int(s) > 0]

                for src_id in src_ids_list:
                    if 0 <= src_id < len(embedding_matrix):
                        X_seq.append(embedding_matrix[src_id].reshape(-1, 1))

                if not X_seq:
                    continue

                X_mat = np.array(X_seq).squeeze(2)
                h_concat = self.bi_lstm.forward(X_mat)
                encoder_context = np.mean(h_concat, axis=0, keepdims=True).T

                Y_seq = [int(y) for y in tgt_ids if int(y) != pad_tgt and int(y) != bos_tgt]

                if not Y_seq:
                    continue

                val_loss = 0.0
                for target_id in Y_seq:
                    logits = self.Wy @ encoder_context + self.by
                    probs = self._softmax(logits)
                    val_loss += -np.log(np.maximum(probs[target_id, 0], 1e-10))

                val_total_loss += val_loss
                val_token_count += len(Y_seq)

            avg_val_loss = val_total_loss / max(val_token_count, 1)
            self.val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save visualization
        combined_losses = np.array([self.losses, self.val_losses])
        self.visualization.plot_loss(combined_losses, title="Training vs Validation Loss", filename="training_loss.png")

        return self

    def _softmax(self, x):
        x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x, axis=0, keepdims=True) + 1e-9)
