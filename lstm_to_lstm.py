import os
import numpy as np
from activation import Activation
from lstm import LSTMCell


class LstmToLstmLanguageTranslation:
    def __init__(
        self,
        vocab_src,
        vocab_tgt,
        input_size=128,
        hidden_size=256,
        embedding_matrix_src=None,
        embedding_matrix_tgt=None,
    ):
        self.activation = Activation()

        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding_src = (
            embedding_matrix_src
            if embedding_matrix_src is not None
            else np.random.randn(len(vocab_src.word2idx), input_size) * 0.01
        )
        self.embedding_tgt = (
            embedding_matrix_tgt
            if embedding_matrix_tgt is not None
            else np.random.randn(len(vocab_tgt.word2idx), input_size) * 0.01
        )

        self.encoder = LSTMCell(input_size, hidden_size)
        self.decoder = LSTMCell(input_size, hidden_size)

        self.Wy = np.random.randn(len(vocab_tgt.word2idx), hidden_size) * 0.01
        self.by = np.zeros((len(vocab_tgt.word2idx), 1))
        self.dWy = np.zeros_like(self.Wy)
        self.dby = np.zeros_like(self.by)

    def _encoder_forward(self, x_ids):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        caches = []

        for tok in x_ids:
            x = self.embedding_src[tok].reshape(-1, 1)
            h, c, cache = self.encoder.forward(x, h, c)
            caches.append((tok, cache))

        return h, c, caches

    def _decoder_forward_train(self, h, c, y_ids):
        cache = []
        loss = 0.0
        prev_id = self.vocab_tgt.word2idx["<SOS>"]
        pad_id = self.vocab_tgt.word2idx["<PAD>"]
        eos_id = self.vocab_tgt.word2idx["<EOS>"]

        step_count = 0

        for t in range(len(y_ids) - 1):
            x = self.embedding_tgt[prev_id].reshape(-1, 1)
            h, c, lstm_cache = self.decoder.forward(x, h, c)

            logits = self.Wy @ h + self.by
            probs = self.activation.softmax(logits)

            target = y_ids[t + 1]

            if target == pad_id or target == eos_id:
                prev_id = target
                continue

            loss += -np.log(probs[target, 0] + 1e-9)
            step_count += 1

            cache.append((prev_id, h, probs, target, lstm_cache))
            prev_id = target

        avg_loss = loss / max(1, step_count)
        return avg_loss, cache

    def _backward_pass(self, dec_cache, enc_cache, lr):
        dh = np.zeros((self.hidden_size, 1))
        dc = np.zeros((self.hidden_size, 1))

        self.dWy.fill(0)
        self.dby.fill(0)

        for prev_id, h, probs, target, lstm_cache in reversed(dec_cache):
            dlogits = probs.copy()
            dlogits[target] -= 1

            self.dWy += dlogits @ h.T
            self.dby += dlogits

            dh_t = self.Wy.T @ dlogits + dh
            dx, dh, dc = self.decoder.backward(dh_t, dc, lstm_cache)

            self.embedding_tgt[prev_id] -= lr * dx.squeeze()

        self.Wy -= lr * self.dWy
        self.by -= lr * self.dby

        for tok, cache in reversed(enc_cache):
            dx, dh, dc = self.encoder.backward(dh, dc, cache)
            self.embedding_src[tok] -= lr * dx.squeeze()

        self.encoder.apply_gradients(lr)
        self.decoder.apply_gradients(lr)

    def train_step(self, x_ids, y_ids, lr=0.01):
        h, c, enc_cache = self._encoder_forward(x_ids)
        loss, dec_cache = self._decoder_forward_train(h, c, y_ids)
        self._backward_pass(dec_cache, enc_cache, lr)
        return loss

    def train(self, epochs, learning_rate, X_train, Y_train, X_val, Y_val, model_path):
        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_loss = 0
            for x_ids, y_ids in zip(X_train, Y_train):
                loss = self.train_step(x_ids, y_ids, learning_rate)
                train_loss += loss
            train_loss /= len(X_train)

            val_loss = 0
            for x_ids, y_ids in zip(X_val, Y_val):
                h, c, enc_cache = self._encoder_forward(x_ids)
                loss, _ = self._decoder_forward_train(h, c, y_ids)
                val_loss += loss
            val_loss /= len(X_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(model_path.replace(".npy", "_best.npy"))
                print(f"Best model saved (val_loss: {val_loss:.4f})")

            self.save_model(model_path.replace(".npy", "_latest.npy"))

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                + (" (BEST)" if val_loss == best_val_loss else "")
            )

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            "embedding_src": self.embedding_src,
            "embedding_tgt": self.embedding_tgt,
            "Wy": self.Wy,
            "by": self.by,
            "encoder_Wf": self.encoder.Wf,
            "encoder_Uf": self.encoder.Uf,
            "encoder_bf": self.encoder.bf,
            "encoder_Wi": self.encoder.Wi,
            "encoder_Ui": self.encoder.Ui,
            "encoder_bi": self.encoder.bi,
            "encoder_Wo": self.encoder.Wo,
            "encoder_Uo": self.encoder.Uo,
            "encoder_bo": self.encoder.bo,
            "encoder_Wc": self.encoder.Wc,
            "encoder_Uc": self.encoder.Uc,
            "encoder_bc": self.encoder.bc,
            "decoder_Wf": self.decoder.Wf,
            "decoder_Uf": self.decoder.Uf,
            "decoder_bf": self.decoder.bf,
            "decoder_Wi": self.decoder.Wi,
            "decoder_Ui": self.decoder.Ui,
            "decoder_bi": self.decoder.bi,
            "decoder_Wo": self.decoder.Wo,
            "decoder_Uo": self.decoder.Uo,
            "decoder_bo": self.decoder.bo,
            "decoder_Wc": self.decoder.Wc,
            "decoder_Uc": self.decoder.Uc,
            "decoder_bc": self.decoder.bc,
        }
        np.save(path, model_data)

    def load_model(self, path):
        model_data = np.load(path, allow_pickle=True).item()
        self.embedding_src = model_data["embedding_src"]
        self.embedding_tgt = model_data["embedding_tgt"]
        self.Wy = model_data["Wy"]
        self.by = model_data["by"]
        self.encoder.Wf = model_data["encoder_Wf"]
        self.encoder.Uf = model_data["encoder_Uf"]
        self.encoder.bf = model_data["encoder_bf"]
        self.encoder.Wi = model_data["encoder_Wi"]
        self.encoder.Ui = model_data["encoder_Ui"]
        self.encoder.bi = model_data["encoder_bi"]
        self.encoder.Wo = model_data["encoder_Wo"]
        self.encoder.Uo = model_data["encoder_Uo"]
        self.encoder.bo = model_data["encoder_bo"]
        self.encoder.Wc = model_data["encoder_Wc"]
        self.encoder.Uc = model_data["encoder_Uc"]
        self.encoder.bc = model_data["encoder_bc"]
        self.decoder.Wf = model_data["decoder_Wf"]
        self.decoder.Uf = model_data["decoder_Uf"]
        self.decoder.bf = model_data["decoder_bf"]
        self.decoder.Wi = model_data["decoder_Wi"]
        self.decoder.Ui = model_data["decoder_Ui"]
        self.decoder.bi = model_data["decoder_bi"]
        self.decoder.Wo = model_data["decoder_Wo"]
        self.decoder.Uo = model_data["decoder_Uo"]
        self.decoder.bo = model_data["decoder_bo"]
        self.decoder.Wc = model_data["decoder_Wc"]
        self.decoder.Uc = model_data["decoder_Uc"]
        self.decoder.bc = model_data["decoder_bc"]

    def load_best_model(self, path):
        best_path = path.replace(".npy", "_best.npy")
        if os.path.exists(best_path):
            self.load_model(best_path)
            print(f"Best model loaded from {best_path}")
        else:
            raise FileNotFoundError(f"Model not found: {best_path}")

    def translate(self, tokens, max_len=100):
        x_ids = [self.vocab_src.word2idx.get(t, self.vocab_src.word2idx["<UNK>"]) for t in tokens]

        max_len = min(max_len, len(tokens) * 2 + 5)

        h, c, _ = self._encoder_forward(x_ids)

        output_ids = []
        prev_id = self.vocab_tgt.word2idx["<SOS>"]
        pad_id = self.vocab_tgt.word2idx["<PAD>"]
        eos_id = self.vocab_tgt.word2idx["<EOS>"]

        for step in range(max_len):
            x = self.embedding_tgt[prev_id].reshape(-1, 1)
            h, c, _ = self.decoder.forward(x, h, c)

            logits = self.Wy @ h + self.by
            logits = logits / 1.2

            probs = self.activation.softmax(logits)
            flat_probs = probs.ravel()

            flat_probs[pad_id] = 0

            if step == 0:
                flat_probs[eos_id] = 0

            min_len = max(1, len(tokens))
            if step >= min_len:
                flat_probs[eos_id] *= 1.5

            for used_id in output_ids[-3:]:
                flat_probs[used_id] *= 0.3
            k = 5
            top_k_ids = flat_probs.argsort()[-k:]
            top_k_probs = flat_probs[top_k_ids]

            top_k_probs = np.clip(top_k_probs, 1e-9, None)
            prob_sum = np.sum(top_k_probs)

            if prob_sum == 0 or not np.isfinite(prob_sum):
                pred_id = int(np.argmax(flat_probs))
            else:
                top_k_probs = top_k_probs / prob_sum
                pred_id = int(np.random.choice(top_k_ids, p=top_k_probs))

            print("Step", step, "Pred:", self.vocab_tgt.idx2word.get(pred_id, "<UNK>"))

            if pred_id == eos_id:
                break

            output_ids.append(pred_id)
            prev_id = pred_id

        if not output_ids:
            return "[Model needs more training - only predicting PAD tokens]"

        result = " ".join([self.vocab_tgt.idx2word.get(idx, "<UNK>") for idx in output_ids])
        return result
