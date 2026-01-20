import os
import time
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np
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
        device='cpu',
    ):
        self.device = device
        self.xp = cp if device == 'gpu' and cp != np else np
        self.activation = Activation(self.xp)

        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding_src = (
            self.xp.asarray(embedding_matrix_src, dtype=self.xp.float32)
            if embedding_matrix_src is not None
            else self.xp.random.randn(len(vocab_src.word2idx), input_size).astype(self.xp.float32) * 0.01
        )
        self.embedding_tgt = (
            self.xp.asarray(embedding_matrix_tgt, dtype=self.xp.float32)
            if embedding_matrix_tgt is not None
            else self.xp.random.randn(len(vocab_tgt.word2idx), input_size).astype(self.xp.float32) * 0.01
        )

        self.encoder = LSTMCell(input_size, hidden_size, self.xp)
        self.decoder = LSTMCell(input_size, hidden_size, self.xp)

        self.Wy = self.xp.random.randn(len(vocab_tgt.word2idx), hidden_size).astype(self.xp.float32) * 0.01
        self.by = self.xp.zeros((len(vocab_tgt.word2idx), 1), dtype=self.xp.float32)
        self.dWy = self.xp.zeros_like(self.Wy)
        self.dby = self.xp.zeros_like(self.by)

    def train(self, epochs, learning_rate, X_train, Y_train, X_val, Y_val, model_path, batch_size=32):
        best_val_loss = float("inf")

        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss = self._compute_train_loss(X_train, Y_train, learning_rate, batch_size)
            val_loss = self._compute_val_loss(X_val, Y_val)
            epoch_time = time.time() - epoch_start

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(model_path, val_loss, is_best=True)

            self._save_checkpoint(model_path, val_loss, is_best=False)
            self._print_epoch_stats(epoch, epochs, train_loss, val_loss, best_val_loss, epoch_time, len(X_train))

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        def to_cpu(arr):
            return arr.get() if hasattr(arr, 'get') else arr
        
        model_data = {
            "embedding_src": to_cpu(self.embedding_src),
            "embedding_tgt": to_cpu(self.embedding_tgt),
            "Wy": to_cpu(self.Wy),
            "by": to_cpu(self.by),
            "encoder_Wf": to_cpu(self.encoder.Wf),
            "encoder_Uf": to_cpu(self.encoder.Uf),
            "encoder_bf": to_cpu(self.encoder.bf),
            "encoder_Wi": to_cpu(self.encoder.Wi),
            "encoder_Ui": to_cpu(self.encoder.Ui),
            "encoder_bi": to_cpu(self.encoder.bi),
            "encoder_Wo": to_cpu(self.encoder.Wo),
            "encoder_Uo": to_cpu(self.encoder.Uo),
            "encoder_bo": to_cpu(self.encoder.bo),
            "encoder_Wc": to_cpu(self.encoder.Wc),
            "encoder_Uc": to_cpu(self.encoder.Uc),
            "encoder_bc": to_cpu(self.encoder.bc),
            "decoder_Wf": to_cpu(self.decoder.Wf),
            "decoder_Uf": to_cpu(self.decoder.Uf),
            "decoder_bf": to_cpu(self.decoder.bf),
            "decoder_Wi": to_cpu(self.decoder.Wi),
            "decoder_Ui": to_cpu(self.decoder.Ui),
            "decoder_bi": to_cpu(self.decoder.bi),
            "decoder_Wo": to_cpu(self.decoder.Wo),
            "decoder_Uo": to_cpu(self.decoder.Uo),
            "decoder_bo": to_cpu(self.decoder.bo),
            "decoder_Wc": to_cpu(self.decoder.Wc),
            "decoder_Uc": to_cpu(self.decoder.Uc),
            "decoder_bc": to_cpu(self.decoder.bc),
        }
        np.save(path, model_data)

    def load_model(self, path):
        model_data = np.load(path, allow_pickle=True).item()
        def to_device(arr):
            return self.xp.asarray(arr, dtype=self.xp.float32)
        
        if "encoder_Wf" not in model_data:
            print("Warning: Loading old model format. Please retrain for optimal GPU performance.")
            return
        
        self.embedding_src = to_device(model_data["embedding_src"])
        self.embedding_tgt = to_device(model_data["embedding_tgt"])
        self.Wy = to_device(model_data["Wy"])
        self.by = to_device(model_data["by"])
        self.encoder.Wf = to_device(model_data["encoder_Wf"])
        self.encoder.Uf = to_device(model_data["encoder_Uf"])
        self.encoder.bf = to_device(model_data["encoder_bf"])
        self.encoder.Wi = to_device(model_data["encoder_Wi"])
        self.encoder.Ui = to_device(model_data["encoder_Ui"])
        self.encoder.bi = to_device(model_data["encoder_bi"])
        self.encoder.Wo = to_device(model_data["encoder_Wo"])
        self.encoder.Uo = to_device(model_data["encoder_Uo"])
        self.encoder.bo = to_device(model_data["encoder_bo"])
        self.encoder.Wc = to_device(model_data["encoder_Wc"])
        self.encoder.Uc = to_device(model_data["encoder_Uc"])
        self.encoder.bc = to_device(model_data["encoder_bc"])
        self.decoder.Wf = to_device(model_data["decoder_Wf"])
        self.decoder.Uf = to_device(model_data["decoder_Uf"])
        self.decoder.bf = to_device(model_data["decoder_bf"])
        self.decoder.Wi = to_device(model_data["decoder_Wi"])
        self.decoder.Ui = to_device(model_data["decoder_Ui"])
        self.decoder.bi = to_device(model_data["decoder_bi"])
        self.decoder.Wo = to_device(model_data["decoder_Wo"])
        self.decoder.Uo = to_device(model_data["decoder_Uo"])
        self.decoder.bo = to_device(model_data["decoder_bo"])
        self.decoder.Wc = to_device(model_data["decoder_Wc"])
        self.decoder.Uc = to_device(model_data["decoder_Uc"])
        self.decoder.bc = to_device(model_data["decoder_bc"])

    def load_best_model(self, path):
        best_path = path.replace(".npy", "_best.npy")
        if os.path.exists(best_path):
            self.load_model(best_path)
            print(f"Best model loaded from {best_path}")
        else:
            raise FileNotFoundError(f"Model not found: {best_path}")

    def translate(self, tokens, max_len=100, seed=42):
        if seed is not None:
            self.xp.random.seed(seed)
        
        x_ids, max_len = self._prepare_translation_input(tokens, max_len)
        
        if not x_ids:
            return "[Empty input]"
        
        h, c, _ = self._encoder_forward(x_ids)

        output_ids = []
        prev_id = self.vocab_tgt.word2idx["<SOS>"]
        eos_id = self.vocab_tgt.word2idx["<EOS>"]

        for step in range(max_len):
            x = self.embedding_tgt[prev_id].reshape(-1, 1)
            h, c, _ = self.decoder.forward(x, h, c)

            logits = self.Wy @ h + self.by
            logits = logits / 1.2

            probs = self.activation.softmax(logits)
            flat_probs = probs.ravel()

            flat_probs = self._adjust_translation_probs(flat_probs, step, output_ids, len(tokens))
            pred_id = self._sample_next_token(flat_probs, top_k=5)

            print("Step", step, "Pred:", self.vocab_tgt.idx2word.get(pred_id, "<UNK>"))

            if pred_id == eos_id:
                break

            output_ids.append(pred_id)
            prev_id = pred_id

        return self._format_translation_output(output_ids)
    
    def _train_step(self, x_ids, y_ids, lr=0.01):
        h, c, enc_cache = self._encoder_forward(x_ids)
        loss, dec_cache = self._decoder_forward_train(h, c, y_ids)
        self._backward_pass(dec_cache, enc_cache, lr)
        return loss

    def _train_step_no_update(self, x_ids, y_ids):
        h, c, enc_cache = self._encoder_forward(x_ids)
        loss, dec_cache = self._decoder_forward_train(h, c, y_ids)
        self._backward_pass_accumulate(dec_cache, enc_cache)
        return loss

    def _encoder_forward(self, x_ids):
        h = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)
        c = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)
        caches = []

        for tok in x_ids:
            x = self.embedding_src[tok].reshape(-1, 1)
            h, c, cache = self.encoder.forward(x, h, c)
            caches.append((tok, cache))

        return h, c, caches

    def _compute_train_loss(self, X_train, Y_train, learning_rate, batch_size=32):
        total_loss = 0
        n_samples = len(X_train)
        
        for i in range(0, n_samples, batch_size):
            batch_loss = 0
            batch_end = min(i + batch_size, n_samples)
            actual_batch_size = batch_end - i
            
            for j in range(i, batch_end):
                x_ids, y_ids = X_train[j], Y_train[j]
                loss = self._train_step_no_update(x_ids, y_ids)
                batch_loss += loss
                total_loss += loss
            
            self._apply_accumulated_gradients(learning_rate, actual_batch_size)
        
        return total_loss / max(1, n_samples)

    def _compute_val_loss(self, X_val, Y_val):
        total_loss = 0
        for x_ids, y_ids in zip(X_val, Y_val):
            h, c, enc_cache = self._encoder_forward(x_ids)
            loss, _ = self._decoder_forward_train(h, c, y_ids)
            total_loss += loss
        return total_loss / max(1, len(X_val))

    def _save_checkpoint(self, model_path, val_loss, is_best):
        if is_best:
            path = model_path.replace(".npy", "_best.npy")
            self.save_model(path)
            print(f"Best model saved (val_loss: {val_loss:.4f})")
        else:
            path = model_path.replace(".npy", "_latest.npy")
            self.save_model(path)

    def _print_epoch_stats(self, epoch, total_epochs, train_loss, val_loss, best_val_loss, epoch_time=None, n_samples=None):
        is_best = " (BEST)" if val_loss == best_val_loss else ""
        time_str = f" | Time: {epoch_time:.1f}s" if epoch_time else ""
        speed_str = ""
        if epoch_time and n_samples:
            samples_per_sec = n_samples / epoch_time
            speed_str = f" | Speed: {samples_per_sec:.1f} samples/s"
        print(f"Epoch {epoch+1}/{total_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{is_best}{time_str}{speed_str}")

    def _compute_cross_entropy_loss(self, probs, target):
        return -self.xp.log(probs[target, 0] + 1e-9)

    def _compute_cross_entropy_loss_gradient(self, probs, target):
        dlogits = probs.copy()
        dlogits[target] -= 1
        return dlogits

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

            loss += self._compute_cross_entropy_loss(probs, target)
            step_count += 1

            cache.append((prev_id, h, probs, target, lstm_cache))
            prev_id = target

        avg_loss = loss / max(1, step_count)
        return avg_loss, cache

    def _backward_pass(self, dec_cache, enc_cache, lr):
        dh = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)
        dc = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)

        self.dWy.fill(0)
        self.dby.fill(0)

        for prev_id, h, probs, target, lstm_cache in reversed(dec_cache):
            dlogits = self._compute_cross_entropy_loss_gradient(probs, target)

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

    def _backward_pass_accumulate(self, dec_cache, enc_cache):
        dh = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)
        dc = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)

        for prev_id, h, probs, target, lstm_cache in reversed(dec_cache):
            dlogits = self._compute_cross_entropy_loss_gradient(probs, target)

            self.dWy += dlogits @ h.T
            self.dby += dlogits

            dh_t = self.Wy.T @ dlogits + dh
            dx, dh, dc = self.decoder.backward(dh_t, dc, lstm_cache)

        for tok, cache in reversed(enc_cache):
            dx, dh, dc = self.encoder.backward(dh, dc, cache)

    def _apply_accumulated_gradients(self, lr, batch_size):
        avg_lr = lr / batch_size
        
        self.Wy -= avg_lr * self.dWy
        self.by -= avg_lr * self.dby
        
        self.encoder.apply_gradients(avg_lr)
        self.decoder.apply_gradients(avg_lr)
        
        self.dWy.fill(0)
        self.dby.fill(0)

    def _prepare_translation_input(self, tokens, max_len):
        x_ids = [self.vocab_src.word2idx.get(t, self.vocab_src.word2idx["<UNK>"]) for t in tokens]
        max_len = min(max_len, len(tokens) * 2 + 5)
        return x_ids, max_len

    def _adjust_translation_probs(self, flat_probs, step, output_ids, source_len):
        pad_id = self.vocab_tgt.word2idx["<PAD>"]
        eos_id = self.vocab_tgt.word2idx["<EOS>"]
        sos_id = self.vocab_tgt.word2idx["<SOS>"]
        unk_id = self.vocab_tgt.word2idx["<UNK>"]

        flat_probs[pad_id] = 0
        flat_probs[sos_id] = 0
        flat_probs[unk_id] = 0

        min_len = max(1, source_len)
        if step < min_len:
            flat_probs[eos_id] = 0
        elif step >= source_len + 1:
            flat_probs[eos_id] *= 1.3
        
        if step >= source_len * 2:
            flat_probs[eos_id] *= 2.5

        for used_id in output_ids[-5:]:
            flat_probs[used_id] *= 0.05

        return flat_probs

    def _sample_next_token(self, flat_probs, top_k=5):
        top_k_ids = flat_probs.argsort()[-top_k:]
        top_k_probs = flat_probs[top_k_ids]

        top_k_probs = self.xp.clip(top_k_probs, 1e-9, None)
        prob_sum = self.xp.sum(top_k_probs)

        if prob_sum == 0 or not self.xp.isfinite(prob_sum):
            pred_id = int(self.xp.argmax(flat_probs))
        else:
            top_k_probs = top_k_probs / prob_sum
            if self.device == 'gpu':
                pred_id = int(self.xp.random.choice(top_k_ids, size=1, p=top_k_probs)[0])
            else:
                pred_id = int(self.xp.random.choice(top_k_ids, p=top_k_probs))

        return pred_id

    def _format_translation_output(self, output_ids):
        if not output_ids:
            return "[Model needs more training - only predicting PAD tokens]"
        
        special_tokens = {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}
        filtered_words = []
        for idx in output_ids:
            word = self.vocab_tgt.idx2word.get(idx, "<UNK>")
            if word not in special_tokens:
                filtered_words.append(word)
        
        if not filtered_words:
            return "[Model needs more training]"
        
        return " ".join(filtered_words)
