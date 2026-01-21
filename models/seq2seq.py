import os
import time

import numpy as np

from layers.activation import Activation
from layers.attention import BahdanauAttention
from layers.bi_lstm import BiLSTM
from layers.lstm_cell import LSTMCell
from utils.constants import (
    EOS_TOKEN,
    PAD_TOKEN,
    SOS_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
)
from utils.device import get_array_module, get_device_name
from utils.losses import CrossEntropyLoss
from utils.utils import compute_corpus_bleu
from visualization.visualization import Visualization


class Seq2SeqLanguageTranslation:
    def __init__(
        self,
        vocab_src,
        vocab_tgt,
        input_size=128,
        hidden_size=256,
        embedding_matrix_src=None,
        embedding_matrix_tgt=None,
        device="cpu",
    ):
        self.device = device
        self.xp = get_array_module(device)
        self.activation = Activation(self.xp)
        self.loss_fn = CrossEntropyLoss(self.xp)

        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.input_size = input_size
        self.hidden_size = hidden_size

        if embedding_matrix_src is not None:
            self.embedding_src = self.xp.asarray(embedding_matrix_src, dtype=self.xp.float32)
        else:
            self.embedding_src = (
                self.xp.random.randn(len(vocab_src.word2idx), input_size).astype(self.xp.float32) * 0.01
            )

        if embedding_matrix_tgt is not None:
            self.embedding_tgt = self.xp.asarray(embedding_matrix_tgt, dtype=self.xp.float32)
        else:
            self.embedding_tgt = (
                self.xp.random.randn(len(vocab_tgt.word2idx), input_size).astype(self.xp.float32) * 0.01
            )

        self.encoder = BiLSTM(input_size, hidden_size, self.xp)

        self.attention = BahdanauAttention(
            encoder_output_dim=hidden_size * 2,
            hidden_size=hidden_size,
            xp=self.xp,
        )

        self.decoder = LSTMCell(input_size + hidden_size * 2, hidden_size, self.xp)

        self.Wy = self.xp.random.randn(len(vocab_tgt.word2idx), hidden_size).astype(self.xp.float32) * 0.01
        self.by = self.xp.zeros((len(vocab_tgt.word2idx), 1), dtype=self.xp.float32)
        self.dWy = self.xp.zeros_like(self.Wy)
        self.dby = self.xp.zeros_like(self.by)

    def train(
        self, epochs, learning_rate, X_train, Y_train, X_val, Y_val, model_path, batch_size=32, bleu_samples=None
    ):
        best_val_loss = float("inf")
        best_bleu = 0.0
        train_losses = []
        val_losses = []
        bleu_scores = []

        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss = self._compute_train_loss(X_train, Y_train, learning_rate, batch_size)
            val_loss = self._compute_val_loss(X_val, Y_val)
            epoch_time = time.time() - epoch_start

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            bleu_score = None
            if bleu_samples and (epoch % 5 == 0 or epoch == epochs - 1):
                bleu_score = self._compute_bleu_score(bleu_samples)
                bleu_scores.append((epoch, bleu_score))
                if bleu_score > best_bleu:
                    best_bleu = bleu_score

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(model_path, val_loss, is_best=True)

            self._save_checkpoint(model_path, val_loss, is_best=False)
            self._print_epoch_stats(
                epoch, epochs, train_loss, val_loss, best_val_loss, epoch_time, len(X_train), bleu_score, best_bleu
            )

        viz = Visualization(output_dir="result")
        model_name = os.path.basename(model_path).replace(".npy", "")
        plot_path = os.path.join("result", f"{model_name}_loss_plot.png")
        viz.plot_training_history(train_losses, val_losses, plot_path)

        if bleu_scores:
            bleu_plot_path = os.path.join("result", f"{model_name}_bleu_plot.png")
            viz.plot_bleu_scores(bleu_scores, bleu_plot_path)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def to_cpu(arr):
            return arr.get() if hasattr(arr, "get") else arr

        def save_lstm(lstm):
            return {
                "Wf": to_cpu(lstm.Wf),
                "Uf": to_cpu(lstm.Uf),
                "bf": to_cpu(lstm.bf),
                "Wi": to_cpu(lstm.Wi),
                "Ui": to_cpu(lstm.Ui),
                "bi": to_cpu(lstm.bi),
                "Wo": to_cpu(lstm.Wo),
                "Uo": to_cpu(lstm.Uo),
                "bo": to_cpu(lstm.bo),
                "Wc": to_cpu(lstm.Wc),
                "Uc": to_cpu(lstm.Uc),
                "bc": to_cpu(lstm.bc),
                "input_size": lstm.input_size,
                "hidden_size": lstm.hidden_size,
            }

        def save_bi_lstm(bi_lstm):
            return {
                "forward_lstm": save_lstm(bi_lstm.forward_lstm),
                "backward_lstm": save_lstm(bi_lstm.backward_lstm),
                "input_size": bi_lstm.input_size,
                "hidden_size": bi_lstm.hidden_size,
            }

        def save_attention(attention):
            return {
                "Wa": to_cpu(attention.Wa),
                "Ua": to_cpu(attention.Ua),
                "va": to_cpu(attention.va),
                "hidden_size": attention.hidden_size,
                "encoder_output_dim": attention.encoder_output_dim,
            }

        model_data = {
            "embedding_src": to_cpu(self.embedding_src),
            "embedding_tgt": to_cpu(self.embedding_tgt),
            "Wy": to_cpu(self.Wy),
            "by": to_cpu(self.by),
            "encoder": save_bi_lstm(self.encoder),
            "decoder": save_lstm(self.decoder),
            "attention": save_attention(self.attention),
            "vocab_src": self.vocab_src,
            "vocab_tgt": self.vocab_tgt,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
        }

        np.save(path, model_data, allow_pickle=True)

    def load_model(self, path):
        model_data = np.load(path, allow_pickle=True).item()

        def to_device(arr):
            return self.xp.asarray(arr, dtype=self.xp.float32)

        def load_lstm(lstm_data, lstm_obj):
            lstm_obj.Wf = to_device(lstm_data["Wf"])
            lstm_obj.Uf = to_device(lstm_data["Uf"])
            lstm_obj.bf = to_device(lstm_data["bf"])
            lstm_obj.Wi = to_device(lstm_data["Wi"])
            lstm_obj.Ui = to_device(lstm_data["Ui"])
            lstm_obj.bi = to_device(lstm_data["bi"])
            lstm_obj.Wo = to_device(lstm_data["Wo"])
            lstm_obj.Uo = to_device(lstm_data["Uo"])
            lstm_obj.bo = to_device(lstm_data["bo"])
            lstm_obj.Wc = to_device(lstm_data["Wc"])
            lstm_obj.Uc = to_device(lstm_data["Uc"])
            lstm_obj.bc = to_device(lstm_data["bc"])
            lstm_obj._init_gradients()

        def load_bi_lstm(bi_lstm_data, bi_lstm_obj):
            load_lstm(bi_lstm_data["forward_lstm"], bi_lstm_obj.forward_lstm)
            load_lstm(bi_lstm_data["backward_lstm"], bi_lstm_obj.backward_lstm)

        def load_attention(attention_data, attention_obj):
            attention_obj.Wa = to_device(attention_data["Wa"])
            attention_obj.Ua = to_device(attention_data["Ua"])
            attention_obj.va = to_device(attention_data["va"])
            attention_obj._init_gradients()

        self.embedding_src = to_device(model_data["embedding_src"])
        self.embedding_tgt = to_device(model_data["embedding_tgt"])
        self.Wy = to_device(model_data["Wy"])
        self.by = to_device(model_data["by"])

        load_bi_lstm(model_data["encoder"], self.encoder)
        load_lstm(model_data["decoder"], self.decoder)
        load_attention(model_data["attention"], self.attention)

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

        enc_outputs, h, c = self._encoder_forward(x_ids)

        output_ids = []
        prev_id = self.vocab_tgt.word2idx[SOS_TOKEN]
        eos_id = self.vocab_tgt.word2idx[EOS_TOKEN]

        for step in range(max_len):
            y_embed = self.embedding_tgt[prev_id].reshape(-1, 1)

            context, _ = self.attention.forward(h, enc_outputs)
            dec_input = self.xp.vstack((y_embed, context))

            h, c, _ = self.decoder.forward(dec_input, h, c)

            logits = self.Wy @ h + self.by
            logits = logits / 1.2
            probs = self.activation.softmax(logits).ravel()

            probs = self._adjust_translation_probs(probs, step, output_ids, len(tokens))
            pred_id = self._sample_next_token(probs, top_k=5)

            if pred_id == eos_id:
                break

            output_ids.append(pred_id)
            prev_id = pred_id

        return self._format_translation_output(output_ids)

    def _train_step(self, x_ids, y_ids, lr=0.001):
        enc_outputs, h, c = self._encoder_forward(x_ids)
        loss, dec_cache = self._decoder_forward_train(enc_outputs, h, c, y_ids)
        self._backward_pass(dec_cache, enc_outputs, lr)
        return loss

    def _train_step_no_update(self, x_ids, y_ids):
        """Train step that accumulates gradients without applying them"""
        enc_outputs, h, c = self._encoder_forward(x_ids)
        loss, dec_cache = self._decoder_forward_train(enc_outputs, h, c, y_ids)
        self._backward_pass_accumulate(dec_cache, enc_outputs)
        return loss

    def _encoder_forward(self, x_ids):
        x_embed_seq = []
        for tok in x_ids:
            idx = int(tok) if hasattr(tok, "item") else tok
            emb = self.embedding_src[idx].copy().reshape(-1, 1)
            x_embed_seq.append(emb)
        enc_outputs = self.encoder.forward(x_embed_seq)

        h0 = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)
        c0 = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)

        return enc_outputs, h0, c0

    def _decoder_forward_train(self, enc_outputs, h, c, y_ids):
        cache = []
        loss = 0.0

        prev_id = self.vocab_tgt.word2idx[SOS_TOKEN]
        pad_id = self.vocab_tgt.word2idx[PAD_TOKEN]
        eos_id = self.vocab_tgt.word2idx[EOS_TOKEN]

        step_count = 0

        for t in range(len(y_ids) - 1):
            y_embed = self.embedding_tgt[prev_id].reshape(-1, 1)

            context, alpha = self.attention.forward(h, enc_outputs)

            dec_input = self.xp.vstack((y_embed, context))

            h, c, lstm_cache = self.decoder.forward(dec_input, h, c)

            logits = self.Wy @ h + self.by
            probs = self.activation.softmax(logits)

            target = y_ids[t + 1]

            if target == pad_id or target == eos_id:
                prev_id = target
                continue

            loss += self.loss_fn.compute_loss(probs, target)
            step_count += 1

            cache.append(
                {
                    "prev_id": prev_id,
                    "h": h,
                    "c": c,
                    "probs": probs,
                    "target": target,
                    "lstm_cache": lstm_cache,
                    "context": context,
                    "alpha": alpha,
                }
            )

            prev_id = target

        avg_loss = loss / max(1, step_count)
        return avg_loss, cache

    def _backward_pass(self, dec_cache, enc_outputs, lr):
        dh_next = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)
        dc_next = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)

        self.dWy.fill(0)
        self.dby.fill(0)

        for step in reversed(dec_cache):
            probs = step["probs"]
            target = step["target"]
            h = step["h"]
            lstm_cache = step["lstm_cache"]

            dlogits = probs.copy()
            dlogits[target] -= 1

            self.dWy += dlogits @ h.T
            self.dby += dlogits

            dh = self.Wy.T @ dlogits + dh_next

            ddec_input, dh_next, dc_next = self.decoder.backward(dh, dc_next, lstm_cache)

            dy_embed = ddec_input[: self.input_size]
            dcontext = ddec_input[self.input_size :]

            prev_id = step["prev_id"]
            self.embedding_tgt[prev_id] -= lr * dy_embed.squeeze()

            dh_attn, denc = self.attention.backward(dcontext)

            dh_next += dh_attn

            self.encoder.backward(denc)

        self.encoder.apply_gradients(lr)
        self.decoder.apply_gradients(lr)
        self.attention.apply_gradients(lr)

        self.Wy -= lr * self.dWy
        self.by -= lr * self.dby

    def _backward_pass_accumulate(self, dec_cache, enc_outputs):
        """Backward pass that accumulates gradients without applying them"""
        dh_next = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)
        dc_next = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)

        for step in reversed(dec_cache):
            probs = step["probs"]
            target = step["target"]
            h = step["h"]
            lstm_cache = step["lstm_cache"]

            dlogits = self.loss_fn.compute_gradient(probs, target)

            self.dWy += dlogits @ h.T
            self.dby += dlogits

            dh = self.Wy.T @ dlogits + dh_next
            ddec_input, dh_next, dc_next = self.decoder.backward(dh, dc_next, lstm_cache)

            dy_embed = ddec_input[: self.input_size]
            dcontext = ddec_input[self.input_size :]

            dh_attn, denc = self.attention.backward(dcontext)
            dh_next += dh_attn

            self.encoder.backward(denc)

    def _apply_accumulated_gradients(self, lr, batch_size):
        """Apply accumulated gradients with averaging over batch"""
        avg_lr = lr / batch_size

        self.encoder.apply_gradients(avg_lr)
        self.decoder.apply_gradients(avg_lr)
        self.attention.apply_gradients(avg_lr)

        self.Wy -= avg_lr * self.dWy
        self.by -= avg_lr * self.dby

        self.dWy.fill(0)
        self.dby.fill(0)

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
            enc_outputs, h, c = self._encoder_forward(x_ids)
            loss, _ = self._decoder_forward_train(enc_outputs, h, c, y_ids)
            total_loss += loss
        return total_loss / max(1, len(X_val))

    def _compute_bleu_score(self, samples, max_samples=50):
        references = []
        hypotheses = []

        n_samples = min(len(samples), max_samples)
        for i in range(n_samples):
            src_tokens, ref_tokens = samples[i]

            hyp_translation = self.translate(src_tokens, max_len=50, seed=None)

            if isinstance(ref_tokens[0], (int, np.integer)) or hasattr(ref_tokens[0], "item"):
                ref_text = " ".join(
                    [
                        self.vocab_tgt.idx2word.get(int(idx), UNK_TOKEN)
                        for idx in ref_tokens
                        if int(idx)
                        not in [
                            self.vocab_tgt.word2idx[PAD_TOKEN],
                            self.vocab_tgt.word2idx[SOS_TOKEN],
                            self.vocab_tgt.word2idx[EOS_TOKEN],
                        ]
                    ]
                )
            else:
                ref_text = " ".join([word for word in ref_tokens if word not in SPECIAL_TOKENS])

            references.append(ref_text)
            hypotheses.append(hyp_translation)

            if i < 3:
                print(f"\nBLEU Sample {i+1}:")
                print(f"  Source: {' '.join(src_tokens[:10])}...")
                print(f"  Reference: {ref_text[:80]}...")
                print(f"  Hypothesis: {hyp_translation[:80]}...")

        bleu = compute_corpus_bleu(references, hypotheses)
        print(f"\nComputed BLEU on {n_samples} samples: {bleu:.2f}")
        return bleu

    def _save_checkpoint(self, model_path, val_loss, is_best):
        if is_best:
            path = model_path.replace(".npy", "_best.npy")
            self.save_model(path)
            print(f"Best model saved (val_loss: {val_loss:.4f})")
        else:
            path = model_path.replace(".npy", "_latest.npy")
            self.save_model(path)

    def _print_epoch_stats(
        self,
        epoch,
        total_epochs,
        train_loss,
        val_loss,
        best_val_loss,
        epoch_time=None,
        n_samples=None,
        bleu_score=None,
        best_bleu=None,
    ):
        is_best = " (BEST)" if val_loss == best_val_loss else ""
        time_str = f" | Time: {epoch_time:.1f}s" if epoch_time else ""
        speed_str = ""
        if epoch_time and n_samples:
            samples_per_sec = n_samples / epoch_time
            speed_str = f" | Speed: {samples_per_sec:.1f} samples/s"
        bleu_str = ""
        if bleu_score is not None:
            bleu_best = " (BEST)" if bleu_score == best_bleu else ""
            bleu_str = f" | BLEU: {bleu_score:.2f}{bleu_best}"
        print(
            f"Epoch {epoch+1}/{total_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{is_best}{bleu_str}{time_str}{speed_str}"
        )

    def _prepare_translation_input(self, tokens, max_len):
        x_ids = [self.vocab_src.word2idx.get(t, self.vocab_src.word2idx[UNK_TOKEN]) for t in tokens]
        max_len = min(max_len, len(tokens) * 2 + 5)
        return x_ids, max_len

    def _adjust_translation_probs(self, flat_probs, step, output_ids, source_len):
        pad_id = self.vocab_tgt.word2idx[PAD_TOKEN]
        eos_id = self.vocab_tgt.word2idx[EOS_TOKEN]
        sos_id = self.vocab_tgt.word2idx[SOS_TOKEN]
        unk_id = self.vocab_tgt.word2idx[UNK_TOKEN]

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
            # CuPy requires size parameter for random.choice
            if self.device == "gpu":
                pred_id = int(self.xp.random.choice(top_k_ids, size=1, p=top_k_probs)[0])
            else:
                pred_id = int(self.xp.random.choice(top_k_ids, p=top_k_probs))

        return pred_id

    def _format_translation_output(self, output_ids):
        if not output_ids:
            return "[Model needs more training]"

        filtered_words = []
        for idx in output_ids:
            word = self.vocab_tgt.idx2word.get(idx, UNK_TOKEN)
            if word not in SPECIAL_TOKENS:
                filtered_words.append(word)

        if not filtered_words:
            return "[Model needs more training]"

        return " ".join(filtered_words)
