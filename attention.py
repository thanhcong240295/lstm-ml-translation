import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np
from activation import Activation


class BahdanauAttention:
    def __init__(self, encoder_output_dim, hidden_size=256, xp=np):
        self.hidden_size = hidden_size
        self.encoder_output_dim = encoder_output_dim
        self.xp = xp
        self.activation = Activation(xp)

        limit = self.xp.sqrt(6 / (hidden_size + encoder_output_dim))
        self.Wa = self.xp.random.uniform(-limit, limit, (hidden_size, hidden_size)).astype(self.xp.float32)
        self.Ua = self.xp.random.uniform(-limit, limit, (hidden_size, encoder_output_dim)).astype(self.xp.float32)
        self.va = self.xp.random.uniform(-limit, limit, (1, hidden_size)).astype(self.xp.float32)

        self._init_gradients()

    def _init_gradients(self):
        self.dWa = self.xp.zeros_like(self.Wa)
        self.dUa = self.xp.zeros_like(self.Ua)
        self.dva = self.xp.zeros_like(self.va)

    def forward(self, decoder_state, encoder_outputs):
        # encoder_outputs: (T, H_enc, 1)
        T = encoder_outputs.shape[0]
        scores = []

        for t in range(T):
            h_enc_t = encoder_outputs[t]
            z = self.Wa @ decoder_state + self.Ua @ h_enc_t
            score = self.va @ self.activation.tanh(z)
            scores.append(score[0, 0])

        scores = self.xp.array(scores).reshape(T, 1)
        scores = scores - self.xp.max(scores, axis=0, keepdims=True)
        alpha = self.activation.softmax(scores)

        context = self.xp.zeros((encoder_outputs.shape[1], 1), dtype=self.xp.float32)
        for t in range(T):
            context += alpha[t, 0] * encoder_outputs[t]

        self.cache = (decoder_state, encoder_outputs, alpha)
        return context, alpha

    def backward(self, dcontext):
        # Reset gradients
        self._init_gradients()

        decoder_state, encoder_outputs, alpha = self.cache
        T = encoder_outputs.shape[0]

        dalpha = self.xp.zeros_like(alpha)
        dencoder_outputs = self.xp.zeros_like(encoder_outputs)

        # dcontext -> dalpha
        for t in range(T):
            h_enc_t = encoder_outputs[t]
            dalpha[t] = dcontext.T @ h_enc_t
            dencoder_outputs[t] += alpha[t] * dcontext

        # Softmax backward
        sum_alpha_dalpha = self.xp.sum(alpha * dalpha, axis=0, keepdims=True)
        dscores = alpha * (dalpha - sum_alpha_dalpha)

        ddecoder_state_total = self.xp.zeros_like(decoder_state)

        for t in range(T):
            h_enc_t = encoder_outputs[t]
            dscore_t = dscores[t, 0]

            z = self.Wa @ decoder_state + self.Ua @ h_enc_t
            tanh_val = self.activation.tanh(z)

            self.dva += dscore_t * tanh_val.T

            dtanh = (self.va.T * dscore_t) * self.activation.tanh_derivative(tanh_val)

            self.dWa += dtanh @ decoder_state.T
            self.dUa += dtanh @ h_enc_t.T

            ddecoder_state_total += self.Wa.T @ dtanh
            dencoder_outputs[t] += self.Ua.T @ dtanh

        return ddecoder_state_total, dencoder_outputs

    def apply_gradients(self, lr=0.1):
        self.Wa -= lr * self.dWa
        self.Ua -= lr * self.dUa
        self.va -= lr * self.dva
        self._init_gradients()
