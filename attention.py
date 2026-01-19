import numpy as np
from activation import Activation


class BahdanauAttention:
    def __init__(self, encoder_output_dim, hidden_size=256):
        self.hidden_size = hidden_size
        self.encoder_output_dim = encoder_output_dim
        self.activation = Activation()

        limit = np.sqrt(6 / (hidden_size + encoder_output_dim))
        self.Wa = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.Ua = np.random.uniform(-limit, limit, (hidden_size, encoder_output_dim))
        self.va = np.random.uniform(-limit, limit, (1, hidden_size))
        
        self._init_gradients()

    def _init_gradients(self):
        self.dWa = np.zeros_like(self.Wa)
        self.dUa = np.zeros_like(self.Ua)
        self.dva = np.zeros_like(self.va)

    def forward(self, decoder_state, encoder_outputs):
        T = encoder_outputs.shape[0]

        scores = []

        for t in range(T):
            h_enc_t = encoder_outputs[t].reshape(-1, 1)

            score = self.va @ self.activation.tanh(
                self.Wa @ decoder_state + self.Ua @ h_enc_t
            )
            scores.append(score[0, 0])

        scores = np.array(scores).reshape(-1, 1)

        scores = scores - np.max(scores, axis=0, keepdims=True)
        alpha = self.activation.softmax(scores)

        context = np.zeros((encoder_outputs.shape[1], 1))
        for t in range(T):
            h_enc_t = encoder_outputs[t].reshape(-1, 1)
            context += alpha[t, 0] * h_enc_t

        self.cache = (decoder_state, encoder_outputs, alpha)
        return context, alpha

    def backward(self, dcontext, dalpha_from_decoder=0):
        decoder_state, encoder_outputs, alpha = self.cache
        T = encoder_outputs.shape[0]
        
        dalpha = np.zeros_like(alpha)
        dencoder_outputs = np.zeros_like(encoder_outputs)
        
        for t in range(T):
            h_enc_t = encoder_outputs[t].reshape(-1, 1)
            dalpha[t] = np.dot(dcontext.T, h_enc_t) + dalpha_from_decoder
            dencoder_outputs[t] = (alpha[t] * dcontext).flatten()

        sum_alpha_dalpha = np.sum(alpha * dalpha, axis=0, keepdims=True)
        dscores = alpha * (dalpha - sum_alpha_dalpha)

        ddecoder_state_total = np.zeros_like(decoder_state)
        
        for t in range(T):
            h_enc_t = encoder_outputs[t].reshape(-1, 1)
            dscore_t = dscores[t, 0]
            
            tanh_val = self.activation.tanh(self.Wa @ decoder_state + self.Ua @ h_enc_t)
            self.dva += dscore_t * tanh_val.T
            
            dtanh = (self.va.T * dscore_t) * self.activation.tanh_derivative(self.Wa @ decoder_state + self.Ua @ h_enc_t)
            
            self.dWa += dtanh @ decoder_state.T
            self.dUa += dtanh @ h_enc_t.T
            
            ddecoder_state_total += self.Wa.T @ dtanh
            
            dencoder_outputs[t] += (self.Ua.T @ dtanh).flatten()

        return ddecoder_state_total, dencoder_outputs

    def apply_gradients(self, lr=0.1):
        self.Wa -= lr * self.dWa
        self.Ua -= lr * self.dUa
        self.va -= lr * self.dva
        self._init_gradients()
