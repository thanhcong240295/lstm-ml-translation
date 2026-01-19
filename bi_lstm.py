import numpy as np

from lstm import LSTMCell


class BiLSTM:
    def __init__(self, input_size=256, hidden_size=256):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forward_lstm = LSTMCell(input_size, hidden_size)
        self.backward_lstm = LSTMCell(input_size, hidden_size)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            T = x.shape[0]
            X_seq = [x[t] for t in range(T)]
        else:
            X_seq = x
            T = len(X_seq)

        h_forward = []
        h_prev_f = np.zeros((self.hidden_size, 1))
        c_prev_f = np.zeros((self.hidden_size, 1))

        for t in range(T):
            x_t = X_seq[t]
            if x_t.ndim == 1:
                x_t = x_t.reshape(-1, 1)

            h_prev_f, c_prev_f = self.forward_lstm.forward(x_t, h_prev_f, c_prev_f)
            h_forward.append(h_prev_f)

        h_backward = []
        h_prev_b = np.zeros((self.hidden_size, 1))
        c_prev_b = np.zeros((self.hidden_size, 1))

        for t in range(T - 1, -1, -1):
            x_t = X_seq[t]
            if x_t.ndim == 1:
                x_t = x_t.reshape(-1, 1)

            h_prev_b, c_prev_b = self.backward_lstm.forward(x_t, h_prev_b, c_prev_b)
            h_backward.append(h_prev_b)

        h_backward.reverse()

        h_concat = [np.vstack((hf, hb)) for hf, hb in zip(h_forward, h_backward)]

        y = np.concatenate([h.T for h in h_concat], axis=0)
        return y
