import numpy as np
from lstm import LSTMCell


class BiLSTM:
    def __init__(self, input_size=256, hidden_size=256):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forward_lstm = LSTMCell(input_size, hidden_size)
        self.backward_lstm = LSTMCell(input_size, hidden_size)

    def forward(self, x_seq):
        # Normalize input to list of column vectors
        if isinstance(x_seq, np.ndarray):
            if x_seq.ndim == 2:
                X_seq = [x_seq[t].reshape(-1, 1) for t in range(x_seq.shape[0])]
            elif x_seq.ndim == 3:
                X_seq = [x_seq[t] for t in range(x_seq.shape[0])]
            else:
                raise ValueError("Invalid x_seq shape")
        else:
            X_seq = [np.asarray(x).reshape(-1, 1) for x in x_seq]

        T = len(X_seq)

        self.forward_caches = []
        self.backward_caches = [None] * T

        h_f, c_f = np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))
        h_b, c_b = np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))

        h_f_list = []
        for t in range(T):
            h_f, c_f, cache = self.forward_lstm.forward(X_seq[t], h_f, c_f)
            h_f_list.append(h_f)
            self.forward_caches.append(cache)

        h_b_list = [None] * T
        for t in reversed(range(T)):
            h_b, c_b, cache = self.backward_lstm.forward(X_seq[t], h_b, c_b)
            h_b_list[t] = h_b
            self.backward_caches[t] = cache

        # Keep (2H, 1) shape
        h_concat = [np.vstack((f, b)) for f, b in zip(h_f_list, h_b_list)]
        return np.stack(h_concat, axis=0)  # (T, 2H, 1)

    def backward(self, dh_total):
        # dh_total: (T, 2H) or (T, 2H, 1)
        if dh_total.ndim == 2:
            dh_total = dh_total[:, :, None]

        T = dh_total.shape[0]
        dx_total = np.zeros((T, self.input_size, 1))

        dh_f_all = dh_total[:, : self.hidden_size, :]
        dh_b_all = dh_total[:, self.hidden_size :, :]

        # Reset gradients
        self.forward_lstm._init_gradients()
        self.backward_lstm._init_gradients()

        # Forward LSTM backward
        dh_f_next = np.zeros((self.hidden_size, 1))
        dc_f_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            dh_t = dh_f_all[t] + dh_f_next
            dx_t, dh_f_next, dc_f_next = self.forward_lstm.backward(dh_t, dc_f_next, self.forward_caches[t])
            dx_total[t] += dx_t

        # Backward LSTM backward
        dh_b_prev = np.zeros((self.hidden_size, 1))
        dc_b_prev = np.zeros((self.hidden_size, 1))

        for t in range(T):
            dh_t = dh_b_all[t] + dh_b_prev
            dx_t, dh_b_prev, dc_b_prev = self.backward_lstm.backward(dh_t, dc_b_prev, self.backward_caches[t])
            dx_total[t] += dx_t

        return dx_total

    def apply_gradients(self, lr):
        self.forward_lstm.apply_gradients(lr)
        self.backward_lstm.apply_gradients(lr)
