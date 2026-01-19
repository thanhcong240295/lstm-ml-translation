import numpy as np

from activation import Activation
from lstm import LSTMCell


class BiLSTM:
    def __init__(self, input_size=256, hidden_size=256):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = Activation()

        self.forward_lstm = LSTMCell(input_size, hidden_size)
        self.backward_lstm = LSTMCell(input_size, hidden_size)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            T = x.shape[0]
            X_seq = [x[t] for t in range(T)]
        else:
            X_seq = x
            T = len(X_seq)

        self.forward_caches = []
        self.backward_caches = [None] * T

        h_f, c_f = np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))
        h_b, c_b = np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))

        h_f_list = []
        for t in range(T):
            x_t = np.asarray(X_seq[t], dtype=np.float32).reshape(-1, 1)
            h_f, c_f, cache = self.forward_lstm.forward(x_t, h_f, c_f)
            h_f_list.append(h_f)
            self.forward_caches.append(cache)

        h_b_list = [None] * T
        for t in reversed(range(T)):
            x_t = np.asarray(X_seq[t], dtype=np.float32).reshape(-1, 1)
            h_b, c_b, cache = self.backward_lstm.forward(x_t, h_b, c_b)
            h_b_list[t] = h_b
            self.backward_caches[t] = cache

        h_concat = [np.vstack((f, b)).flatten() for f, b in zip(h_f_list, h_b_list)]
        return np.array(h_concat)

    def backward(self, dh_total, learning_rate=0.1):
        T = dh_total.shape[0]
        dx_total = np.zeros((T, self.input_size, 1))

        dh_f_all = dh_total[:, : self.hidden_size]
        dh_b_all = dh_total[:, self.hidden_size :]

        dh_f_next = np.zeros((self.hidden_size, 1))
        dc_f_next = np.zeros((self.hidden_size, 1))
        for t in reversed(range(T)):
            dh_t = dh_f_all[t].reshape(-1, 1) + dh_f_next
            dx_t, dh_f_next, dc_f_next = self.forward_lstm.backward(dh_t, dc_f_next, self.forward_caches[t])
            dx_total[t] += dx_t

        dh_b_prev = np.zeros((self.hidden_size, 1))
        dc_b_prev = np.zeros((self.hidden_size, 1))
        for t in range(T):
            dh_t = dh_b_all[t].reshape(-1, 1) + dh_b_prev
            dx_t, dh_b_prev, dc_b_prev = self.backward_lstm.backward(dh_t, dc_b_prev, self.backward_caches[t])
            dx_total[t] += dx_t

        self.forward_lstm.apply_gradients(learning_rate)
        self.backward_lstm.apply_gradients(learning_rate)

        return dx_total
