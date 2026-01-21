import numpy as np

from layers.lstm_cell import LSTMCell


class BiLSTM:
    def __init__(self, input_size=256, hidden_size=256, xp=np):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.xp = xp

        self.forward_lstm = LSTMCell(input_size, hidden_size, xp)
        self.backward_lstm = LSTMCell(input_size, hidden_size, xp)

    def forward(self, x_seq):
        if isinstance(x_seq, (np.ndarray, self.xp.ndarray)):
            if x_seq.ndim == 2:
                X_seq = [x_seq[t].reshape(-1, 1) for t in range(x_seq.shape[0])]
            elif x_seq.ndim == 3:
                X_seq = [x_seq[t] for t in range(x_seq.shape[0])]
            else:
                raise ValueError("Invalid x_seq shape")
        else:
            X_seq = [self.xp.asarray(x).reshape(-1, 1) for x in x_seq]

        T = len(X_seq)

        self.forward_caches = []
        self.backward_caches = [None] * T

        h_f, c_f = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32), self.xp.zeros(
            (self.hidden_size, 1), dtype=self.xp.float32
        )
        h_b, c_b = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32), self.xp.zeros(
            (self.hidden_size, 1), dtype=self.xp.float32
        )

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

        h_concat = [self.xp.vstack((f, b)) for f, b in zip(h_f_list, h_b_list)]
        return self.xp.stack(h_concat, axis=0)  # (T, 2H, 1)

    def backward(self, dh_total):
        if dh_total.ndim == 2:
            dh_total = dh_total[:, :, None]

        T = dh_total.shape[0]
        dx_total = self.xp.zeros((T, self.input_size, 1), dtype=self.xp.float32)

        dh_f_all = dh_total[:, : self.hidden_size, :]
        dh_b_all = dh_total[:, self.hidden_size :, :]

        self.forward_lstm._init_gradients()
        self.backward_lstm._init_gradients()

        dh_f_next = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)
        dc_f_next = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)

        for t in reversed(range(T)):
            dh_t = dh_f_all[t] + dh_f_next
            dx_t, dh_f_next, dc_f_next = self.forward_lstm.backward(dh_t, dc_f_next, self.forward_caches[t])
            dx_total[t] += dx_t

        dh_b_prev = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)
        dc_b_prev = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)

        for t in range(T):
            dh_t = dh_b_all[t] + dh_b_prev
            dx_t, dh_b_prev, dc_b_prev = self.backward_lstm.backward(dh_t, dc_b_prev, self.backward_caches[t])
            dx_total[t] += dx_t

        return dx_total

    def apply_gradients(self, lr):
        self.forward_lstm.apply_gradients(lr)
        self.backward_lstm.apply_gradients(lr)
