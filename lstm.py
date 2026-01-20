import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np
from activation import Activation


class LSTMCell:
    def __init__(self, input_size, hidden_size, xp=np):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.xp = xp
        self.activation = Activation(xp)

        self._init_weights()
        self._init_gradients()

    def forward(self, x_t, h_prev, c_prev):
        f_t = self.activation.sigmoid(self.Wf @ x_t + self.Uf @ h_prev + self.bf)
        i_t = self.activation.sigmoid(self.Wi @ x_t + self.Ui @ h_prev + self.bi)
        o_t = self.activation.sigmoid(self.Wo @ x_t + self.Uo @ h_prev + self.bo)
        c_tilde = self.activation.tanh(self.Wc @ x_t + self.Uc @ h_prev + self.bc)

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * self.activation.tanh(c_t)

        cache = (x_t, h_prev, c_prev, f_t, i_t, o_t, c_tilde, c_t)
        return h_t, c_t, cache

    def backward(self, dh_t, dc_t, cache):
        x_t, h_prev, c_prev, f_t, i_t, o_t, c_tilde, c_t = cache

        tanh_ct = self.activation.tanh(c_t)
        do = dh_t * tanh_ct
        do_raw = do * self.activation.sigmoid_derivative(o_t)

        dc_total = dh_t * o_t * self.activation.tanh_derivative(c_t) + dc_t

        df = dc_total * c_prev
        df_raw = df * self.activation.sigmoid_derivative(f_t)

        di = dc_total * c_tilde
        di_raw = di * self.activation.sigmoid_derivative(i_t)

        dc_tilde = dc_total * i_t
        dc_tilde_raw = dc_tilde * self.activation.tanh_derivative(c_tilde)

        self.dWf += df_raw @ x_t.T
        self.dUf += df_raw @ h_prev.T
        self.dbf += df_raw

        self.dWi += di_raw @ x_t.T
        self.dUi += di_raw @ h_prev.T
        self.dbi += di_raw

        self.dWo += do_raw @ x_t.T
        self.dUo += do_raw @ h_prev.T
        self.dbo += do_raw

        self.dWc += dc_tilde_raw @ x_t.T
        self.dUc += dc_tilde_raw @ h_prev.T
        self.dbc += dc_tilde_raw

        dh_prev = self.Uf.T @ df_raw + self.Ui.T @ di_raw + self.Uo.T @ do_raw + self.Uc.T @ dc_tilde_raw
        dc_prev = dc_total * f_t
        dx_t = self.Wf.T @ df_raw + self.Wi.T @ di_raw + self.Wo.T @ do_raw + self.Wc.T @ dc_tilde_raw

        return dx_t, dh_prev, dc_prev

    def apply_gradients(self, lr):
        for name in ["f", "i", "o", "c"]:
            setattr(self, f"W{name}", getattr(self, f"W{name}") - lr * getattr(self, f"dW{name}"))
            setattr(self, f"U{name}", getattr(self, f"U{name}") - lr * getattr(self, f"dU{name}"))
            setattr(self, f"b{name}", getattr(self, f"b{name}") - lr * getattr(self, f"db{name}"))

        self._init_gradients()

    def get_weights(self):
        return {
            "Wf": self.Wf,
            "Uf": self.Uf,
            "bf": self.bf,
            "Wi": self.Wi,
            "Ui": self.Ui,
            "bi": self.bi,
            "Wo": self.Wo,
            "Uo": self.Uo,
            "bo": self.bo,
            "Wc": self.Wc,
            "Uc": self.Uc,
            "bc": self.bc,
        }

    def set_weights(self, weights):
        for k, v in weights.items():
            setattr(self, k, v)

    def _init_weights(self):
        limit = float(self.xp.sqrt(1 / self.input_size))

        self.Wf = self.xp.random.uniform(-limit, limit, (self.hidden_size, self.input_size)).astype(self.xp.float32)
        self.Uf = self.xp.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size)).astype(self.xp.float32)
        self.bf = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)

        self.Wi = self.xp.random.uniform(-limit, limit, (self.hidden_size, self.input_size)).astype(self.xp.float32)
        self.Ui = self.xp.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size)).astype(self.xp.float32)
        self.bi = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)

        self.Wo = self.xp.random.uniform(-limit, limit, (self.hidden_size, self.input_size)).astype(self.xp.float32)
        self.Uo = self.xp.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size)).astype(self.xp.float32)
        self.bo = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)

        self.Wc = self.xp.random.uniform(-limit, limit, (self.hidden_size, self.input_size)).astype(self.xp.float32)
        self.Uc = self.xp.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size)).astype(self.xp.float32)
        self.bc = self.xp.zeros((self.hidden_size, 1), dtype=self.xp.float32)

    def _init_gradients(self):
        self.dWf = self.xp.zeros_like(self.Wf)
        self.dUf = self.xp.zeros_like(self.Uf)
        self.dbf = self.xp.zeros_like(self.bf)

        self.dWi = self.xp.zeros_like(self.Wi)
        self.dUi = self.xp.zeros_like(self.Ui)
        self.dbi = self.xp.zeros_like(self.bi)

        self.dWo = self.xp.zeros_like(self.Wo)
        self.dUo = self.xp.zeros_like(self.Uo)
        self.dbo = self.xp.zeros_like(self.bo)

        self.dWc = self.xp.zeros_like(self.Wc)
        self.dUc = self.xp.zeros_like(self.Uc)
        self.dbc = self.xp.zeros_like(self.bc)
