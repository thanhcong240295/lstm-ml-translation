import numpy as np

from activation import Activation


class LSTMCell:
    def __init__(self, input_size=256, hidden_size=256, vocab_size=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self._init_weights()
        self.activation = Activation()
        self.Wy = np.random.randn(self.vocab_size, self.hidden_size) * 0.01
        self.by = np.zeros((self.vocab_size, 1))

    def forward(self, x_t, h_prev, c_prev):
        x_t = np.asarray(x_t, dtype=np.float32)
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)

        h_prev = np.asarray(h_prev, dtype=np.float32)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)

        c_prev = np.asarray(c_prev, dtype=np.float32)
        if c_prev.ndim == 1:
            c_prev = c_prev.reshape(-1, 1)

        f_t = self._forget_gate(x_t, h_prev)
        i_t = self._input_gate(x_t, h_prev)
        c_tilde = self._candidate(x_t, h_prev)
        c_t = self._cell_state(f_t, c_prev, i_t, c_tilde)
        o_t = self._output_gate(x_t, h_prev)
        h_t = self._hidden_state(o_t, c_t)

        cache = (x_t, h_prev, c_prev, f_t, i_t, o_t, c_tilde, c_t)

        return h_t, c_t, cache

    def backward(self, dh_t, dc_t, cache):
        x_t, h_prev, c_prev, f_t, i_t, o_t, c_tilde, c_t = cache
        tanh_ct = self.activation.tanh(c_t)

        do_t = dh_t * tanh_ct
        do_t_raw = do_t * self.activation.sigmoid_derivative(o_t)

        dc_t_total = dh_t * o_t * self.activation.tanh_derivative(c_t) + dc_t

        df_t = dc_t_total * c_prev
        df_t_raw = df_t * self.activation.sigmoid_derivative(f_t)

        di_t = dc_t_total * c_tilde
        di_t_raw = di_t * self.activation.sigmoid_derivative(i_t)

        dc_tilde = dc_t_total * i_t
        dc_tilde_raw = dc_tilde * self.activation.tanh_derivative(c_tilde)

        self.dWf += df_t_raw @ x_t.T
        self.dUf += df_t_raw @ h_prev.T
        self.dbf += df_t_raw

        self.dWi += di_t_raw @ x_t.T
        self.dUi += di_t_raw @ h_prev.T
        self.dbi += di_t_raw

        self.dWo += do_t_raw @ x_t.T
        self.dUo += do_t_raw @ h_prev.T
        self.dbo += do_t_raw

        self.dWc += dc_tilde_raw @ x_t.T
        self.dUc += dc_tilde_raw @ h_prev.T
        self.dbc += dc_tilde_raw

        dh_prev = self.Uf.T @ df_t_raw + self.Ui.T @ di_t_raw + self.Uo.T @ do_t_raw + self.Uc.T @ dc_tilde_raw

        dc_prev = dc_t_total * f_t

        dx_t = self.Wf.T @ df_t_raw + self.Wi.T @ di_t_raw + self.Wo.T @ do_t_raw + self.Wc.T @ dc_tilde_raw

        return dx_t, dh_prev, dc_prev

    def apply_gradients(self, lr=0.1):
        for attr in ["f", "i", "o", "c"]:
            setattr(self, f"W{attr}", getattr(self, f"W{attr}") - lr * getattr(self, f"dW{attr}"))
            setattr(self, f"U{attr}", getattr(self, f"U{attr}") - lr * getattr(self, f"dU{attr}"))
            setattr(self, f"b{attr}", getattr(self, f"b{attr}") - lr * getattr(self, f"db{attr}"))

        self._init_gradients()

    def predict_word(self, h_t):
        logits = self.Wy @ h_t + self.by
        return logits

    def _init_weights(self):
        self.Wf = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.Uf = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        self.bf = np.zeros((self.hidden_size, 1))

        self.Wi = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.Ui = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        self.bi = np.zeros((self.hidden_size, 1))

        self.Wo = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.Uo = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        self.bo = np.zeros((self.hidden_size, 1))

        self.Wc = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.Uc = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        self.bc = np.zeros((self.hidden_size, 1))

    def _init_gradients(self):
        self.dWf, self.dUf, self.dbf = np.zeros_like(self.Wf), np.zeros_like(self.Uf), np.zeros_like(self.bf)
        self.dWi, self.dUi, self.dbi = np.zeros_like(self.Wi), np.zeros_like(self.Ui), np.zeros_like(self.bi)
        self.dWo, self.dUo, self.dbo = np.zeros_like(self.Wo), np.zeros_like(self.Uo), np.zeros_like(self.bo)
        self.dWc, self.dUc, self.dbc = np.zeros_like(self.Wc), np.zeros_like(self.Uc), np.zeros_like(self.bc)

    def _forget_gate(self, x_t, h_prev):
        Wf_xt = self.Wf @ x_t
        Uf_htprev = self.Uf @ h_prev

        return self.activation.sigmoid(Wf_xt + Uf_htprev + self.bf)

    def _input_gate(self, x_t, h_prev):
        Wi_xt = self.Wi @ x_t
        Ui_htprev = self.Ui @ h_prev

        return self.activation.sigmoid(Wi_xt + Ui_htprev + self.bi)

    def _output_gate(self, x_t, h_prev):
        Wo_xt = self.Wo @ x_t
        Uo_htprev = self.Uo @ h_prev

        return self.activation.sigmoid(Wo_xt + Uo_htprev + self.bo)

    def _cell_state(self, f_t, c_prev, i_t, c_tilde):
        return f_t * c_prev + i_t * c_tilde

    def _hidden_state(self, o_t, c_t):
        return o_t * self.activation.tanh(c_t)

    def _candidate(self, x_t, h_prev):
        Wc_xt = self.Wc @ x_t
        Uc_htprev = self.Uc @ h_prev

        return self.activation.tanh(Wc_xt + Uc_htprev + self.bc)
