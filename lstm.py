import numpy as np


class LSTMCell:
    def __init__(self, input_size=256, hidden_size=256):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._init_weights()

    def forward(self, x_t, h_prev, c_prev):
        f_t = self._forget_gate(x_t, h_prev)
        i_t = self._input_gate(x_t, h_prev)
        c_tilde = self._candidate(x_t, h_prev)
        c_t = self._cell_state(f_t, c_prev, i_t, c_tilde)
        o_t = self._output_gate(x_t, h_prev)
        h_t = self._hidden_state(o_t, c_t)

        return h_t, c_t

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

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x):
        return np.tanh(x)

    def _forget_gate(self, x_t, h_prev):
        Wf_xt = self.Wf @ x_t
        Uf_htprev = self.Uf @ h_prev

        return self._sigmoid(Wf_xt + Uf_htprev + self.bf)

    def _input_gate(self, x_t, h_prev):
        Wi_xt = self.Wi @ x_t
        Ui_htprev = self.Ui @ h_prev

        return self._sigmoid(Wi_xt + Ui_htprev + self.bi)

    def _output_gate(self, x_t, h_prev):
        Wo_xt = self.Wo @ x_t
        Uo_htprev = self.Uo @ h_prev

        return self._sigmoid(Wo_xt + Uo_htprev + self.bo)

    def _cell_state(self, f_t, c_prev, i_t, c_tilde):
        return f_t * c_prev + i_t * c_tilde

    def _hidden_state(self, o_t, c_t):
        return o_t * self._tanh(c_t)

    def _candidate(self, x_t, h_prev):
        Wc_xt = self.Wc @ x_t
        Uc_htprev = self.Uc @ h_prev

        return self._tanh(Wc_xt + Uc_htprev + self.bc)
