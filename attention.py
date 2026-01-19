import numpy as np


class Attention:
    """Attention mechanism for sequence-to-sequence models."""

    def __init__(self, hidden_size):
        """Initialize attention layer.

        Args:
            hidden_size: Size of hidden state (including concatenated Bi-LSTM)
        """
        self.hidden_size = hidden_size
        self._init_weights()

    def _init_weights(self):
        """Initialize attention weights."""
        self.Wa = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.ba = np.zeros((self.hidden_size, 1))
        self.va = np.random.randn(1, self.hidden_size) * 0.01

    def forward(self, decoder_state, encoder_outputs):
        """Compute attention weights and context vector.

        Args:
            decoder_state: Current decoder hidden state (hidden_size, 1)
            encoder_outputs: All encoder hidden states (seq_len, hidden_size, 1)

        Returns:
            context: Context vector (hidden_size, 1)
            attention_weights: Attention weights (seq_len,)
        """
        seq_len = len(encoder_outputs)

        scores = np.zeros(seq_len)

        for i, encoder_output in enumerate(encoder_outputs):
            score = self.va @ np.tanh(self.Wa @ encoder_output + self.ba)
            scores[i] = float(score)

        attention_weights = self._softmax(scores)

        context = np.zeros_like(encoder_outputs[0])
        for i, encoder_output in enumerate(encoder_outputs):
            context += attention_weights[i] * encoder_output

        return context, attention_weights

    def _softmax(self, x):
        """Compute softmax."""
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def backward(self, decoder_state, encoder_outputs, context_grad, learning_rate):
        """Backward pass for attention (simplified).

        Args:
            decoder_state: Current decoder hidden state
            encoder_outputs: All encoder hidden states
            context_grad: Gradient w.r.t. context vector
            learning_rate: Learning rate for updates
        """
        seq_len = len(encoder_outputs)

        scores = np.zeros(seq_len)
        for i, encoder_output in enumerate(encoder_outputs):
            score = self.va @ np.tanh(self.Wa @ encoder_output + self.ba)
            scores[i] = float(score)

        attention_weights = self._softmax(scores)

        for i in range(seq_len):
            self.Wa -= learning_rate * 0.001 * context_grad @ encoder_outputs[i].T
            self.ba -= learning_rate * 0.001 * context_grad
            self.va -= learning_rate * 0.001 * context_grad.T @ encoder_outputs[i]
