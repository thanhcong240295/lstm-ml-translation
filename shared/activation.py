import numpy as np


class Activation:
    def __init__(self, xp=np):
        self.xp = xp

    def sigmoid(self, x):
        return 1 / (1 + self.xp.exp(-self.xp.clip(x, -50, 50)))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        return self.xp.tanh(x)

    def tanh_derivative(self, x):
        return 1 - self.xp.tanh(x) ** 2

    def relu(self, x):
        return self.xp.maximum(0, x)

    def relu_derivative(self, x):
        return self.xp.where(x > 0, 1, 0)

    def softmax(self, x):
        x_max = self.xp.max(x, axis=0, keepdims=True)
        exp_x = self.xp.exp(self.xp.clip(x - x_max, -50, 50))
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    def softmax_derivative(self, x):
        s = self.softmax(x)
        return s * (1 - s)
