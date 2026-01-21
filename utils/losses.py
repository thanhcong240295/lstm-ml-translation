import numpy as np


class CrossEntropyLoss:
    def __init__(self, xp=np):
        self.xp = xp

    def compute_loss(self, probs, target):
        return -self.xp.log(probs[target, 0] + 1e-9)

    def compute_gradient(self, probs, target):
        dlogits = probs.copy()
        dlogits[target] -= 1
        return dlogits
