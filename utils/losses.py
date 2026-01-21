"""Loss functions for neural network training."""

import numpy as np


class CrossEntropyLoss:
    """Cross-entropy loss for classification tasks."""

    def __init__(self, xp=np):
        """
        Initialize the cross-entropy loss.

        Args:
            xp: numpy or cupy module for array operations
        """
        self.xp = xp

    def compute_loss(self, probs, target):
        """
        Compute cross-entropy loss.

        Args:
            probs: Probability distribution (softmax output)
            target: Target class index

        Returns:
            Scalar loss value
        """
        return -self.xp.log(probs[target, 0] + 1e-9)

    def compute_gradient(self, probs, target):
        """
        Compute gradient of cross-entropy loss with respect to logits.

        Args:
            probs: Probability distribution (softmax output)
            target: Target class index

        Returns:
            Gradient with respect to logits
        """
        dlogits = probs.copy()
        dlogits[target] -= 1
        return dlogits
