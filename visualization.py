import os

import matplotlib.pyplot as plt
import numpy as np


class Visualization:
    """Visualization utilities for training metrics and results."""

    def __init__(self, output_dir="result"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_loss(self, losses, title="Training Loss", filename="loss.png"):
        """Plot training loss over epochs.

        Args:
            losses: List of loss values per epoch
            title: Title of the plot
            filename: Output filename
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, "b-", linewidth=2, marker="o", markersize=6)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=100)
        plt.close()
        print(f"Loss diagram saved to {filepath}")

    def plot_loss_statistics(self, losses, filename="loss_statistics.png"):
        """Plot loss statistics including min, max, mean, and std.

        Args:
            losses: List of loss values
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        epochs = range(1, len(losses) + 1)

        axes[0, 0].plot(epochs, losses, "b-", linewidth=2)
        axes[0, 0].set_title("Loss Over Epochs")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(losses, bins=20, color="green", alpha=0.7, edgecolor="black")
        axes[0, 1].set_title("Loss Distribution")
        axes[0, 1].set_xlabel("Loss Value")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3, axis="y")

        min_loss = np.min(losses)
        max_loss = np.max(losses)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        stats_text = f"""
        Loss Statistics:
        Min Loss: {min_loss:.6f}
        Max Loss: {max_loss:.6f}
        Mean Loss: {mean_loss:.6f}
        Std Dev: {std_loss:.6f}
        """
        axes[1, 0].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment="center", family="monospace")
        axes[1, 0].axis("off")

        axes[1, 1].plot(epochs, np.gradient(losses), "r-", linewidth=2)
        axes[1, 1].set_title("Loss Gradient (Change)")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss Change")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=100)
        plt.close()
        print(f"Loss statistics diagram saved to {filepath}")
