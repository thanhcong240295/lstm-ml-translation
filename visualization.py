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

    def plot_losses_comparison(
        self, train_losses, val_losses=None, title="Training and Validation Loss", filename="loss_comparison.png"
    ):
        """Plot training and validation loss comparison.

        Args:
            train_losses: List of training loss values
            val_losses: List of validation loss values (optional)
            title: Title of the plot
            filename: Output filename
        """
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, "b-", linewidth=2, marker="o", label="Training Loss")

        if val_losses is not None:
            plt.plot(epochs, val_losses, "r-", linewidth=2, marker="s", label="Validation Loss")

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=100)
        plt.close()
        print(f"Loss comparison diagram saved to {filepath}")

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

    def plot_metrics_dashboard(self, metrics_dict, filename="metrics_dashboard.png"):
        """Plot multiple metrics in a dashboard format.

        Args:
            metrics_dict: Dictionary with metric names as keys and lists of values as values
            filename: Output filename
        """
        num_metrics = len(metrics_dict)
        cols = 2
        rows = (num_metrics + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        axes = axes.flatten()

        for idx, (metric_name, values) in enumerate(metrics_dict.items()):
            epochs = range(1, len(values) + 1)
            axes[idx].plot(epochs, values, "b-", linewidth=2, marker="o")
            axes[idx].set_title(metric_name, fontsize=12)
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("Value")
            axes[idx].grid(True, alpha=0.3)

        for idx in range(num_metrics, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=100)
        plt.close()
        print(f"Metrics dashboard saved to {filepath}")
