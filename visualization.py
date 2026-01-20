import os

import matplotlib.pyplot as plt
import numpy as np


class Visualization:
    def __init__(self, output_dir="result"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_loss(self, train_losses, val_losses=None, filename="loss.png"):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, "b-", linewidth=2, marker="o", markersize=6, label="Train Loss")
        if val_losses is not None and len(val_losses) > 0:
            plt.plot(epochs, val_losses, "r-", linewidth=2, marker="s", markersize=6, label="Val Loss")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss", fontsize=14)
        if val_losses is not None and len(val_losses) > 0:
            plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=100)
        plt.close()
        print(f"Loss diagram saved to {filepath}")

    def plot_loss_statistics(self, losses, filename="loss_statistics.png"):
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

    def plot_data_split_cycle(self, train_count, val_count, filename="data_split_cycle.png"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        total_count = train_count + val_count
        sizes = [train_count, val_count]
        labels = [
            f"Training Data\n{train_count} samples\n({train_count/total_count*100:.1f}%)",
            f"Validation Data\n{val_count} samples\n({val_count/total_count*100:.1f}%)",
        ]
        colors = ["#3498db", "#e74c3c"]
        explode = (0.05, 0.05)

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=explode,
            textprops={"fontsize": 11, "weight": "bold"},
        )

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(12)
            autotext.set_weight("bold")

        ax.set_title(f"Data Split Distribution (Total: {total_count} samples)", fontsize=14, weight="bold", pad=20)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Data split pie chart saved to {filepath}")

    def plot_roc_curve(self, all_probs, all_targets, vocab_size, filename="roc_curve.png"):
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)

        if len(all_targets) < 2:
            print("Not enough samples for ROC curve")
            return

        all_targets_bin = label_binarize(all_targets, classes=range(vocab_size))

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.nipy_spectral(np.linspace(0, 1, min(10, vocab_size)))
        fpr_list = []
        tpr_list = []
        roc_auc_list = []

        for i in range(min(10, vocab_size)):
            try:
                fpr, tpr, _ = roc_curve(
                    all_targets_bin[:, i] if i < all_targets_bin.shape[1] else np.zeros(len(all_targets)),
                    all_probs[:, i],
                )
                roc_auc = auc(fpr, tpr)
                fpr_list.append(fpr)
                tpr_list.append(tpr)
                roc_auc_list.append(roc_auc)

                ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"Token {i} (AUC = {roc_auc:.3f})")
            except Exception:
                continue

        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves for Token Prediction (One-vs-Rest)", fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"ROC curve diagram saved to {filepath}")
