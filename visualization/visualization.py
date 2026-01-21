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

    def plot_dataset_statistics(self, en_sentences, vi_sentences, train_count, val_count, prefix="dataset"):
        total_pairs = len(en_sentences)
        en_lengths = [len(sent.split()) if isinstance(sent, str) else len(sent) for sent in en_sentences]
        vi_lengths = [len(sent.split()) if isinstance(sent, str) else len(sent) for sent in vi_sentences]

        en_vocab = set()
        vi_vocab = set()
        for sent in en_sentences:
            if isinstance(sent, str):
                en_vocab.update(sent.split())
            else:
                en_vocab.update(sent)
        for sent in vi_sentences:
            if isinstance(sent, str):
                vi_vocab.update(sent.split())
            else:
                vi_vocab.update(sent)

        en_vocab_size = len(en_vocab)
        vi_vocab_size = len(vi_vocab)

        plt.figure(figsize=(10, 6))
        plt.bar(
            ["Train", "Validation"],
            [train_count, val_count],
            color=["#3498db", "#e74c3c"],
            edgecolor="black",
            linewidth=1.5,
        )
        plt.ylabel("Number of Sentence Pairs", fontsize=11, weight="bold")
        plt.title(f"Dataset Split (Total: {total_pairs} pairs)", fontsize=12, weight="bold")
        plt.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate([train_count, val_count]):
            plt.text(i, v + max(train_count, val_count) * 0.02, str(v), ha="center", fontsize=10, weight="bold")
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{prefix}_split.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Dataset split chart saved to {filepath}")

        plt.figure(figsize=(10, 6))
        plt.hist(en_lengths, bins=30, color="#2ecc71", alpha=0.7, edgecolor="black", label="English")
        plt.hist(vi_lengths, bins=30, color="#f39c12", alpha=0.7, edgecolor="black", label="Vietnamese")
        plt.xlabel("Sentence Length (words)", fontsize=11)
        plt.ylabel("Frequency", fontsize=11)
        plt.title("Sentence Length Distribution", fontsize=12, weight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{prefix}_length_distribution.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Length distribution chart saved to {filepath}")

        plt.figure(figsize=(10, 6))
        plt.bar(
            ["English", "Vietnamese"],
            [en_vocab_size, vi_vocab_size],
            color=["#9b59b6", "#e67e22"],
            edgecolor="black",
            linewidth=1.5,
        )
        plt.ylabel("Vocabulary Size", fontsize=11, weight="bold")
        plt.title("Vocabulary Statistics", fontsize=12, weight="bold")
        plt.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate([en_vocab_size, vi_vocab_size]):
            plt.text(i, v + max(en_vocab_size, vi_vocab_size) * 0.02, str(v), ha="center", fontsize=10, weight="bold")
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{prefix}_vocabulary.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Vocabulary statistics chart saved to {filepath}")

    def plot_training_history(self, train_losses, val_losses, output_path):
        try:
            train_losses_cpu = [float(loss.get()) if hasattr(loss, "get") else float(loss) for loss in train_losses]
            val_losses_cpu = [float(loss.get()) if hasattr(loss, "get") else float(loss) for loss in val_losses]

            plt.figure(figsize=(10, 6))
            epochs = range(1, len(train_losses_cpu) + 1)
            plt.plot(epochs, train_losses_cpu, "b-", linewidth=2, marker="o", markersize=4, label="Train Loss")
            plt.plot(epochs, val_losses_cpu, "r-", linewidth=2, marker="s", markersize=4, label="Val Loss")
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.title("Training and Validation Loss", fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(output_path, dpi=100)
            plt.close()
            print(f"\nLoss plot saved to {output_path}")
        except Exception as e:
            print(f"\nWarning: Could not create loss plot: {e}")

    def plot_bleu_scores(self, bleu_scores, output_path):
        try:
            if not bleu_scores or len(bleu_scores) == 0:
                return

            epochs, scores = zip(*bleu_scores)

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, scores, "g-", linewidth=2, marker="D", markersize=6, label="BLEU Score")
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("BLEU Score", fontsize=12)
            plt.title("BLEU Score Progress", fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
            plt.tight_layout()

            plt.savefig(output_path, dpi=100)
            plt.close()
            print(f"BLEU score plot saved to {output_path}")
        except Exception as e:
            print(f"\nWarning: Could not create BLEU plot: {e}")
