import os

import numpy as np

from embeddings.word2vec import Word2VecEmbedding
from models.lstm2lstm import Lstm2LstmLanguageTranslation
from models.seq2seq import Seq2SeqLanguageTranslation
from utils.constants import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, UNK_TOKEN
from utils.device import get_array_module, to_device
from utils.preprocessor import Preprocessor
from utils.utils import split_train_val, validate_arguments
from utils.vocab import Vocab
from visualization.visualization import Visualization


def main():
    EPOCHS = 20
    LEARNING_RATE = 0.01
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 256
    MAX_LEN = 100
    BATCH_SIZE = 32

    dataset_path, model_path, translate_text, architecture, device = validate_arguments()

    print(f"Using device: {device.upper()}")

    if translate_text:
        _run_translation(translate_text, model_path, architecture, EMBEDDING_DIM, HIDDEN_SIZE, MAX_LEN, device)
    elif dataset_path and model_path:
        _run_training(
            dataset_path,
            model_path,
            architecture,
            EPOCHS,
            LEARNING_RATE,
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            MAX_LEN,
            device,
            BATCH_SIZE,
        )


def _run_translation(translate_text, model_path, architecture, embedding_dim, hidden_size, max_len, device):
    preprocessor = Preprocessor()
    tokens = preprocessor.preprocess_str(translate_text)

    vocab_en = Vocab()
    vocab_en.load_vocab(model_path + "_train.en.txt_vocab.json")

    vocab_vi = Vocab()
    vocab_vi.load_vocab(model_path + "_train.vi.txt_vocab.json")

    word2vec_embedding = Word2VecEmbedding(sg=1)
    word2vec_model = word2vec_embedding.load_model(model_path + "_word2vec.model")
    embedding_matrix = word2vec_embedding.build_embedding_matrix(vocab_en, word2vec_model, embedding_dim)

    print(f"Loading {architecture.upper()} model...")
    language_translation = _create_model(
        architecture, vocab_en, vocab_vi, embedding_dim, hidden_size, embedding_matrix, device=device
    )
    language_translation.load_best_model(model_path + ".npy")

    vietnamese = language_translation.translate(tokens, max_len=max_len)
    print(f"\nOutput: {vietnamese}")


def _run_training(
    dataset_path,
    model_path,
    architecture,
    epochs,
    learning_rate,
    embedding_dim,
    hidden_size,
    max_len,
    device="cpu",
    batch_size=32,
):
    print("Starting training process...")
    preprocessor = Preprocessor()
    source_dict = preprocessor.preprocess(dataset_path)

    en_sentences = source_dict["train.en.txt"]
    vi_sentences = source_dict["train.vi.txt"]

    vocab_en, encoded_en = _build_and_encode_vocab(en_sentences, model_path, "train.en.txt", max_len)
    vocab_vi, encoded_vi = _build_and_encode_vocab(vi_sentences, model_path, "train.vi.txt", max_len)

    X_train, Y_train, X_val, Y_val = _prepare_training_data(encoded_en, encoded_vi)

    bleu_samples = _prepare_bleu_samples(en_sentences, vi_sentences, X_val, Y_val, vocab_en, vocab_vi, n_samples=100)

    viz = Visualization()
    viz.plot_dataset_statistics(
        en_sentences, vi_sentences, len(X_train), len(X_val), prefix=os.path.basename(model_path)
    )

    embedding_matrix, word2vec_model = _train_word2vec(vocab_en, en_sentences, embedding_dim, model_path)

    print(f"Training Language Translation model with {architecture.upper()}...")
    language_translation = _create_model(
        architecture, vocab_en, vocab_vi, embedding_dim, hidden_size, embedding_matrix, device=device
    )

    if device == "gpu":
        try:
            xp = get_array_module(device)
            X_train = [to_device(x, xp) for x in X_train]
            Y_train = [to_device(y, xp) for y in Y_train]
            X_val = [to_device(x, xp) for x in X_val]
            Y_val = [to_device(y, xp) for y in Y_val]
            print(f"Data transferred to GPU successfully")
        except ImportError as e:
            print(f"[WARNING] {e}")
            print("Falling back to CPU mode")
            device = "cpu"
        except Exception as e:
            print(f"[WARNING] Failed to convert data to GPU: {e}")
            print("Falling back to CPU mode")
            device = "cpu"

    print("Starting training...")
    print(f"Batch size: {batch_size} (gradient accumulation)")
    language_translation.train(
        epochs,
        learning_rate,
        X_train,
        Y_train,
        X_val,
        Y_val,
        model_path + ".npy",
        batch_size=batch_size,
        bleu_samples=bleu_samples,
    )


def _create_model(architecture, vocab_src, vocab_tgt, input_size, hidden_size, embedding_matrix_src, device="cpu"):
    if architecture == "lstm-lstm":
        return Lstm2LstmLanguageTranslation(
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            input_size=input_size,
            hidden_size=hidden_size,
            embedding_matrix_src=embedding_matrix_src,
            device=device,
        )
    else:
        return Seq2SeqLanguageTranslation(
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            input_size=input_size,
            hidden_size=hidden_size,
            embedding_matrix_src=embedding_matrix_src,
            device=device,
        )


def _build_and_encode_vocab(sentences, model_path, dataset_name, max_len):
    print(f"Building vocabulary for {dataset_name}...")
    vocab = Vocab()
    vocab.build(sentences)
    encoded = vocab.encode(sentences, max_len=max_len)
    vocab.save_vocab(model_path + f"_{dataset_name}_vocab.json")
    return vocab, encoded


def _prepare_training_data(encoded_en, encoded_vi):
    print("\nSplitting data 80/20...")
    X_train, Y_train, X_val, Y_val = split_train_val(encoded_en, encoded_vi, split_ratio=0.8)

    print(f"Training samples: {len(X_train)} (80%)")
    print(f"Validation samples: {len(X_val)} (20%)")

    return X_train, Y_train, X_val, Y_val


def _prepare_bleu_samples(en_sentences, vi_sentences, X_val, Y_val, vocab_en, vocab_vi, n_samples=100):
    bleu_samples = []
    n_samples = min(n_samples, len(X_val))

    indices = np.random.choice(len(X_val), size=n_samples, replace=False)

    for idx in indices:
        x_ids = X_val[idx]
        y_ids = Y_val[idx]

        if hasattr(x_ids, "get"):
            x_ids = x_ids.get()
        if hasattr(y_ids, "get"):
            y_ids = y_ids.get()

        src_tokens = [
            vocab_en.idx2word.get(int(idx), UNK_TOKEN)
            for idx in x_ids
            if int(idx)
            not in [vocab_en.word2idx[PAD_TOKEN], vocab_en.word2idx[SOS_TOKEN], vocab_en.word2idx[EOS_TOKEN]]
        ]

        bleu_samples.append((src_tokens, y_ids))

    print(f"Prepared {len(bleu_samples)} samples for BLEU evaluation")
    return bleu_samples


def _train_word2vec(vocab_en, en_sentences, embedding_dim, model_path):
    print("Training Word2Vec embeddings...")
    word2vec_embedding = Word2VecEmbedding(sg=1)
    res = word2vec_embedding.train(vocab_en, {"train.en.txt": en_sentences}, embed_dim=embedding_dim)

    embedding_matrix = res["W"]
    word2vec_model = res["model"]

    word2vec_embedding.save_model(word2vec_model, model_path + "_word2vec.model")
    return embedding_matrix, word2vec_model


if __name__ == "__main__":
    main()
