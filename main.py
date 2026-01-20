from lstm_to_lstm import LstmToLstmLanguageTranslation
from seq2seq import Seq2SeqLanguageTranslation
from preprocessor import Preprocessor
from utils import split_train_val, validate_arguments
from visualization import Visualization
from vocab import Vocab
from word2vec import Word2VecEmbedding


def main():
    EPOCHS = 20
    LEARNING_RATE = 0.01
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 256
    MAX_LEN = 100

    dataset_path, model_path, translate_text, architecture, device = validate_arguments()

    print(f"Using device: {device.upper()}")

    if translate_text:
        _run_translation(translate_text, model_path, architecture, EMBEDDING_DIM, HIDDEN_SIZE, MAX_LEN, device)
    elif dataset_path and model_path:
        _run_training(dataset_path, model_path, architecture, EPOCHS, LEARNING_RATE, EMBEDDING_DIM, HIDDEN_SIZE, MAX_LEN, device)


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
    language_translation = _create_model(architecture, vocab_en, vocab_vi, embedding_dim, hidden_size, embedding_matrix, device=device)
    language_translation.load_best_model(model_path + ".npy")

    vietnamese = language_translation.translate(tokens, max_len=max_len)
    print(f"\nOutput: {vietnamese}")


def _run_training(dataset_path, model_path, architecture, epochs, learning_rate, embedding_dim, hidden_size, max_len, device):
    print("Starting training process...")
    preprocessor = Preprocessor()
    source_dict = preprocessor.preprocess(dataset_path)

    en_sentences = source_dict["train.en.txt"]
    vi_sentences = source_dict["train.vi.txt"]

    vocab_en, encoded_en = _build_and_encode_vocab(en_sentences, model_path, "train.en.txt", max_len)
    vocab_vi, encoded_vi = _build_and_encode_vocab(vi_sentences, model_path, "train.vi.txt", max_len)

    X_train, Y_train, X_val, Y_val = _prepare_training_data(encoded_en, encoded_vi)
    embedding_matrix, word2vec_model = _train_word2vec(vocab_en, en_sentences, embedding_dim, model_path)

    print(f"Training Language Translation model with {architecture.upper()}...")
    language_translation = _create_model(architecture, vocab_en, vocab_vi, embedding_dim, hidden_size, embedding_matrix, device=device)

    if device == 'gpu':
        try:
            import cupy as cp
            X_train = [cp.asarray(x, dtype=cp.int32) for x in X_train]
            Y_train = [cp.asarray(y, dtype=cp.int32) for y in Y_train]
            X_val = [cp.asarray(x, dtype=cp.int32) for x in X_val]
            Y_val = [cp.asarray(y, dtype=cp.int32) for y in Y_val]
        except Exception as e:
            print(f"[WARNING] Failed to convert data to GPU: {e}")
            print("Falling back to CPU mode")
            device = 'cpu'

    print("Starting training...")
    language_translation.train(epochs, learning_rate, X_train, Y_train, X_val, Y_val, model_path + ".npy")


def _create_model(architecture, vocab_src, vocab_tgt, input_size, hidden_size, embedding_matrix_src, device='cpu'):
    if architecture == "lstm-lstm":
        return LstmToLstmLanguageTranslation(
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

    visualization = Visualization()
    visualization.plot_data_split_cycle(len(X_train), len(X_val))

    return X_train, Y_train, X_val, Y_val


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
