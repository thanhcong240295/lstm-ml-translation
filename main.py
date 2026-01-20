from lstm_to_lstm import LstmToLstmLanguageTranslation
from preprocessor import Preprocessor
from utils import split_train_val, validate_arguments
from visualization import Visualization
from vocab import Vocab
from word2vec import Word2VecEmbedding


def main():
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 256
    MAX_LEN = 100

    dataset_path, model_path, translate_text = validate_arguments()

    if translate_text:
        preprocessor = Preprocessor()
        tokens = preprocessor.preprocess_str(translate_text)

        vocab_en = Vocab()
        vocab_en.load_vocab(model_path + "_train.en.txt_vocab.json")

        vocab_vi = Vocab()
        vocab_vi.load_vocab(model_path + "_train.vi.txt_vocab.json")

        word2vec_embedding = Word2VecEmbedding(sg=1)
        word2vec_model = word2vec_embedding.load_model(model_path + "_word2vec.model")

        embedding_matrix = word2vec_embedding.build_embedding_matrix(vocab_en, word2vec_model, EMBEDDING_DIM)

        language_translation = LstmToLstmLanguageTranslation(
            vocab_src=vocab_en,
            vocab_tgt=vocab_vi,
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            embedding_matrix_src=embedding_matrix,
        )

        language_translation.load_best_model(model_path + ".npy")

        vietnamese = language_translation.translate(tokens, max_len=MAX_LEN)
        print(f"\nOutput: {vietnamese}")

    elif dataset_path and model_path:
        print("Starting training process...")
        preprocessor = Preprocessor()
        source_dict = preprocessor.preprocess(dataset_path)

        en_sentences = source_dict["train.en.txt"]
        vi_sentences = source_dict["train.vi.txt"]

        print("Building vocabulary and encoding data...")
        vocab_en = Vocab()
        vocab_en.build(en_sentences)
        encoded_en = vocab_en.encode(en_sentences, max_len=MAX_LEN)
        vocab_en.save_vocab(model_path + "_train.en.txt_vocab.json")

        vocab_vi = Vocab()
        vocab_vi.build(vi_sentences)
        encoded_vi = vocab_vi.encode(vi_sentences, max_len=MAX_LEN)
        vocab_vi.save_vocab(model_path + "_train.vi.txt_vocab.json")

        print("\nSplitting data 80/20...")
        X_train, Y_train, X_val, Y_val = split_train_val(encoded_en, encoded_vi, split_ratio=0.8)

        print(f"Training samples: {len(X_train)} (80%)")
        print(f"Validation samples: {len(X_val)} (20%)")

        visualization = Visualization()
        visualization.plot_data_split_cycle(len(X_train), len(X_val))

        print("Training Word2Vec embeddings...")
        word2vec_embedding = Word2VecEmbedding(sg=1)

        res = word2vec_embedding.train(vocab_en, {"train.en.txt": en_sentences}, embed_dim=EMBEDDING_DIM)

        embedding_matrix = res["W"]
        word2vec_model = res["model"]

        word2vec_embedding.save_model(
            word2vec_model,
            model_path + "_word2vec.model",
        )

        print("Training Language Translation model...")
        language_translation = LstmToLstmLanguageTranslation(
            vocab_src=vocab_en,
            vocab_tgt=vocab_vi,
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            embedding_matrix_src=embedding_matrix,
        )

        language_translation.train(
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            model_path=model_path + ".npy",
        )


if __name__ == "__main__":
    main()
