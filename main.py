from language_translation import LanguageTranslation
from preprocessor import Preprocessor
from utils import validate_arguments
from vocab import Vocab
from word2vec import Word2VecEmbedding


def main():
    EPOCHS = 50
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 256
    MAX_LEN = 100

    dataset_path, model_path, translate_text = validate_arguments()

    if translate_text:
        print("\nLoading model and embeddings...")
        preprocessor = Preprocessor()
        translate_text_tokenized = preprocessor.preprocess_str(translate_text)

        vocab_en = Vocab()
        vocab_path = model_path + '_train.en.txt' + "_vocab.json"
        vocab_en_data = vocab_en.load_vocab(vocab_path)

        vocab_vi = Vocab()
        vocab_path = model_path + '_train.vi.txt' + "_vocab.json"
        vocab_vi_data = vocab_vi.load_vocab(vocab_path)
        
        word2vecEmbedding = Word2VecEmbedding(sg=1)
        word2vec_path = model_path + "_word2vec.model"
        word2vec_model = word2vecEmbedding.load_model(word2vec_path)
        embedding_matrix = word2vecEmbedding.build_embedding_matrix(vocab_en_data, word2vec_model, EMBEDDING_DIM)

        language_translation = LanguageTranslation(
            input_size=EMBEDDING_DIM,
            hidden_size=EMBEDDING_DIM,
            use_attention=False,
        )
        lstm_path = model_path + "_lstm.model.npy"
        language_translation.load_model(lstm_path)

        vietnamese = language_translation.translate(translate_text_tokenized, vocab_en_data, vocab_vi_data, embedding_matrix, max_len=MAX_LEN)
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

        print("Training Word2Vec embeddings...")
        word2vecEmbedding = Word2VecEmbedding(sg=1)
        word2vec_models = word2vecEmbedding.train(vocab_en, {"train.en.txt": en_sentences}, embed_dim=EMBEDDING_DIM)
        embedding_matrix = word2vec_models["train.en.txt"]["W"]

        word2vecEmbedding.save_model(
            word2vec_models["train.en.txt"]["model"],
            model_path + "_word2vec.model",
        )

        print("Training Language Translation model...")
        language_translation = LanguageTranslation(
            input_size=EMBEDDING_DIM,
            hidden_size=EMBEDDING_DIM,
            use_attention=False,
        )
        language_translation.train(
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            vocab_source=vocab_en,
            vocab_target=vocab_vi,
            X_ids=encoded_en,
            Y_ids=encoded_vi,
            embedding_matrix=embedding_matrix
        )
        language_translation.save_model(model_path + "_lstm.model")


if __name__ == "__main__":
    main()
