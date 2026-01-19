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

    dataset_path, model_path, _ = validate_arguments()

    if dataset_path and model_path:
        print("Starting training process...")
        preprocessor = Preprocessor()
        source_dict = preprocessor.preprocess(dataset_path)

        print("Building vocabulary and encoding data...")
        vocab = Vocab()
        vocab.build(source_dict)
        vocab.encode(source_dict, max_len=MAX_LEN)

        print("Training Word2Vec embeddings...")
        word2vecEmbedding = Word2VecEmbedding(sg=1)
        word2vec_models = word2vecEmbedding.train(vocab, source_dict, embed_dim=EMBEDDING_DIM)

        print("Training Language Translation model...")
        language_translation = LanguageTranslation(
            input_size=EMBEDDING_DIM,
            hidden_size=EMBEDDING_DIM,
            use_attention=True,
        )
        language_translation.train(
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            vocab=vocab,
            dict_data=source_dict,
            word2vec_models=word2vec_models,
        )
        # language_translation.save_model(model_path)


if __name__ == "__main__":
    main()
