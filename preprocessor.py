import os
import sys

from nltk.tokenize import word_tokenize


class Preprocessor:
    def preprocess(self, dir_path: str) -> dict:
        dataset_folder = os.path.join(os.path.dirname(__file__), dir_path)
        if not os.path.exists(dataset_folder) or not os.path.isdir(dataset_folder):
            sys.exit(1)

        source_text_files = [f for f in os.listdir(dataset_folder) if f.endswith(".txt")]
        source_text = {}
        for file_name in source_text_files:
            file_path = os.path.join(dataset_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.read().strip().split("\n")
                print("Processing file:", file_name, "with", len(lines), "lines")

            processed = []
            for line in lines:
                tokens = self._preprocess_data(line)
                processed.append(tokens)

            source_text[file_name] = processed

        return source_text

    def _preprocess_data(self, source_text: str) -> list:
        lowercase_text = self._lowercase(source_text)
        cleaned_text = self._remove_punctuation(lowercase_text)
        tokenized_text = self._tokenize(cleaned_text)

        return tokenized_text

    def _lowercase(self, text: str) -> str:
        return text.lower()

    def _remove_punctuation(self, text: str) -> str:
        import string

        return text.translate(str.maketrans("", "", string.punctuation))

    def _tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)
