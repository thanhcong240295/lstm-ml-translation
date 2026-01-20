import os
import sys
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)


class Preprocessor:
    def preprocess_str(self, text: str, lang="en") -> list[str]:
        tokens = self._preprocess_data(text, lang)
        return tokens

    def preprocess(self, dir_path: str, src_lang="en", tgt_lang="vi") -> dict:
        if os.path.isabs(dir_path):
            dataset_folder = dir_path
        else:
            dataset_folder = os.path.join(os.getcwd(), dir_path)

        if not os.path.exists(dataset_folder) or not os.path.isdir(dataset_folder):
            print(f"Dataset folder not found: {dataset_folder}")
            sys.exit(1)

        source_text_files = [f for f in os.listdir(dataset_folder) if f.endswith(".txt")]
        source_text = {}

        for file_name in source_text_files:
            file_path = os.path.join(dataset_folder, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.read().strip().split("\n")
                print("Processing file:", file_name, "with", len(lines), "lines")

            processed = []
            lang = src_lang if "en" in file_name else tgt_lang

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                tokens = self._preprocess_data(line, lang)
                if tokens:
                    processed.append(tokens)

            source_text[file_name] = processed

        return source_text

    def _preprocess_data(self, source_text: str, lang="en") -> list[str]:
        text = self._lowercase(source_text)
        text = self._normalize(text)
        text = self._remove_punctuation(text, keep="?!")
        tokens = self._tokenize(text, lang)
        return tokens

    def _lowercase(self, text: str) -> str:
        return text.lower()

    def _normalize(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _remove_punctuation(self, text: str, keep="") -> str:
        import string

        punctuation = string.punctuation
        for ch in keep:
            punctuation = punctuation.replace(ch, "")

        return text.translate(str.maketrans("", "", punctuation))

    def _tokenize(self, text: str, lang="en") -> list[str]:
        if not text:
            return []

        if lang == "vi":
            return text.split()

        try:
            return word_tokenize(text)
        except LookupError:
            nltk.download("punkt")
            return word_tokenize(text)
