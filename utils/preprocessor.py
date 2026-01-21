import re
import string
import nltk
from nltk.tokenize import word_tokenize

from utils.constants import EOS_TOKEN, SOS_TOKEN

nltk.download("punkt", quiet=True)


class Preprocessor:
    def preprocess_str(self, text: str, lang="en") -> list[str]:
        tokens = self._preprocess_data(text, lang)
        return tokens

    def preprocess(self, file_path, is_src = True) -> dict:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.read().strip().split("\n")
            print("Processing file:", file_path, "with", len(lines), "lines")

        processed = []
        lang = 'en' if is_src else 'vi'

        for line in lines:
            line = line.strip()
            if not line:
                continue

            tokens = self._preprocess_data(line, lang)
            if tokens:
                if not is_src:
                    tokens.insert(0, SOS_TOKEN)
                    tokens.append(EOS_TOKEN)
                processed.append(tokens)

        return processed

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
