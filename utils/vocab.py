import json
import os

from utils.constants import (
    EOS_IDX,
    EOS_TOKEN,
    PAD_IDX,
    PAD_TOKEN,
    SOS_IDX,
    SOS_TOKEN,
    SPECIAL_TOKENS,
    UNK_IDX,
    UNK_TOKEN,
)


class Vocab:
    def __init__(self):
        self.word2idx = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
        }
        self.idx2word = {
            PAD_IDX: PAD_TOKEN,
            UNK_IDX: UNK_TOKEN,
            SOS_IDX: SOS_TOKEN,
            EOS_IDX: EOS_TOKEN,
        }
        self.size = 4

    def build(self, token_lists: list[list[str]]) -> None:
        for tokens in token_lists:
            for word in tokens:
                if word not in self.word2idx:
                    self.word2idx[word] = self.size
                    self.idx2word[self.size] = word
                    self.size += 1

    def encode(self, token_lists: list[list[str]], max_len: int) -> list[list[int]]:
        padded_lists = []
        for tokens in token_lists:
            padded_ids = self._encode_and_pad(tokens, max_len)
            padded_lists.append(padded_ids)
        return padded_lists

    def decode(self, indices: list[int]) -> list[str]:
        words = []
        for i in indices:
            word = self.idx2word.get(i, UNK_TOKEN)
            if word in SPECIAL_TOKENS:
                continue
            words.append(word)
        return words

    def save_vocab(self, file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            vocab_data = {
                "word2idx": self.word2idx,
                "idx2word": self.idx2word,
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=4)
            print(f"Vocabulary saved to {file_path}")
        except Exception as e:
            print(f"Error saving vocabulary: {e}")

    def load_vocab(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
                self.word2idx = vocab_data["word2idx"]
                self.idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}
                self.size = len(self.word2idx)
                print(f"Vocabulary loaded from {file_path}")
                return self
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
            return None

    def _encode_and_pad(self, tokens: list[str], max_len: int) -> list[int]:
        ids = self._encode(tokens)

        padded_ids = self._pad_sequence(ids, max_len)
        return padded_ids

    def _encode(self, tokens: list[str]) -> list[int]:
        return [self.word2idx.get(w, self.word2idx[UNK_TOKEN]) for w in tokens]

    def _pad_sequence(self, seq: list[int], max_len: int) -> list[int]:
        if len(seq) < max_len:
            return seq + [self.word2idx[PAD_TOKEN]] * (max_len - len(seq))
        else:
            return seq[:max_len]
