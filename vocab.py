class Vocab:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.size = 2

    def build(self, data_dict: dict) -> None:
        for token_lists in data_dict.values():
            for tokens in token_lists:
                for word in tokens:
                    if word not in self.word2idx:
                        self.word2idx[word] = self.size
                        self.idx2word[self.size] = word
                        self.size += 1

    def encode(self, data_dict: dict, max_len: int) -> dict[str, list[list[int]]]:
        result = {}

        for filename, token_lists in data_dict.items():
            padded_lists = []
            for tokens in token_lists:
                padded_ids = self._encode_and_pad(tokens, max_len)
                padded_lists.append(padded_ids)
            result[filename] = padded_lists

        return result

    def decode(self, indices):
        return [self.idx2word.get(i, "<UNK>") for i in indices]

    def _encode_and_pad(self, tokens: list[str], max_len: int) -> list[int]:
        ids = self._encode(tokens)
        padded_ids = self._pad_sequence(ids, max_len)
        return padded_ids

    def _encode(self, tokens: list[str]) -> list[int]:
        return [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in tokens]

    def _pad_sequence(self, seq, max_len):
        if len(seq) < max_len:
            return seq + [self.word2idx["<PAD>"]] * (max_len - len(seq))
        else:
            return seq[:max_len]
