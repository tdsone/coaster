"""DNA and RNA tokenizers for the Coaster model."""
from __future__ import annotations


class DNATokenizer:
    PAD = 0
    A = 1
    T = 2
    G = 3
    C = 4
    N = 5
    VOCAB_SIZE = 6

    _char2id: dict[str, int] = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5}
    _id2char: dict[int, str] = {1: "A", 2: "T", 3: "G", 4: "C", 5: "N"}

    def encode(self, seq: str) -> list[int]:
        return [self._char2id.get(c.upper(), self.N) for c in seq]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id2char.get(i, "N") for i in ids if i != self.PAD)


class RNATokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    A = 3
    U = 4
    G = 5
    C = 6
    N = 7
    VOCAB_SIZE = 8

    _char2id: dict[str, int] = {"A": 3, "U": 4, "G": 5, "C": 6, "N": 7}
    _id2char: dict[int, str] = {3: "A", 4: "U", 5: "G", 6: "C", 7: "N"}

    def encode(self, seq: str, add_special: bool = True) -> list[int]:
        ids = [self._char2id.get(c.upper(), self.N) for c in seq]
        if add_special:
            ids = [self.BOS] + ids + [self.EOS]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        chars = []
        for i in ids:
            if i == self.EOS:
                break
            if skip_special and i in (self.BOS, self.PAD):
                continue
            chars.append(self._id2char.get(i, "N"))
        return "".join(chars)
