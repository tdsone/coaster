"""DNA tokenizer for the pointer-factorized read model.

Vocab: {PAD, A, C, G, T, N, MASK}. PAD=0 so it's the natural padding index;
MASK is substituted into DNA at MLM-masked positions.
"""
from __future__ import annotations


class DNATokenizer:
    PAD = 0
    A = 1
    C = 2
    G = 3
    T = 4
    N = 5
    MASK = 6
    VOCAB_SIZE = 7

    _char2id: dict[str, int] = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
    _id2char: dict[int, str] = {1: "A", 2: "C", 3: "G", 4: "T", 5: "N", 6: "<M>"}

    # Token-id → complement token-id table for reverse-complement augmentation.
    # PAD/N/MASK map to themselves; A↔T; C↔G.
    COMPLEMENT: tuple[int, ...] = (PAD, T, G, C, A, N, MASK)

    def encode(self, seq: str) -> list[int]:
        return [self._char2id.get(c.upper(), self.N) for c in seq]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id2char.get(i, "?") for i in ids if i != self.PAD)
