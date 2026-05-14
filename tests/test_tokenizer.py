"""Tests for the DNA tokenizer (incl. MASK + complement table)."""
from coaster.tokenizer import DNATokenizer


def test_encode_basic():
    tok = DNATokenizer()
    assert tok.encode("ACGTN") == [tok.A, tok.C, tok.G, tok.T, tok.N]


def test_encode_lowercase():
    tok = DNATokenizer()
    assert tok.encode("acgt") == [tok.A, tok.C, tok.G, tok.T]


def test_encode_unknown_is_N():
    tok = DNATokenizer()
    assert tok.encode("X") == [tok.N]


def test_encode_empty():
    tok = DNATokenizer()
    assert tok.encode("") == []


def test_decode_skips_pad():
    tok = DNATokenizer()
    assert tok.decode([tok.A, tok.PAD, tok.T]) == "AT"


def test_roundtrip():
    tok = DNATokenizer()
    seq = "ACGTACGTN"
    assert tok.decode(tok.encode(seq)) == seq


def test_vocab_size_and_pad_id():
    assert DNATokenizer.VOCAB_SIZE == 7
    assert DNATokenizer.PAD == 0


def test_mask_token_distinct():
    tok = DNATokenizer()
    assert tok.MASK not in {tok.PAD, tok.A, tok.C, tok.G, tok.T, tok.N}


def test_complement_table_is_involution():
    """Complementing twice should return the original token id for every token."""
    tok = DNATokenizer()
    for tid in range(tok.VOCAB_SIZE):
        assert tok.COMPLEMENT[tok.COMPLEMENT[tid]] == tid


def test_complement_pairs():
    tok = DNATokenizer()
    assert tok.COMPLEMENT[tok.A] == tok.T
    assert tok.COMPLEMENT[tok.T] == tok.A
    assert tok.COMPLEMENT[tok.C] == tok.G
    assert tok.COMPLEMENT[tok.G] == tok.C
    # Self-complementary / non-base tokens
    assert tok.COMPLEMENT[tok.N] == tok.N
    assert tok.COMPLEMENT[tok.PAD] == tok.PAD
    assert tok.COMPLEMENT[tok.MASK] == tok.MASK
