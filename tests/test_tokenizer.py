import pytest
from coaster.tokenizer import DNATokenizer, RNATokenizer


def test_dna_encode_basic():
    tok = DNATokenizer()
    assert tok.encode("ATGCN") == [1, 2, 3, 4, 5]


def test_dna_encode_lowercase():
    tok = DNATokenizer()
    assert tok.encode("atgc") == [1, 2, 3, 4]


def test_dna_encode_unknown():
    tok = DNATokenizer()
    assert tok.encode("X") == [DNATokenizer.N]


def test_dna_encode_empty():
    tok = DNATokenizer()
    assert tok.encode("") == []


def test_dna_decode_basic():
    tok = DNATokenizer()
    assert tok.decode([1, 2, 3, 4, 5]) == "ATGCN"


def test_dna_decode_skips_pad():
    tok = DNATokenizer()
    assert tok.decode([1, 0, 2]) == "AT"


def test_dna_roundtrip():
    tok = DNATokenizer()
    seq = "ATGCATGCN"
    assert tok.decode(tok.encode(seq)) == seq


def test_dna_vocab_size():
    assert DNATokenizer.VOCAB_SIZE == 6


def test_dna_pad_id():
    assert DNATokenizer.PAD == 0


def test_rna_encode_with_special():
    tok = RNATokenizer()
    ids = tok.encode("AUG", add_special=True)
    assert ids[0] == RNATokenizer.BOS
    assert ids[-1] == RNATokenizer.EOS
    assert len(ids) == 5  # BOS + 3 chars + EOS


def test_rna_encode_without_special():
    tok = RNATokenizer()
    ids = tok.encode("AUG", add_special=False)
    assert len(ids) == 3
    assert RNATokenizer.BOS not in ids
    assert RNATokenizer.EOS not in ids


def test_rna_decode_skips_special():
    tok = RNATokenizer()
    ids = [RNATokenizer.BOS, RNATokenizer.A, RNATokenizer.U, RNATokenizer.G, RNATokenizer.EOS]
    assert tok.decode(ids) == "AUG"


def test_rna_decode_stops_at_eos():
    tok = RNATokenizer()
    ids = [RNATokenizer.BOS, RNATokenizer.A, RNATokenizer.EOS, RNATokenizer.G]
    assert tok.decode(ids) == "A"


def test_rna_roundtrip():
    tok = RNATokenizer()
    seq = "AUGCUAGCN"
    assert tok.decode(tok.encode(seq)) == seq


def test_rna_vocab_size():
    assert RNATokenizer.VOCAB_SIZE == 8


def test_rna_pad_id():
    assert RNATokenizer.PAD == 0


def test_rna_bos_eos_distinct():
    assert RNATokenizer.BOS != RNATokenizer.EOS
    assert RNATokenizer.BOS != RNATokenizer.PAD
    assert RNATokenizer.EOS != RNATokenizer.PAD
