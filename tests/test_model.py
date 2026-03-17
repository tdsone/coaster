import pytest
import torch
import torch.nn.functional as F

from coaster.model.config import EncoderConfig, DecoderConfig
from coaster.model.layers import RMSNorm, SinusoidalPosEmb
from coaster.model.encoder import DNAEncoder
from coaster.model.decoder import RNADecoder
from coaster.model.transformer import CoasterModel
from coaster.tokenizer import RNATokenizer

# Small configs for fast CPU tests
ENC = EncoderConfig(d_model=32, n_heads=2, n_layers=1, ffn_dim=64, dna_len=64, conv_kernel=8, conv_stride=8)
DEC = DecoderConfig(d_model=32, n_heads=2, n_layers=1, ffn_dim=64, max_rna_len=30)
B = 2


@pytest.fixture
def model():
    return CoasterModel(ENC, DEC)


def test_rms_norm_shape():
    norm = RMSNorm(32)
    x = torch.randn(2, 10, 32)
    assert norm(x).shape == (2, 10, 32)


def test_rms_norm_output_finite():
    norm = RMSNorm(32)
    x = torch.randn(2, 10, 32)
    assert torch.isfinite(norm(x)).all()


def test_sinusoidal_pos_emb_shape():
    emb = SinusoidalPosEmb(100, 32)
    x = torch.randn(2, 10, 32)
    assert emb(x).shape == (2, 10, 32)


def test_encoder_output_shape():
    enc = DNAEncoder(ENC)
    dna = torch.randint(1, 6, (B, ENC.dna_len))
    out, mask = enc(dna)
    assert out.shape == (B, ENC.enc_seq_len, ENC.d_model)
    assert mask is None  # no padding mask supplied


def test_encoder_no_padding_mask_by_default():
    enc = DNAEncoder(ENC)
    dna = torch.randint(1, 6, (1, ENC.dna_len))
    _, mask = enc(dna)
    assert mask is None


def test_decoder_output_shape():
    dec = RNADecoder(DEC)
    memory = torch.randn(B, ENC.enc_seq_len, DEC.d_model)
    tgt = torch.randint(1, 8, (B, 10))
    logits = dec(tgt, memory)
    assert logits.shape == (B, 10, DEC.vocab_size)


def test_model_forward_shape(model):
    dna = torch.randint(1, 6, (B, ENC.dna_len))
    rna = torch.randint(1, 8, (B, 10))
    logits = model(dna, rna)
    assert logits.shape == (B, 10, DEC.vocab_size)


def test_model_loss_finite(model):
    dna = torch.randint(1, 6, (B, ENC.dna_len))
    rna_input = torch.randint(1, 8, (B, 10))
    rna_target = torch.randint(1, 8, (B, 10))
    logits = model(dna, rna_input)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), rna_target.reshape(-1))
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_model_backward(model):
    dna = torch.randint(1, 6, (B, ENC.dna_len))
    rna_input = torch.randint(1, 8, (B, 8))
    rna_target = torch.randint(1, 8, (B, 8))
    logits = model(dna, rna_input)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), rna_target.reshape(-1))
    loss.backward()
    # At least one parameter should have a gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_model_generate_returns_strings(model):
    rna_tok = RNATokenizer()
    dna = torch.randint(1, 6, (1, ENC.dna_len))
    results = model.generate(dna, rna_tok, max_len=10)
    assert len(results) == 1
    assert isinstance(results[0], str)


def test_model_generate_valid_chars(model):
    rna_tok = RNATokenizer()
    dna = torch.randint(1, 6, (2, ENC.dna_len))
    results = model.generate(dna, rna_tok, max_len=15)
    assert len(results) == 2
    for seq in results:
        assert all(c in "AUGCN" for c in seq), f"Invalid chars in: {seq!r}"


def test_model_generate_batch_size(model):
    rna_tok = RNATokenizer()
    dna = torch.randint(1, 6, (3, ENC.dna_len))
    results = model.generate(dna, rna_tok, max_len=10)
    assert len(results) == 3


def test_enc_seq_len_matches_config():
    # (dna_len - conv_kernel) // conv_stride + 1 = (64 - 8) // 8 + 1 = 8
    assert ENC.enc_seq_len == 8
