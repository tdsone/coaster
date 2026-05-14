# Pointer-Factorized Generative S2E: Modeling Reads as $(\sigma, s, e)$

**Status:** Refined project spec; supersedes `generative_s2e_handoff.md`.
**Audience:** Implementer picking up the project. The existing codebase is a throwaway prototype — feel free to start fresh.

---

## 1. Project Overview

### Motivation

Current sequence-to-expression (s2e) models (e.g. Yorzoi) are **discriminative**: they predict summary statistics — per-nucleotide read counts across strands and cell types — directly from DNA. A failure mode is that they can produce coverage profiles that are physically implausible given the read-length distribution of the underlying sequencing assay. No realistic set of fixed-length reads could ever aggregate into the predicted shape.

### Hypothesis

If we model the **reads themselves** — generate $(\sigma, s, e)$ triples autoregressively conditioned on DNA, then aggregate to coverage — the resulting profiles will be physically realizable by construction, and the model should improve on downstream coverage prediction.

### Preliminary Evidence

A small encoder–decoder transformer that generated *read sequences* (not positions) for 500k steps on a single yeast Illumina RNA-seq experiment reached mean Pearson R = 0.46 on test windows (Yorzoi baseline: R = 0.61). Heavily undertrained and underparameterized, but the gap with a tiny model is the encouraging signal motivating this project.

---

## 2. Problem Formalization

For a DNA window $D$ of length $L$, the data are a set of aligned reads $\{(s_n, e_n, \sigma_n)\}_{n=1}^N$, where:

- $s_n, e_n \in \{0, \ldots, L-1\}$ are reference-coordinate positions with $s_n \leq e_n$. We keep this convention universally — $s$ is `pos_min`, $e$ is `pos_max`, **not** the 5′/3′ ends of the read.
- $\sigma_n \in \{+, -\}$ is the strand from which the read originated. $\sigma_n$ tells you which of $s_n, e_n$ is the 5′ end.

Each read is modeled as a sample from a conditional joint distribution, factored via the chain rule:

$$
p(s, e, \sigma \mid D) = p(\sigma \mid D) \cdot p(s \mid \sigma, D) \cdot p(e \mid s, \sigma, D).
$$

The biological reading: a transcription event happens on some strand ($\sigma$), starts somewhere ($s$), ends somewhere ($e$). All three factors are categorical distributions, learned jointly with shared parameters. Keeping $s \leq e$ as the universal convention (regardless of strand) means the end-head mask `end ≥ start` stays consistent across both strands. DNA is always presented in reference (+) orientation so the encoder sees both strands' regulatory context simultaneously — this is what preserves convergent-transcription and other strand-interaction signal.

A coverage profile is a derived statistic of $N$ iid reads:

$$
C(j) = \sum_{n=1}^N \mathbb{1}[s_n \leq j \leq e_n], \qquad j \in \{0, \ldots, L-1\}.
$$

Stranded coverage profiles are obtained by restricting the sum to reads with a specific $\sigma_n$.

---

## 3. Architecture

A single shared encoder over the DNA window, with several heads attached. **Resolution is native (one token per base): no convolutional downsampling.** $L = 4992$ is the working window length.

### Encoder backbone

Standard pre-norm transformer encoder over the DNA window. Per-position embeddings $H \in \mathbb{R}^{B \times L \times d}$.

- **Tokenization:** one token per base, vocab `{A, C, G, T, N, MASK}` (size 6). No k-mers, no conv stem.
- **Attention:** full self-attention over all $L = 4992$ positions, every layer. Use FlashAttention for memory and throughput. The $O(L^2)$ cost is the dominant compute concern (see §7); for now we eat it. Future variants may interleave state-space / Mamba blocks with attention layers — the per-position interface (`(B, L, d)` in/out) is unchanged, so the heads don't care.
- **Positional encoding: RoPE.** Applied inside each attention layer's Q/K projections (not added to embeddings). Standard rotary, half-dim or full-dim — pick one and commit. Long-context tricks (NTK scaling, YaRN) are not needed at $L = 4992$ but the door is open.
- **Norm: RMSNorm**, pre-norm placement (before attn and FFN, plus a final RMSNorm at the top of the stack). No bias in norm.
- **FFN: SwiGLU**, $d_{\text{ffn}} \approx \tfrac{8}{3} d$ (rounded to a multiple of 64). No biases on the gate/up/down linears.
- **No dropout in the backbone for v1.** Add if/when overfitting shows up.

### Conditioning tokens

Following Karollus et al. (2024) — who showed that a single species token substantially improves representations at near-zero cost — prepend one learned token per axis of variation: **assay, cell type, strand-of-protocol** (and species, if multi-organism). Embeddings are learned. Essential for multi-experiment training; cheap to add. Tokens participate in self-attention and are stripped from $H$ before the per-position heads run.

### Heads

- **[CLS]-style strand head.** Prepend a learned `[CLS]` token to the input. Its post-encoder representation $h_\text{CLS}$ feeds a linear $\to 2$ logits, defining $p(\sigma \mid D)$. (Mean-pooling would also work but CLS is cleaner since we already need a pooled vector.)
- **Start head.** Per-position logits, conditioned on $\sigma$ via a learned strand embedding mixed into the per-position features. Output: $(B, L)$ logits → $p(s \mid \sigma, D)$.
- **End head.** Per-position logits, conditioned on the start position $s$ (gathered from $H$) and $\sigma$ (via embedding). Output: $(B, L)$ logits, with $j < s$ masked to $-\infty$ to enforce $e \geq s$. Also mask $j > s + \ell_{\max}$ if the assay has a known max fragment length. Defines $p(e \mid s, \sigma, D)$. **Compute note:** a 3·d-concat MLP at every position at $L=4992$ is heavy. Prefer a **bilinear form** $h_j^\top W_\sigma h_s + u_\sigma^\top h_j$ where $W_\sigma, u_\sigma$ are strand-indexed parameters — much cheaper, similar expressivity.
- **MLM head.** Per-position linear $\mathbb{R}^d \to \mathbb{R}^{|V|}$. Predicts the original nucleotide at masked positions. Trained alongside the read heads (§4) and gives the encoder Karollus-style denoising representations as a free pretraining signal.

All heads share the encoder, so gradients from any active loss flow back through the backbone.

---

## 4. Training

### Teacher forcing on the read losses

The chain-rule factorization lets us train without sampling in the forward pass. For each ground-truth read $(s^*, e^*, \sigma^*)$ in the batch:

$$
\mathcal{L}_\text{read}(D, s^*, e^*, \sigma^*) = -\log p_\theta(\sigma^* \mid D) - \log p_\theta(s^* \mid \sigma^*, D) - \log p_\theta(e^* \mid s^*, \sigma^*, D).
$$

All three terms are categorical cross-entropies. The start and end heads are conditioned on the **ground-truth** $\sigma^*$ (and $s^*$) during training, not on samples. This avoids the non-differentiability of discrete sampling (`argmax`, inverse-CDF over a step function) while still training the model to produce a coherent autoregressive joint.

### Joint MLM + read training via in-batch task mixing

The MLM objective (Karollus et al., 2024) is a strong, cheap pretraining signal for the encoder. We integrate it via **in-batch task mixing** rather than running it as a separate pretraining phase. Each batch contains samples drawn from three populations:

1. **MLM-only.** Masked DNA in. MLM loss applied at masked positions. Read heads unused.
2. **Read-only.** Clean (unmasked) DNA in. All three read losses applied. MLM head unused.
3. **Joint.** Masked DNA in. All four heads contribute losses on the same sample.

A reasonable starting mix is **40 / 40 / 20**. Populations 1 and 2 give clean per-task signal; population 3 provides the cross-task coupling that pushes the encoder toward representations jointly useful for both objectives. The mixing ratio is a hyperparameter; a simpler 50/50 with just populations 1 and 2 is a fine first cut.

Total batch loss:

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_\text{MLM}^{\text{(populations 1, 3)}} + \beta \cdot \mathcal{L}_\text{read}^{\text{(populations 2, 3)}}.
$$

Per-sample loss masking (zero out the head contributions that don't apply to a given sample before reducing) is straightforward in PyTorch — see the code sketch. Tune $\alpha, \beta$ to balance loss magnitudes; if the tasks fight, try GradNorm or PCGrad.

### Reverse-complement augmentation

Orthogonal to strand modeling. At training time, randomly substitute $(D, \text{reads}) \to (\text{RC}(D), \text{RC}(\text{reads}))$ — flip the DNA, complement the nucleotides, remap $(s, e, \sigma) \to (L - 1 - e,\; L - 1 - s,\; 1 - \sigma)$. This regularizes the model and approximates the RC symmetry without baking it into the architecture. RC-equivariant architectures are the principled-but-heavier alternative (see §7); for v1 use augmentation.

### If we ever need to sample inside the forward pass

(Not the v1 plan.) Discrete sampling blocks gradients. Standard escape hatches if needed later: Gumbel-softmax / Concrete relaxation, straight-through estimator, or REINFORCE.

---

## 5. Inference

Ancestral sampling per read, in factorization order:

1. Encode DNA once: $H = \text{encoder}(D)$. Cache it.
2. Sample strand: $\hat{\sigma} \sim \text{Categorical}(p_\theta(\cdot \mid D))$.
3. Sample start: $\hat{s} \sim \text{Categorical}(p_\theta(\cdot \mid \hat{\sigma}, D))$.
4. Sample end: $\hat{e} \sim \text{Categorical}(p_\theta(\cdot \mid \hat{s}, \hat{\sigma}, D))$.

For a coverage profile, repeat steps 2–4 $N$ times against the same cached $H$ and aggregate via the indicator sum. For stranded profiles, condition the sum on $\hat{\sigma}_n$.

The encoder is the expensive part and runs once per window. The heads are cheap and run $N$ times. Batch the $N$ samples for the per-window draws — don't loop in Python.

---

## 6. Code Sketch

PyTorch skeleton with RoPE, RMSNorm, SwiGLU. Heads use the bilinear end-head form noted in §3. Illustrative — production version needs proper init, FlashAttention, and BAM-based data loading.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- building blocks ---------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x, cos, sin):
    """x: (B, H, L, D); rotate even/odd pairs by position-dependent angles."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., ::2] = rx1
    out[..., 1::2] = rx2
    return out


class RoPECache(nn.Module):
    def __init__(self, head_dim: int, max_len: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("l,d->ld", t, inv_freq)              # (L, D/2)
        self.register_buffer("cos", freqs.cos().repeat_interleave(2, dim=-1), persistent=False)
        self.register_buffer("sin", freqs.sin().repeat_interleave(2, dim=-1), persistent=False)

    def forward(self, L):
        return self.cos[:L], self.sin[:L]


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ffn, bias=False)
        self.w_up   = nn.Linear(d, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ffn: int):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d // n_heads
        self.norm_attn = RMSNorm(d)
        self.qkv  = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm_ffn = RMSNorm(d)
        self.ffn = SwiGLU(d, d_ffn)

    def forward(self, x, cos, sin):
        B, L, d = x.shape
        h = self.norm_attn(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # F.scaled_dot_product_attention picks FlashAttention when available.
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, L, d)
        x = x + self.proj(attn)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ----- model -------------------------------------------------------------

class ReadModel(nn.Module):
    """Joint model for p(strand, start, end | DNA) with an MLM head.

    Factorization: p(σ | D) · p(s | σ, D) · p(e | s, σ, D).
    Convention: s <= e always (reference min/max); σ tells you which end is 5'.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 6,
        d_ffn: int | None = None,
        vocab_size: int = 6,         # A, C, G, T, N, MASK
        max_len: int = 5008,         # 4992 + prefix tokens
        n_assays: int = 0,
        n_cell_types: int = 0,
    ):
        super().__init__()
        d_ffn = d_ffn or int(round(8 / 3 * d_model / 64) * 64)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.assay_embed = nn.Embedding(n_assays, d_model) if n_assays > 0 else None
        self.cell_embed  = nn.Embedding(n_cell_types, d_model) if n_cell_types > 0 else None

        self.rope = RoPECache(d_model // n_heads, max_len)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ffn) for _ in range(n_layers)])
        self.norm_out = RMSNorm(d_model)

        # Strand embedding (additive bias) used by start/end heads.
        self.strand_embed = nn.Embedding(2, d_model)        # 0: +, 1: -

        # Heads
        self.strand_head = nn.Linear(d_model, 2, bias=False)
        # Start head: position-wise from h_j and strand bias.
        self.start_head = nn.Linear(d_model, 1, bias=False)
        self.start_strand_bias = nn.Linear(d_model, 1, bias=False)  # uses σ embedding
        # End head: bilinear in (h_j, h_s) plus strand-conditioned position bias.
        self.end_bilinear = nn.Parameter(torch.zeros(2, d_model, d_model))   # one matrix per σ
        self.end_pos_bias = nn.Linear(d_model, 1, bias=False)                # h_j -> scalar
        self.end_strand_bias = nn.Embedding(2, d_model)                      # σ -> d_model
        self.mlm_head = nn.Linear(d_model, vocab_size, bias=False)

    # ---- shared encoding -------------------------------------------------
    def encode(self, dna, assay=None, cell_type=None):
        """dna: (B, L) int64, may be partially masked for MLM samples."""
        B = dna.size(0)
        h = self.embed(dna)
        prefixes = [self.cls.expand(B, 1, -1)]
        if self.assay_embed is not None and assay is not None:
            prefixes.append(self.assay_embed(assay).unsqueeze(1))
        if self.cell_embed is not None and cell_type is not None:
            prefixes.append(self.cell_embed(cell_type).unsqueeze(1))
        h = torch.cat(prefixes + [h], dim=1)
        cos, sin = self.rope(h.size(1))
        for blk in self.blocks:
            h = blk(h, cos, sin)
        h = self.norm_out(h)
        n_prefix = len(prefixes)
        h_cls, h_seq = h[:, 0], h[:, n_prefix:]
        return h_cls, h_seq                                  # (B, d), (B, L, d)

    # ---- heads -----------------------------------------------------------
    def strand_logits(self, h_cls):
        return self.strand_head(h_cls)                       # (B, 2)

    def start_logits(self, h_seq, strand):
        sigma_e = self.strand_embed(strand)                  # (B, d)
        per_pos = self.start_head(h_seq).squeeze(-1)         # (B, L)
        strand_bias = self.start_strand_bias(sigma_e)        # (B, 1) — global per σ
        return per_pos + strand_bias

    def end_logits(self, h_seq, start, strand, max_len: int | None = None):
        B, L, d = h_seq.shape
        h_s = h_seq.gather(1, start.view(B, 1, 1).expand(B, 1, d)).squeeze(1)   # (B, d)
        W = self.end_bilinear[strand]                        # (B, d, d)
        # Bilinear scores: (B, L, d) @ (B, d, d) @ (B, d, 1) -> (B, L)
        bil = torch.einsum("bld,bde,be->bl", h_seq, W, h_s)
        pos_bias = self.end_pos_bias(h_seq).squeeze(-1)      # (B, L)
        strand_bias = (h_seq * self.end_strand_bias(strand).unsqueeze(1)).sum(-1)  # (B, L)
        logits = bil + pos_bias + strand_bias
        positions = torch.arange(L, device=h_seq.device).unsqueeze(0)
        mask = positions < start.unsqueeze(1)
        if max_len is not None:
            mask = mask | (positions > start.unsqueeze(1) + max_len)
        return logits.masked_fill(mask, float("-inf"))

    def mlm_logits(self, h_seq):
        return self.mlm_head(h_seq)                          # (B, L, V)


# --- training: in-batch task mixing --------------------------------------

def training_step(
    model: ReadModel,
    dna,            # (B, L) int64, possibly masked
    mlm_mask,       # (B, L) bool — True where MLM loss applies
    target_nucl,    # (B, L) int64 — original (pre-mask) nucleotides
    has_reads,      # (B,) bool — samples that contribute read losses
    strand_true,    # (B,) int64
    start_true,     # (B,) int64
    end_true,       # (B,) int64
    assay=None, cell_type=None,
    alpha: float = 1.0, beta: float = 1.0,
):
    h_cls, h_seq = model.encode(dna, assay=assay, cell_type=cell_type)

    if mlm_mask.any():
        logits = model.mlm_logits(h_seq)
        mlm_loss = F.cross_entropy(logits[mlm_mask], target_nucl[mlm_mask])
    else:
        mlm_loss = h_seq.new_zeros(())

    if has_reads.any():
        hc, hs = h_cls[has_reads], h_seq[has_reads]
        s_true = strand_true[has_reads]
        s = start_true[has_reads]
        e = end_true[has_reads]

        loss_sigma = F.cross_entropy(model.strand_logits(hc), s_true)
        loss_start = F.cross_entropy(model.start_logits(hs, s_true), s)
        loss_end   = F.cross_entropy(model.end_logits(hs, s, s_true), e)
        read_loss = loss_sigma + loss_start + loss_end
    else:
        read_loss = h_seq.new_zeros(())

    return alpha * mlm_loss + beta * read_loss


# --- inference -----------------------------------------------------------

@torch.no_grad()
def sample_reads(model: ReadModel, dna, n_reads, assay=None, cell_type=None):
    """Ancestral sampling. Encode once, draw n_reads (σ, s, e) triples per window."""
    h_cls, h_seq = model.encode(dna, assay=assay, cell_type=cell_type)
    sigma_dist = torch.distributions.Categorical(logits=model.strand_logits(h_cls))

    strands, starts, ends = [], [], []
    for _ in range(n_reads):
        sigma = sigma_dist.sample()
        s = torch.distributions.Categorical(logits=model.start_logits(h_seq, sigma)).sample()
        e = torch.distributions.Categorical(logits=model.end_logits(h_seq, s, sigma)).sample()
        strands.append(sigma); starts.append(s); ends.append(e)
    return (torch.stack(strands, dim=1),
            torch.stack(starts,  dim=1),
            torch.stack(ends,    dim=1))


def coverage_from_reads(starts, ends, L, strands=None, target_strand=None):
    """Aggregate sampled reads into a coverage profile (B, L)."""
    if target_strand is not None:
        keep = (strands == target_strand)
        starts = torch.where(keep, starts, torch.full_like(starts, -1))
        ends   = torch.where(keep, ends,   torch.full_like(ends,   -2))
    positions = torch.arange(L, device=starts.device)
    covered = (starts.unsqueeze(-1) <= positions) & (ends.unsqueeze(-1) >= positions)
    return covered.sum(dim=1)
```

### Notes on the sketch

- **RoPE is applied inside attention**, on Q and K, not as an additive embedding. The cache stores cos/sin tables up to `max_len` (window + prefix tokens).
- **RMSNorm pre-norm** throughout; final RMSNorm before the heads.
- **SwiGLU FFN** with `d_ffn ≈ 8/3 · d`, no biases. Standard LLaMA recipe.
- **No conv stem.** $L = 4992$ goes straight into self-attention. Use FlashAttention via `F.scaled_dot_product_attention`. This is the dominant cost; profile early.
- **State-space hybrids are deferred.** The block interface is just `(B, L, d) -> (B, L, d)`, so swapping in Mamba/S4 blocks alongside attention layers later is a drop-in change.
- **CLS token for strand head.** Prepended alongside any conditioning tokens; its post-encoder vector pools the window.
- **End head is bilinear**, not a wide-concat MLP. One matrix per strand; per-position bias adds back the linear-in-$h_j$ term. Memory-friendly at $L=4992$.
- **`end_logits` mask** enforces `end ≥ start` and optionally `end ≤ start + ℓ_max`.
- **Conditioning tokens** are prepended to the sequence and stripped after encoding (the per-position heads see only the $L$ DNA positions; the CLS sees everything via attention).
- **In-batch population assignment lives in the dataloader.** For each sample, decide independently whether to mask (probability $p_\text{mlm}$) and whether to expose a read triple (probability $p_\text{reads}$). The three populations from §4 emerge from setting these two probabilities, e.g. $(1, 0) \to$ MLM-only, $(0, 1) \to$ read-only, $(1, 1) \to$ joint.
- **Sampling loop in `sample_reads`** is sequential per read but reuses the cached encoding. For larger `n_reads`, batch the strand samples and run start/end heads on a batched `(B × n_reads, L, d)` input rather than looping.
- **RC augmentation** lives in the dataloader, not the model. Flip the DNA, complement the nucleotides, remap $(s, e, \sigma) \to (L - 1 - e,\; L - 1 - s,\; 1 - \sigma)$.

---

## 7. Open Questions / Risks

1. **Compute scaling.** Full self-attention at $L = 4992$ is $O(L^2)$ per layer. With FlashAttention this is tractable on a single A100/H100 for moderate depth/width, but it sets the ceiling on window length for the mammalian setting. Mitigations: efficient attention kernels (already planned), encoder output caching during sampling (free), state-space / linear-attention hybrids if the quadratic term bites (deferred).
2. **Are we just learning experimental biases?** Read distributions reflect biology *and* assay quirks (fragmentation, PCR bias, mappability). Worth measuring whether the learned read-length distributions match the empirical ones, and how well the model generalizes across experiments. The assay conditioning token should help disentangle biology from assay-specific biases — verify this empirically.
3. **Expected performance ceiling.** How much of the discriminative baseline's error is attributable to implausible read distributions? Probe: take Yorzoi's predicted profiles, find the read-length distribution that best explains each one, quantify how implausible the implied distributions are. This sets an upper bound on the gain we can expect. (Realizability scoring is already implemented in `evals/realizability/`.)
4. **MLM ↔ read loss interference.** The in-batch mixing setup (§4) is the working assumption. Verify empirically that adding the MLM loss doesn't degrade read prediction vs. read-only training. If it does, fall back to a pretrain-then-finetune pipeline (MLM-only pretraining, then unfreeze and add read heads).
5. **RC-equivariant architecture as a principled alternative.** Instead of relying on RC augmentation, an RC-equivariant architecture (Borzoi-style equivariant attention/conv blocks) would *guarantee* that flipping the input flips the predictions consistently. Strictly more correct, substantially more involved. Worth considering once v1 is working and if augmentation turns out to be insufficient.
6. **Architectural alternatives worth investigating.**
   - Pointer-network-style heads predicting $(s, e)$ jointly rather than autoregressively.
   - Reads-as-mass moving along the sequence axis → optimal transport / flow matching, if read-length distributions are stable across windows.
   - State-space / Mamba blocks interleaved with attention to subquadratic-ify the backbone at long windows.

---

## 8. Next Steps

1. **Dataloader from BAMs.** Re-extract reads to persist $(s, e, \sigma)$ alongside the DNA window — these are already in the BAM (`reference_start`, `reference_end`, `is_reverse`), so it's a one-line addition to the extraction pipeline. Output schema: `(dna_window, [(s, e, σ), ...])`. Handle stranded protocols correctly. Decide on window length (start: $L = 4992$) and stride. Implement RC augmentation here.
2. **Implement the model in §6.** RMSNorm / RoPE / SwiGLU, full self-attention, bilinear end head.
3. **Sanity-check on yeast Illumina RNA-seq** with a stripped-down config first: single strand, no MLM, no conditioning tokens. Reproduce the R = 0.46 baseline.
4. **Turn on strand modeling.** Verify the joint $(s, e, \sigma)$ factorization trains and that read prediction doesn't regress vs. the single-strand baseline.
5. **Turn on MLM joint training.** Sweep the in-batch mixing ratio (start with 40/40/20) and $\alpha, \beta$. Verify both losses go down and that read prediction improves (or at least doesn't degrade) vs. read-only training.
6. **Add assay / cell-type conditioning tokens.** Verify multi-experiment training works and that the model uses the tokens (ablation: swap tokens, see if predictions change appropriately — like the Karollus species-token swap experiment).
7. **Profile compute.** FlashAttention on, measure tokens/s at $L = 4992$. Decide whether state-space hybrids are needed before scaling to mammalian windows.
8. **Run the "how implausible are Yorzoi's profiles" probe** (open question 3) to set expectations on gains.
