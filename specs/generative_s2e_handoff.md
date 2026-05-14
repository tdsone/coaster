# Generative Sequence-to-Expression: Modeling Reads Directly

**Status:** Project proposal + reference implementation sketch for hand-off.
**Audience:** Implementer picking up the project.

---

## 1. Project Overview

### Motivation

Current sequence-to-expression (s2e) models (e.g. Yorzoi) are **discriminative**: they predict summary statistics — per-nucleotide read counts across strands and cell types — directly from DNA. A failure mode of these models is that they can produce coverage profiles that are physically implausible given the read-length and other properties of the underlying sequencing assay. No realistic set of fixed-length reads could ever aggregate into the predicted shape.

### Hypothesis

If we model the **reads themselves** — generate them autoregressively conditioned on DNA, then aggregate to produce coverage — the resulting profiles will be physically realizable by construction, and the model should improve on downstream coverage prediction.

### Preliminary Evidence

A small encoder–decoder transformer trained for 500k steps on a single yeast Illumina RNA-seq experiment (Yorzoi data split) achieves mean Pearson R = 0.46 on test windows when sampling 1k reads per sequence and aggregating into a coverage profile. Yorzoi's baseline on the same split is R = 0.61. The generative model is heavily undertrained and underparameterized, so the gap is expected; reaching this level with a tiny model is the encouraging signal motivating the project.

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

A single shared encoder over the DNA window, with several heads attached:

- **Encoder (shared backbone).** Transformer encoder over the DNA window, producing per-position embeddings $H \in \mathbb{R}^{B \times L \times d}$.
- **Conditioning tokens.** Following Karollus et al. (2024) — who showed that adding a single species token substantially improves representations at near-zero cost — prepend one learned token per axis of variation: **assay, cell type, strand-of-protocol** (and species, if multi-organism). Embeddings are learned. Essential for multi-experiment training; cheap to add.
- **Strand head.** Pooled encoder output → linear → 2 logits → $p(\sigma \mid D)$. Captures the strand bias of expression in the window.
- **Start head.** Per-position logits, conditioned on $\sigma$ via a learned strand embedding mixed into the per-position features. Output: $(B, L)$ logits → $p(s \mid \sigma, D)$.
- **End head.** Per-position logits, conditioned on the start position $s$ (gathered from $H$) and $\sigma$ (via embedding). Output: $(B, L)$ logits, with $j < s$ masked to $-\infty$ to enforce $e \geq s$. Defines $p(e \mid s, \sigma, D)$.
- **MLM head.** Per-position linear $\mathbb{R}^d \to \mathbb{R}^{|V|}$. Predicts the original nucleotide at masked positions. Trained alongside the read heads (see §4) and gives the encoder Karollus-style denoising representations as a free pretraining signal.

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

A reasonable starting mix is something like **40 / 40 / 20**. Populations 1 and 2 give clean per-task signal; population 3 provides the cross-task coupling that pushes the encoder toward representations jointly useful for both objectives. The mixing ratio is a hyperparameter; a simpler 50/50 with just populations 1 and 2 is a fine first cut.

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

---

## 6. Code Sketch

PyTorch skeleton. Illustrative — production version will need positional encodings (the snippet below has none), proper BAM-based data loading, and efficient attention for long windows.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadModel(nn.Module):
    """Joint model for p(strand, start, end | DNA) with an MLM head.

    Factorization: p(σ | D) · p(s | σ, D) · p(e | s, σ, D).
    Convention: s <= e always (reference min/max); σ tells you which end is 5'.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        vocab_size: int = 5,                  # A, C, G, T, N
        n_assays: int = 0,                    # 0 disables the embedding
        n_cell_types: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # Karollus-style conditioning tokens (prepended; optional)
        self.assay_embed = nn.Embedding(n_assays, d_model) if n_assays > 0 else None
        self.cell_embed  = nn.Embedding(n_cell_types, d_model) if n_cell_types > 0 else None

        layer = nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Strand embedding feeds the start/end heads
        self.strand_embed = nn.Embedding(2, d_model)        # 0: +, 1: -

        # Heads sharing the encoder
        self.strand_head = nn.Linear(d_model, 2)
        self.start_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.end_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.mlm_head = nn.Linear(d_model, vocab_size)

    # ---- shared encoding -------------------------------------------------
    def encode(self, dna, assay=None, cell_type=None):
        """
        dna: (B, L) int64, may be partially masked for MLM samples.
        Returns per-position embeddings (B, L, d). Conditioning tokens are
        prepended inside the encoder and stripped before returning.
        """
        h = self.embed(dna)
        prefixes = []
        if self.assay_embed is not None and assay is not None:
            prefixes.append(self.assay_embed(assay).unsqueeze(1))
        if self.cell_embed is not None and cell_type is not None:
            prefixes.append(self.cell_embed(cell_type).unsqueeze(1))
        if prefixes:
            h = torch.cat(prefixes + [h], dim=1)
        h = self.encoder(h)
        return h[:, len(prefixes):]                          # strip prefix tokens

    # ---- heads -----------------------------------------------------------
    def strand_logits(self, h):
        return self.strand_head(h.mean(dim=1))               # (B, 2)

    def start_logits(self, h, strand):                       # strand: (B,)
        B, L, d = h.shape
        sigma = self.strand_embed(strand).unsqueeze(1).expand(B, L, d)
        return self.start_head(torch.cat([h, sigma], dim=-1)).squeeze(-1)   # (B, L)

    def end_logits(self, h, start, strand):                  # start, strand: (B,)
        B, L, d = h.shape
        idx = start.view(B, 1, 1).expand(B, 1, d)
        h_s = h.gather(1, idx).expand(B, L, d)
        sigma = self.strand_embed(strand).unsqueeze(1).expand(B, L, d)
        logits = self.end_head(torch.cat([h, h_s, sigma], dim=-1)).squeeze(-1)
        positions = torch.arange(L, device=h.device).unsqueeze(0)
        return logits.masked_fill(positions < start.unsqueeze(1), float("-inf"))

    def mlm_logits(self, h):
        return self.mlm_head(h)                              # (B, L, V)


# --- training: in-batch task mixing --------------------------------------

def training_step(
    model: ReadModel,
    dna,            # (B, L) int64, possibly masked
    mlm_mask,       # (B, L) bool — True where MLM loss applies
    target_nucl,    # (B, L) int64 — original (pre-mask) nucleotides
    has_reads,      # (B,) bool — samples that contribute read losses
    strand_true,    # (B,) int64 — ground-truth strand (valid where has_reads)
    start_true,     # (B,) int64
    end_true,       # (B,) int64
    assay=None, cell_type=None,
    alpha: float = 1.0, beta: float = 1.0,
):
    """
    Each sample belongs to one of three populations (MLM-only, Read-only, Joint),
    encoded via `mlm_mask` (any True positions → MLM applies) and `has_reads`.
    """
    h = model.encode(dna, assay=assay, cell_type=cell_type)         # shared encoding

    # MLM loss over all masked positions across the batch
    if mlm_mask.any():
        logits = model.mlm_logits(h)                                # (B, L, V)
        mlm_loss = F.cross_entropy(logits[mlm_mask], target_nucl[mlm_mask])
    else:
        mlm_loss = h.new_zeros(())

    # Read losses only on samples flagged has_reads
    if has_reads.any():
        hr = h[has_reads]
        s_true = strand_true[has_reads]
        s = start_true[has_reads]
        e = end_true[has_reads]

        loss_sigma = F.cross_entropy(model.strand_logits(hr), s_true)
        loss_start = F.cross_entropy(model.start_logits(hr, s_true), s)
        loss_end   = F.cross_entropy(model.end_logits(hr, s, s_true), e)
        read_loss = loss_sigma + loss_start + loss_end
    else:
        read_loss = h.new_zeros(())

    return alpha * mlm_loss + beta * read_loss


# --- inference -----------------------------------------------------------

@torch.no_grad()
def sample_reads(model: ReadModel, dna, n_reads, assay=None, cell_type=None):
    """Ancestral sampling. Encode once, draw n_reads (σ, s, e) triples per window."""
    h = model.encode(dna, assay=assay, cell_type=cell_type)
    sigma_dist = torch.distributions.Categorical(logits=model.strand_logits(h))

    strands, starts, ends = [], [], []
    for _ in range(n_reads):
        sigma = sigma_dist.sample()                                 # (B,)
        s = torch.distributions.Categorical(
            logits=model.start_logits(h, sigma)
        ).sample()
        e = torch.distributions.Categorical(
            logits=model.end_logits(h, s, sigma)
        ).sample()
        strands.append(sigma); starts.append(s); ends.append(e)
    return (torch.stack(strands, dim=1),
            torch.stack(starts,  dim=1),
            torch.stack(ends,    dim=1))                            # each (B, n_reads)


def coverage_from_reads(starts, ends, L, strands=None, target_strand=None):
    """
    Aggregate sampled reads into a coverage profile (B, L).
    If `target_strand` is given, restrict to reads with that σ value.
    """
    if target_strand is not None:
        keep = (strands == target_strand)
        starts = torch.where(keep, starts, torch.full_like(starts, -1))
        ends   = torch.where(keep, ends,   torch.full_like(ends,   -2))
    positions = torch.arange(L, device=starts.device)
    covered = (starts.unsqueeze(-1) <= positions) & (ends.unsqueeze(-1) >= positions)
    return covered.sum(dim=1)
```

### Notes on the sketch

- **No positional encoding** in the snippet — the implementer will need to add one (sinusoidal, learned, or rotary). The model as written is permutation-invariant in the DNA, which is wrong.
- **Vocab size 5** assumes `A, C, G, T, N`. Adjust if using k-mers or BPE-style tokenization.
- **Conditioning tokens** are prepended to the sequence and stripped after encoding. Simplest workable option; an alternative is adding them as a learned bias to every position. Karollus et al. prepend.
- **`end_logits` masking** enforces `end >= start`. If there's a known maximum read length $\ell_{\max}$, also mask $j > s + \ell_{\max}$.
- **Strand head pools by mean-averaging.** A learned `[CLS]`-style token is a cleaner alternative and worth trying.
- **In-batch population assignment lives in the dataloader.** Recipe: for each sample, decide independently whether to mask (probability $p_\text{mlm}$) and whether to expose a read triple (probability $p_\text{reads}$). The three populations from §4 emerge from setting these two probabilities, e.g. $(1, 0) \to$ MLM-only, $(0, 1) \to$ read-only, $(1, 1) \to$ joint.
- **Sampling loop in `sample_reads`** is sequential per read but reuses the cached encoding. For larger `n_reads`, batch the strand samples and run start/end heads on a batched `(B × n_reads, L, d)` input rather than looping.
- **RC augmentation** lives in the dataloader, not the model. Flip the DNA, complement the nucleotides, remap $(s, e, \sigma) \to (L - 1 - e,\; L - 1 - s,\; 1 - \sigma)$.

---

## 7. Open Questions / Risks

1. **Compute scaling.** Each coverage profile aggregates thousands of reads. Training and inference cost is the main risk. Mitigations: cache the encoder output and reuse across reads, start on small genomes (yeast is already in hand), batch read sampling efficiently.
2. **Are we just learning experimental biases?** Read distributions reflect biology *and* assay quirks (fragmentation, PCR bias, mappability). Worth measuring whether the learned read-length distributions match the empirical ones, and how well the model generalizes across experiments. The assay conditioning token should help disentangle biology from assay-specific biases — verify this empirically.
3. **Expected performance ceiling.** How much of the discriminative baseline's error is attributable to implausible read distributions? Probe: take Yorzoi's predicted profiles, find the read-length distribution that best explains each one, quantify how implausible the implied distributions are. This sets an upper bound on the gain we can expect.
4. **MLM ↔ read loss interference.** The in-batch mixing setup (§4) is the working assumption. Verify empirically that adding the MLM loss doesn't degrade read prediction vs. read-only training. If it does, fall back to a pretrain-then-finetune pipeline (MLM-only pretraining, then unfreeze and add read heads).
5. **RC-equivariant architecture as a principled alternative.** Instead of relying on RC augmentation, an RC-equivariant architecture (Borzoi-style equivariant attention/conv blocks) would *guarantee* that flipping the input flips the predictions consistently. Strictly more correct, substantially more involved. Worth considering once v1 is working and if augmentation turns out to be insufficient.
6. **Architectural alternatives worth investigating.**
   - Pointer-network-style heads predicting $(s, e)$ jointly rather than autoregressively.
   - Reads-as-mass moving along the sequence axis → optimal transport / flow matching, if read-length distributions are stable across windows.

---

## 8. Next Steps

1. Build dataloader: from aligned BAMs, produce `(dna_window, [(s, e, σ), ...])` tuples. Handle stranded protocols correctly. Decide on window length and stride. Implement RC augmentation here.
2. Implement the model in §6 with proper positional encoding.
3. **Sanity-check on yeast Illumina RNA-seq** with a stripped-down config first: single strand, no MLM, no conditioning tokens. Reproduce the R = 0.46 baseline.
4. **Turn on strand modeling.** Verify the joint $(s, e, \sigma)$ factorization trains and that read prediction doesn't regress vs. the single-strand baseline.
5. **Turn on MLM joint training.** Sweep the in-batch mixing ratio (start with 40/40/20) and $\alpha, \beta$. Verify both losses go down and that read prediction improves (or at least doesn't degrade) vs. read-only training.
6. **Add assay / cell-type conditioning tokens.** Verify multi-experiment training works and that the model uses the tokens (ablation: swap tokens, see if predictions change appropriately — like the Karollus species-token swap experiment).
7. Profile compute. Likely needs efficient attention (FlashAttention or similar) for the full mammalian setting.
8. Run the "how implausible are Yorzoi's profiles" probe (open question 3) to set expectations on gains.
