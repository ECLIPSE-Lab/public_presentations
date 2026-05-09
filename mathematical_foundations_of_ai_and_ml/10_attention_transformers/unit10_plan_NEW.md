# Unit 10 Plan (REORGANIZED) — Attention & Transformers

> Note: This is the **new** Unit 10 plan after the SS26 mid-semester reorganization.
> The old plan (`unit10_plan.md`) covered "latent spaces and t-SNE"; that content
> is folded into the new Unit 9. This slot is now the transformer unit, the
> single biggest gap in the original SS26 syllabus. See
> `CURRICULUM_NOTES_NEXT_YEAR.md` in the project root for context.

## Audience + constraints
- BSc AI-Material Technology, 4th semester
- Prior knowledge: Units 1-9; especially Unit 4 (CNN architectures), Unit 8 (probabilistic), Unit 9 (latent embeddings, contrastive)
- Lecture: 90 min + 90-min exercise
- Language: English

## Why this unit, why now
Transformers are the dominant architecture of the past 7+ years and the substrate of every modern foundation model (GPT, BERT, ViT, DINO, CLIP, AlphaFold-2). For a 2026 ML foundations course to omit them would be indefensible. The story builds naturally on Unit 4 (CNNs encode locality + weight-sharing as a *prior*) — attention asks: *what if we let the model decide which positions interact?*

## Learning objectives
By the end of the unit, students can:
1. State why CNN locality is a useful but limiting inductive bias and identify problem classes where it fails.
2. Derive the scaled dot-product self-attention formula $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(QK^T / \sqrt{d_k}) V$ from a content-based addressing motivation.
3. Explain the role of multi-head attention as parallel subspaces.
4. Describe positional encoding (sinusoidal and learned) and why it is needed.
5. Sketch a transformer block: attention + MLP + residual + LayerNorm.
6. Describe the Vision Transformer (ViT) as "transformer on patch sequences" and identify when ViT beats CNNs.
7. Place transformers in the broader landscape of foundation models and explain why scaling drove their dominance.

## Book + paper mapping (priority order)
1. **Vaswani et al. (2017) "Attention is All You Need"** — *required reading for the derivation*.
2. **Dosovitskiy et al. (2020) "An Image is Worth 16x16 Words" (ViT)** — required reading for the vision portion.
3. **Bishop (2024) 2nd ed.** Ch. 12 (transformers) — textbook depth, recommended.
4. **Murphy (2023)** Ch. 16 (attention and transformers) — textbook depth.
5. **The Illustrated Transformer** (Jay Alammar) — visual intuition, optional but excellent for students.
6. **Karpathy "Let's build GPT"** (YouTube) — 2-hour from-scratch implementation, optional.
7. **d2l.ai** Ch. 11 (attention mechanisms) — alternative textbook.

## 90-minute structure

| Time | Block | Content |
|------|-------|---------|
| 0–5 | Recap | Unit 4 CNN priors. The question: what if locality is wrong? |
| 5–12 | When CNNs hit their limits | Long-range dependencies, sets, graphs, sequences. Concrete cases: SMILES strings, XRD spectra with peak interactions, language. |
| 12–25 | **Self-attention** — the idea | Content-based addressing motivation. Query, key, value as a soft dictionary lookup. Hand-trace one attention computation on 4 tokens. |
| 25–35 | Self-attention — the formula | $\mathrm{softmax}(QK^T / \sqrt{d_k}) V$. Why $\sqrt{d_k}$ scaling. Geometric intuition: similarity-weighted average. |
| 35–45 | **Multi-head attention** | Parallel subspaces; why heads help; concatenate + project. Worked size example. |
| 45–55 | **Positional encoding** | Why attention is permutation-equivariant and what to do about it. Sinusoidal vs learned vs RoPE (mention only). |
| 55–65 | **The transformer block** | Attention sublayer + MLP sublayer + residual + LayerNorm. Stack of blocks. Encoder-only vs decoder-only vs encoder-decoder (one slide each). |
| 65–78 | **Vision Transformer (ViT)** | Image as a sequence of patches. CLS token. Comparison to CNNs: ViT wins with enough data, CNNs with less data + locality prior. Materials: ViT for STEM/SEM micrographs. |
| 78–85 | Foundation models (briefly) | Tokenization (very brief). GPT, BERT, ViT, CLIP, DINO patterns. Why scaling worked. |
| 85–90 | Wrap | Three must-knows; reading; bridge to Unit 11 (VAE + diffusion: now we generate, not just discriminate). |

## Slide budget (~60 slides)

- **Block A — Recap + framing (1–6):** title, where we are, the locality-prior limitation, motivating examples, learning outcomes, roadmap.
- **Block B — From CNN to attention (7–11):** what CNNs encode, when locality is wrong, "let the model decide which positions interact" framing.
- **Block C — Self-attention idea (12–20):** soft dictionary lookup analogy; query/key/value definitions; hand-traced attention on 4 tokens; the matrix view.
- **Block D — Self-attention formula (21–28):** scaled dot-product, $\sqrt{d_k}$ scaling explained, softmax-weighted V, masking (causal vs padding) briefly.
- **Block E — Multi-head (29–34):** why one head is not enough; parallel subspaces; concat + project; worked size example (e.g., $d_{\text{model}}=512, h=8, d_k=64$).
- **Block F — Positional encoding (35–40):** permutation equivariance issue; sinusoidal formula + intuition; learned alternative; RoPE pointer.
- **Block G — Transformer block (41–47):** sublayer 1 (attention), sublayer 2 (MLP), residual + LayerNorm, "pre-norm vs post-norm" briefly, stacking.
- **Block H — Architecture variants (48–51):** encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5, original transformer); one slide each.
- **Block I — ViT (52–57):** patch embedding, CLS token, ViT block (same as transformer), data-efficiency vs CNN, ViT vs CNN decision rule.
- **Block J — Foundation models brief (58–60):** scaling laws, the foundation-model recipe (pre-train at scale, fine-tune cheaply), materials angle (foundation models for spectra, sequences).

## Reusable equations

- Scaled dot-product attention: $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$
- Multi-head: $\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W^O$ with $\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
- Sinusoidal positional encoding: $\mathrm{PE}(\mathrm{pos}, 2i) = \sin(\mathrm{pos} / 10000^{2i/d_{\text{model}}})$, $\mathrm{PE}(\mathrm{pos}, 2i+1) = \cos(\mathrm{pos} / 10000^{2i/d_{\text{model}}})$
- Transformer block (pre-norm): $x' = x + \mathrm{MHA}(\mathrm{LN}(x))$; $x'' = x' + \mathrm{MLP}(\mathrm{LN}(x'))$
- ViT patch embedding: $z_0 = [x_{\text{cls}}; x_p^1 E; x_p^2 E; \ldots; x_p^N E] + E_{\text{pos}}$

## Materials examples to weave in

- **STEM/SEM ViT classification**: classify defect types in atomic-resolution micrographs; attention maps highlight the atomic columns the model uses for the decision.
- **Composition tokens**: treat each element fraction as a token; transformer learns interactions; attention map reveals "this Cr-Mo pair is what makes this alloy clusterable as martensitic."
- **XRD peak interactions**: 1-D transformer over peak positions; attention captures phase-mixture relationships.
- **Multimodal materials**: composition + processing + characterization tokens in one transformer (foundation-model preview).

## Exercise (90 min)

1. **Implement single-head self-attention from scratch in PyTorch** (10-line core). Run on a 6-token toy sequence and inspect attention weights.
2. **Hand-trace one attention computation** for $n=3$ tokens with $d_k=2$. Match by hand vs autograd.
3. **Train a tiny ViT** on a small materials dataset (or CIFAR-10 if no dataset is ready). Compare to a CNN baseline of similar parameter count.
4. **Visualize attention maps** on the ViT: confirm the model attends to physically meaningful image regions.
5. **Bonus**: replace ViT's learned positional encoding with sinusoidal — compare training curves.

## Assessment alignment
Three exam-must-knows:
1. Self-attention computes $\mathrm{softmax}(QK^T / \sqrt{d_k}) V$ — a similarity-weighted average of value vectors, where the similarity is content-based (Q vs K dot product).
2. A transformer block stacks multi-head attention and an MLP, each with a residual connection and LayerNorm; positional encoding is added at input because attention is permutation-equivariant.
3. ViT treats an image as a sequence of patch tokens and applies a standard transformer; it beats CNNs with enough data, loses with less data because it lacks the locality prior.

## Slide production notes
- Style: match Units 1-9 RevealJS pattern.
- Critical figures to produce: attention weight heatmap on a toy sequence; ViT patch tokenization diagram; transformer block diagram (residual + LayerNorm boxes); CNN-vs-ViT data-efficiency curve.
- Strongly consider linking to Karpathy's nanoGPT or `transformer-from-scratch` notebook as a take-home extension.
- Avoid going deep into BPE/WordPiece tokenization — students don't need it for materials work.
