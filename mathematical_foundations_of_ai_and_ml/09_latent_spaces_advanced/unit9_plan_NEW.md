# Unit 9 Plan (REORGANIZED) — Latent Spaces & Advanced Representation Learning

> Note: This is the **new** Unit 9 plan after the SS26 mid-semester reorganization.
> The old plan (`unit9_plan.md`) covered AE-heavy "representation learning" content;
> AE basics now live in Unit 5, freeing this slot for the deeper representation
> story. See `CURRICULUM_NOTES_NEXT_YEAR.md` in the project root for context.

## Audience + constraints
- BSc AI-Material Technology, 4th semester
- Prior knowledge: Units 1-8 completed (especially Unit 5 clustering+AEs and Unit 8 probabilistic foundations)
- Lecture: 90 min + 90-min exercise
- Language: English

## Why this unit, why now
Unit 5 introduced autoencoders and **the idea** of a latent space. This unit asks the deeper questions: **what makes a latent space *good*?** How do we visualize it? How do we shape it without reconstruction targets? And why do today's foundation models work? This is the bridge between classical representation learning and modern self-supervised learning.

## Learning objectives
By the end of the unit, students can:
1. Explain what makes a latent space "good" (clusters, interpolation, smoothness, transferability) and why standard AEs underdeliver.
2. Derive the t-SNE objective (high-dim Gaussian similarities → low-dim Student-t similarities → KL minimization) and avoid common misinterpretations of t-SNE plots.
3. Compare t-SNE and UMAP at a conceptual level (local vs global structure trade-offs).
4. Describe **contrastive learning** as a self-supervised representation objective: pulling positives together and pushing negatives apart in latent space.
5. Articulate what a **foundation model embedding** is, why it generalizes, and how to use it as a feature extractor for downstream materials tasks.
6. Critique a representation choice for a given task (visualization vs downstream classification vs retrieval).

## Book + paper mapping (priority order)
1. **van der Maaten & Hinton (2008)** — original t-SNE paper. *Required for the derivation slides.*
2. **McInnes et al. (2018)** — UMAP paper. *Reference depth.*
3. **Chen et al. (2020) "SimCLR"** — contrastive learning canonical paper.
4. **Radford et al. (2021) "CLIP"** — image-text contrastive embedding.
5. **Bishop (2024) 2nd ed.** Ch. 16 (representation learning) — textbook depth.
6. **Murphy (2023) Ch. 32** (representation learning) — overview.
7. **Goodfellow et al. (2016) Ch. 15** (representation learning).

## 90-minute structure

| Time | Block | Content |
|------|-------|---------|
| 0–5 | Recap | Where we are; recap AEs from Unit 5; what is *missing* from a vanilla AE latent space. |
| 5–12 | What makes a latent space good | Compactness within concept, separation across concepts, smooth interpolation, transferability. |
| 12–35 | **t-SNE deep dive** | High-dim Gaussian similarities; perplexity; Student-t in low dim (the *crowding problem* and why a heavy-tailed kernel fixes it); KL minimization; what t-SNE preserves (local) vs not (global); common misinterpretations. |
| 35–43 | UMAP (briefly) | Manifold-approximation framing; preserves global structure better; faster; default for many materials use cases now. |
| 43–63 | **Contrastive learning** | Self-supervision without reconstruction. SimCLR loss (InfoNCE). Positive pair = augmentations of same sample; negatives = other samples in batch. Why it produces semantically organized embeddings. Materials angle: augmentations for spectra and micrographs. |
| 63–80 | **Foundation embeddings** | What a foundation model is (briefly: trained at scale, transfers broadly). Using a frozen pre-trained encoder as a feature extractor. CLIP-style multimodal embeddings. Materials examples: pre-trained ViT embeddings for micrograph similarity search. |
| 80–87 | Latent space geometry, briefly | Interpolation paths; nearest-neighbor retrieval; latent arithmetic and its limits. Set up Unit 11 VAE as the principled fix. |
| 87–90 | Summary + bridge to Unit 10 | Three must-knows; reading; transition: "the encoder we just praised — let's open it up. It's a transformer." |

## Slide budget (~55 slides)

- **Block A — Where we are (slides 1–5):** title, recap of Unit 5 AE, what was missing, learning outcomes, roadmap.
- **Block B — What is a good latent space? (6–9):** four desiderata, examples that fail each.
- **Block C — t-SNE (10–22):** motivation, high-dim similarities, perplexity sweep, low-dim similarities, **the crowding problem** (one slide that earns its keep), Student-t fix, KL objective, gradient intuition, perplexity interactive (if available), what t-SNE plots **do not** mean (cluster size, between-cluster distance), Fashion-MNIST or alloy-composition example.
- **Block D — UMAP (23–26):** UMAP idea, comparison to t-SNE on the same dataset, when to prefer which, computational considerations.
- **Block E — Contrastive learning (27–38):** the self-supervised reframe; positive/negative pairs; InfoNCE loss; SimCLR pipeline; augmentation choice; why this works (intuition + 1-slide loss landscape); supervised vs self-supervised contrastive; materials augmentation choices.
- **Block F — Foundation embeddings (39–48):** what a foundation model is (brief); pre-train + freeze + downstream-head pattern; CLIP intuition; embedding similarity for retrieval; materials use case (micrograph similarity search, spectral retrieval).
- **Block G — Latent geometry recap (49–52):** interpolation, retrieval, arithmetic limits; pointer to Unit 11 VAE.
- **Block H — Wrap (53–55):** three must-knows, reading, Unit 10 bridge.

## Reusable equations

- t-SNE high-dim similarity: $p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$, $p_{ij} = (p_{j|i} + p_{i|j})/(2N)$
- Perplexity: $\mathrm{Perp}(P_i) = 2^{H(P_i)}$ where $H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$
- t-SNE low-dim similarity (Student-t, $\nu=1$): $q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$
- t-SNE objective: $C = \mathrm{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$
- InfoNCE (SimCLR): $\mathcal{L}_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j) / \tau)}{\sum_{k=1, k\neq i}^{2N} \exp(\mathrm{sim}(z_i, z_k) / \tau)}$ where $\mathrm{sim}(u,v) = u^T v / (\|u\| \|v\|)$.

## Exercise (90 min)

1. **t-SNE perplexity sweep** on Fashion-MNIST: plot 2-D embeddings at perplexity = 5, 30, 100; observe how cluster shapes change; reflect on what is *not* preserved.
2. **UMAP vs t-SNE** on the same dataset: compare the two embeddings; report which preserves alloy-family structure better on a provided composition dataset.
3. **Train a small SimCLR model** on Fashion-MNIST or a small materials dataset: implement the InfoNCE loss, train an encoder, compare the resulting embeddings to a vanilla AE trained on the same data.
4. **Foundation embedding hands-on**: load a pre-trained image encoder (e.g., DINOv2 or a ViT from `timm`), embed a stack of micrographs, do nearest-neighbor retrieval — show that "this micrograph looks like that one" works without any training.

## Assessment alignment
Three exam-must-knows:
1. t-SNE minimizes the KL between high-dim Gaussian similarities and low-dim Student-t similarities; cluster sizes and between-cluster distances in a t-SNE plot are **not** quantitatively meaningful.
2. Contrastive learning produces semantically organized embeddings without labels by pulling augmentations of the same sample together and pushing different samples apart.
3. Foundation models work as feature extractors because pre-training on massive datasets produces embeddings that transfer; freezing the encoder + training a small head is often a strong baseline.

## Slide production notes
- Style: match Units 1-5 RevealJS pattern (incremental bullets, two-column layouts, speaker notes).
- Reuse old `unit10_plan.md` (latent spaces) for t-SNE figures and crowding-problem visuals.
- New visuals needed: SimCLR positive/negative pair diagram; CLIP-style multimodal embedding sketch; foundation-model use-case flow.
- The old `unit9_plan.md` and `unit10_plan.md` deck content can be cannibalized — pull figures into this deck and discard the rest.
