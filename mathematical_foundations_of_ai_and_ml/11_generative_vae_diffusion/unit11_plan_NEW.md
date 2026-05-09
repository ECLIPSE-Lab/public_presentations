# Unit 11 Plan (REORGANIZED) — Generative Models: VAE & Diffusion

> Note: This is the **new** Unit 11 plan after the SS26 mid-semester reorganization.
> The old plan (`unit11_plan.md`) covered "unsupervised learning: K-means + GMM/EM",
> which has been folded into the new Unit 5 (Clustering & Autoencoders). This slot
> is now the generative-models unit. See `CURRICULUM_NOTES_NEXT_YEAR.md` in the
> project root for context.

## Audience + constraints
- BSc AI-Material Technology, 4th semester
- Prior knowledge: Units 1-10; especially Unit 5 (autoencoders), Unit 8 (probabilistic, KL primer), Unit 10 (transformers)
- Lecture: 90 min + 90-min exercise
- Language: English

## Why this unit, why now
Autoencoders compress; **generative** models go further: they learn $p(x)$ and let us *sample new* $x$. For materials, this is **inverse design** — generate a microstructure with target properties. Two paradigms cover ~95% of modern generative work: variational autoencoders (probabilistic latent-variable models with explicit likelihood bounds) and diffusion models (the dominant paradigm since 2020 for images, audio, molecules, materials). The unit teaches both at intuition depth.

## Learning objectives
By the end of the unit, students can:
1. Explain why a vanilla autoencoder cannot generate new samples and what is missing (a prior on $z$, plus a tractable sampling procedure).
2. Derive the VAE ELBO (evidence lower bound) and the reparameterization trick at intuition depth — full generality is not required.
3. Use a trained VAE: encode → sample latent → decode; explain why the KL prior term shapes the latent distribution.
4. Describe the **forward (noising) process** and **reverse (denoising) process** of a diffusion model, and identify the loss as noise prediction.
5. Sketch the connection between diffusion and score matching (one slide).
6. Apply classifier-free guidance for conditional generation.
7. Place GANs and normalizing flows as alternatives, and articulate the trade-offs.
8. Identify materials use cases for generative models: inverse design, microstructure generation, data augmentation under physics constraints.

## Book + paper mapping (priority order)
1. **Kingma & Welling (2013) "Auto-Encoding Variational Bayes"** — VAE original paper. Required.
2. **Ho et al. (2020) "DDPM"** — diffusion canonical paper. Required for the diffusion section.
3. **Sohl-Dickstein et al. (2015)** — original diffusion thermodynamic framing.
4. **Song & Ermon (2019)** — score matching connection (cite, do not derive).
5. **Ho & Salimans (2022)** — classifier-free guidance.
6. **Bishop (2024) 2nd ed.** Ch. 19 (deep generative models) — textbook.
7. **Murphy (2023) Ch. 28** (deep generative models) — textbook depth.
8. **Lilian Weng's blog** "What are diffusion models?" — best free overview.

## 90-minute structure

| Time | Block | Content |
|------|-------|---------|
| 0–5 | Recap | Unit 5 AE limitations: no prior on $z$, can't sample new data. |
| 5–10 | What is a generative model | $p(x)$; sampling; conditional generation; inverse design framing. |
| 10–35 | **VAE** | Encoder produces a *distribution* over $z$, not a single point. Prior $p(z) = \mathcal{N}(0, I)$. ELBO derivation (intuition + KL term). Reparameterization trick (one slide). Sampling. β-VAE briefly. Posterior collapse. Generation example. |
| 35–40 | Bridge | VAEs sample from a single learned prior. What if generation were a *sequence* of small denoising steps? Enter diffusion. |
| 40–70 | **Diffusion** | Forward noising process: $x_0 \to x_1 \to \ldots \to x_T \approx \mathcal{N}(0,I)$. Closed-form $q(x_t \mid x_0)$. Reverse denoising process: train $\epsilon_\theta(x_t, t)$ to predict noise. Simple loss = MSE on noise. Sampling: start from $\mathcal{N}(0,I)$, iterate the learned reverse process. |
| 70–77 | Diffusion: extras | Conditional diffusion. Classifier-free guidance. Score-matching connection (1 slide). Latent diffusion (Stable Diffusion idea, 1 slide). |
| 77–84 | Alternatives | GANs (mode collapse, no likelihood). Normalizing flows (exact likelihood, restricted architectures). When to choose what. |
| 84–90 | Materials examples + bridge | Inverse design via conditional diffusion; microstructure generation; data augmentation under physics constraints (preview Unit 13). Three must-knows; reading. |

## Slide budget (~62 slides)

- **Block A — Recap + framing (1–6):** title, recap of AE, generation as the missing capability, learning outcomes, roadmap.
- **Block B — Generative model definition (7–9):** $p(x)$, sampling, conditional sampling, inverse design framing.
- **Block C — VAE setup (10–14):** encoder produces a distribution, prior on $z$, decoder as before, generation = sample $z$ then decode.
- **Block D — VAE loss (15–22):** the marginal likelihood is intractable, ELBO as a tractable lower bound, reconstruction term + KL term, KL closed form for Gaussian, reparameterization trick.
- **Block E — Training and using a VAE (23–28):** training loop, sampling new data, latent interpolation, β-VAE, posterior collapse.
- **Block F — VAE → diffusion bridge (29–32):** "what if we use many small steps?"; the limitation of a single Gaussian prior.
- **Block G — Forward diffusion (33–40):** schedule of $\beta_t$, marginal $q(x_t \mid x_0)$, Gaussian properties (closed-form), the data destruction picture.
- **Block H — Reverse diffusion (41–48):** training objective = noise prediction, the simple loss, U-Net as the typical $\epsilon_\theta$.
- **Block I — Sampling (49–52):** DDPM sampler, DDIM (1 slide), how many steps in practice.
- **Block J — Conditional generation (53–56):** classifier-free guidance, prompt conditioning, materials angle.
- **Block K — Score matching connection (57–58):** one slide each: score = $\nabla_x \log p(x)$, denoising = score estimation.
- **Block L — Alternatives (59–60):** GANs, flows, when to use which.
- **Block M — Materials applications + wrap (61–62):** inverse design, microstructure generation, three must-knows, reading, bridge to Unit 12 (uncertainty).

## Reusable equations

- VAE ELBO: $\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x) \,\|\, p(z))$
- Reparameterization: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$
- KL between Gaussians (closed form): $\mathrm{KL}(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)) = \tfrac{1}{2}\sum (\mu^2 + \sigma^2 - \log \sigma^2 - 1)$
- Forward diffusion: $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}\, x_{t-1}, \beta_t I)$
- Closed-form marginal: $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}\, x_0, (1 - \bar\alpha_t) I)$ with $\bar\alpha_t = \prod_{s=1}^t (1 - \beta_s)$
- Simple noise-prediction loss (DDPM): $\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\| \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon, t)\|^2\right]$
- Classifier-free guidance: $\tilde\epsilon_\theta(x_t, c) = (1+w)\epsilon_\theta(x_t, c) - w\epsilon_\theta(x_t, \emptyset)$

## Materials examples to weave in

- **Inverse design**: condition a diffusion model on target Young's modulus + density; sample candidate compositions; verify with simulator.
- **Microstructure generation**: 2-D diffusion model trained on micrographs; conditional on processing parameters; generate plausible synthetic micrographs.
- **Spectral synthesis**: 1-D diffusion or VAE for synthetic spectra; useful for data augmentation when labeled measurements are scarce.
- **Physics-constrained generation**: bridge to Unit 13 — encode conservation laws as soft penalties on the diffusion model's output.

## Exercise (90 min)

1. **Train a small VAE** on Fashion-MNIST or a 1-D spectrum dataset. Inspect the latent: 2-D scatter, interpolation between two samples, KL term magnitude during training.
2. **Sample from the trained VAE** and qualitatively assess generation quality.
3. **Toy diffusion implementation in NumPy/PyTorch**: 2-D Swiss roll dataset, train a tiny $\epsilon_\theta$ MLP, run the sampler, plot the trajectory. (Karpathy-style 200-line implementation.)
4. **Conditional diffusion** with classifier-free guidance on Fashion-MNIST: condition on class label, generate samples with various guidance scales.
5. **Bonus**: replace the VAE prior with a Gaussian mixture (ties back to Unit 5 GMMs) and compare generation quality.

## Assessment alignment
Three exam-must-knows:
1. The VAE optimizes the **ELBO** = reconstruction term − KL prior term; the reparameterization trick makes the KL gradient tractable. Sampling: $z \sim \mathcal{N}(0,I)$, then decode.
2. Diffusion training: predict the noise added to a clean sample at a random timestep. Sampling: start from $\mathcal{N}(0,I)$, iterate the learned denoiser. The forward noising process is a **fixed** Gaussian schedule with closed-form marginals.
3. Trade-off: VAE is fast at sampling and has explicit likelihoods, but tends to produce blurry samples. Diffusion is slow at sampling (many steps) but currently produces the highest-quality samples.

## Slide production notes
- Style: match Units 1-10 RevealJS pattern.
- Critical figures: VAE schematic with stochastic encoder; reparameterization trick diagram; forward-noising schedule visualization; reverse-denoising trajectory; conditional generation grid (e.g., FashionMNIST class-conditioned samples).
- Avoid the full ELBO derivation on the board — students don't have the variational-inference background. State, motivate, use.
- Diffusion math is dense; lean on the Lilian Weng blog and `diffusion_intro.ipynb`-style hand-on rather than algebra slides.
- The score-matching slide is a *pointer*, not a derivation.
