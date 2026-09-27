"""Figures for DSEM Week 10 — Attention, transformers & graph networks for EM.

Run from the repo root:
    .venv/bin/python data_science_for_em/10_transformers_graphs/img/make_figures.py

Schematic figures are deterministic. The two "result" figures (mini-ViT, GNN) train the
same small models as notebooks/week10_attention_gnn.ipynb (same generators, same seeds),
so the numbers quoted on the slides match the notebook. CPU only, ~1-2 min total.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
torch.set_num_threads(4)
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "figure.dpi": 110})
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Shared data generators (identical copies live in the notebook)
# ---------------------------------------------------------------------------
LATTICES = ["square", "hexagonal", "rectangular"]


def diffraction_pattern(kind, rng, size=32, dose=200.0):
    """Synthetic zone-axis spot pattern: random orientation, scale, excitation; Poisson noise.

    Friedel symmetry I(g) = I(-g) is enforced. Returns log(1+counts) normalised to [0, 1].
    """
    theta = rng.uniform(0, 2 * np.pi)
    a = rng.uniform(5.0, 7.0)                          # |a*| in pixels
    if kind == "square":
        b, gamma = a, np.pi / 2
    elif kind == "hexagonal":
        b, gamma = a, np.pi / 3
    else:                                              # rectangular
        b, gamma = a * rng.uniform(1.3, 1.6), np.pi / 2
    a_vec = a * np.array([np.cos(theta), np.sin(theta)])
    b_vec = b * np.array([np.cos(theta + gamma), np.sin(theta + gamma)])
    c = (size - 1) / 2
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size))
    for h in range(-4, 5):
        for k in range(-4, 5):
            if (h, k) < (0, 0):                        # handle each Friedel pair once
                continue
            g = h * a_vec + k * b_vec
            r = np.linalg.norm(g)
            if r > c - 1:
                continue
            inten = 30.0 if (h, k) == (0, 0) else np.exp(-(r / 9.0) ** 2) * rng.uniform(0.4, 1.6)
            for s in ([1] if (h, k) == (0, 0) else [1, -1]):
                gx, gy = c + s * g[0], c + s * g[1]
                img += inten * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * 0.7 ** 2))
    counts = rng.poisson(dose * img / img.sum() * 50 + 0.05)
    out = np.log1p(counts)
    return (out / out.max()).astype(np.float32)


def make_dataset(n, rng, size=32, dose=200.0):
    X = np.zeros((n, size, size), np.float32)
    y = rng.integers(0, 3, n)
    for i in range(n):
        X[i] = diffraction_pattern(LATTICES[y[i]], rng, size, dose)
    return X, y


# ---------------------------------------------------------------------------
# Mini-ViT with explicit (hand-written) attention so we can read the weights
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.last_attn = None

    def forward(self, x):                              # x: (B, L, d)
        B, L, d = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        A = torch.softmax(q @ k.transpose(-2, -1) / self.dk ** 0.5, dim=-1)  # (B, h, L, L)
        self.last_attn = A.detach()
        z = (A @ v).transpose(1, 2).reshape(B, L, d)
        return self.out(z)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))                  # pre-norm residual
        return x + self.mlp(self.n2(x))


class MiniViT(nn.Module):
    def __init__(self, size=32, patch=4, d=48, heads=4, depth=3, n_classes=3, use_pos=True, seed=0):
        super().__init__()
        torch.manual_seed(seed)                        # reproducible initialisation
        self.patch, self.use_pos = patch, use_pos
        L = (size // patch) ** 2
        self.embed = nn.Linear(patch * patch, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(0.02 * torch.randn(1, L + 1, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def tokens(self, x):                               # (B, H, W) -> (B, L, P*P)
        p = self.patch
        B, H, W = x.shape
        return x.reshape(B, H // p, p, W // p, p).permute(0, 1, 3, 2, 4).reshape(B, -1, p * p)

    def forward(self, x):
        t = self.embed(self.tokens(x))
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], dim=1)
        if self.use_pos:
            t = t + self.pos
        for blk in self.blocks:
            t = blk(t)
        return self.head(self.norm(t[:, 0]))


def train_vit(model, Xtr, ytr, Xte, yte, epochs=25, lr=2e-3, bs=64, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(Xtr_t[idx]), ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        hist.append(accuracy(model, Xte, yte))
    return hist


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    return (model(torch.tensor(X)).argmax(1).numpy() == y).mean()


# ---------------------------------------------------------------------------
# Atom-column graph generator + hand-written message passing
# ---------------------------------------------------------------------------
def atom_column_graph(rng, n=12, p_sub=0.08, p_vac=0.05, cutoff=1.2):
    """Square lattice of columns; substitutional (bright) columns and vacancies.

    Node feature: noisy column intensity. Label: host column with >=1 substitutional neighbour.
    """
    ij = np.array([(i, j) for i in range(n) for j in range(n)], float)
    keep = rng.random(len(ij)) > p_vac
    pos = ij[keep] + rng.normal(0, 0.05, (keep.sum(), 2))
    is_sub = rng.random(len(pos)) < p_sub
    inten = np.where(is_sub, 1.6, 1.0) + rng.normal(0, 0.08, len(pos))
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    src, dst = np.where((d < cutoff) & (d > 0))
    nb_sub = np.zeros(len(pos), bool)
    np.logical_or.at(nb_sub, dst, is_sub[src])
    label = (~is_sub & nb_sub).astype(int)
    x = np.stack([inten, np.ones_like(inten)], 1).astype(np.float32)
    return dict(pos=pos, x=x, edges=np.stack([src, dst]), y=label, is_sub=is_sub)


class MessagePassing(nn.Module):
    """h_i' = ReLU( W_self h_i + sum_{j in N(i)} MLP_msg(h_j) )  — sum = permutation invariant."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.self_lin = nn.Linear(d_in, d_out)
        self.msg = nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(), nn.Linear(d_out, d_out))

    def forward(self, h, edges):
        src, dst = edges
        agg = torch.zeros(h.shape[0], self.msg[-1].out_features).index_add_(0, dst, self.msg(h[src]))
        return F.relu(self.self_lin(h) + agg)


class TinyGNN(nn.Module):
    def __init__(self, d_in=2, d=16, n_layers=1):
        super().__init__()
        dims = [d_in] + [d] * n_layers
        self.layers = nn.ModuleList([MessagePassing(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.head = nn.Linear(dims[-1], 2)
        self.pre = nn.Sequential(nn.Linear(d_in, d), nn.ReLU()) if n_layers == 0 else None
        if n_layers == 0:
            self.head = nn.Linear(d, 2)

    def forward(self, x, edges):
        h = x
        if self.pre is not None:                       # 0 layers = per-node MLP baseline
            return self.head(self.pre(h))
        for layer in self.layers:
            h = layer(h, edges)
        return self.head(h)


def batch_graphs(graphs):
    xs, es, ys, off = [], [], [], 0
    for g in graphs:
        xs.append(g["x"]); es.append(g["edges"] + off); ys.append(g["y"]); off += len(g["x"])
    return torch.tensor(np.concatenate(xs)), torch.tensor(np.concatenate(es, 1)), torch.tensor(np.concatenate(ys))


def balanced_acc(pred, y):
    return 0.5 * ((pred[y == 1] == 1).float().mean() + (pred[y == 0] == 0).float().mean()).item()


def train_gnn(n_layers, train_graphs, test_graphs, epochs=300, seed=0):
    torch.manual_seed(seed)
    model = TinyGNN(n_layers=n_layers)
    xtr, etr, ytr = batch_graphs(train_graphs)
    xte, ete, yte = batch_graphs(test_graphs)
    w = torch.tensor([1.0, (ytr == 0).sum().item() / max((ytr == 1).sum().item(), 1)])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        loss = F.cross_entropy(model(xtr, etr), ytr, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return model, balanced_acc(model(xte, ete).argmax(1), yte)


# ---------------------------------------------------------------------------
# 1. Unifying picture: CNN = grid graph, GNN = sparse graph, transformer = complete graph
# ---------------------------------------------------------------------------
def fig_inductive_bias_graphs():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    # grid
    ax = axes[0]
    g = np.array([(i, j) for i in range(5) for j in range(5)])
    for (i, j) in g:
        for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            if 0 <= i + di < 5 and 0 <= j + dj < 5:
                ax.plot([i, i + di], [j, j + dj], color=C_GREY, lw=0.8, zorder=1)
    ax.scatter(g[:, 0], g[:, 1], s=180, c=C_BLUE, zorder=2)
    nb = [(i, j) for i in range(1, 4) for j in range(1, 4)]
    ax.scatter(*np.array(nb).T, s=260, facecolors="none", edgecolors=C_ORANGE, lw=2.5, zorder=3)
    ax.scatter([2], [2], s=260, c=C_ORANGE, zorder=4)
    ax.set_title("CNN: fixed grid graph\n3×3 neighbourhood, shared weights")
    # sparse atom graph
    ax = axes[1]
    rng = np.random.default_rng(3)
    pos = np.array([(i + 0.5 * (j % 2), j * 0.87) for i in range(5) for j in range(5)], float)
    pos = np.delete(pos, [7, 16], axis=0) + rng.normal(0, 0.05, (len(pos) - 2, 2))
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    for a, b in zip(*np.where((d < 1.15) & (d > 0))):
        if a < b:
            ax.plot(*pos[[a, b]].T, color=C_GREY, lw=1.0, zorder=1)
    ax.scatter(*pos.T, s=180, c=C_GREEN, zorder=2)
    c = 11
    for b in np.where((d[c] < 1.15) & (d[c] > 0))[0]:
        ax.plot(*pos[[c, b]].T, color=C_ORANGE, lw=2.5, zorder=2)
    ax.scatter(*pos[c], s=260, c=C_ORANGE, zorder=4)
    ax.set_title("GNN: sparse graph from physics\natom columns + cutoff neighbours")
    # complete graph
    ax = axes[2]
    n = 12
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.stack([np.cos(ang), np.sin(ang)], 1) * 2 + 2
    for a in range(n):
        for b in range(a + 1, n):
            ax.plot(*p[[a, b]].T, color=C_GREY, lw=0.5, alpha=0.6, zorder=1)
    w = np.random.default_rng(1).dirichlet(np.ones(n) * 0.4)
    for b in range(1, n):
        ax.plot(*p[[0, b]].T, color=C_ORANGE, lw=0.5 + 12 * w[b], zorder=2)
    ax.scatter(*p.T, s=180, c=C_RED, zorder=3)
    ax.scatter(*p[0], s=260, c=C_ORANGE, zorder=4)
    ax.set_title("Transformer: complete graph\nedge weights = attention (input-dependent)")
    for ax in axes:
        ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("Architecture = assumption about which inputs talk to each other", fontsize=15, y=1.02)
    save(fig, "inductive_bias_graphs.png")


# ---------------------------------------------------------------------------
# 2. Hand-traced attention on 4 tokens (numbers match the slide)
# ---------------------------------------------------------------------------
def toy_attention():
    Q = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], float)
    K = np.array([[1, 0], [0, 1], [1, 1], [-1, 0]], float)
    V = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], float)
    S = Q @ K.T / np.sqrt(2)
    A = np.exp(S) / np.exp(S).sum(1, keepdims=True)
    return Q, K, V, S, A, A @ V


def fig_attention_toy():
    Q, K, V, S, A, Z = toy_attention()
    names = ["A: bright disc", "B: weak disc", "C: disc pair", "D: background"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1]})
    for ax, M, t, cm in [(axes[0], S, r"scores $QK^\top/\sqrt{d_k}$", "RdBu_r"), (axes[1], A, "attention $A$ = row-softmax", "viridis")]:
        vmax = np.abs(M).max()
        im = ax.imshow(M, cmap=cm, vmin=-vmax if cm == "RdBu_r" else 0, vmax=vmax)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=12,
                        color="white" if (cm == "viridis" and M[i, j] < 0.35) else "black")
        ax.set_xticks(range(4), [n.split(":")[0] for n in names]); ax.set_yticks(range(4), [n.split(":")[0] for n in names])
        ax.set_xlabel("key $j$"); ax.set_ylabel("query $i$"); ax.set_title(t)
        plt.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "attention_toy.png")
    print("toy A row C:", np.round(A[2], 3), " z_C =", np.round(Z[2], 3))


# ---------------------------------------------------------------------------
# 3. Why sqrt(d_k): softmax saturation (real numbers, random unit-variance q, k)
# ---------------------------------------------------------------------------
def fig_softmax_saturation():
    rng = np.random.default_rng(0)
    dk, L, n = 64, 64, 2000
    q = rng.normal(size=(n, dk)); k = rng.normal(size=(L, dk))
    s = q @ k.T
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].hist(s.ravel(), bins=80, alpha=0.6, color=C_RED, density=True, label=f"raw: std={s.std():.1f}")
    axes[0].hist((s / np.sqrt(dk)).ravel(), bins=80, alpha=0.6, color=C_BLUE, density=True, label=f"scaled: std={(s / np.sqrt(dk)).std():.2f}")
    axes[0].set_xlabel(r"score $\mathbf{q}^\top\mathbf{k}$"); axes[0].set_title(f"score distribution ($d_k={dk}$)"); axes[0].legend()
    for scale, col, lab in [(1, C_RED, "raw"), (np.sqrt(dk), C_BLUE, r"$/\sqrt{d_k}$")]:
        z = s / scale
        a = np.exp(z - z.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        axes[1].hist(a.max(1), bins=40, range=(0, 1), alpha=0.6, color=col, label=f"{lab}: mean max weight={a.max(1).mean():.2f}")
    axes[1].set_xlabel("largest attention weight per query"); axes[1].set_title(f"softmax over L={L} keys"); axes[1].legend()
    save(fig, "softmax_saturation.png")


# ---------------------------------------------------------------------------
# 4. Positional encoding: 1-D sinusoidal + permutation demo
# ---------------------------------------------------------------------------
def fig_positional_encoding():
    L, d = 64, 48
    pos = np.arange(L)[:, None]; i = np.arange(d // 2)[None]
    pe = np.zeros((L, d))
    pe[:, 0::2] = np.sin(pos / 10000 ** (2 * i / d)); pe[:, 1::2] = np.cos(pos / 10000 ** (2 * i / d))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    im = axes[0].imshow(pe, aspect="auto", cmap="RdBu_r")
    axes[0].set_xlabel("embedding dimension"); axes[0].set_ylabel("token position (e.g. energy window)")
    axes[0].set_title("sinusoidal positional encoding"); plt.colorbar(im, ax=axes[0])
    sim = pe @ pe.T
    im = axes[1].imshow(sim, cmap="viridis")
    axes[1].set_xlabel("position"); axes[1].set_ylabel("position")
    axes[1].set_title(r"$\mathrm{PE}_i^\top\mathrm{PE}_j$ depends on $|i-j|$ only"); plt.colorbar(im, ax=axes[1])
    save(fig, "positional_encoding.png")


# ---------------------------------------------------------------------------
# 5. Tokenising EM data: diffraction patches, per-disc tokens, spectrum windows
# ---------------------------------------------------------------------------
def fig_tokenise_em():
    rng = np.random.default_rng(5)
    dp = diffraction_pattern("hexagonal", rng, size=64, dose=2000)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    ax = axes[0]
    ax.imshow(dp, cmap="magma")
    for t in range(0, 65, 8):
        ax.axhline(t - 0.5, color="w", lw=0.6); ax.axvline(t - 0.5, color="w", lw=0.6)
    ax.set_title("(a) square patches: 8×8 px → 64 tokens\n(discs can straddle seams)"); ax.axis("off")
    ax = axes[1]
    ax.imshow(dp, cmap="magma")
    from scipy.ndimage import maximum_filter
    pk = (dp == maximum_filter(dp, 5)) & (dp > 0.3)
    yy, xx = np.where(pk)
    ax.scatter(xx, yy, s=160, facecolors="none", edgecolors="cyan", lw=1.8)
    ax.set_title(f"(b) one token per Bragg disc: {len(xx)} tokens\n(k_r, k_θ, I) embedded"); ax.axis("off")
    ax = axes[2]
    E = np.linspace(400, 800, 400)
    spec = 2e6 * E ** -2.2
    for e0, h, w in [(456, 1.0, 3), (532, 1.4, 4), (708, 0.9, 3), (721, 0.5, 3)]:
        spec += h * (1 / (1 + np.exp(-(E - e0) / 1.5))) * np.exp(-(E - e0) / 120) + 1.5 * h * np.exp(-0.5 * ((E - e0 - 2) / w) ** 2)
    spec = rng.poisson(spec * 200) / 200
    ax.plot(E, spec, color="k", lw=1)
    for j, e in enumerate(np.arange(400, 801, 25)):
        ax.axvline(e, color=C_BLUE, lw=0.6, alpha=0.6)
    for e0, lab in [(456, "Ti L"), (532, "O K"), (708, "Fe L")]:
        ax.annotate(lab, (e0, spec[np.argmin(abs(E - e0 - 3))]), xytext=(0, 12), textcoords="offset points", ha="center")
    ax.set_ylim(spec.min() * 0.9, spec.max() * 1.15)
    ax.set_xlabel("energy loss (eV)"); ax.set_ylabel("counts (a.u.)")
    ax.set_title("(c) spectrum: 25 eV windows → 16 tokens\nposition = absolute energy")
    save(fig, "tokenise_em.png")


# ---------------------------------------------------------------------------
# 6. Atom-column graph + receptive field of message passing
# ---------------------------------------------------------------------------
def fig_atom_graph():
    rng = np.random.default_rng(11)
    g = atom_column_graph(rng, n=9)
    pos, (src, dst) = g["pos"], g["edges"]
    # render a HAADF-like image of the columns
    s = 20; H = 9 * s
    yy, xx = np.mgrid[0:H, 0:H] / s - 0.5
    img = np.zeros_like(xx)
    for p, I in zip(pos, g["x"][:, 0]):
        img += I ** 2 * np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * 0.15 ** 2))
    img = rng.poisson(img * 30 + 2) / 30
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img, cmap="gray", extent=[-0.5, 8.5, 8.5, -0.5])
    axes[0].set_title("synthetic HAADF: bright = substitution,\nmissing = vacancy"); axes[0].axis("off")
    ax = axes[1]
    for a, b in zip(src, dst):
        if a < b:
            ax.plot(*pos[[a, b]].T, color=C_GREY, lw=1, zorder=1)
    col = np.where(g["is_sub"], C_ORANGE, np.where(g["y"] == 1, C_RED, C_BLUE))
    ax.scatter(*pos.T, c=col, s=140, zorder=2, edgecolors="k", lw=0.5)
    ax.set_title("graph: nodes = columns, edges = d < r_c\nred = host next to a substitution (label)")
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    ax = axes[2]
    centre = np.argmin(np.linalg.norm(pos - np.array([4, 4]), axis=1))
    A = np.zeros((len(pos), len(pos))); A[src, dst] = 1
    hop1 = A[centre] > 0
    hop2 = ((A @ A)[centre] > 0) & ~hop1
    for a, b in zip(src, dst):
        if a < b:
            ax.plot(*pos[[a, b]].T, color=C_GREY, lw=0.6, zorder=1, alpha=0.5)
    ax.scatter(*pos.T, c="lightgrey", s=110, zorder=2)
    ax.scatter(*pos[hop2].T, c=C_GREEN, s=140, zorder=3, label="after 2 layers")
    ax.scatter(*pos[hop1].T, c=C_ORANGE, s=140, zorder=4, label="after 1 layer")
    ax.scatter(*pos[centre], c="k", s=200, zorder=5, marker="*", label="node i")
    ax.set_title("receptive field grows one hop per\nmessage-passing layer")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=11)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    save(fig, "atom_graph.png")


# ---------------------------------------------------------------------------
# 7. Mini-ViT results: examples, training curves, PE ablation, attention vs occlusion
# ---------------------------------------------------------------------------
def fig_vit_results():
    rng = np.random.default_rng(0)
    Xtr, ytr = make_dataset(3000, rng)
    Xte, yte = make_dataset(1000, rng)
    Xlow, ylow = make_dataset(1000, rng, dose=20.0)
    torch.set_num_threads(4)
    vit = MiniViT(use_pos=True)
    h_pos = train_vit(vit, Xtr, ytr, Xte, yte)
    vit_nopos = MiniViT(use_pos=False)
    h_nopos = train_vit(vit_nopos, Xtr, ytr, Xte, yte)
    acc_low = accuracy(vit, Xlow, ylow)
    print(f"ViT test acc: with PE {h_pos[-1]:.3f}, without PE {h_nopos[-1]:.3f}, low-dose {acc_low:.3f}")
    print("n params:", sum(p.numel() for p in vit.parameters()))

    # permutation check on the no-PE model: shuffling patches leaves prediction unchanged
    xb = torch.tensor(Xte[:8])
    with torch.no_grad():
        t = vit_nopos.tokens(xb)
        perm = torch.randperm(t.shape[1])
        emb = vit_nopos.embed(t)
        def run(e):
            z = torch.cat([vit_nopos.cls.expand(len(e), -1, -1), e], 1)
            for b in vit_nopos.blocks:
                z = b(z)
            return vit_nopos.head(vit_nopos.norm(z[:, 0]))
        diff = (run(emb) - run(emb[:, perm])).abs().max().item()
    print(f"no-PE ViT: max logit change under patch shuffle = {diff:.2e}")

    # attention (last layer CLS, head-mean) vs occlusion sensitivity per patch
    vit.eval()
    n_eval = 100
    corr = []
    ex = {}
    for idx in range(n_eval):
        x = torch.tensor(Xte[idx:idx + 1])
        with torch.no_grad():
            logit = vit(x)
            c = logit.argmax(1).item()
            att = vit.blocks[-1].attn.last_attn[0].mean(0)[0, 1:].numpy()
            base = torch.softmax(logit, 1)[0, c].item()
            occ = np.zeros(64)
            for pidx in range(64):
                xo = x.clone()
                r, cc = divmod(pidx, 8)
                xo[0, r * 4:(r + 1) * 4, cc * 4:(cc + 1) * 4] = 0.0
                occ[pidx] = base - torch.softmax(vit(xo), 1)[0, c].item()
        rk = lambda v: np.argsort(np.argsort(v))
        corr.append(np.corrcoef(rk(att), rk(occ))[0, 1])
        if idx in (0, 1, 2):
            ex[idx] = (Xte[idx], att.reshape(8, 8), occ.reshape(8, 8), LATTICES[c], LATTICES[yte[idx]])
    corr = np.array(corr)
    print(f"Spearman(attention, occlusion) over {n_eval} test patterns: mean {corr.mean():.2f}, "
          f"median {np.median(corr):.2f}, frac<0.3 {np.mean(corr < 0.3):.2f}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    for j, k in enumerate(range(3)):
        axes[j].imshow(Xtr[np.where(ytr == k)[0][0]], cmap="magma")
        axes[j].set_title(f"{LATTICES[k]}"); axes[j].axis("off")
    axes[3].plot(np.arange(1, len(h_pos) + 1), h_pos, "o-", color=C_BLUE, label=f"with pos. emb. ({h_pos[-1]:.2f})")
    axes[3].plot(np.arange(1, len(h_nopos) + 1), h_nopos, "s-", color=C_RED, label=f"no pos. emb. ({h_nopos[-1]:.2f})")
    axes[3].axhline(1 / 3, color=C_GREY, ls="--", label="chance")
    axes[3].set_xlabel("epoch"); axes[3].set_ylabel("test accuracy"); axes[3].set_ylim(0, 1.02)
    axes[3].set_title("mini-ViT on 3 000 patterns (CPU)"); axes[3].legend(fontsize=11, loc="lower right")
    save(fig, "vit_diffraction_results.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9.2))
    for j, idx in enumerate(ex):
        img, att, occ, pred, true = ex[idx]
        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].imshow(np.kron(att, np.ones((4, 4))), cmap="inferno", alpha=0.55)
        axes[0, j].set_title(f"CLS attention (last layer)\npred {pred} / true {true}", fontsize=12)
        axes[1, j].imshow(img, cmap="gray")
        axes[1, j].imshow(np.kron(np.clip(occ, 0, None), np.ones((4, 4))), cmap="inferno", alpha=0.55)
        axes[1, j].set_title(f"occlusion Δp (Spearman ρ = {corr[idx]:.2f})", fontsize=12)
        axes[0, j].axis("off"); axes[1, j].axis("off")
    fig.suptitle(f"Where the model 'looks' vs what actually changes the output — mean ρ over {n_eval} patterns = {corr.mean():.2f}", fontsize=14)
    save(fig, "attention_vs_occlusion.png")
    return dict(acc_pos=h_pos[-1], acc_nopos=h_nopos[-1], acc_low=acc_low, rho=corr.mean())


def fig_gnn_results():
    rng = np.random.default_rng(1)
    train = [atom_column_graph(rng) for _ in range(40)]
    test = [atom_column_graph(rng) for _ in range(20)]
    res = {}
    for L in [0, 1, 2]:
        _, res[L] = train_gnn(L, train, test)
    print("GNN balanced acc (0=per-node MLP, 1, 2 MP layers):", {k: round(v, 3) for k, v in res.items()})
    # permutation equivariance check
    model, _ = train_gnn(1, train, test)
    g = test[0]
    x, e = torch.tensor(g["x"]), torch.tensor(g["edges"])
    perm = torch.randperm(len(x)); inv = torch.argsort(perm)
    with torch.no_grad():
        out = model(x, e)
        out_p = model(x[perm], inv[e])
    print(f"GNN equivariance: max |f(Px) - P f(x)| = {(out_p - out[perm]).abs().max().item():.2e}")
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    labs = ["per-node MLP\n(0 layers)", "1 message-\npassing layer", "2 message-\npassing layers"]
    ax.bar(labs, [res[0], res[1], res[2]], color=[C_GREY, C_GREEN, C_GREEN])
    for i, v in enumerate([res[0], res[1], res[2]]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    ax.axhline(0.5, ls="--", color="k", lw=1); ax.text(2.45, 0.52, "chance", ha="right", fontsize=11)
    ax.set_ylim(0, 1.1); ax.set_ylabel("balanced accuracy (20 held-out images)")
    ax.set_title("'host next to a substitution?' needs neighbours")
    save(fig, "gnn_results.png")
    return res


if __name__ == "__main__":
    fig_inductive_bias_graphs()
    fig_attention_toy()
    fig_softmax_saturation()
    fig_positional_encoding()
    fig_tokenise_em()
    fig_atom_graph()
    fig_gnn_results()
    fig_vit_results()
