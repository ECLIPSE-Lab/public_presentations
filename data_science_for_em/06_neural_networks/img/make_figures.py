"""Reproducible figures for DSEM Week 6 (neural networks & backpropagation).

Generates the figures added in the WS 26/27 update:
  - backprop_worked_example.png   worked 1-2-1 ReLU network, forward values + backward gradients
  - init_activation_stats.png     activation std vs depth for different initialisations
  - loss_curve_diagnostics.png    four typical training-curve patterns (real PyTorch runs)
  - lr_finder.png                 learning-rate range test (real PyTorch run)

Run from repo root:
    .venv/bin/python data_science_for_em/06_neural_networks/img/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)  # tiny models: single thread is fastest
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13})
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#e67e22", "#2ca02c", "#d62728", "#555555"


# ---------------------------------------------------------------------------
# 1. Worked backprop example on a computational graph
# ---------------------------------------------------------------------------
def fig_backprop_worked_example():
    # Same numbers as the slides and the notebook gradient-check cell
    x, y = 1.0, 0.5
    W1 = np.array([2.0, -1.0]); b1 = np.array([-0.5, 0.5])
    w2 = np.array([1.0, 2.0]); b2 = 0.0
    z1 = W1 * x + b1                    # [1.5, -0.5]
    a1 = np.maximum(z1, 0)              # [1.5, 0.0]
    yhat = w2 @ a1 + b2                 # 1.5
    L = 0.5 * (yhat - y) ** 2           # 0.5
    d_yhat = yhat - y                   # 1.0
    d_a1 = w2 * d_yhat                  # [1.0, 2.0]
    d_z1 = d_a1 * (z1 > 0)              # [1.0, 0.0]

    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6.2); ax.axis("off")

    def node(xc, yc, label, fwd, bwd, color="#eaf2fb"):
        ax.add_patch(FancyBboxPatch((xc - 0.95, yc - 0.55), 1.9, 1.1,
                                    boxstyle="round,pad=0.05", fc=color, ec=GREY, lw=1.5))
        ax.text(xc, yc + 0.15, label, ha="center", va="center", fontsize=15, weight="bold")
        ax.text(xc, yc - 0.28, fwd, ha="center", va="center", fontsize=12, color="black")
        if bwd:
            ax.text(xc, yc - 0.95, bwd, ha="center", va="center", fontsize=12, color=RED)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, lw=1.6, color="black"))

    node(1.2, 3.1, r"$x$", f"= {x:.1f}", None, "#f4f4f4")
    node(4.2, 4.6, r"$z_1^{(1)}$", f"= {z1[0]:.1f}", rf"$\delta$ = {d_z1[0]:.1f}")
    node(4.2, 1.6, r"$z_2^{(1)}$", f"= {z1[1]:.1f}", rf"$\delta$ = {d_z1[1]:.1f}")
    node(7.2, 4.6, r"$a_1 = \mathrm{ReLU}$", f"= {a1[0]:.1f}", rf"$\partial L/\partial a$ = {d_a1[0]:.1f}")
    node(7.2, 1.6, r"$a_2 = \mathrm{ReLU}$", f"= {a1[1]:.1f}", rf"$\partial L/\partial a$ = {d_a1[1]:.1f}")
    node(10.2, 3.1, r"$\hat y$", f"= {yhat:.1f}", rf"$\partial L/\partial \hat y$ = {d_yhat:.1f}")
    node(13.2, 3.1, r"$L=\frac{1}{2}(\hat y-y)^2$", f"= {L:.2f}  (y = {y})", r"$\partial L/\partial L$ = 1", "#fdecea")

    arrow((2.15, 3.3), (3.25, 4.45)); arrow((2.15, 2.9), (3.25, 1.75))
    arrow((5.15, 4.6), (6.25, 4.6)); arrow((5.15, 1.6), (6.25, 1.6))
    arrow((8.15, 4.45), (9.25, 3.3)); arrow((8.15, 1.75), (9.25, 2.9))
    arrow((11.15, 3.1), (12.1, 3.1))

    ax.text(1.35, 4.55, r"$W^{(1)}_1\!=\!2,\ b^{(1)}_1\!=\!-0.5$", fontsize=11.5)
    ax.text(1.35, 1.45, r"$W^{(1)}_2\!=\!-1,\ b^{(1)}_2\!=\!0.5$", fontsize=11.5)
    ax.text(8.75, 4.3, r"$w^{(2)}_1\!=\!1$", fontsize=11.5)
    ax.text(8.75, 1.75, r"$w^{(2)}_2\!=\!2$", fontsize=11.5)

    ax.text(7.5, 0.05,
            r"Weight gradients:  $\partial L/\partial \mathbf{w}^{(2)} = \delta_{\hat y}\,\mathbf{a}^{(1)} = [1.5,\ 0]$,   "
            r"$\partial L/\partial \mathbf{W}^{(1)} = \boldsymbol{\delta}^{(1)} x = [1.0,\ 0]$,   "
            r"$\partial L/\partial \mathbf{b}^{(1)} = [1.0,\ 0]$",
            ha="center", fontsize=12.5, color=RED)
    ax.text(0.2, 5.9, "black: forward pass (values)    red: backward pass (gradients)", fontsize=12, color=GREY)
    fig.tight_layout()
    fig.savefig(OUT / "backprop_worked_example.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Initialisation: activation statistics through depth
# ---------------------------------------------------------------------------
def fig_init_activation_stats():
    rng = np.random.default_rng(0)
    D, depth, N = 256, 20, 1000
    X = rng.standard_normal((N, D))
    schemes = {
        r"small $\mathcal{N}(0,0.01^2)$": lambda din, dout: rng.normal(0, 0.01, (din, dout)),
        r"large $\mathcal{N}(0,1)$": lambda din, dout: rng.normal(0, 1.0, (din, dout)),
        r"Xavier $\mathrm{Var}=2/(D_{in}+D_{out})$": lambda din, dout: rng.normal(0, np.sqrt(2 / (din + dout)), (din, dout)),
        r"He $\mathrm{Var}=2/D_{in}$": lambda din, dout: rng.normal(0, np.sqrt(2 / din), (din, dout)),
    }
    colors = [GREY, RED, ORANGE, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (act_name, act) in zip(axes, [("ReLU", lambda z: np.maximum(z, 0)), ("tanh", np.tanh)]):
        for (name, init), c in zip(schemes.items(), colors):
            h = X.copy(); stds = []
            for _ in range(depth):
                h = act(h @ init(D, D))
                stds.append(h.std())
            stds = np.clip(np.array(stds), 1e-30, 1e30)
            ax.semilogy(np.arange(1, depth + 1), stds, "o-", color=c, lw=2, ms=4, label=name)
        ax.set_xlabel("layer index"); ax.set_ylabel("std of activations (log)")
        ax.set_title(f"{act_name} network, width {D}, depth {depth}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=11, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "init_activation_stats.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared toy data for the training-diagnostics figures
# ---------------------------------------------------------------------------
def make_data(n, seed):
    g = np.random.default_rng(seed)
    d = g.uniform(0.3, 9.0, n); T = g.uniform(500, 900, n)
    extra = g.standard_normal((n, 3))
    H = 150 + 90 / np.sqrt(d) - 0.15 * (T - 700) + 5 * np.sin(extra[:, 0]) + g.normal(0, 8, n)
    X = np.column_stack([d, T, extra])
    return X, H


def to_tensors(X, H, mu, sd, hm, hs):
    return (torch.tensor((X - mu) / sd, dtype=torch.float32),
            torch.tensor((H - hm) / hs, dtype=torch.float32).unsqueeze(1))


def mlp(width=64, depth=2):
    layers, din = [], 5
    for _ in range(depth):
        layers += [torch.nn.Linear(din, width), torch.nn.ReLU()]; din = width
    layers.append(torch.nn.Linear(din, 1))
    return torch.nn.Sequential(*layers)


def train(Xtr, ytr, Xva, yva, opt_fn, epochs, width=64, depth=2, seed=0, batch=32):
    torch.manual_seed(seed)
    net = mlp(width, depth); opt = opt_fn(net.parameters()); lossf = torch.nn.MSELoss()
    tr, va = [], []
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(); loss = lossf(net(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
        with torch.no_grad():
            tr.append(lossf(net(Xtr), ytr).item()); va.append(lossf(net(Xva), yva).item())
    return np.array(tr), np.array(va)


# ---------------------------------------------------------------------------
# 3. Four loss-curve patterns
# ---------------------------------------------------------------------------
def fig_loss_curve_diagnostics():
    X, H = make_data(600, 1)
    Xtr_np, Htr = X[:300], H[:300]; Xva_np, Hva = X[300:], H[300:]
    mu, sd, hm, hs = Xtr_np.mean(0), Xtr_np.std(0), Htr.mean(), Htr.std()
    Xtr, ytr = to_tensors(Xtr_np, Htr, mu, sd, hm, hs)
    Xva, yva = to_tensors(Xva_np, Hva, mu, sd, hm, hs)
    # small training set for the overfitting panel
    Xs, ys = Xtr[:30], ytr[:30]

    runs = [
        ("LR too high (SGD, η = 0.5)", train(Xtr, ytr, Xva, yva, lambda p: torch.optim.SGD(p, lr=0.5), 150)),
        ("LR too low (SGD, η = 1e-4)", train(Xtr, ytr, Xva, yva, lambda p: torch.optim.SGD(p, lr=1e-4), 150)),
        ("Healthy (Adam, η = 1e-3)", train(Xtr, ytr, Xva, yva, lambda p: torch.optim.Adam(p, lr=1e-3), 150)),
        ("Overfitting (N = 30, 256×3 net, no reg.)",
         train(Xs, ys, Xva, yva, lambda p: torch.optim.Adam(p, lr=1e-3), 400, width=256, depth=3, batch=30)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    for ax, (title, (tr, va)) in zip(axes, runs):
        ep = np.arange(1, len(tr) + 1)
        tr = np.where(np.isfinite(tr), tr, np.nan); va = np.where(np.isfinite(va), va, np.nan)
        ax.semilogy(ep, tr, color=BLUE, lw=2, label="train")
        ax.semilogy(ep, va, color=ORANGE, lw=2, label="validation")
        ax.set_title(title, fontsize=13); ax.set_xlabel("epoch"); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("MSE (standardised units, log)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "loss_curve_diagnostics.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Learning-rate range test ("LR finder")
# ---------------------------------------------------------------------------
def fig_lr_finder():
    X, H = make_data(600, 2)
    mu, sd, hm, hs = X.mean(0), X.std(0), H.mean(), H.std()
    Xt, yt = to_tensors(X, H, mu, sd, hm, hs)
    torch.manual_seed(0)
    net = mlp(64, 2); opt = torch.optim.SGD(net.parameters(), lr=1e-5); lossf = torch.nn.MSELoss()
    lrs = np.logspace(-5, 1, 200); losses = []; smooth = None
    g = torch.Generator().manual_seed(0)
    for lr in lrs:
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = torch.randint(0, len(Xt), (32,), generator=g)
        opt.zero_grad(); loss = lossf(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        v = loss.item()
        smooth = v if smooth is None else 0.9 * smooth + 0.1 * v
        losses.append(smooth)
        if not np.isfinite(v) or smooth > 4 * min(losses):
            break
    losses = np.array(losses); lrs = lrs[:len(losses)]
    i_min = int(np.argmin(losses))
    grad = np.gradient(losses, np.log10(lrs))
    i_steep = int(np.argmin(grad[:i_min])) if i_min > 2 else 0
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.semilogx(lrs, losses, color=BLUE, lw=2.5)
    ax.axvline(lrs[i_steep], color=GREEN, ls="--", lw=2, label=f"steepest descent ≈ {lrs[i_steep]:.0e}")
    ax.axvline(lrs[i_min], color=RED, ls=":", lw=2, label=f"loss minimum ≈ {lrs[i_min]:.0e}")
    ax.set_xlabel("learning rate η (increased exponentially every mini-batch)")
    ax.set_ylabel("smoothed training loss")
    ax.set_title("LR range test: pick η ~ steepest region, ~10× below the minimum")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "lr_finder.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_backprop_worked_example()
    fig_init_activation_stats()
    fig_loss_curve_diagnostics()
    fig_lr_finder()
    print("Figures written to", OUT)
