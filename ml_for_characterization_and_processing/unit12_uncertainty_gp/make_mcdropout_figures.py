"""Generate the Case-study-B MC-Dropout segmentation figures (slides 14-18)
from the real MetalDAM dataset (ArcelorMittal/DaSCI, 42 labeled SEM
micrographs of additively manufactured steel, 5 classes incl. defects).

The "SEM #2" tool shift is SIMULATED (detector gamma/contrast/brightness/
noise perturbation of the held-out micrographs) and must be labeled as
such on the slide.

Run from the unit folder with the repo venv:
    ../../.venv/bin/python make_mcdropout_figures.py
Downloads MetalDAM into data_metaldam/ on first run (~100 MB, GitHub
release of ari-dasci/OD-MetalDAM). Writes PNGs into images/ and prints
the measured numbers quoted on the slides.
"""

import os
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

CLASSES = ["matrix", "austenite", "mart./aust.", "precipitate", "defect"]
N_CLS = 5
DEFECT = 4
CLASS_COLORS = ["#9e9e9e", "#4878a8", "#ff9d3c", "#4daf4a", "#d62728"]
CLS_CMAP = ListedColormap(CLASS_COLORS)

DATA_DIR = Path("data_metaldam")
URL = ("https://github.com/ari-dasci/OD-MetalDAM/releases/download/1.0/"
       "MetalDAM_labeled.zip")

# ---------------------------------------------------------------- data
def fetch_data():
    if not (DATA_DIR / "MetalDAM" / "images").exists():
        DATA_DIR.mkdir(exist_ok=True)
        zp = DATA_DIR / "MetalDAM_labeled.zip"
        if not zp.exists():
            print("downloading MetalDAM ...")
            urllib.request.urlretrieve(URL, zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(DATA_DIR)


def load_images():
    imgs, labs, names = [], [], []
    img_dir = DATA_DIR / "MetalDAM" / "images"
    lab_dir = DATA_DIR / "MetalDAM" / "labels"
    for f in sorted(img_dir.iterdir()):
        lab = np.array(Image.open(lab_dir / (f.stem + ".png")))
        if lab.ndim == 3:  # one file ships as RGB with identical channels
            lab = lab[..., 0]
        img = np.array(Image.open(f).convert("L")).astype(np.float32) / 255.0
        img = img[: lab.shape[0], : lab.shape[1]]  # crop info band
        imgs.append(img)
        labs.append(lab.astype(np.int64))
        names.append(f.stem)
    return imgs, labs, names


def simulate_tool_shift(img):
    """SEM #2 (simulated): different detector gain/bias plus slight focus
    drift -> gamma + contrast + brightness shift, mild blur, extra noise."""
    out = np.clip(img, 0, 1) ** 0.70           # gamma (detector response)
    out = (out - 0.5) * 1.35 + 0.5 + 0.05      # contrast gain + brightness bias
    k = np.array([0.25, 0.5, 0.25])            # ~0.7 px separable blur
    out = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, out)
    out = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, out)
    out = out + rng.normal(0, 0.035, out.shape)  # extra shot/readout noise
    return np.clip(out, 0, 1).astype(np.float32)


def make_tiles(imgs, labs, size=256, stride=128):
    X, Y = [], []
    for im, lb in zip(imgs, labs):
        H, W = lb.shape
        ys = list(range(0, H - size + 1, stride))
        xs = list(range(0, W - size + 1, stride))
        if ys[-1] != H - size:
            ys.append(H - size)
        if xs[-1] != W - size:
            xs.append(W - size)
        for y in ys:
            for x in xs:
                X.append(im[y : y + size, x : x + size])
                Y.append(lb[y : y + size, x : x + size])
    return np.stack(X), np.stack(Y)


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout, p_drop=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        ]
        if p_drop > 0:
            layers.append(nn.Dropout2d(p_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Small U-Net; Dropout2d(p) in the bottleneck and every decoder block
    (the placement recommended on slide 14)."""

    def __init__(self, n_cls=N_CLS, base=32, p_drop=0.2):
        super().__init__()
        b = base
        self.e1 = Block(1, b)
        self.e2 = Block(b, 2 * b)
        self.e3 = Block(2 * b, 4 * b)
        self.e4 = Block(4 * b, 8 * b)
        self.bott = Block(8 * b, 16 * b, p_drop)
        self.up4 = nn.ConvTranspose2d(16 * b, 8 * b, 2, 2)
        self.d4 = Block(16 * b, 8 * b, p_drop)
        self.up3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2)
        self.d3 = Block(8 * b, 4 * b, p_drop)
        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2)
        self.d2 = Block(4 * b, 2 * b, p_drop)
        self.up1 = nn.ConvTranspose2d(2 * b, b, 2, 2)
        self.d1 = Block(2 * b, b, p_drop)
        self.head = nn.Conv2d(b, n_cls, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        bt = self.bott(self.pool(e4))
        d4 = self.d4(torch.cat([self.up4(bt), e4], 1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


def enable_dropout(model):
    """Keep ONLY the dropout layers stochastic at inference (slide 14
    anti-pattern: model.eval() silently switches dropout off)."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


# ---------------------------------------------------------------- train
def train(model, Xtr, Ytr, epochs=60, bs=16):
    freq = np.bincount(Ytr.ravel(), minlength=N_CLS) / Ytr.size
    w = 1.0 / np.sqrt(freq + 1e-6)
    w = w / w.sum() * N_CLS
    print("class freq:", np.round(freq, 4), " CE weights:", np.round(w, 2))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEV))
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb = torch.from_numpy(Xtr[idx]).unsqueeze(1).to(DEV)
            yb = torch.from_numpy(Ytr[idx]).to(DEV)
            # augmentation: random flips / 90-deg rotations
            k = int(rng.integers(4))
            xb = torch.rot90(xb, k, (2, 3)); yb = torch.rot90(yb, k, (1, 2))
            if rng.random() < 0.5:
                xb = torch.flip(xb, (3,)); yb = torch.flip(yb, (2,))
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep+1:3d}  loss {tot/n:.4f}")
    return model


# ------------------------------------------------------- MC-dropout inference
@torch.no_grad()
def mc_predict(model, img, T=30, temp=1.0):
    """T stochastic passes on a full (padded) micrograph.
    Returns mean softmax probs (logits scaled by `temp` per pass) [C,H,W]
    and mean logits [C,H,W]."""
    H, W = img.shape
    ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
    x = torch.from_numpy(img).float()[None, None].to(DEV)
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    model.eval()
    enable_dropout(model)
    probs = torch.zeros(N_CLS, x.shape[2], x.shape[3], device=DEV)
    logits_sum = torch.zeros_like(probs)
    for _ in range(T):
        lg = model(x)[0]
        probs += F.softmax(lg / temp, dim=0)
        logits_sum += lg
    probs /= T
    logits_sum /= T
    return probs[:, :H, :W].cpu().numpy(), logits_sum[:, :H, :W].cpu().numpy()


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(0)


# ---------------------------------------------------------------- calibration
def fit_temperature(logits_list, label_list, n_sub=300_000):
    """Fit a single scalar temperature on (mean) logits, LBFGS on NLL."""
    lg = np.concatenate([l.reshape(N_CLS, -1).T for l in logits_list])
    yy = np.concatenate([y.ravel() for y in label_list])
    sub = rng.choice(len(yy), min(n_sub, len(yy)), replace=False)
    lg = torch.from_numpy(lg[sub]).float().to(DEV)
    yy = torch.from_numpy(yy[sub]).to(DEV)
    logT = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / torch.exp(logT), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).item())


def reliability(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    mids, accs, weights = [], [], []
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc = correct[m].mean()
        c = conf[m].mean()
        w = m.mean()
        ece += w * abs(acc - c)
        mids.append(c); accs.append(acc); weights.append(w)
    return np.array(mids), np.array(accs), np.array(weights), ece


# =====================================================================
def main():
    fetch_data()
    imgs, labs, names = load_images()
    order = rng.permutation(len(imgs))
    tr_idx, cal_idx, te_idx = order[:30], order[30:34], order[34:]
    print(f"split: train {len(tr_idx)} / calibration {len(cal_idx)} / test {len(te_idx)} micrographs")
    print("test micrographs:", [names[i] for i in te_idx])

    Xtr, Ytr = make_tiles([imgs[i] for i in tr_idx], [labs[i] for i in tr_idx])
    print(f"train tiles: {len(Xtr)} (256x256)")

    model = UNet().to(DEV)
    ckpt = Path("mcdropout_unet.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEV))
        print("loaded checkpoint", ckpt)
    else:
        train(model, Xtr, Ytr)
        torch.save(model.state_dict(), ckpt)

    # ---------------- MC dropout on held-out micrographs, both "tools"
    T = 30
    res = {}
    for tool, shift in [("SEM1", False), ("SEM2", True)]:
        probs_l, logits_l, ys = [], [], []
        for i in te_idx:
            im = simulate_tool_shift(imgs[i]) if shift else imgs[i]
            p, lg = mc_predict(model, im, T=T)
            probs_l.append(p); logits_l.append(lg); ys.append(labs[i])
        res[tool] = (probs_l, logits_l, ys)

    # accuracy / IoU on both tools
    for tool in ("SEM1", "SEM2"):
        probs_l, _, ys = res[tool]
        pred = [p.argmax(0) for p in probs_l]
        yall = np.concatenate([y.ravel() for y in ys])
        pall = np.concatenate([p.ravel() for p in pred])
        acc = (yall == pall).mean()
        ious = []
        for c in range(N_CLS):
            inter = ((yall == c) & (pall == c)).sum()
            union = ((yall == c) | (pall == c)).sum()
            ious.append(inter / union if union else np.nan)
        print(f"\n{tool} test: pixel acc {acc:.3f}, mIoU {np.nanmean(ious):.3f}, "
              f"IoU per class {np.round(ious,3)}")

    # per-tile timing for the compute-budget note (slide 15)
    tile = imgs[te_idx[0]][:256, :256]
    mc_predict(model, tile, T=T)  # warm-up
    t0 = time.time(); mc_predict(model, tile, T=T)
    dt_gpu = time.time() - t0
    print(f"T={T} on one 256x256 tile [{DEV}, "
          f"{torch.cuda.get_device_name() if DEV=='cuda' else 'cpu'}]: {dt_gpu:.2f} s")

    # ---------------- figure 1: input | prediction | entropy maps
    # pick the two test micrographs with the most defect pixels
    dcount = [ (labs[i] == DEFECT).sum() for i in te_idx ]
    show = np.argsort(dcount)[-2:][::-1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    for r, k in enumerate(show):
        i = te_idx[k]
        p = res["SEM1"][0][k]
        H = entropy(p) / np.log(N_CLS)  # normalized [0,1]
        axes[r, 0].imshow(imgs[i], cmap="gray")
        axes[r, 0].set_title("SEM input" if r == 0 else None)
        axes[r, 1].imshow(p.argmax(0), cmap=CLS_CMAP, vmin=-0.5, vmax=N_CLS - 0.5,
                          interpolation="nearest")
        axes[r, 1].set_title("MC-mean prediction" if r == 0 else None)
        im2 = axes[r, 2].imshow(H, cmap="inferno", vmin=0, vmax=H.max())
        axes[r, 2].set_title("predictive entropy" if r == 0 else None)
        fig.colorbar(im2, ax=axes[r, 2], fraction=0.037, pad=0.02)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in CLASS_COLORS]
    axes[1, 1].legend(handles, CLASSES, loc="upper center",
                      bbox_to_anchor=(0.5, -0.03), ncol=5, fontsize=10,
                      frameon=False)
    fig.tight_layout()
    fig.savefig("images/mcdropout_maps.png")
    plt.close(fig)

    # ---------------- figure 2: reject-for-review operating curve
    # Training-tool test set only: under a strong tool shift the model
    # over-predicts the defect class, which inflates recall while precision
    # collapses -- that story belongs on the calibration slide, not here.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    probs_l, _, ys = res["SEM1"]
    Hs = np.concatenate([entropy(p).ravel() for p in probs_l])
    yy = np.concatenate([y.ravel() for y in ys])
    pp = np.concatenate([p.argmax(0).ravel() for p in probs_l])
    is_def = yy == DEFECT
    auto_ok = (pp == DEFECT) & is_def
    auto_recall = auto_ok[is_def].mean()
    order_H = np.argsort(-Hs)  # highest entropy first
    qs = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05,
                   0.08, 0.12, 0.20])
    rec = []
    flagged = np.zeros(len(Hs), bool)
    for q in qs:
        k = int(q * len(Hs))
        flagged[:] = False
        flagged[order_H[:k]] = True
        rec.append((auto_ok | flagged)[is_def].mean())
    rec = np.array(rec)
    ax.axhline(auto_recall * 100, color="gray", lw=1.2, ls="--")
    ax.text(0.6, auto_recall * 100 + 0.12,
            f"fully automatic (no review): {auto_recall*100:.1f}%",
            fontsize=11, color="gray")
    ax.plot(qs * 100, rec * 100, "-", color="#1f77b4", marker="o",
            label="entropy-ranked review, held-out micrographs")
    for q_mark in (0.01, 0.05):
        r_mark = rec[qs == q_mark][0]
        ax.annotate(f"{r_mark*100:.1f}%", (q_mark * 100, r_mark * 100),
                    textcoords="offset points", xytext=(6, -16),
                    fontsize=11, color="#1f77b4")
    ax.set_xscale("log")
    ax.set_xlabel("human-review rate (% of pixels flagged by entropy)")
    ax.set_ylabel("defect-pixel recall (%)")
    ax.axvline(5, color="gray", lw=0.8, ls=":")
    ax.axvline(1, color="gray", lw=0.8, ls=":")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("images/mcdropout_operating_curve.png")
    plt.close(fig)
    print(f"operating curve SEM1: auto-only defect recall {auto_recall*100:.1f}%")
    for q_mark in (0.01, 0.05, 0.20):
        print(f"operating curve SEM1: review {q_mark*100:g}% -> "
              f"defect recall {rec[qs == q_mark][0]*100:.1f}%")

    # ---------------- figure 3: tool-shift reliability + temperature scaling
    # temperature fitted on the small calibration set, acquired on the NEW tool
    cal_logits, cal_ys = [], []
    for i in cal_idx:
        _, lg = mc_predict(model, simulate_tool_shift(imgs[i]), T=T)
        cal_logits.append(lg); cal_ys.append(labs[i])
    T_scal = fit_temperature(cal_logits, cal_ys)
    print(f"fitted temperature on {len(cal_idx)} shifted calibration "
          f"micrographs: T = {T_scal:.2f}")

    # re-run MC inference on SEM2 with the fitted temperature applied per pass,
    # so before/after use the identical predictive-mean construction
    probs2c = []
    for i in te_idx:
        p, _ = mc_predict(model, simulate_tool_shift(imgs[i]), T=T, temp=T_scal)
        probs2c.append(p)

    panels = []
    probs_l, _, ys = res["SEM1"]
    panels.append(("SEM #1 (training tool)", probs_l, ys))
    probs2, logits2, ys2 = res["SEM2"]
    panels.append(("SEM #2 (simulated shift)", probs2, ys2))
    panels.append((f"SEM #2, temp-scaled (T={T_scal:.2f})", probs2c, ys2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (title, pl, yl) in zip(axes, panels):
        conf = np.concatenate([p.max(0).ravel() for p in pl])
        corr = np.concatenate([(p.argmax(0) == y).ravel() for p, y in zip(pl, yl)])
        mids, accs, w, ece = reliability(conf, corr.astype(float))
        ax.bar(mids, accs, width=0.055, color="#4878a8", edgecolor="k",
               linewidth=0.4, label="observed accuracy")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("confidence")
        ax.text(0.04, 0.93, f"ECE = {ece*100:.1f}%", transform=ax.transAxes,
                fontsize=12, va="top",
                bbox=dict(fc="white", ec="gray", alpha=0.8))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        print(f"reliability {title}: ECE {ece*100:.2f}%")
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig("images/mcdropout_toolshift_reliability.png")
    plt.close(fig)

    print("\nwrote images/mcdropout_maps.png, images/mcdropout_operating_curve.png,"
          " images/mcdropout_toolshift_reliability.png")


if __name__ == "__main__":
    main()
