"""Reproducible figures for DSEM Week 7 (CNNs & U-Nets for microscopy).

Run from the repo root:
    .venv/bin/python data_science_for_em/07_cnns_for_microscopy/img/make_figures.py

Generates (into this img/ folder):
    equivariance_vs_rotation.png  - translation equivariance holds, rotation equivariance does not
    iou_dice_imbalance.png        - IoU/Dice geometry + accuracy trap on an imbalanced mask
    unet_tiny_results.png         - tiny U-Net on synthetic particles (test examples)
    unet_failure_saliency.png     - OOD failure (scratch + oversized particle) + saliency of false positives
    shortcut_xai.png              - shortcut-learning demo: gradient saliency, Grad-CAM, occlusion

The data generators / models are the same as in notebooks/week07_cnn_segmentation.ipynb
(same seeds), so the numbers printed here match the notebook.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import convolve, gaussian_filter
from skimage.filters import threshold_otsu

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12})


# ----------------------------------------------------------------------------
# 1. Translation vs rotation equivariance
# ----------------------------------------------------------------------------
def fig_equivariance():
    rng = np.random.default_rng(0)
    H = 64
    yy, xx = np.mgrid[:H, :H]
    img = 0.3 + 0.03 * rng.standard_normal((H, H))
    img[(yy - 22) ** 2 + ((xx - 20) / 1.8) ** 2 < 60] += 0.5  # elongated particle
    img[(yy > 40) & (xx > 36) & (xx < 44)] += 0.35  # vertical rod
    k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)  # vertical Sobel
    f = lambda z: convolve(z, k, mode="wrap")
    shift = lambda z: np.roll(z, (8, 12), axis=(0, 1))
    rot = lambda z: np.rot90(z)

    d_t = np.abs(f(shift(img)) - shift(f(img))).max()
    d_r = np.abs(f(rot(img)) - rot(f(img))).max()
    rows = [
        ("translation $T$", shift, d_t),
        ("rotation $R$ (90°)", rot, d_r),
    ]
    fig, ax = plt.subplots(2, 5, figsize=(17, 7.2))
    for r, (name, g, d) in enumerate(rows):
        fm = f(img)
        v = np.abs(fm).max()
        panels = [
            (img, "gray", "input $x$"),
            (g(img), "gray", f"transformed input\n{name}"),
            (f(g(img)), "RdBu_r", "$f(g\\,x)$: conv of\ntransformed input"),
            (g(fm), "RdBu_r", "$g\\,f(x)$: transformed\nfeature map"),
            (f(g(img)) - g(fm), "RdBu_r", f"difference\nmax |Δ| = {d:.2f}"),
        ]
        for a, (z, cm, t) in zip(ax[r], panels):
            if cm == "gray":
                a.imshow(z, cmap=cm)
            else:
                a.imshow(z, cmap=cm, vmin=-v, vmax=v)
            a.set_title(t)
            a.axis("off")
    ax[0, 0].text(-0.25, 0.5, "equivariant ✓", transform=ax[0, 0].transAxes, rotation=90,
                  va="center", ha="center", fontsize=14, color="tab:green")
    ax[1, 0].text(-0.25, 0.5, "not equivariant ✗", transform=ax[1, 0].transAxes, rotation=90,
                  va="center", ha="center", fontsize=14, color="tab:red")
    fig.suptitle("Convolution (vertical Sobel kernel $f$) commutes with shifts, but not with rotations", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "equivariance_vs_rotation.png", dpi=130)
    plt.close(fig)
    print(f"[equivariance] max|diff| translation={d_t:.2e}  rotation={d_r:.2f}")


# ----------------------------------------------------------------------------
# 2. Synthetic particle segmentation + tiny U-Net (same as notebook)
# ----------------------------------------------------------------------------
IMG = 48
SEED = 7


def make_particle_image(rng, n_min=2, n_max=6, r_min=2.5, r_max=6.0, dose=40.0, scratch=False, big=False):
    """Noisy EM-like image with bright particles on a textured, unevenly lit support."""
    yy, xx = np.mgrid[:IMG, :IMG]
    bg = 0.30 + gaussian_filter(rng.standard_normal((IMG, IMG)), 3) * 0.25
    g = rng.uniform(-0.12, 0.12, 2)
    bg += g[0] * (yy / IMG - 0.5) + g[1] * (xx / IMG - 0.5)
    mask = np.zeros((IMG, IMG), bool)
    obj = np.zeros((IMG, IMG))
    for _ in range(rng.integers(n_min, n_max + 1)):
        r = rng.uniform(r_min, r_max)
        cy, cx = rng.uniform(r, IMG - r, 2)
        e = rng.uniform(0.7, 1.3)
        d = ((yy - cy) / (r * e)) ** 2 + ((xx - cx) * e / r) ** 2 <= 1
        mask |= d
        obj[d] = np.maximum(obj[d], rng.uniform(0.15, 0.40))
    if big:
        d = (yy - 24) ** 2 + (xx - 24) ** 2 <= 11 ** 2
        mask |= d
        obj[d] = 0.30
    img = bg + gaussian_filter(obj, 0.8)
    if scratch:
        img = img + 0.30 * (np.abs(yy - 0.9 * xx - 5) < 1.2)
    img = np.clip(img, 0.01, None)
    img = rng.poisson(img * dose) / dose
    return img.astype(np.float32), mask.astype(np.float32)


def make_set(n, seed, **kw):
    rng = np.random.default_rng(seed)
    X, Y = zip(*[make_particle_image(rng, **kw) for _ in range(n)])
    return np.stack(X), np.stack(Y)


def block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    def __init__(self, c=8, use_skips=True):
        super().__init__()
        self.use_skips = use_skips
        k = 2 if use_skips else 1
        self.enc1, self.enc2, self.bott = block(1, c), block(c, 2 * c), block(2 * c, 4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = block(k * 2 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = block(k * c, c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bott(F.max_pool2d(e2, 2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], 1) if self.use_skips else d2)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], 1) if self.use_skips else d1)
        return self.head(d1)


def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    tot = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (tot + eps)).mean()


def iou_dice(pred, target):
    pred, target = pred.astype(bool), target.astype(bool)
    inter = (pred & target).sum()
    union = (pred | target).sum()
    return inter / max(union, 1), 2 * inter / max(pred.sum() + target.sum(), 1)


def train_unet(Xtr, Ytr, epochs=15, lr=3e-3):
    torch.manual_seed(0)
    np.random.seed(0)
    model = TinyUNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, Yt = torch.tensor(Xtr[:, None]), torch.tensor(Ytr[:, None])
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 16):
            idx = perm[i:i + 16]
            xb, yb = Xt[idx], Yt[idx]
            if np.random.rand() < 0.5:
                xb, yb = xb.flip(-1), yb.flip(-1)
            lo = model(xb)
            loss = F.binary_cross_entropy_with_logits(lo, yb) + soft_dice_loss(lo, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def predict(model, X):
    with torch.no_grad():
        return (torch.sigmoid(model(torch.tensor(X[:, None])))[:, 0] > 0.5).numpy()


def fig_segmentation():
    Xtr, Ytr = make_set(256, SEED)
    Xte, Yte = make_set(96, SEED + 2)
    model = train_unet(Xtr, Ytr)
    P = predict(model, Xte)
    iou_u, dice_u = iou_dice(P, Yte)
    acc_u = (P == Yte.astype(bool)).mean()
    P_bg = np.zeros_like(P)
    acc_bg = (P_bg == Yte.astype(bool)).mean()
    P_otsu = np.stack([gaussian_filter(x, 1) > threshold_otsu(gaussian_filter(x, 1)) for x in Xte])
    iou_o, dice_o = iou_dice(P_otsu, Yte)
    acc_o = (P_otsu == Yte.astype(bool)).mean()
    print(f"[unet] foreground fraction test = {Yte.mean():.3f}")
    print(f"[unet] all-background: acc={acc_bg:.3f} IoU=0 Dice=0")
    print(f"[unet] smoothed Otsu:  acc={acc_o:.3f} IoU={iou_o:.3f} Dice={dice_o:.3f}")
    print(f"[unet] tiny U-Net:     acc={acc_u:.3f} IoU={iou_u:.3f} Dice={dice_u:.3f}")

    # ---- IoU / Dice / imbalance figure ----
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1, 1.25]})
    a = ax[0]
    a.add_patch(mpatches.Circle((0.42, 0.5), 0.28, color="tab:blue", alpha=0.45, label="ground truth $A$"))
    a.add_patch(mpatches.Circle((0.62, 0.5), 0.28, color="tab:orange", alpha=0.45, label="prediction $B$"))
    a.text(0.52, 0.5, "$A\\cap B$", ha="center", va="center", fontsize=15)
    a.text(0.5, 0.08, r"IoU $=\frac{|A\cap B|}{|A\cup B|}$      Dice $=\frac{2|A\cap B|}{|A|+|B|}=\frac{2\,\mathrm{IoU}}{1+\mathrm{IoU}}$",
           ha="center", fontsize=15)
    a.set_xlim(0, 1.04)
    a.set_ylim(0, 1)
    a.set_aspect("equal")
    a.axis("off")
    a.legend(loc="upper center", ncol=2, frameon=False)
    a.set_title("Overlap metrics ignore the (huge) true-negative background")
    a = ax[1]
    names = ["all background", "smoothed Otsu", "tiny U-Net"]
    accs = [acc_bg, acc_o, acc_u]
    ious = [0.0, iou_o, iou_u]
    x = np.arange(3)
    b1 = a.bar(x - 0.18, accs, 0.36, label="pixel accuracy", color="0.6")
    b2 = a.bar(x + 0.18, ious, 0.36, label="IoU (particle class)", color="tab:blue")
    for bars in (b1, b2):
        for bb in bars:
            a.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.01, f"{bb.get_height():.2f}",
                   ha="center", fontsize=12)
    a.set_xticks(x, names)
    a.set_ylim(0, 1.1)
    a.legend(loc="upper left", frameon=False)
    a.set_title(f"Particles cover {100 * Yte.mean():.0f}% of pixels: accuracy {acc_bg:.2f} for predicting nothing")
    fig.tight_layout()
    fig.savefig(OUT / "iou_dice_imbalance.png", dpi=130)
    plt.close(fig)

    # ---- example predictions ----
    per = np.array([iou_dice(P[i], Yte[i])[0] for i in range(len(Xte))])
    order = np.argsort(per)
    picks = [order[len(order) // 2], order[3 * len(order) // 4], order[0]]
    fig, ax = plt.subplots(3, 3, figsize=(9.5, 9.5))
    for r, i in enumerate(picks):
        for c, (z, t) in enumerate([(Xte[i], "input"), (Yte[i], "ground truth"), (P[i], f"U-Net (IoU {per[i]:.2f})")]):
            ax[r, c].imshow(z, cmap="gray")
            ax[r, c].set_title(t + (" — worst test image" if (r == 2 and c == 2) else ""), fontsize=11)
            ax[r, c].axis("off")
    fig.suptitle(f"Tiny U-Net (29.6k params, 15 epochs CPU): test IoU {iou_u:.2f}, Dice {dice_u:.2f}")
    fig.tight_layout()
    fig.savefig(OUT / "unet_tiny_results.png", dpi=130)
    plt.close(fig)
    print(f"[unet] per-image IoU median={np.median(per):.3f} worst={per.min():.3f}")

    # ---- OOD failure + saliency ----
    rng = np.random.default_rng(99)
    xs, ys = make_particle_image(rng, scratch=True, big=True)
    ps = predict(model, xs[None])[0]
    fp = ps & ~ys.astype(bool)
    fn = ~ps & ys.astype(bool)
    x = torch.tensor(xs[None, None], requires_grad=True)
    lo = model(x)[0, 0]
    lo[torch.tensor(fp)].sum().backward()
    sal = x.grad[0, 0].abs().numpy()
    sal /= sal.max()
    iou_s, _ = iou_dice(ps, ys)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.8))
    ax[0].imshow(xs, cmap="gray")
    ax[0].set_title("OOD input: scratch + oversized particle")
    ax[1].imshow(ys, cmap="gray")
    ax[1].set_title("ground truth")
    ov = np.dstack([fp, ps & ys.astype(bool), fn]).astype(float)
    ax[2].imshow(ov)
    ax[2].set_title(f"prediction: TP green, FP red, FN blue\nIoU {iou_s:.2f}")
    ax[3].imshow(xs, cmap="gray")
    ax[3].imshow(sal, cmap="hot", alpha=0.6)
    ax[3].set_title("saliency of the false-positive logits")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "unet_failure_saliency.png", dpi=130)
    plt.close(fig)
    print(f"[unet] OOD image IoU={iou_s:.3f} FP pixels={fp.sum()} FN pixels={fn.sum()}")


# ----------------------------------------------------------------------------
# 3. Shortcut learning demo (classification) + gradient saliency, Grad-CAM, occlusion
# ----------------------------------------------------------------------------
S, CORNER, R_DEF, AMP = 32, 6, 4, 0.15
yy32, xx32 = np.mgrid[:S, :S]


def bg32(rng, n):
    return np.stack([
        np.clip(0.35 + rng.standard_normal((S, S)) * 0.12
                + gaussian_filter(rng.standard_normal((S, S)) * 0.10, 2.5), 0, 1)
        for _ in range(n)]).astype(np.float32)


def add_defect(rng, X):
    X = X.copy()
    masks = np.zeros((len(X), S, S), bool)
    for i in range(len(X)):
        cy, cx = rng.uniform(R_DEF + 2, S - R_DEF - 2, 2)
        d = (yy32 - cy) ** 2 + (xx32 - cx) ** 2 <= R_DEF ** 2
        X[i][d] += AMP
        masks[i] = d
    return np.clip(X, 0, 1), masks


def add_corner(X):
    X = X.copy()
    X[:, :CORNER, :CORNER] = 0.0  # dark scan-start / detector-shadow artifact
    return X


def build32(seed, n, artifact):
    rng = np.random.default_rng(seed)
    X0 = bg32(rng, n)
    X1, m = add_defect(rng, bg32(rng, n))
    if artifact:
        X1 = add_corner(X1)
    return np.concatenate([X0, X1]), np.array([0] * n + [1] * n), m


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3)))


def train_cls(X, y, epochs=25):
    torch.manual_seed(0)
    m = SmallCNN()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xt, yt = torch.tensor(X[:, None]), torch.tensor(y)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            idx = perm[i:i + 32]
            loss = F.cross_entropy(m(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()


def accuracy(m, X, y):
    with torch.no_grad():
        return (m(torch.tensor(X[:, None])).argmax(1).numpy() == y).mean()


def grad_saliency(m, x):
    x = torch.tensor(x[None, None], requires_grad=True)
    m(x)[0, 1].backward()
    return x.grad[0, 0].abs().numpy()


def grad_cam(m, x):
    x = torch.tensor(x[None, None])
    A = m.features(x)
    A.retain_grad()
    m.head(A.mean((2, 3)))[0, 1].backward()
    w = A.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(S, S), mode="bilinear", align_corners=False)
    return cam[0, 0].detach().numpy()


def occlusion(m, x, patch=4, value=0.35):
    with torch.no_grad():
        p0 = torch.softmax(m(torch.tensor(x[None, None])), 1)[0, 1].item()
        smap = np.zeros((S, S))
        for r in range(0, S, patch):
            for c in range(0, S, patch):
                xo = x.copy()
                xo[r:r + patch, c:c + patch] = value
                p1 = torch.softmax(m(torch.tensor(xo[None, None])), 1)[0, 1].item()
                smap[r:r + patch, c:c + patch] = max(p0 - p1, 0)
    return smap


def fig_shortcut():
    Xc, yc, _ = build32(1, 200, artifact=False)
    Xs, ys, _ = build32(1, 200, artifact=True)
    X_iid, y_te, m_iid = build32(2, 50, artifact=True)   # artifact present (= shortcut training distribution)
    X_def, _, _ = build32(2, 50, artifact=False)           # deployment: defect, no artifact
    clean, short = train_cls(Xc, yc), train_cls(Xs, ys)
    res = {}
    for name, m in [("clean", clean), ("shortcut", short)]:
        res[name] = (accuracy(m, X_iid, y_te), accuracy(m, X_def, y_te))
        print(f"[shortcut] {name:8s}: acc on test WITH artifact={res[name][0]:.2f}  "
              f"acc on defect-only deployment set={res[name][1]:.2f}")

    # explanations on class-1 test images that contain defect + artifact
    idx = np.arange(50, 100)
    corner = np.zeros((S, S), bool)
    corner[:CORNER, :CORNER] = True
    maps = {}
    for name, m in [("clean", clean), ("shortcut", short)]:
        agg = {"grad": 0, "cam": 0, "occ": 0}
        for i in idx:
            for key, fn in [("grad", grad_saliency), ("cam", grad_cam), ("occ", occlusion)]:
                z = fn(m, X_iid[i])
                agg[key] = agg[key] + z / (z.sum() + 1e-12)  # each map as a distribution over pixels
        maps[name] = agg
        # fraction of attribution mass in corner vs in defect (averaged over images)
        fr = {k: v / len(idx) for k, v in agg.items()}
        dmass = np.mean([
            (grad_saliency(m, X_iid[i]) / (grad_saliency(m, X_iid[i]).sum() + 1e-12))[m_iid[i - 50]].sum()
            for i in idx])
        print(f"[shortcut] {name:8s}: grad-saliency mass in corner={fr['grad'][corner].sum():.2f} "
              f"(corner = {corner.mean():.1%} of pixels), in defect disk={dmass:.2f}; "
              f"Grad-CAM corner={fr['cam'][corner].sum():.2f}; occlusion corner={fr['occ'][corner].sum():.2f}")

    i0 = 50 + int(np.argmax([m.sum() > 0 and m[:16, :16].sum() == 0 for m in m_iid]))  # defect away from corner
    fig, ax = plt.subplots(2, 4, figsize=(17, 8.8))
    for r, (name, m) in enumerate([("clean", clean), ("shortcut", short)]):
        x = X_iid[i0]
        panels = [
            (x, "gray", f"{name} model\nacc: w/ artifact {res[name][0]:.2f} | deploy {res[name][1]:.2f}"),
            (grad_saliency(m, x), "hot", "gradient saliency $|\\partial z_1/\\partial x|$"),
            (grad_cam(m, x), "hot", "Grad-CAM (last conv layer, 8×8)"),
            (occlusion(m, x), "hot", "occlusion (4×4 patch)"),
        ]
        for a, (z, cm, t) in zip(ax[r], panels):
            a.imshow(z, cmap=cm)
            a.set_title(t)
            a.axis("off")
            a.add_patch(mpatches.Rectangle((-0.5, -0.5), CORNER, CORNER, ec="cyan", fc="none", lw=2, ls="--"))
            cy, cx = np.argwhere(m_iid[i0 - 50]).mean(0)
            a.add_patch(mpatches.Circle((cx, cy), R_DEF + 0.5, ec="lime", fc="none", lw=2, ls="--"))
    fig.suptitle("Same training accuracy, different evidence: lime = real defect, cyan = scan artifact "
                 "(present in every class-1 training image of the shortcut model)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "shortcut_xai.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_equivariance()
    fig_segmentation()
    fig_shortcut()
