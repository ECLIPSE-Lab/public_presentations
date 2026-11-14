"""Build xp/data/models.json: every model result of the Week 5 deck, from xp/data/columns.json.

Usage (needs numpy, scikit-learn, shap):
    uv run --no-project --with scikit-learn==1.7.0 --with shap python xp/prep/make_models.py
The script prints the numbers that the slides and the speaker notes quote.

Task: classify each candidate of the column finder as substrate column / film column / no column.
Rows: 1 934 labelled candidates. Groups: the four images (A-D). Two validation schemes:
  random : stratified 5-fold over candidates (candidates of one image are in training AND test)
  image  : leave one image out (the claim "works on a new image")
Parts of the output
  ladder   : accuracy of six models x four feature sets x two schemes, with the test predictions
  depth    : accuracy against tree depth (training, random folds, new image)
  votes    : the vote of every tree for the candidates of image B (bagging and random forest)
  ens      : accuracy against the number of trees, tree correlation rho
  boost    : log-loss against the boosting stage for four learning rates (new image)
  imp      : MDI, permutation importance on training images and on the new image; a noise feature
  corr     : permutation importance with exact copies of d4; accuracy after dropping d1, d4 or both
  shap     : TreeSHAP values of the random forest for the candidates of image B
"""
import base64, json, warnings
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
XP = Path(__file__).resolve().parents[1]


def dec(v):
    a = np.frombuffer(base64.b64decode(v["b64"]), dtype={"f4": "<f4", "u1": "u1", "u2": "<u2", "i2": "<i2"}[v["dtype"]])
    return a.reshape(v["shape"]).astype(float if v["dtype"] == "f4" else int)


def arr(a, dtype="f4"):
    a = np.ascontiguousarray(np.asarray(a).astype({"f4": "<f4", "u1": "u1", "u2": "<u2", "i2": "<i2"}[dtype]))
    return {"shape": list(a.shape), "dtype": dtype, "b64": base64.b64encode(a.tobytes()).decode()}


D = json.loads((XP / "data" / "columns.json").read_text())
names = D["names"]
Xall, yall, gall, A = dec(D["X"]), dec(D["y"]), dec(D["g"]), dec(D["acsf"])
idx = np.nonzero(yall >= 0)[0]
X, y, g, A = Xall[idx], yall[idx], gall[idx], A[idx]
N = len(y)
col = lambda *c: X[:, [names.index(q) for q in c]]
GEOM = ["I_rel", "d1", "d4", "d_spread", "n_near", "psi4"]
SETS = {
    "raw": (["amp", "bg", "sigma"], col("amp", "bg", "sigma")),
    "all": (names, X),
    "invariant": (GEOM, col(*GEOM)),
    "acsf": (["I_rel"] + [f"G2({r:.1f})" for r in D["acsf_rs"]], np.c_[col("I_rel"), A]),
}
MODELS = {
    "majority": lambda: DummyClassifier(),
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "tree (depth 3)": lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
    "tree (full)": lambda: DecisionTreeClassifier(random_state=0),
    "random forest": lambda: RandomForestClassifier(300, random_state=0, n_jobs=8),
    "boosting": lambda: HistGradientBoostingClassifier(random_state=0),
}
RANDOM = StratifiedKFold(5, shuffle=True, random_state=0)
LOGO = LeaveOneGroupOut()
out = {"idx": arr(idx, "u2"), "n": N, "sets": {k: v[0] for k, v in SETS.items()}, "models": list(MODELS)}
print(f"{N} labelled candidates; classes {np.bincount(y)}; per image {np.bincount(g)}")

# ---------------------------------------------------------------------------------------------- ladder
ladder = {}
for sn, (_, F) in SETS.items():
    print(f"== features: {sn} ({F.shape[1]})")
    ladder[sn] = {}
    for mn, mk in MODELS.items():
        pr = cross_val_predict(mk(), F, y, cv=RANDOM)
        pi = cross_val_predict(mk(), F, y, cv=LOGO, groups=g)
        per = [accuracy_score(y[g == i], pi[g == i]) for i in range(4)]
        ladder[sn][mn] = {"random": accuracy_score(y, pr), "image": accuracy_score(y, pi), "image_bal": balanced_accuracy_score(y, pi),
                          "per_image": per, "pred_random": arr(pr, "u1"), "pred_image": arr(pi, "u1")}
        print(f"  {mn:15s} random {accuracy_score(y, pr):.3f} | new image {accuracy_score(y, pi):.3f} "
              f"(balanced {balanced_accuracy_score(y, pi):.3f}) per image {np.round(per, 2)}")
    if sn in ("raw", "invariant"):
        print("  confusion (random forest, new image; rows = truth):\n", confusion_matrix(y, dec(ladder[sn]["random forest"]["pred_image"])))
out["ladder"] = ladder

# ---------------------------------------------------------------------------------------------- depth
depth = {}
for sn in ("raw", "invariant"):
    F = SETS[sn][1]; rows = []
    for d in range(1, 15):
        m = DecisionTreeClassifier(max_depth=d, random_state=0)
        tr = accuracy_score(y, m.fit(F, y).predict(F))
        rows.append([d, tr, accuracy_score(y, cross_val_predict(m, F, y, cv=RANDOM)),
                     accuracy_score(y, cross_val_predict(m, F, y, cv=LOGO, groups=g)), int(m.get_n_leaves())])
    depth[sn] = rows
    print(f"depth curve ({sn}): depth, train, random, new image, leaves\n", np.round(rows, 3)[[0, 1, 2, 3, 5, 9, 13]])
out["depth"] = depth

# ---------------------------------------------------------------------------------------------- ensemble
B = 200
F = SETS["invariant"][1]
ens = {}
for kind, mf in (("bagging", None), ("random forest", "sqrt")):
    acc = np.zeros((4, B)); rho = []; s2 = []; single = []
    for i in range(4):
        tr, te = g != i, g == i
        rf = RandomForestClassifier(B, max_features=mf, random_state=0, n_jobs=8).fit(F[tr], y[tr])
        P = np.stack([t.predict_proba(F[te]) for t in rf.estimators_])          # (B, n, 3)
        cum = np.cumsum(P, 0)
        acc[i] = [(cum[b].argmax(1) == y[te]).mean() for b in range(B)]
        single.append(np.mean([(P[b].argmax(1) == y[te]).mean() for b in range(B)]))
        # correlation of the trees: P(film) of tree a against tree b over the test candidates
        C = np.corrcoef(P[:60, :, 1]); rho.append(C[np.triu_indices(60, 1)].mean()); s2.append(P[:, :, 1].var(0).mean())
        if i == 1:
            ens_votes = P.argmax(2)
            out.setdefault("votes", {})[kind] = arr(ens_votes, "u1")
    w = np.bincount(g) / N
    ens[kind] = {"acc": (acc * w[:, None]).sum(0).tolist(), "rho": float(np.mean(rho)), "single": float(np.dot(single, w)),
                 "var_tree": float(np.mean(s2))}
    a = ens[kind]["acc"]
    print(f"{kind}: one tree {ens[kind]['single']:.3f}; B=1 {a[0]:.3f}, 5 {a[4]:.3f}, 10 {a[9]:.3f}, 30 {a[29]:.3f}, 100 {a[99]:.3f}, 200 {a[-1]:.3f}; "
          f"tree correlation rho = {ens[kind]['rho']:.2f}")
out["ens"] = ens
out["votes_rows"] = arr(np.nonzero(g == 1)[0], "u2")

# ---------------------------------------------------------------------------------------------- boosting
STAGES = 300
boost = {}
for eta in (0.03, 0.1, 0.3, 1.0):
    te_l, tr_l, te_a = np.zeros(STAGES), np.zeros(STAGES), np.zeros(STAGES)
    for i in range(4):
        tr, te = g != i, g == i
        m = GradientBoostingClassifier(n_estimators=STAGES, learning_rate=eta, max_depth=2, random_state=0).fit(F[tr], y[tr])
        w = te.sum() / N
        for s, (pa, pb) in enumerate(zip(m.staged_predict_proba(F[te]), m.staged_predict_proba(F[tr]))):
            te_l[s] += w * log_loss(y[te], pa, labels=[0, 1, 2]); tr_l[s] += w * log_loss(y[tr], pb, labels=[0, 1, 2])
            te_a[s] += w * (pa.argmax(1) == y[te]).mean()
    boost[str(eta)] = {"test": te_l.tolist(), "train": tr_l.tolist(), "acc": te_a.tolist()}
    b = int(np.argmin(te_l))
    print(f"boosting eta={eta}: best stage {b + 1} (new-image log-loss {te_l[b]:.3f}, accuracy {te_a[b]:.3f}); "
          f"stage {STAGES}: log-loss {te_l[-1]:.3f}, accuracy {te_a[-1]:.3f}, training log-loss {tr_l[-1]:.4f}")
out["boost"] = boost

# ---------------------------------------------------------------------------------------------- importance
rng = np.random.default_rng(0)
Fn = np.c_[X, rng.normal(size=N)]; fn = names + ["noise"]
mdi, ptr, pte, acc_te = [], [], [], []
for i in range(4):
    tr, te = g != i, g == i
    rf = RandomForestClassifier(300, random_state=0, n_jobs=8).fit(Fn[tr], y[tr])
    mdi.append(rf.feature_importances_)
    ptr.append(permutation_importance(rf, Fn[tr], y[tr], n_repeats=10, random_state=0, n_jobs=8).importances_mean)
    pte.append(permutation_importance(rf, Fn[te], y[te], n_repeats=10, random_state=0, n_jobs=8).importances_mean)
    acc_te.append(rf.score(Fn[te], y[te]))
mdi, ptr, pte = np.array(mdi), np.array(ptr), np.array(pte)
out["imp"] = {"names": fn, "mdi": mdi.mean(0).tolist(), "mdi_sd": mdi.std(0).tolist(), "perm_train": ptr.mean(0).tolist(),
              "perm_test": pte.mean(0).tolist(), "perm_test_sd": pte.std(0).tolist(), "perm_test_folds": pte.tolist()}
print("importance (random forest on all 11 features + 1 noise feature; mean over 4 held-out images)")
print("  feature      MDI    perm(train)  perm(new image) +- sd")
for j in np.argsort(-mdi.mean(0)):
    print(f"  {fn[j]:10s} {mdi.mean(0)[j]:.3f}   {ptr.mean(0)[j]:+.3f}      {pte.mean(0)[j]:+.3f} +- {pte.std(0)[j]:.3f}")
print("  accuracy on the new image:", np.round(acc_te, 3))

# correlated features: add exact copies of d4; drop d1, d4, or both
def perm_new_image(Fx):
    p = []
    for i in range(4):
        tr, te = g != i, g == i
        r = RandomForestClassifier(300, random_state=0, n_jobs=8).fit(Fx[tr], y[tr])
        p.append(permutation_importance(r, Fx[te], y[te], n_repeats=10, random_state=0, n_jobs=8).importances_mean)
    return np.mean(p, 0)
def acc_new_image(Fx):
    return float(sum(RandomForestClassifier(300, random_state=0, n_jobs=8).fit(Fx[g != i], y[g != i]).score(Fx[g == i], y[g == i])
                     * (g == i).sum() / N for i in range(4)))
k4 = GEOM.index("d4")
p0, p1, p2 = perm_new_image(F), perm_new_image(np.c_[F, F[:, k4]]), perm_new_image(np.c_[F, F[:, k4], F[:, k4]])
drop = {}
for dn in ([], ["d1"], ["d4"], ["d1", "d4"]):
    drop[" + ".join(dn) or "none"] = acc_new_image(F[:, [q for q, n in enumerate(GEOM) if n not in dn]])
out["corr"] = {"names": GEOM, "p0": p0.tolist(), "p1": p1.tolist(), "p2": p2.tolist(), "drop": drop,
               "r_d1_d4": float(np.corrcoef(F[:, 1], F[:, 2])[0, 1])}
print("correlated features: permutation importance on the new image; no copy / one exact copy of d4 / two copies")
for q, n in enumerate(GEOM):
    print(f"  {n:9s} {p0[q]:+.3f}  {p1[q]:+.3f}  {p2[q]:+.3f}")
print(f"  the copies themselves: {p1[-1]:+.3f} | {p2[-2]:+.3f}, {p2[-1]:+.3f}")
print("  accuracy on the new image after dropping features:", {k: round(v, 3) for k, v in drop.items()})
print(f"  correlation of d1 and d4: r = {out['corr']['r_d1_d4']:.2f}")

# ---------------------------------------------------------------------------------------------- SHAP
import shap
tr, te = g != 1, g == 1
rf = RandomForestClassifier(300, random_state=0, n_jobs=8, min_samples_leaf=3).fit(F[tr], y[tr])
ex = shap.TreeExplainer(rf)
sv = np.asarray(ex.shap_values(F[te]))
if sv.shape[0] == 3 and sv.ndim == 3 and sv.shape[1] == te.sum():
    sv = np.moveaxis(sv, 0, 2)                                             # -> (n, features, classes)
base = np.asarray(ex.expected_value).ravel()
proba = rf.predict_proba(F[te])
err = np.abs(sv.sum(1) + base[None] - proba).max()
out["shap"] = {"names": GEOM, "values": arr(sv), "base": base.tolist(), "proba": arr(proba), "x": arr(F[te]),
               "rows": arr(np.nonzero(te)[0], "u2"), "mean_abs": np.abs(sv).mean(0).tolist()}
print(f"SHAP (random forest, invariant descriptors, trained on A, C, D; explained on image B, {te.sum()} candidates)")
print(f"  base values (mean prediction on the training images): {np.round(base, 3)}; additivity error {err:.1e}")
print(f"  accuracy on image B: {(proba.argmax(1) == y[te]).mean():.3f}")
print("  mean |SHAP| for the class 'film column':", {n: round(float(v), 3) for n, v in zip(GEOM, np.abs(sv[:, :, 1]).mean(0))})

(XP / "data" / "models.json").write_text(json.dumps(out))
print("models.json", (XP / "data" / "models.json").stat().st_size // 1024, "kB")
