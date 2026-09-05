"""IMC21 slide panels for the MEMS vortex-beam characterization.

Each subfigure of the paper becomes its own standalone figure, with no panel letter, so
the story can be spread over several slides. Nothing is recomputed: every analysis
script in `mems_ptychography/scripts/` writes a companion `.npz` holding all plotted
numbers, and this module only re-plots them at slide scale.

Run:  python vortex_panels.py            (env: PAPER_FIGS, DARK)
"""
from __future__ import annotations

import os

import numpy as np

from talkstyle import BOX, FG, REF, cbar, newfig, savepanel, style

# Where the paper's companion npz live.
PAPER_FIGS = os.environ.get(
    "PAPER_FIGS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "mems_ptychography", "typst", "figures"),
)

QMAX = 2.0          # plot L(q) out to 2 x q_probe, as in the paper (fig_ssnr.QMAX)
R_BF_PX = 38.0      # bright-field disk radius in px (common.R_BF_PX)
# Envelope-map crop fractions of the 192 px grid. The paper uses 0.4 in real space; on a
# slide the ~3 px real-space envelope vanishes in that window, so crop to +-12 px.
FR_R, FR_F = 0.125, 0.6

# Representative charges for the coherence maps, mapped to acquisition stems. From
# fig_coherence.SHOW_L with common.L_MAP under NEGATE_LMAP=1.
COH_REPS = [(-10, "0015"), (-5, "0020"), (-1, "0024"), (0, "0025"),
            (1, "0027"), (5, "0031"), (10, "0036")]


def load(name: str):
    p = os.path.join(PAPER_FIGS, f"{name}.npz")
    if not os.path.exists(p):
        raise SystemExit(
            f"missing {p}\n"
            f"Regenerate it in the paper repo: cd mems_ptychography/scripts && python {name}.py"
        )
    return np.load(p, allow_pickle=True)


def delivered_slope():
    """Programmed -> delivered charge gain (~1.68) from the ptychographic OAM fit.

    `fig_ssnr.npz` and `fig_oam_sorter.npz` store the *programmed* charge; the paper
    multiplies by this slope before plotting, and so must we.
    """
    return float(load("fig_oam_sorter")["fit_mean"][0])


def ell_cmap(ell):
    """Diverging colour map over the delivered-charge range, as in the paper."""
    import matplotlib.pyplot as plt
    lmin, lmax = float(ell.min()), float(ell.max())
    norm = plt.Normalize(lmin, lmax)
    return plt.cm.coolwarm, norm


# --------------------------------------------------------------------------- mode powers

def mode_powers_stacked():
    """Stacked fractional mode powers p_k per delivered charge."""
    d = load("fig_mode_powers")
    powers, ell = d["powers"], d["ell"]
    fig, ax = newfig(9.0, 5.6)
    x = np.arange(len(ell))
    bottom = np.zeros(len(ell))
    for k in range(powers.shape[1]):
        ax.bar(x, powers[:, k], bottom=bottom, label=f"mode {k}")
        bottom += powers[:, k]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(round(l)):+d}" for l in ell], rotation=45, ha="right",
                       fontsize=13)
    ax.set_xlabel(r"delivered charge $\ell$")
    ax.set_ylabel(r"fractional mode power $p_k$")
    ax.set_ylim(0, 1)
    # Above the axes: the bars fill the frame, so an inset legend would sit on the data.
    ax.legend(fontsize=12, ncol=8, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              columnspacing=1.0, handlelength=1.2)
    savepanel(fig, "mode_powers_stacked")


def purity_vs_l():
    """Dominant-mode power, purity and von Neumann entropy vs |ell|, with linear fits."""
    d = load("fig_mode_powers")
    powers, ell, purity = d["powers"], d["ell"], d["purity"]
    al = np.abs(ell)
    fit_p0 = np.polyfit(al, powers[:, 0], 1)
    fit_pu = np.polyfit(al, purity, 1)
    xf = np.array([al.min(), al.max()])

    def vn(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    S = np.array([vn(powers[i]) for i in range(len(powers))])
    fit_S = np.polyfit(al, S, 1)

    fig, ax = newfig(8.0, 5.6)
    ax.plot(al, powers[:, 0], "o", color="C0",
            label=rf"$p_0$ dominant mode: ${fit_p0[0]:+.3f}$ per $\hbar$")
    ax.plot(xf, np.polyval(fit_p0, xf), "-", color="C0", lw=1.6, alpha=0.8)
    ax.plot(al, purity, "s", color="C1",
            label=rf"purity $\mathrm{{Tr}}\,\rho^2$: ${fit_pu[0]:+.3f}$ per $\hbar$")
    ax.plot(xf, np.polyval(fit_pu, xf), "-", color="C1", lw=1.6, alpha=0.8)
    axS = ax.twinx()
    axS.plot(al, S, "^", color="C2", label=rf"entropy $S$: ${fit_S[0]:+.3f}$ per $\hbar$")
    axS.plot(xf, np.polyval(fit_S, xf), "-", color="C2", lw=1.6, alpha=0.8)
    axS.set_ylabel(r"von Neumann entropy $S$ (nats)", color="C2")
    axS.tick_params(axis="y", colors="C2", labelsize=17)
    axS.spines["right"].set_color("C2")
    axS.set_ylim(0, np.log(powers.shape[1]) * 1.03)
    ax.set_xlabel(r"$|\ell|$ (delivered charge)")
    ax.set_ylabel("power / purity")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axS.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=13, loc="lower left")
    savepanel(fig, "purity_vs_l")


# ---------------------------------------------------------------------------- coherence

def _crop(a, frac):
    H, W = a.shape[-2:]
    ry, rx = int(H * frac / 2), int(W * frac / 2)
    return a[H // 2 - ry:H // 2 + ry, W // 2 - rx:W // 2 + rx]


def _coherence_row(domain: str, name: str):
    """One row of |gamma| envelope maps across representative charges."""
    import matplotlib.pyplot as plt
    d = load("fig_coherence")
    slope = float(load("fig_oam_sorter")["fit_mean"][0])
    reps = [(l, s) for l, s in COH_REPS if f"env_{domain}_{s}" in d.files]
    n = len(reps)
    frac = FR_R if domain == "real" else FR_F
    unit = r"$\Delta x$ (px)" if domain == "real" else r"$\Delta k / q_\mathrm{probe}$"

    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.9))
    for ax, (lprog, s) in zip(np.atleast_1d(axes), reps):
        env = _crop(np.abs(d[f"env_{domain}_{s}"]), frac)
        half = env.shape[0] / 2
        if domain != "real":
            half /= R_BF_PX
        im = ax.imshow(env, cmap="magma", vmin=0, vmax=1,
                       extent=(-half, half, -half, half))
        ax.set_title(rf"$\ell = {int(round(slope * lprog)):+d}$", fontsize=18)
        ax.tick_params(labelsize=12)
        ax.set_xlabel(unit, fontsize=15)
    # Both axes carry the same separation coordinate; |gamma| lives on the colourbar.
    np.atleast_1d(axes)[0].set_ylabel(unit, fontsize=17)
    for ax in np.atleast_1d(axes)[1:]:
        ax.set_yticklabels([])
    cbar(fig, list(np.atleast_1d(axes)), im, r"$|\gamma|$", fraction=0.02)
    savepanel(fig, name)


def coherence_real():
    """Sample-plane coherence envelope |gamma(dx)| across charges."""
    _coherence_row("real", "coherence_real")


def coherence_fourier():
    """Aperture-plane coherence envelope |gamma(dk)| across charges."""
    _coherence_row("fourier", "coherence_fourier")


def coherence_widths():
    """Envelope width at |gamma| = 0.5 vs delivered charge, both domains."""
    d = load("fig_coherence")
    ell = d["ell_measured"]
    fig, ax = newfig(8.0, 5.6)
    ax.plot(ell, d["width_real"], "o", color="C0", label="real space (px)")
    ax.set_xlabel(r"delivered charge $\ell$")
    ax.set_ylabel("coherence width at\n" + r"$|\gamma| = 0.5$  (px)", color="C0")
    ax.tick_params(axis="y", colors="C0")
    ax.spines["left"].set_color("C0")
    ax2 = ax.twinx()
    ax2.plot(ell, d["width_fourier_qprobe"], "s", color="C1",
             label=r"aperture plane ($\Delta k/q_\mathrm{probe}$)")
    ax2.set_ylabel(r"width ($\Delta k / q_\mathrm{probe}$)", color="C1")
    ax2.tick_params(axis="y", colors="C1", labelsize=17)
    ax2.spines["right"].set_color("C1")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Both series peak at l = 0, so the bottom centre is the only clear region.
    ax.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=14)
    savepanel(fig, "coherence_widths")


# --------------------------------------------------------------------------------- SSNR

def _ssnr_best(d):
    """Index of the best fully coherent vortex mode, by band-integrated dose efficiency.

    Selected on the *programmed* charge (|l| >= 1 excludes the round-beam controls),
    matching fig_ssnr.py.
    """
    ell, dose_coh = d["ell"], d["dose_coherent"]
    vortex = np.abs(ell) >= 1
    return int(np.arange(len(ell))[vortex][np.argmax(dose_coh[vortex])])


def ssnr_q():
    """Dose-normalized SSNR(q) for every delivered charge, against the pure references."""
    d = load("fig_ssnr")
    q, ell = d["q_frac"], delivered_slope() * d["ell"]
    cmap, norm = ell_cmap(ell)
    best = _ssnr_best(d)
    fig, ax = newfig(8.6, 5.8)
    for i in range(len(ell)):
        ax.plot(q, d["ssnr_mixed"][i], color=cmap(norm(ell[i])), lw=1.4, alpha=0.85)
    ax.plot(q, d["ssnr_coherent_m0"][best], color="C2", lw=2.6,
            label=rf"coherent mode 0, $\ell={int(round(ell[best])):+d}$")
    ax.plot(q, d["ssnr_ideal"], color=REF, ls="--", lw=2.4, label="ideal round aperture")
    ax.plot(q, d["ssnr_defocus"], color="C1", ls="-.", lw=2.4,
            label=r"defocused ($\Delta f = 2\,\mathrm{DOF}$)")
    ax.plot(q, d["ssnr_holo"], color="C4", ls=":", lw=2.4, label="STEM holography")
    ax.axvline(1.0, color=FG, ls=":", lw=1, alpha=0.5)
    ax.set_xlabel(r"$q / q_\mathrm{probe}$")
    ax.set_ylabel(r"dose-normalized SSNR$(q)$")
    ax.set_xlim(0, QMAX)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=13)
    savepanel(fig, "ssnr_q")


def dcoh_q():
    """Generalized SSB coherence envelope D_coh(q) per delivered charge."""
    d = load("fig_ssnr")
    q, ell = d["q_frac"], delivered_slope() * d["ell"]
    cmap, norm = ell_cmap(ell)
    fig, ax = newfig(8.2, 5.6)
    for i in range(len(ell)):
        ax.plot(q, d["coherence_envelope"][i], color=cmap(norm(ell[i])), lw=1.6, alpha=0.9)
    ax.axvline(1.0, color=FG, ls=":", lw=1, alpha=0.5)
    ax.set_xlabel(r"$q / q_\mathrm{probe}$")
    ax.set_ylabel(r"$D_\mathrm{coh}(q) = L_\rho / L_\mathrm{up}$")
    ax.set_xlim(0, QMAX)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    import matplotlib.pyplot as plt
    cbar(fig, ax, plt.cm.ScalarMappable(cmap=cmap, norm=norm), r"delivered $\ell$")
    savepanel(fig, "dcoh_q")


def dqe_q():
    """DQE(q) of the delivered mixed states against the pure references."""
    d = load("fig_ssnr")
    q, ell = d["q_frac"], delivered_slope() * d["ell"]
    cmap, norm = ell_cmap(ell)
    best = _ssnr_best(d)
    fig, ax = newfig(8.6, 5.8)
    for i in range(len(ell)):
        ax.plot(q, d["ssnr_mixed"][i] ** 2 / 4.0, color=cmap(norm(ell[i])), lw=1.4, alpha=0.85)
    ax.plot(q, d["ssnr_coherent_m0"][best] ** 2 / 4.0, color="C2", lw=2.6,
            label=rf"coherent mode 0, $\ell={int(round(ell[best])):+d}$")
    ax.plot(q, d["ssnr_ideal"] ** 2 / 4.0, color=REF, ls="--", lw=2.4,
            label="ideal round aperture")
    ax.plot(q, d["ssnr_defocus"] ** 2 / 4.0, color="C1", ls="-.", lw=2.4,
            label=r"defocused ($\Delta f = 2\,\mathrm{DOF}$)")
    ax.plot(q, d["ssnr_holo"] ** 2 / 4.0, color="C4", ls=":", lw=2.4,
            label="STEM holography")
    ax.axhline(0.5, color=FG, ls="-.", lw=1.2, alpha=0.6, label="iterative-ptychography bound")
    ax.axvline(1.0, color=FG, ls=":", lw=1, alpha=0.5)
    ax.set_xlabel(r"$q / q_\mathrm{probe}$")
    ax.set_ylabel(r"$\mathrm{DQE}_\rho(q)$")
    ax.set_xlim(0, QMAX)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=13)
    savepanel(fig, "dqe_q")


def dose_efficiency_vs_l():
    """Band-integrated dose efficiency vs delivered charge: mixed vs purified vs references."""
    d = load("fig_ssnr")
    ell = delivered_slope() * d["ell"]
    fig, ax = newfig(8.2, 5.6)
    ax.plot(ell, d["dose_mixed"], "o", color="C0", label="mixed state (delivered)")
    ax.plot(ell, d["dose_coherent"], "s", mfc="none", color="C2",
            label="coherent (mode 0 only)")
    ax.axhline(float(d["dose_ideal"]), color=REF, ls="--", lw=2, label="ideal round aperture")
    ax.axhline(float(d["dose_defocus"]), color="C1", ls="-.", lw=2,
               label=r"defocused ($\Delta f = 2\,\mathrm{DOF}$)")
    ax.axhline(float(d["dose_holo"]), color="C4", ls=":", lw=2, label="STEM holography")
    ax.set_xlabel(r"delivered charge $\ell$")
    ax.set_ylabel(r"dose efficiency $\int \mathrm{DQE}\,\mathrm{d}^2q$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=13)
    savepanel(fig, "dose_efficiency_vs_l")


def dose_to_snr5():
    """Electrons per resolution element needed to reach SNR = 5, vs spatial frequency."""
    d = load("fig_ssnr")
    q, ell = d["q_frac"], delivered_slope() * d["ell"]
    cmap, norm = ell_cmap(ell)
    best = _ssnr_best(d)
    snr0, phi0 = float(d["snr0"]), float(d["phi0"])
    floor = snr0 ** 2 / (4.0 * phi0 ** 2)

    fig, ax = newfig(10.0, 5.8)
    for i in range(len(ell)):
        ax.plot(q, d["n_res_mixed"][i], color=cmap(norm(ell[i])), lw=1.2, alpha=0.55)
    lbest = int(round(ell[best]))
    ax.plot(q, d["n_res_mixed"][best], color="C3", lw=2.8,
            label=rf"delivered vortex, $\ell={lbest:+d}$ (mixed)")
    ax.plot(q, d["n_res_coherent_best"], color="C2", lw=2.4,
            label=rf"coherent mode 0, $\ell={lbest:+d}$")
    ax.plot(q, d["n_res_ideal"], color=REF, ls="--", lw=2.4, label="ideal round aperture")
    ax.plot(q, d["n_res_defocus"], color="C1", ls="-.", lw=2.4,
            label=r"defocused ($\Delta f = 2\,\mathrm{DOF}$)")
    ax.plot(q, d["n_res_holo"], color="C4", ls=":", lw=2.4, label="STEM holography")
    ax.axhline(floor, color=FG, lw=1.2, alpha=0.6,
               label=rf"ideal-Zernike floor ({floor:.0f} $e^-$/resel)")
    ax.axvline(1.0, color=FG, ls=":", lw=1, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlim(0, QMAX)
    ax.set_ylim(floor * 0.6, 1e6)
    ax.set_xlabel(r"$q / q_\mathrm{probe}$")
    ax.set_ylabel(rf"dose to reach SNR$ = {snr0:.0f}$" + "\n" + r"($e^-$ / resolution element)")
    ax.grid(alpha=0.25, which="both")
    # Seven entries and curves that fill both top corners: park the legend below.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=13)

    # The crossover is the actionable number: below it the vortex is the cheaper probe.
    ratio = d["n_res_ideal"] / np.maximum(d["n_res_mixed"][best], 1e-9)
    sel = (q > 0.02) & (q < 1.5)
    cross = q[sel][np.where(np.diff(np.sign(ratio[sel] - 1.0)))[0]]
    if len(cross):
        ax.text(0.985, 0.03,
                rf"crossover at $q = {cross[0]:.2f}\,q_\mathrm{{probe}}$",
                transform=ax.transAxes, fontsize=14, va="bottom", ha="right",
                color=FG, bbox=BOX)
    savepanel(fig, "dose_to_snr5")


# -------------------------------------------------------------------------- OAM sorter

def sorter_voltage_calib():
    """Measured OAM spectrum per dataset, as a heatmap against applied bias."""
    d = load("fig_oam_sorter")
    spectra, lax, volts = d["spectra"], d["l_axis"], d["applied_V"]
    fig, ax = newfig(8.6, 5.8)
    im = ax.imshow(spectra.T, aspect="auto", origin="lower", cmap="magma",
                   extent=(-0.5, len(volts) - 0.5, lax[0] - 0.5, lax[-1] + 0.5))
    step = 2                                  # every second tick: 21 labels will not fit
    ax.set_xticks(np.arange(len(volts))[::step])
    ax.set_xticklabels([f"{v:+.2f}" for v in volts[::step]], rotation=90, fontsize=12)
    ax.set_xlabel(r"applied voltage $V$ (V)")
    ax.set_ylabel(r"OAM $l$")
    cbar(fig, ax, im, r"$P_l$")
    savepanel(fig, "sorter_voltage_calib")


def sorter_spectrum():
    """Azimuthal-harmonic (OAM) spectra for representative delivered charges."""
    d = load("fig_oam_sorter")
    spectra, lax, ell = d["spectra"], d["l_axis"], d["ell"]
    slope = float(d["fit_mean"][0])
    fig, ax = newfig(8.2, 5.6)
    for l_want, color in [(-10, "C0"), (-5, "C1"), (0, "C2"), (5, "C3"), (10, "C4")]:
        idx = [i for i, e in enumerate(ell) if int(e) == l_want]
        if idx:
            ax.plot(lax, spectra[idx[0]], "-o", ms=5, color=color,
                    label=rf"$\ell = {int(round(slope * l_want)):+d}$")
    ax.set_xlabel(r"OAM $l$")
    ax.set_ylabel(r"$P_l$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=14)
    savepanel(fig, "sorter_spectrum")


def sorter_mean_l_vs_bias():
    """Device calibration: measured OAM against applied chopstick bias."""
    d = load("fig_oam_sorter")
    volts, l_mean, l_argmax = d["applied_V"], d["l_mean"], d["l_argmax"]
    mV, bV, r2V = (float(x) for x in d["fit_voltage"])
    ok = ~np.isnan(volts)
    fig, ax = newfig(8.0, 5.6)
    ax.plot(volts[ok], l_mean[ok], "o", color="C0", label=r"$\langle l \rangle$ (mean)")
    ax.plot(volts[ok], l_argmax[ok], "s", mfc="none", color="C1", label=r"$\arg\max_l P_l$")
    xf = np.array([np.nanmin(volts), np.nanmax(volts)])
    ax.plot(xf, mV * xf + bV, "-", color="C0", lw=1.6, alpha=0.8)
    ax.set_xlabel(r"applied bias $V$ (V)")
    ax.set_ylabel(r"measured OAM $\langle l \rangle$")
    ax.text(0.04, 0.86, rf"${mV:.1f}\ \hbar$/V" + "\n" + rf"($R^2 = {r2V:.3f}$)",
            transform=ax.transAxes, fontsize=17, color=FG, va="top", bbox=BOX)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=14, loc="lower right")
    savepanel(fig, "sorter_mean_l_vs_bias")


def sorter_channel_purity():
    """Power in the dominant OAM channel, with and without the microscope aberrations."""
    d = load("fig_oam_sorter")
    l_mean = d["l_mean"]
    fig, ax = newfig(8.0, 5.6)
    ax.plot(l_mean, d["purity"], "o", color="C0", label="aberration-corrected")
    ax.plot(l_mean, d["purity_raw"], "s", mfc="none", color="C1",
            label=r"raw (with microscope $\chi$)")
    ax.set_xlabel(r"$\ell$ (measured mean OAM)")
    ax.set_ylabel(r"$P_{l=\hat{l}}$ (dominant channel)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=14)
    savepanel(fig, "sorter_channel_purity")


def sorter_density_matrix():
    """|rho_ll'| for a representative charge: diagonal populations, off-diagonal coherence."""
    import matplotlib.pyplot as plt
    d = load("fig_oam_sorter")
    rho, lax = d["rho_repr"], d["rho_repr_laxis"]
    i0 = int(np.argmax(np.diag(rho).real))
    l_dom = int(lax[i0])
    a, b = max(0, i0 - 4), min(len(lax), i0 + 5)
    sub = np.abs(rho[a:b, a:b])
    laxsub = lax[a:b].astype(int)
    nl = sub.shape[0]

    fig = plt.figure(figsize=(7.6, 6.2))
    ax = fig.add_subplot(projection="3d")
    xp, yp = np.meshgrid(np.arange(nl), np.arange(nl), indexing="ij")
    xp, yp, dz = xp.ravel(), yp.ravel(), sub.ravel()
    cols = plt.cm.viridis(plt.Normalize(0, dz.max())(dz))
    ax.bar3d(xp - 0.35, yp - 0.35, np.zeros_like(dz), 0.7, 0.7, dz,
             color=cols, shade=True, edgecolor=FG, linewidth=0.25)
    # 3-D axes carry their own panes and per-axis colours, which the rcParams miss.
    for pane, axis in ((ax.xaxis, "x"), (ax.yaxis, "y"), (ax.zaxis, "z")):
        pane.set_pane_color((0, 0, 0, 0))
        pane.line.set_color(FG)
        pane._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 0.25)
    ax.set_xticks(np.arange(nl)); ax.set_xticklabels(laxsub, fontsize=13)
    ax.set_yticks(np.arange(nl)); ax.set_yticklabels(laxsub, fontsize=13)
    ax.set_xlabel(r"OAM $l$", labelpad=8, fontsize=18)
    ax.set_ylabel(r"OAM $l'$", labelpad=8, fontsize=18)
    ax.tick_params(colors=FG, labelsize=13)
    # The z-label sits outside the tight bounding box of a zoomed 3-D axes and gets
    # clipped, so the quantity goes in the title instead.
    ax.set_title(rf"$|\rho_{{l l'}}|$  at  $\ell = {l_dom:+d}$", fontsize=20, color=FG)
    ax.view_init(elev=28, azim=-52)
    ax.set_box_aspect((1, 1, 0.8), zoom=1.25)
    savepanel(fig, "sorter_density_matrix")


def sorter_adjacent_coherence():
    """Adjacent-OAM coherence g: ~0 is an incoherent mixture, ->1 a coherent superposition."""
    d = load("fig_oam_sorter")
    l_mean, coh = d["l_mean"], d["oam_coherence"]
    fig, ax = newfig(8.0, 5.6)
    ax.plot(l_mean, coh, "o", color="C3")
    ax.axhline(float(np.median(coh)), color="C3", ls="--", lw=1.6,
               label=f"median {np.median(coh):.2f}")
    ax.set_xlabel(r"$\ell$ (measured mean OAM)")
    ax.set_ylabel(r"adjacent-OAM coherence $g_{\hat l,\hat l+1}$")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=14)
    savepanel(fig, "sorter_adjacent_coherence")


PANELS = [
    mode_powers_stacked, purity_vs_l,
    coherence_real, coherence_fourier, coherence_widths,
    ssnr_q, dcoh_q, dqe_q, dose_efficiency_vs_l, dose_to_snr5,
    sorter_voltage_calib, sorter_spectrum, sorter_mean_l_vs_bias,
    sorter_channel_purity, sorter_density_matrix, sorter_adjacent_coherence,
]


def main():
    style()
    for fn in PANELS:
        fn()


if __name__ == "__main__":
    main()
