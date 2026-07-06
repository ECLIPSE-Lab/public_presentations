"""Generate images/empirical_bayes_alpha_selection.png for the Empirical Bayes slide.

Bayesian RBF-basis regression on noisy sin(2*pi*x): three fits (alpha too
small / evidence-optimal / too large) plus the log-evidence curve that
selects the middle one. Run with the repo venv:
    .venv/bin/python mathematical_foundations_of_ai_and_ml/12_uncertainty_in_predictions/make_empirical_bayes_figure.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

# Data: noisy sine, N=25
N = 25
sigma = 0.2
beta = 1.0 / sigma**2
x = rng.uniform(0, 1, N)
y = np.sin(2 * np.pi * x) + rng.normal(0, sigma, N)
xg = np.linspace(0, 1, 300)
f_true = np.sin(2 * np.pi * xg)

# RBF basis, M=12 bumps
M = 12
centers = np.linspace(0, 1, M)
s = 0.08


def Phi(t):
    return np.exp(-(t[:, None] - centers[None, :])**2 / (2 * s**2))


Px, Pg = Phi(x), Phi(xg)


def fit(alpha):
    A = alpha * np.eye(M) + beta * Px.T @ Px
    mN = beta * np.linalg.solve(A, Px.T @ y)
    E = 0.5 * beta * np.sum((y - Px @ mN)**2) + 0.5 * alpha * mN @ mN
    logev = (M/2)*np.log(alpha) + (N/2)*np.log(beta) - E \
            - 0.5*np.linalg.slogdet(A)[1] - (N/2)*np.log(2*np.pi)
    mean = Pg @ mN
    var = 1/beta + np.einsum('ij,jk,ik->i', Pg, np.linalg.inv(A), Pg)
    return mN, mean, np.sqrt(var), logev


log_alphas = np.linspace(-9, 6, 200)
evs = np.array([fit(np.exp(la))[3] for la in log_alphas])
la_opt = log_alphas[np.argmax(evs)]

picks = [(-8.0, 'overfit: $\\alpha$ too small'),
         (la_opt, f'evidence optimum: $\\ln\\alpha={la_opt:.1f}$'),
         (5.0, 'oversmoothed: $\\alpha$ too large')]

fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
colors = ['#d62728', '#2ca02c', '#ff7f0e']
panels = [axes[0, 0], axes[0, 1], axes[1, 0]]
letters = ['(a)', '(b)', '(c)']
for ax, (la, title), c, letter in zip(panels, picks, colors, letters):
    _, mean, sd, _ = fit(np.exp(la))
    ax.fill_between(xg, mean - 2*sd, mean + 2*sd, color=c, alpha=0.15)
    ax.plot(xg, f_true, 'k--', lw=1, label='true function')
    ax.plot(xg, mean, color=c, lw=2, label='posterior mean')
    ax.scatter(x, y, s=18, color='k', zorder=3)
    ax.set_title(f'{letter} {title}', fontsize=12)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    if letter == '(a)':
        ax.legend(fontsize=9, loc='upper right')

ax = axes[1, 1]
ax.plot(log_alphas, evs, 'b-', lw=2)
for (la, _), c, letter in zip(picks, colors, letters):
    ev = fit(np.exp(la))[3]
    ax.plot(la, ev, 'o', ms=9, color=c)
    ax.annotate(letter, (la, ev), textcoords='offset points', xytext=(8, 6), fontsize=12)
ax.axvline(la_opt, color='#2ca02c', ls=':', lw=1)
ax.set_title('(d) log evidence $\\ln p(\\mathbf{y}\\,|\\,\\alpha)$ vs. $\\ln\\alpha$', fontsize=12)
ax.set_xlabel('$\\ln \\alpha$')
ax.set_ylabel('log evidence')
ax.set_ylim(evs.min() - 2, evs.max() + 6)

fig.suptitle('Empirical Bayes: the evidence selects the prior precision $\\alpha$ — no validation set needed', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = 'mathematical_foundations_of_ai_and_ml/12_uncertainty_in_predictions/images/empirical_bayes_alpha_selection.png'
fig.savefig(out, dpi=150, facecolor='white')
print('saved', out, 'opt ln(alpha) =', round(la_opt, 2))
