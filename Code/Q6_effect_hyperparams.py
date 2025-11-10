# ============================================
# Q6 — Figures séparées pour chaque réglage d'hyperparamètre
# Génère:
#  - Comparatif + 3 figures individuelles pour μ0
#  - Comparatif + 3 figures individuelles pour σ0
#  - Comparatif + 3 figures individuelles pour (s,b0) sur p(μ|X)
#  - Comparatif + 3 figures individuelles pour (s,b0) sur p(σ|X)
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from pathlib import Path

# ----------------------
# 1) Données (Q1)
# ----------------------
fig_dir = Path("../Figures")
fig_dir.mkdir(parents=True, exist_ok=True)

X = np.load(fig_dir / "data_Q1.npy")   # ton fichier existant
n = len(X)
logx = np.log(X)
log1mx = np.log(1 - X)
mu_hat = X.mean()

# ----------------------
# 2) Modèle : Beta via (μ,σ) et prior hiérarchique
# ----------------------
def alpha_beta_from_mu_sigma(mu, sigma):
    v = sigma**2
    t = mu * (1 - mu) / v - 1.0
    return mu * t, (1 - mu) * t

def loglik_beta_mu_sigma(mu, sigma):
    # support
    if sigma <= 0 or mu <= 0 or mu >= 1 or sigma**2 >= mu*(1-mu):
        return -np.inf
    a, b = alpha_beta_from_mu_sigma(mu, sigma)
    if a <= 0 or b <= 0:
        return -np.inf
    # log-vraisemblance
    ll_data = (a - 1.0) * np.sum(logx) + (b - 1.0) * np.sum(log1mx)
    ll_norm = n * (gammaln(a) + gammaln(b) - gammaln(a + b))
    return ll_data - ll_norm

def log_prior(mu, sigma, mu0, sigma0, s, b0):
    # p(mu|sigma) ∝ sqrt(sigma) * exp(-0.5 * sigma0 * sigma * (mu - mu0)^2)
    # p(sigma)    ∝ sigma^s * exp(-b0 * sigma)
    if sigma <= 0 or mu <= 0 or mu >= 1:
        return -np.inf
    return (s + 0.5) * np.log(sigma) - 0.5 * sigma0 * sigma * (mu - mu0)**2 - b0 * sigma

def marginal_mu(mu_grid, S_sigma, hyper):
    mu0, sigma0, s, b0 = hyper["mu0"], hyper["sigma0"], hyper["s"], hyper["b0"]
    dens_mu = np.zeros_like(mu_grid, dtype=float)
    for j, mu in enumerate(mu_grid):
        sig_max = np.sqrt(mu * (1 - mu))
        if not np.isfinite(sig_max) or sig_max <= 0:
            continue
        sigma_grid = np.linspace(1e-6, 0.999*sig_max, S_sigma)
        log_int = np.array([loglik_beta_mu_sigma(mu, sg) + log_prior(mu, sg, mu0, sigma0, s, b0)
                            for sg in sigma_grid])
        if np.all(~np.isfinite(log_int)):
            continue
        # stabilisation
        m = np.max(log_int)
        f = np.exp(log_int - m)
        dens_mu[j] = np.trapz(f, sigma_grid) * np.exp(m)
    area = np.trapz(dens_mu, mu_grid)
    if area > 0:
        dens_mu /= area
    return dens_mu

def marginal_sigma(sigma_grid, M_mu, hyper):
    mu0, sigma0, s, b0 = hyper["mu0"], hyper["sigma0"], hyper["s"], hyper["b0"]
    dens_sig = np.zeros_like(sigma_grid, dtype=float)
    mu_grid = np.linspace(1e-3, 1-1e-3, M_mu)
    for k, sig in enumerate(sigma_grid):
        if sig <= 0 or sig >= 0.5:  # max sqrt(mu(1-mu)) = 0.5
            continue
        mask = (sig**2 < mu_grid*(1 - mu_grid))
        if not np.any(mask):
            continue
        mu_valid = mu_grid[mask]
        log_int = np.array([loglik_beta_mu_sigma(m, sig) + log_prior(m, sig, mu0, sigma0, s, b0)
                            for m in mu_valid])
        if np.all(~np.isfinite(log_int)):
            continue
        m = np.max(log_int)
        f = np.exp(log_int - m)
        dens_sig[k] = np.trapz(f, mu_valid) * np.exp(m)
    area = np.trapz(dens_sig, sigma_grid)
    if area > 0:
        dens_sig /= area
    return dens_sig

# ----------------------
# 3) Grilles et réglages
# ----------------------
mu_grid = np.linspace(1e-3, 1-1e-3, 350)
sigma_grid = np.linspace(1e-3, 0.49, 350)   # σ < 0.5
S_sigma = 180
M_mu = 180

# “base” pour isoler un seul hyperparamètre à la fois
base = {"mu0": 0.6, "sigma0": 0.6, "s": 0.8, "b0": 2.0}

mu0_vals   = [0.4, 0.6, 0.75]        # on varie μ0 (reste fixe)
sigma0_vals= [0.2, 0.6, 1.2]         # on varie σ0 (reste fixe)
sb0_vals   = [("concentré", 3.0, 8.0),
              ("moyen",      0.8, 2.0),
              ("diffus",    -0.1, 0.5)]  # on varie (s,b0), μ0 et σ0 fixes

# ----------------------
# 4) Fonctions de rendu (comparatif + individuel)
# ----------------------
def plot_compare_mu(title, tag, settings, set_label_fn):
    plt.figure(figsize=(10,6))
    for st in settings:
        hyp = dict(base); hyp.update(st)
        dens = marginal_mu(mu_grid, S_sigma, hyp)
        plt.plot(mu_grid, dens, lw=2, label=set_label_fn(st))
    plt.axvline(mu_hat, lw=1.5, ls="--", color="k", label=f"μ̂={mu_hat:.3f}")
    plt.title(title); plt.xlabel("μ"); plt.ylabel("p(μ|X)"); plt.legend(); plt.grid(alpha=0.3)
    fname = fig_dir / f"q6_compare_mu_{tag}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)

def plot_individual_mu(tag_prefix, settings, set_label_fn):
    for st in settings:
        hyp = dict(base); hyp.update(st)
        dens = marginal_mu(mu_grid, S_sigma, hyp)
        plt.figure(figsize=(9,5.5))
        plt.plot(mu_grid, dens, lw=2)
        plt.axvline(mu_hat, lw=1.3, ls="--", color="k", label=f"μ̂={mu_hat:.3f}")
        plt.title(f"{tag_prefix} — {set_label_fn(st)}")
        plt.xlabel("μ"); plt.ylabel("p(μ|X)"); plt.legend(); plt.grid(alpha=0.3)
        # nom de fichier unique et lisible
        tag = tag_prefix.lower().replace(" ", "_")
        spec = set_label_fn(st).lower().replace(" ", "_").replace("=", "")
        fname = fig_dir / f"q6_single_mu_{tag}_{spec}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", fname)

def plot_compare_sigma(title, tag, settings, set_label_fn):
    plt.figure(figsize=(10,6))
    for st in settings:
        hyp = dict(base); hyp.update(st)
        dens = marginal_sigma(sigma_grid, M_mu, hyp)
        plt.plot(sigma_grid, dens, lw=2, label=set_label_fn(st))
    plt.title(title); plt.xlabel("σ"); plt.ylabel("p(σ|X)"); plt.legend(); plt.grid(alpha=0.3)
    fname = fig_dir / f"q6_compare_sigma_{tag}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)

def plot_individual_sigma(tag_prefix, settings, set_label_fn):
    for st in settings:
        hyp = dict(base); hyp.update(st)
        dens = marginal_sigma(sigma_grid, M_mu, hyp)
        plt.figure(figsize=(9,5.5))
        plt.plot(sigma_grid, dens, lw=2)
        plt.title(f"{tag_prefix} — {set_label_fn(st)}")
        plt.xlabel("σ"); plt.ylabel("p(σ|X)"); plt.grid(alpha=0.3)
        tag = tag_prefix.lower().replace(" ", "_")
        spec = set_label_fn(st).lower().replace(" ", "_").replace("=", "")
        fname = fig_dir / f"q6_single_sigma_{tag}_{spec}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", fname)

# ----------------------
# 5) MU0 : comparatif + 3 figures individuelles (p(μ|X))
# ----------------------
mu0_settings = [ {"mu0": v} for v in mu0_vals ]
plot_compare_mu("Q6 — Effet de μ0 sur p(μ|X) (σ0,s,b0 fixes)",
                "mu0",
                mu0_settings,
                lambda st: f"μ0={st['mu0']}")
plot_individual_mu("Effet de μ0 (p(μ|X))",
                   mu0_settings,
                   lambda st: f"μ0={st['mu0']}")

# ----------------------
# 6) SIGMA0 : comparatif + 3 figures individuelles (p(μ|X))
# ----------------------
sigma0_settings = [ {"sigma0": v} for v in sigma0_vals ]
plot_compare_mu("Q6 — Effet de σ0 sur p(μ|X) (μ0,s,b0 fixes)",
                "sigma0",
                sigma0_settings,
                lambda st: f"σ0={st['sigma0']}")
plot_individual_mu("Effet de σ0 (p(μ|X))",
                   sigma0_settings,
                   lambda st: f"σ0={st['sigma0']}")

# ----------------------
# 7) (s,b0) : comparatif + 3 individuelles sur p(μ|X)
# ----------------------
sb0_settings = [ {"s": s, "b0": b0} for _, s, b0 in sb0_vals ]
plot_compare_mu("Q6 — Effet de (s,b0) sur p(μ|X) (μ0,σ0 fixes)",
                "sb0_mu",
                sb0_settings,
                lambda st: f"s={st['s']}, b0={st['b0']}")
plot_individual_mu("Effet de (s,b0) (p(μ|X))",
                   sb0_settings,
                   lambda st: f"s={st['s']}, b0={st['b0']}")

# ----------------------
# 8) (s,b0) : comparatif + 3 individuelles sur p(σ|X)
# ----------------------
plot_compare_sigma("Q6 — Effet de (s,b0) sur p(σ|X) (μ0,σ0 fixes)",
                   "sb0_sigma",
                   sb0_settings,
                   lambda st: f"s={st['s']}, b0={st['b0']}")
plot_individual_sigma("Effet de (s,b0) (p(σ|X))",
                      sb0_settings,
                      lambda st: f"s={st['s']}, b0={st['b0']}")

print("Terminé. Toutes les figures sont dans:", fig_dir.resolve())
