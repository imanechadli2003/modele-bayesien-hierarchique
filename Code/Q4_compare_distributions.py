# ============================================
# Question 4 : Comparaison prior / vraisemblance / postérieure
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Grille de valeurs de μ ---
mu_vals = np.linspace(0, 1, 300)

# --- 1. Prior : croyance initiale centrée sur μ0 = 0.5 ---
prior = norm.pdf(mu_vals, loc=0.5, scale=0.15)

# --- 2. Vraisemblance : information issue des données, centrée vers 0.7 ---
likelihood = norm.pdf(mu_vals, loc=0.7, scale=0.08)

# --- 3. Postérieure : produit normalisé du prior et de la vraisemblance ---
posterior = prior * likelihood
posterior /= np.trapz(posterior, mu_vals)  # Normalisation numérique

# --- Tracé ---
plt.figure(figsize=(8,5))
plt.plot(mu_vals, prior, 'b--', lw=2, label='Prior (avant les données)')
plt.plot(mu_vals, likelihood, 'g-.', lw=2, label='Vraisemblance (données)')
plt.plot(mu_vals, posterior, 'r-', lw=2, label='Postérieure (mise à jour)')
plt.title("Comparaison entre prior, vraisemblance et postérieure (paramètre μ)")
plt.xlabel("μ")
plt.ylabel("Densité (échelle relative)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Sauvegarde ---
plt.savefig("../Figures/compare_priors_posteriors.png", dpi=300)
plt.show()

print(" Figure sauvegardée dans ../Figures/compare_priors_posteriors.png")
