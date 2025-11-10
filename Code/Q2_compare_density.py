# ================================
# Question 2 : Comparaison histogramme vs densité estimée
# ================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# --- Chargement de l'échantillon de la Q1 ---
X = np.load("../Figures/data_Q1.npy")
n = len(X)

# --- Estimation empirique ---
mu_hat = np.mean(X)
sigma_hat = np.std(X, ddof=1)

# --- Conversion en alpha et beta estimés ---
alpha_hat = mu_hat * ((mu_hat * (1 - mu_hat)) / (sigma_hat**2) - 1)
beta_hat = (1 - mu_hat) * ((mu_hat * (1 - mu_hat)) / (sigma_hat**2) - 1)

# --- Tracé ---
x_vals = np.linspace(0, 1, 200)
plt.figure(figsize=(8,5))
plt.hist(X, bins=15, density=True, alpha=0.6, color='skyblue', label='Échantillon simulé')
plt.plot(x_vals, beta.pdf(x_vals, alpha_hat, beta_hat), 'r-', lw=2,
         label=r'Densité estimée $f(x;\hat{\alpha},\hat{\beta})$')

# --- Mise en forme ---
plt.title("Comparaison : histogramme vs densité théorique estimée")
plt.xlabel("x")
plt.ylabel("Densité")
plt.legend()
plt.grid(True)

# --- Sauvegarde ---
plt.savefig("../Figures/compare_density.png", dpi=300)
plt.show()

# --- Affichage des estimateurs ---
print(f"μ̂ = {mu_hat:.3f}, σ̂ = {sigma_hat:.3f}")
print(f"α̂ = {alpha_hat:.2f}, β̂ = {beta_hat:.2f}")
