# ================================
# Question 1 : Génération d’un échantillon et histogramme
# ================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# --- Paramètres du modèle ---
mu = 0.7
sigma = 0.1
n = 100

# --- Conversion en alpha et beta ---
alpha = mu * ((mu * (1 - mu)) / (sigma**2) - 1)
beta_param = (1 - mu) * ((mu * (1 - mu)) / (sigma**2) - 1)

# --- Génération de l’échantillon ---
X = beta.rvs(alpha, beta_param, size=n)

# --- Histogramme et densité théorique ---
plt.figure(figsize=(8,5))
plt.hist(X, bins=15, density=True, alpha=0.6, color='skyblue', label='Échantillon simulé')
x_vals = np.linspace(0, 1, 200)
plt.plot(x_vals, beta.pdf(x_vals, alpha, beta_param), 'r-', lw=2, label='Densité théorique')
plt.title("Histogramme de l'échantillon simulé (loi Beta)")
plt.xlabel("x")
plt.ylabel("Densité")
plt.legend()
plt.grid(True)

# --- Sauvegarde des données simulées ---
np.save("../Figures/data_Q1.npy", X)
print("chantillon sauvegardé dans ../Figures/data_Q1.npy")

# --- Sauvegarde de la figure ---
plt.savefig("../Figures/histo_data.png", dpi=300)
plt.show()
