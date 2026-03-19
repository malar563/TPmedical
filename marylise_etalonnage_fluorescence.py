import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
# https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z29.html

df = pd.read_csv('spectres_bruts/fluorescence_csv/etalonnage_SIPIN_75s_G110.29.csv')
Am241 = df.to_numpy().ravel()
print(Am241)


fig, ax = plt.subplots(figsize=(8,5))
ax.plot(range(4096), Am241, lw=0.5, label=r"$^{241}$Am")
ax.legend()
ax.set_xlabel("Canal [-]", size=12)
ax.set_ylabel("Nombre de comptes [-]", size=12)
plt.show()


# Modèle gaussienne + fond linéaire
def gaussian_with_background(x, A, mu, sigma, m, c):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + m*x + c


def fit_peak(counts, peak_center, window=30, plot=True):
    """
    counts : array numpy des comptes
    peak_center : estimation approximative du pic
    window : nombre de canaux autour du pic
    """

    # Définition de la région de fit
    x_min = peak_center - window
    x_max = peak_center + window
    
    x = np.arange(len(counts))
    mask = (x >= x_min) & (x <= x_max)

    x_fit = x[mask]
    y_fit = counts[mask]

    # Valeurs initiales
    A0 = max(y_fit)
    mu0 = peak_center
    sigma0 = window/3
    m0 = 0
    c0 = min(y_fit)

    p0 = [A0, mu0, sigma0, m0, c0]

    # Fit
    popt, pcov = curve_fit(gaussian_with_background, x_fit, y_fit, p0=p0)

    A, mu, sigma, m, c = popt
    mu_err = np.sqrt(np.diag(pcov))[1]
    sigma_err = np.sqrt(np.diag(pcov))[2]

    # Nombre de compte dans le pic
    N_pic = A*sigma*np.sqrt(2*np.pi)


    if plot:
        plt.figure()
        plt.scatter(x_fit, y_fit, s=10)
        plt.plot(x_fit, gaussian_with_background(x_fit, *popt))
        plt.title(f"Pic ajusté : μ = {mu:.2f} ± {mu_err:.2f}")
        plt.xlabel("Canal")
        plt.ylabel("Comptes")
        plt.show()

    return mu, mu_err, sigma, sigma_err, N_pic


peaks, _ = find_peaks(Am241, height=3000)  # ajuster height
print(peaks)

mu_Am1, mu_err_Am1, sigma_Am1, sigma_err_Am1, _ = fit_peak(Am241, peak_center=1447, window=50)

mu_Am2, mu_err_CAm2, sigma_Am2, sigma_err_Am2, _ = fit_peak(Am241, peak_center=1838, window=50)

# mu_Am3, mu_err_Am3, sigma_Am3, sigma_err_Am3, _ = fit_peak(Am241, peak_center=3346, window=50)


channels = np.array([mu_Am1, mu_Am2])#, mu_Am3
energies = np.array([13.95, 17.74])#, 59.54


coeff = np.polyfit(channels, energies, 1)

a = coeff[0]
b = coeff[1]

print("Gain (keV/canal) =", a)
print("Offset (keV) =", b)  

def etalon(channel):
    return a * channel + b

# Calcul de R^2
ss_res = np.sum((energies - etalon(channels))**2)
ss_tot = np.sum((energies - np.mean(energies))**2)
r2 = 1 - ss_res/ss_tot


plt.scatter(channels, energies, label="Points d'étalonnage", color='red')
plt.plot(channels, etalon(channels), label="Régression linéaire")


# Calcul du point milieu
x_mid = np.mean(channels)
y_mid = a * x_mid + b

# Affichage équation sur le graphe
plt.text(x_mid+300,
         y_mid-3,
         r"$E_\gamma$"+f" = {a:.3f} C - {np.abs(b):.3f}\n$R^2$ = {r2:.6f}",
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel("Canal C [-]", size=12)
plt.ylabel(r"Énergie $E_\gamma$ [keV]", size=12)
plt.legend(loc="best")
plt.grid()


plt.show()