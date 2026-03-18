import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit

# https://www.mint.ca/fr/en-savoir-plus/pieces-de-circulation-canadiennes/1-dollar?srsltid=AfmBOorypmF-M8b70BuOue7oS4VPJF5muGZB79KJPt_fMcYd0SWh0BuU
# https://www.mint.ca/en/discover/canadian-circulation/25-cents?srsltid=AfmBOopj7c5tcRLpdrI5NeVL1B41ZRw3klexn558r65Z61dBOHgif6N2
# https://www.mint.ca/fr/en-savoir-plus/pieces-de-circulation-canadiennes/1-cent?srsltid=AfmBOoqG-OhRLVWnGrBHfhNjmIvdeg3su1JOiaA5GiA6SdhXCFv7hX3_

# Modèle gaussienne + fond linéaire
def gaussian_with_background(x, A, mu, sigma, m, c):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + m*x + c

def fit_peak(energies, counts, peak_center, window=0.5, plot=True):
    """
    energies : array numpy des énergies (E = a*canal + b)
    counts : array numpy des comptes
    peak_center : centre du pic en ÉNERGIE
    window : demi-largeur de la fenêtre de fit en ÉNERGIE
    """

    # Définition de la région de fit basée sur l'énergie
    x_min = peak_center - window
    x_max = peak_center + window
    
    # Masque appliqué sur le tableau d'énergie
    mask = (energies >= x_min) & (energies <= x_max)

    x_fit = energies[mask]
    y_fit = counts[mask]

    # Valeurs initiales (estimations)
    A0 = max(y_fit)
    mu0 = peak_center
    sigma0 = window / 3  # Estimation de l'écart-type
    m0 = 0
    c0 = min(y_fit)

    p0 = [A0, mu0, sigma0, m0, c0]

    # Fit
    popt, pcov = curve_fit(gaussian_with_background, x_fit, y_fit, p0=p0)

    A, mu, sigma, m, c = popt
    errors = np.sqrt(np.diag(pcov))
    mu_err = errors[1]
    sigma_err = errors[2]

    # Nombre de coups dans le pic (Intégrale de la Gaussienne)
    # Note : sigma est maintenant en unités d'énergie
    N_pic = A * np.abs(sigma) * np.sqrt(2 * np.pi)

    if plot:
        plt.figure()
        plt.scatter(x_fit, y_fit, s=10, label="Données")
        plt.plot(x_fit, gaussian_with_background(x_fit, *popt), 'r', label="Fit")
        plt.title(f"Pic : μ = {mu:.3f} ± {mu_err:.3f} keV")
        plt.xlabel("Énergie (keV)")
        plt.ylabel("Comptes / s")
        plt.legend()
        plt.show()

    return mu, mu_err, sigma, sigma_err, N_pic

folder = f"spectres_bruts/fluorescence_csv/alliages"
# files = os.listdir(folder)

def pre_process(folder, file_name):

    df = pd.read_csv(os.path.join(folder, file_name))
    time = float(df.columns[0]) 

    index = df.index.to_numpy()
    counts = (df[df.columns[0]]).to_numpy()
    counts_s = counts/time

    energies = (0.009685321409623682*index) - 0.059015714868320815

    diff_energies = np.mean(np.diff(energies))
    
    # plt.plot(energies, counts)
    # plt.show()

    return energies, counts_s

# Cu, Zn, Sn
energies, counts_1CAD = pre_process(folder, "1$_SIPIN_60s_G110.29_150mua_40kV.csv")


# Ni, Cu, Fe?
energies, counts_25c = pre_process(folder, "25c_SIPIN_60s_G110.29_150mua_40kV.csv")

# Fe, Cu, 
energies, counts_1c = pre_process(folder, "1c_SIPIN_60s_G110.29_150mua_40kV.csv")

energies, counts_cle = pre_process(folder, "cle_SIPIN_60s_G110.29_150mua_40kV.csv")


files = os.listdir(folder)
for file in files:
    energies, counts = pre_process(folder, file)
    try:    
        mu_Cu1, mu_err_Cu, sigma_Cu1, sigma_err_Cu1, _ = fit_peak(energies, counts, peak_center=8.05, window=0.5) 
        mu_Cu2, mu_err_Cu2, sigma_Cu2, sigma_err_Cu2, _ = fit_peak(energies, counts, peak_center=8.91, window=0.5)
        mu_Fe1, mu_err_Fe, sigma_Fe1, sigma_err_Fe1, _ = fit_peak(energies, counts, peak_center=6.4, window=0.5) 
        mu_Fe2, mu_err_Fe2, sigma_Fe2, sigma_err_Fe2, _ = fit_peak(energies, counts, peak_center=7.06, window=0.3)
    except:
        pass

# energies, counts_25c = pre_process(folder, "25c_SIPIN_60s_G110.29_150mua_40kV.csv")

    
plt.plot(energies, counts_1CAD)
plt.plot(energies, counts_25c)
plt.plot(energies, counts_1c)
plt.plot(energies, counts_cle)
plt.show()

mu_Cu1, mu_err_Cu, sigma_Cu1, sigma_err_Cu1, _ = fit_peak(energies, counts_1CAD, peak_center=8.05, window=0.5) 
mu_Cu2, mu_err_Cu2, sigma_Cu2, sigma_err_Cu2, _ = fit_peak(energies, counts_1CAD, peak_center=8.91, window=0.5)
mu_Fe1, mu_err_Fe, sigma_Fe1, sigma_err_Fe1, _ = fit_peak(energies, counts_1CAD, peak_center=6.4, window=0.5) 
mu_Fe2, mu_err_Fe2, sigma_Fe2, sigma_err_Fe2, _ = fit_peak(energies, counts_1CAD, peak_center=7.06, window=0.3)



folder = f"spectres_bruts/fluorescence_csv/spectres_purs"
files = os.listdir(folder)

for file in files:
    element = file[0:2]

    df = pd.read_csv(os.path.join(folder, file))
    time = float(df.columns[0]) 

    index = df.index.to_numpy()
    counts = (df[df.columns[0]]).to_numpy()
    counts_s = counts/time

    energies = (0.009685321409623682*index) - 0.059015714868320815

    diff_energies = np.mean(np.diff(energies))
    
    plt.plot(energies, counts)
    plt.show()

    if element == "Ag":
        mu_Ag1, mu_err_Ag, sigma_Ag1, sigma_err_Ag1, _ = fit_peak(energies, counts_s, peak_center=22.16, window=0.5) 
        mu_Ag2, mu_err_CAg2, sigma_Ag2, sigma_err_Ag2, _ = fit_peak(energies, counts_s, peak_center=24.94, window=0.5)
    if element == "Al":
        mu_Al1, mu_err_Al1, sigma_Al1, sigma_err_Al1, _ = fit_peak(energies, counts_s, peak_center=1.49, window=0.5) 
    if element == "Cu":
        mu_Cu1, mu_err_Cu, sigma_Cu1, sigma_err_Cu1, _ = fit_peak(energies, counts_s, peak_center=8.05, window=0.5) 
        mu_Cu2, mu_err_Cu2, sigma_Cu2, sigma_err_Cu2, _ = fit_peak(energies, counts_s, peak_center=8.91, window=0.5)
    if element == "Fe":
        mu_Fe1, mu_err_Fe, sigma_Fe1, sigma_err_Fe1, _ = fit_peak(energies, counts_s, peak_center=6.4, window=0.5) 
        mu_Fe2, mu_err_Fe2, sigma_Fe2, sigma_err_Fe2, _ = fit_peak(energies, counts_s, peak_center=7.06, window=0.3)
    if element == "Pb":
        mu_Pb1, mu_err_Pb, sigma_Pb1, sigma_err_Pb1, _ = fit_peak(energies, counts_s, peak_center=9.18, window=0.5) 
        mu_Pb2, mu_err_Pb2, sigma_Pb2, sigma_err_Pb2, _ = fit_peak(energies, counts_s, peak_center=10.55, window=0.5)
        mu_Pb3, mu_err_Pb3, sigma_Pb3, sigma_err_Pb3, _ = fit_peak(energies, counts_s, peak_center=12.61, window=0.5)
        mu_Pb4, mu_err_Pb4, sigma_Pb4, sigma_err_Pb4, _ = fit_peak(energies, counts_s, peak_center=14.76, window=0.5)



