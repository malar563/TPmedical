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
def gaussian_without_background(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


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

    return mu, mu_err, sigma, sigma_err, N_pic, A


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

    return energies, counts_s, time

def sum_up_sigma(energies, counts, mean, std, n=2):
    # intervalle sur l'axe des ÉNERGIES
    lower_bound = mean - n * std
    upper_bound = mean + n * std

    # crée le masque basé sur l'axe des ÉNERGIES
    mask = (energies >= lower_bound) & (energies <= upper_bound)

    return np.sum(counts[mask])

def calcule_incertitudes_ratio(resultat, valeur1, sigma1, valeur2, sigma2):
    sigma_resultat = resultat*np.sqrt((sigma1/valeur1)**2 + (sigma2/valeur2)**2)# - 2*((sigma1*valeur2)/(valeur1*valeur2)))
    return sigma_resultat



folder = f"spectres_bruts/fluorescence_csv/alliages"
files = os.listdir(folder)

# Cu, Zn, Sn
energies, counts_1CAD, time = pre_process(folder, "1$_SIPIN_60s_G110.29_150mua_40kV.csv")
# Ni, Cu, Fe? : PIC DE 8.05 DU Cu MÉLANGÉ AVEC CELUI À 8.26 DU Ni
energies, counts_25c, time = pre_process(folder, "25c_SIPIN_60s_G110.29_150mua_40kV.csv")
# Fe, Cu, 
energies, counts_1c, time = pre_process(folder, "1c_SIPIN_60s_G110.29_150mua_40kV.csv")
energies, counts_cle, time = pre_process(folder, "cle_SIPIN_60s_G110.29_150mua_40kV.csv")

plt.plot(energies, counts_1c, label="pièce de 1 cent")
plt.plot(energies, counts_25c, label="pièce de 25 cents")
plt.plot(energies, counts_1CAD, label="pièce de 1 dollar")
# plt.plot(energies, counts_cle, label="pièce de 1 dollar")
plt.legend(fontsize=14)
plt.xlabel("Énergie [keV]", fontsize=16)
plt.ylabel("Nombre de comptes par seconde", fontsize=16)
plt.show()


"""ANALYSE DES SPECTRES PURS"""
folder = f"spectres_bruts/fluorescence_csv/spectres_purs"
files = os.listdir(folder)

# Cu, Zn, Sn
energies, counts_Ag, time = pre_process(folder, "Ag_SIPIN_60s_G110.29_150mua_40kV.csv")
energies, counts_Cu, time = pre_process(folder, "Cu_SIPIN_180s_G110.29_150mua_40kV.csv")
energies, counts_Pb, time = pre_process(folder, "Pb_SIPIN_60s_G110.29_150mua_40kV.csv")
energies, counts_Al, time = pre_process(folder, "Al_SIPIN_180s_G110.29_150mua_40kV.csv")
energies, counts_Fe, time = pre_process(folder, "Fe_SIPIN_380s_G110.29_150mua_40kV.csv")

plt.plot(energies, counts_Ag, label="Ag")
plt.plot(energies, counts_Pb, label="Pb")
plt.plot(energies, counts_Cu, label="Cu")
plt.plot(energies, counts_Al, label="Al")
plt.plot(energies, counts_Fe, label="Fe")

plt.legend(fontsize=14)
plt.xlabel("Énergie [keV]", fontsize=16)
plt.ylabel("Nombre de comptes par seconde", fontsize=16)
# plt.yscale("log")
plt.show()


for file in files:
    element = file[0:2]

    df = pd.read_csv(os.path.join(folder, file))
    time = float(df.columns[0]) 

    index = df.index.to_numpy()
    counts = (df[df.columns[0]]).to_numpy()
    # counts_s = counts/time
    counts_s = (df[df.columns[0]]).to_numpy() # iciiii

    energies = (0.009685321409623682*index) - 0.059015714868320815

    diff_energies = np.mean(np.diff(energies))
    
    # plt.plot(energies, counts)
    # plt.show()

    if element == "Ag":
        mu_Ag1, mu_err_Ag, sigma_Ag1, sigma_err_Ag1, _, A = fit_peak(energies, counts_s, peak_center=22.16, window=0.5, plot=False) 
        mu_Ag2, mu_err_CAg2, sigma_Ag2, sigma_err_Ag2, _, A = fit_peak(energies, counts_s, peak_center=24.94, window=0.5, plot=False)
    if element == "Al":
        mu_Al1, mu_err_Al1, sigma_Al1, sigma_err_Al1, _, A = fit_peak(energies, counts_s, peak_center=1.49, window=0.5, plot=False) 
    if element == "Cu":
        time_Cu = time
        mu_Cu1, mu_err_Cu, sigma_Cu1, sigma_err_Cu1, N_Cu1_ref, A1 = fit_peak(energies, counts_s, peak_center=8.05, window=0.5, plot=True)
        total_counts_REF_Cu1 = sum_up_sigma(energies, gaussian_without_background(energies, A1, mu_Cu1, sigma_Cu1), mu_Cu1, sigma_Cu1) 
        mu_Cu2, mu_err_Cu2, sigma_Cu2, sigma_err_Cu2, N_Cu2_ref, A2 = fit_peak(energies, counts_s, peak_center=8.91, window=0.5, plot=True)
        total_counts_REF_Cu2 = sum_up_sigma(energies, gaussian_without_background(energies, A2, mu_Cu2, sigma_Cu2), mu_Cu2, sigma_Cu2) 
    
        total_counts_REF_Cu2_s = total_counts_REF_Cu2/time_Cu# iciii
        total_counts_REF_Cu1_s = total_counts_REF_Cu1/time_Cu# iciii
    if element == "Fe":
        time_Fe = time
        mu_Fe1, mu_err_Fe, sigma_Fe1, sigma_err_Fe1, N_Fe1_ref, A1 = fit_peak(energies, counts_s, peak_center=6.4, window=0.5, plot=False)
        total_counts_REF_Fe1 = sum_up_sigma(energies, gaussian_without_background(energies, A1, mu_Fe1, sigma_Fe1), mu_Fe1, sigma_Fe1)
        mu_Fe2, mu_err_Fe2, sigma_Fe2, sigma_err_Fe2, N_Fe2_ref, A2 = fit_peak(energies, counts_s, peak_center=7.06, window=0.3, plot=False)
        total_counts_REF_Fe2 = sum_up_sigma(energies, gaussian_without_background(energies, A2, mu_Fe2, sigma_Fe2), mu_Fe2, sigma_Fe2)

        total_counts_REF_Fe2_s = total_counts_REF_Fe2/time_Fe # iciii
        total_counts_REF_Fe1_s = total_counts_REF_Fe1/time_Fe# iciii

        print("RÉSOLUTION FER")
        print("6.4 keV", 2.35482*sigma_Fe1, 2.35482*sigma_err_Fe1)
        print("7.06 keV", 2.35482*sigma_Fe2, 2.35482*sigma_err_Fe2)
    if element == "Pb":
        mu_Pb1, mu_err_Pb, sigma_Pb1, sigma_err_Pb1, _, A = fit_peak(energies, counts_s, peak_center=9.18, window=0.5, plot=False) 
        mu_Pb2, mu_err_Pb2, sigma_Pb2, sigma_err_Pb2, _, A = fit_peak(energies, counts_s, peak_center=10.55, window=0.5, plot=False)
        mu_Pb3, mu_err_Pb3, sigma_Pb3, sigma_err_Pb3, _, A = fit_peak(energies, counts_s, peak_center=12.61, window=0.5, plot=False)
        mu_Pb4, mu_err_Pb4, sigma_Pb4, sigma_err_Pb4, _, A = fit_peak(energies, counts_s, peak_center=14.76, window=0.5, plot=False)

# ratio_ref_Fe1_Cu1 = total_counts_REF_Fe1/total_counts_REF_Cu1
# ratio_ref_Fe1_Cu2 = total_counts_REF_Fe1/total_counts_REF_Cu2
ratio_ref_Fe1_Cu1 = total_counts_REF_Fe1_s/total_counts_REF_Cu1_s # iciii
ratio_ref_Fe1_Cu2 = total_counts_REF_Fe1_s/total_counts_REF_Cu2_s # iciii
# ratio_ref_Fe1_Cu1 = N_Fe1_ref/N_Cu1_ref
# ratio_ref_Fe1_Cu2 = N_Fe1_ref/N_Cu2_ref
# i_ratio_ref_Fe1_Cu1 = calcule_incertitudes_ratio(ratio_ref_Fe1_Cu1, total_counts_REF_Fe1, np.sqrt(total_counts_REF_Fe1), total_counts_REF_Cu1, np.sqrt(total_counts_REF_Cu1))
# i_ratio_ref_Fe1_Cu2 = calcule_incertitudes_ratio(ratio_ref_Fe1_Cu2, total_counts_REF_Fe1, np.sqrt(total_counts_REF_Fe1), total_counts_REF_Cu2, np.sqrt(total_counts_REF_Cu2))
i_ratio_ref_Fe1_Cu1 = calcule_incertitudes_ratio(ratio_ref_Fe1_Cu1, total_counts_REF_Fe1_s, np.sqrt(total_counts_REF_Fe1)/time_Fe, total_counts_REF_Cu1_s, np.sqrt(total_counts_REF_Cu1)/time_Cu) #iciii
i_ratio_ref_Fe1_Cu2 = calcule_incertitudes_ratio(ratio_ref_Fe1_Cu2, total_counts_REF_Fe1_s, np.sqrt(total_counts_REF_Fe1)/time_Fe, total_counts_REF_Cu2_s, np.sqrt(total_counts_REF_Cu2)/time_Cu) #iciii



"""SECTION RÉSOLUTION"""
def resolution_theorique(energies):
    return (((260**2) - (120**2) + 2440*energies)**(1/2))/1000

def racine_carre(energies, A, k):
    return ((k + (A*energies))**(1/2))

resolutions = 2.35482*np.array([sigma_Ag1, sigma_Ag2, sigma_Cu1, sigma_Cu2, sigma_Fe1, sigma_Fe2, sigma_Pb1, sigma_Pb1, sigma_Pb3, sigma_Pb4])
i_resolutions = 2.35482*np.array([sigma_err_Ag1, sigma_err_Ag2, sigma_err_Cu1, sigma_err_Cu2, sigma_err_Fe1, sigma_err_Fe2, sigma_err_Pb1, sigma_err_Pb1, sigma_err_Pb3, sigma_err_Pb4])
energies_pics = [22.16, 24.94, 8.05, 8.91, 6.4, 7.06, 9.18, 10.55, 12.61, 14.76]

popt, pcov = curve_fit(racine_carre, energies_pics, resolutions, p0=[0.0012, 0.01], sigma=i_resolutions) #(2440/(1000**0.5)), (((260**2)-(120**2))/(1000**0.5))
i_fit = np.diag(pcov)**0.5

# plt.plot(energies_pics, resolutions, "ro")
plt.errorbar(energies_pics, resolutions, yerr=i_resolutions, fmt='ro', ecolor = 'black',capsize=5, label = 'Énergies maximales')
plt.plot(energies, racine_carre(energies, popt[0], popt[1]), color="red", linestyle="--", label="Fonction ajustée")
plt.plot(energies, resolution_theorique(energies), "blue", label="Résolution théorique")
plt.fill_between(energies, racine_carre(energies, popt[0]-i_fit[0], popt[1]-i_fit[1]), racine_carre(energies, popt[0]+i_fit[0], popt[1]+i_fit[1]), alpha=0.2, color="red")
plt.text(15, 0.25, fr"$FWHM_{{théor.}} = \frac{{\sqrt{{53200 + 2440 E}}}}{{1000}}$", fontsize=14)#
plt.text(0, 0.44, fr"$FWHM_{{fit}} = \sqrt{{{popt[1]:.3g} + {popt[0]:.3g} E}}$", fontsize=14)#
plt.xlabel("Énergies [keV]", fontsize=16)
plt.ylabel("Résolution (FWHM) [keV]", fontsize=16)
plt.legend(fontsize=14)
plt.show()




folder = f"spectres_bruts/fluorescence_csv/alliages"
# files = os.listdir(folder)



files = os.listdir(folder)
for file in files:
    print(file)
    energies, counts, time = pre_process(folder, file)
    try:    
        # mu_Cu1, mu_err_Cu, sigma_Cu1, sigma_err_Cu1, N_Cu1, A1 = fit_peak(energies, counts, peak_center=8.05, window=0.4)
        mu_Cu1, mu_err_Cu, sigma_Cu1, sigma_err_Cu1, N_Cu1, A1 = fit_peak(energies, time*counts, peak_center=8.05, window=0.4)#iciii
        plt.plot(energies, gaussian_without_background(energies, A1, mu_Cu1, sigma_Cu1))
        plt.plot(energies, counts)
        plt.show()
        total_counts_Cu1 = sum_up_sigma(energies, gaussian_without_background(energies, A1, mu_Cu1, sigma_Cu1), mu_Cu1, sigma_Cu1)
        total_counts_Cu1_s = sum_up_sigma(energies, gaussian_without_background(energies, A1, mu_Cu1, sigma_Cu1), mu_Cu1, sigma_Cu1)/time #iciii

  

        # mu_Cu2, mu_err_Cu2, sigma_Cu2, sigma_err_Cu2, N_Cu2, A2 = fit_peak(energies, counts, peak_center=8.91, window=0.3)#0.5 pour cle
        mu_Cu2, mu_err_Cu2, sigma_Cu2, sigma_err_Cu2, N_Cu2, A2 = fit_peak(energies, time*counts, peak_center=8.91, window=0.3)#0.5 pour cle # iciiii
        plt.plot(energies, gaussian_without_background(energies, A2, mu_Cu2, sigma_Cu2))
        plt.plot(energies, counts)
        plt.show()
        total_counts_Cu2 = sum_up_sigma(energies, gaussian_without_background(energies, A2, mu_Cu2, sigma_Cu2), mu_Cu2, sigma_Cu2)
        total_counts_Cu2_s = sum_up_sigma(energies, gaussian_without_background(energies, A2, mu_Cu2, sigma_Cu2), mu_Cu2, sigma_Cu2)/time #iciiii
        # ratio_Cu2 = total_counts_Cu2/total_counts_REF_Cu2
        # print(ratio_Cu2)

        # mu_Fe1, mu_err_Fe, sigma_Fe1, sigma_err_Fe1, N_Fe1, A3 = fit_peak(energies, counts, peak_center=6.4, window=0.5)
        mu_Fe1, mu_err_Fe, sigma_Fe1, sigma_err_Fe1, N_Fe1, A3 = fit_peak(energies, time*counts, peak_center=6.4, window=0.5) #iciiii
        plt.plot(energies, gaussian_without_background(energies, A3, mu_Fe1, sigma_Fe1))
        plt.plot(energies, counts)
        plt.show()
        total_counts_Fe1 = sum_up_sigma(energies, gaussian_without_background(energies, A3, mu_Fe1, sigma_Fe1), mu_Fe1, sigma_Fe1)
        total_counts_Fe1_s = sum_up_sigma(energies, gaussian_without_background(energies, A3, mu_Fe1, sigma_Fe1), mu_Fe1, sigma_Fe1)/time #iciii
        # ratio_Fe1 = total_counts_Fe1/total_counts_REF_Fe1
        # print(ratio_Fe1)

        ratio_Fe1_Cu1 = total_counts_Fe1/total_counts_Cu1
        ratio_Fe1_Cu2 = total_counts_Fe1/total_counts_Cu2
        ratio_Fe1_Cu1 = total_counts_Fe1_s/total_counts_Cu1_s # iciii
        ratio_Fe1_Cu2 = total_counts_Fe1_s/total_counts_Cu2_s # iciii
        # ratio_Fe1_Cu1 = N_Fe1/N_Cu1
        # ratio_Fe1_Cu2 = N_Fe1/N_Cu2
        i_ratio_Fe1_Cu1 = calcule_incertitudes_ratio(ratio_Fe1_Cu1, total_counts_Fe1, np.sqrt(total_counts_Fe1), total_counts_Cu1, np.sqrt(total_counts_Cu1))
        i_ratio_Fe1_Cu2 = calcule_incertitudes_ratio(ratio_Fe1_Cu2, total_counts_Fe1, np.sqrt(total_counts_Fe1), total_counts_Cu2, np.sqrt(total_counts_Cu2))
        i_ratio_Fe1_Cu1 = calcule_incertitudes_ratio(ratio_Fe1_Cu1, total_counts_Fe1_s, np.sqrt(total_counts_Fe1)/time, total_counts_Cu1_s, np.sqrt(total_counts_Cu1)/time) #iciii
        i_ratio_Fe1_Cu2 = calcule_incertitudes_ratio(ratio_Fe1_Cu2, total_counts_Fe1_s, np.sqrt(total_counts_Fe1)/time, total_counts_Cu2_s, np.sqrt(total_counts_Cu2)/time) #iciii


        intensite_rel_1 = ratio_Fe1_Cu1/ratio_ref_Fe1_Cu1
        intensite_rel_2 = ratio_Fe1_Cu2/ratio_ref_Fe1_Cu2
        i_intensite_rel_1 = calcule_incertitudes_ratio(intensite_rel_1, ratio_Fe1_Cu1, i_ratio_Fe1_Cu1, ratio_ref_Fe1_Cu1, ratio_ref_Fe1_Cu1)
        i_intensite_rel_2 = calcule_incertitudes_ratio(intensite_rel_2, ratio_Fe1_Cu2, i_ratio_Fe1_Cu2, ratio_ref_Fe1_Cu2, i_ratio_ref_Fe1_Cu2)
        
        print(intensite_rel_1, i_intensite_rel_1)
        print(intensite_rel_2, i_intensite_rel_2)
        


        mu_Fe2, mu_err_Fe2, sigma_Fe2, sigma_err_Fe2, _, A4 = fit_peak(energies, counts, peak_center=7.06, window=0.3)
        plt.plot(energies, gaussian_without_background(energies, A4, mu_Fe2, sigma_Fe2))
        plt.plot(energies, counts)
        plt.show()
        total_counts_Fe2 = sum_up_sigma(energies, gaussian_without_background(energies, A4, mu_Fe2, sigma_Fe2), mu_Fe2, sigma_Fe2)
        # ratio_Fe2 = total_counts_Fe2/total_counts_REF_Fe2
        # print(ratio_Fe2)

    except:
        pass









