import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit


def mcmc_weighted_mean(energies, counts, n_samples=5000):
    means = []

    for _ in range(n_samples):
        # Sample counts from Poisson
        sampled_counts = np.random.poisson(counts)

        # Avoid division by zero
        if np.sum(sampled_counts) == 0:
            continue

        sampled_counts_s = sampled_counts / np.sum(sampled_counts)

        # Weighted mean
        mean = np.sum(sampled_counts * energies) / np.sum(sampled_counts)
        means.append(mean)

    means = np.array(means)
    return np.mean(means), np.std(means)

def mcmc_energie_max(energies, counts, threshold=8, n_samples=5000):
    emax_samples = []

    for _ in range(n_samples):
        sampled_counts = np.random.poisson(counts)

        # Skip empty spectra
        if np.sum(sampled_counts) == 0:
            continue

        try:
            emax = find_energie_max(energies, sampled_counts, threshold)
            emax_samples.append(emax)
        except IndexError:
            # No bin above threshold → skip
            continue

    emax_samples = np.array(emax_samples)

    if len(emax_samples) == 0:
        return np.nan, np.nan

    return np.mean(emax_samples), np.std(emax_samples)












def find_energie_max(energies, counts, threshold=8):#3 pour tension, 8 pour courant
    # masque > seuil de bruit
    mask = counts > threshold
    # Indice du dernier point qui dépasse le seuil
    indices = np.where(mask)[0]

    idx_max = indices[-1]
    return energies[idx_max]

def droite(x, a, b):
    return (a*x)+b

def constante(x, k):
    return k*np.ones(len(x))


def courant_ou_tension(param_interet="tension"):
    folder = f"spectres_bruts/filtration_csv/{param_interet}_variable"

    energies_moy = []
    i_energies_moy = []
    energies_max = []
    i_energies_moy = []
    files = os.listdir(folder)

    for file in files:
        df = pd.read_csv(os.path.join(folder, file))
        time = float(df.columns[0]) 

        index = df.index.to_numpy()
        counts = (df[df.columns[0]]).to_numpy()
        counts_s = counts/time

        energies = (0.017905141963904726*index) - 0.35239275479242

        diff_energies = np.mean(np.diff(energies))
        
        # plt.plot(energies, counts)
        # plt.show()

        energie_moy = np.average(energies, weights=counts_s)
        energies_moy.append(energie_moy)

        i_energie_moy = np.sqrt(np.sum(counts_s*(energies-energie_moy)**2)/((np.sum(counts))**2))
        # i_energie_moy = np.sqrt(1/(np.sum(counts)**2))
        print(i_energie_moy)
        i_energies_moy.append(i_energie_moy)

        # energie_moy, i_energie_moy = mcmc_weighted_mean(energies, counts)
        # i_energies_moy.append(i_energie_moy)
        # print(i_energie_moy)

        energie_max = find_energie_max(energies, counts)
        energies_max.append(energie_max)
    

        # energie_max, i_energie_max = mcmc_energie_max(energies, counts)
        # energies_max.append(energie_max)
        # print(i_energie_max)


    if param_interet == "tension": 
        tensions = np.array([10, 20, 25, 30, 35, 40, 45, 50])  
        popt_moy, pcov_moy = curve_fit(droite, tensions, energies_moy)
        popt_max, pcov_max = curve_fit(droite, tensions, energies_max)
        x=np.linspace(10, 50, 1000)
        plt.plot(x, droite(x, popt_moy[0], popt_moy[1]), color="blue", linestyle="--")
        plt.plot(x, droite(x, popt_max[0], popt_max[1]), color="red", linestyle="--")
        plt.plot(tensions, energies_moy, 'ro')
        plt.plot(tensions, energies_max, 'bo')
        i_moy = np.diag(pcov_moy)**0.5
        i_max = np.diag(pcov_max)**0.5
        print(i_moy, popt_moy)
        droite_min_moy = (popt_moy[0]-(2*i_moy[0]))*tensions + (popt_moy[1]-2*i_moy[1])
        droite_max_moy = (popt_moy[0]+(2*i_moy[0]))*tensions + (popt_moy[1]+2*i_moy[1])
        droite_min_max = (popt_max[0]-(2*i_max[0]))*tensions + (popt_max[1]-2*i_max[1])
        droite_max_max = (popt_max[0]+(2*i_max[0]))*tensions + (popt_max[1]+2*i_max[1])
        erreur_test = (diff_energies*np.ones(len(i_energies_moy)))
        plt.errorbar(tensions, energies_moy, yerr=4*erreur_test, fmt='bo', ecolor = 'black', label = 'Énergies moyennes')
        plt.fill_between(tensions, droite_min_moy, droite_max_moy, alpha=0.2, color="blue")
        # plt.plot(courants, energies_max, 'bo')
        plt.errorbar(tensions, energies_max, yerr=4*erreur_test, fmt='ro', ecolor = 'black', label = 'Énergies maximales')
        plt.fill_between(tensions, droite_min_max, droite_max_max, alpha=0.2, color="red")
        plt.legend()
        plt.xlabel(r"Tension [kV]", fontsize=14)
        plt.text(10, 38, r"$E_{max}$"+ f" = {popt_max[0]:.2f} kV {popt_max[1]:.2f}", fontsize=12)#
        plt.text(30, 10, r"$E_{moy}$"+ f' = {popt_moy[0]:.2f} kV +{popt_moy[1]:.2f}', fontsize=12)#
        plt.ylabel(r"Énergie [keV]", fontsize=14)
        plt.show()

    else:
        courants = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        popt_moy, pcov_moy = curve_fit(constante, courants, energies_moy)
        popt_max, pcov_max = curve_fit(constante, courants, energies_max)

        x=np.linspace(10, 100, 1000)
        plt.plot(x, constante(x, popt_moy[0]), color="blue", linestyle="--")
        plt.plot(x, constante(x, popt_max[0]), color="red", linestyle="--")
        print(pcov_moy[0]**0.5, pcov_max[0]**0.5)

        erreur_test = np.array(energies_moy) * 0.05
        erreur_test = i_energies_moy + (diff_energies*np.ones(len(i_energies_moy)))
        plt.errorbar(courants, energies_moy, yerr=4*erreur_test, fmt='bo', ecolor = 'black', label = 'Énergies moyennes')
        plt.fill_between(courants, energies_moy-3*(pcov_moy[0]**0.5), energies_moy+3*(pcov_moy[0]**0.5), alpha=0.2, color="blue")
        # plt.plot(courants, energies_max, 'bo')
        plt.errorbar(courants, energies_max, yerr=4*erreur_test, fmt='ro', ecolor = 'black', label = 'Énergies maximales')
        plt.fill_between(courants, energies_max-3*(pcov_max[0]**0.5), energies_max+3*(pcov_max[0]**0.5), alpha=0.2, color="red")
        plt.legend()
        plt.xlabel(r"Courant [$\mu$A]", fontsize=14)
        plt.text(10, 38, r"$E_{max}$"+ f' = ({popt_max[0]:.1f} ± {(pcov_max[0]**0.5)[0]:.1f}) keV', fontsize=12)#
        plt.text(10, 27, r"$E_{moy}$"+ f' = ({popt_moy[0]:.1f} ± {(pcov_moy[0]**0.5)[0]:.1f}) keV', fontsize=12)#
        plt.ylabel(r"Énergie [keV]", fontsize=14)
        plt.show()

# incertitude du bruit de fond

courant_ou_tension("tension")
courant_ou_tension("courant")