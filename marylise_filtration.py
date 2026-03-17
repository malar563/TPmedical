import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit


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
        
        # plt.plot(energies, counts)
        # plt.show()

        energie_moy = np.average(energies, weights=counts_s)
        energies_moy.append(energie_moy)

        i_energie_moy = np.sqrt(np.sum(counts_s*(energies-energie_moy)**2)/((np.sum(counts))**2))
        print(i_energie_moy)
        i_energies_moy.append(i_energie_moy)

        energie_max = find_energie_max(energies, counts)
        energies_max.append(energie_max)


    if param_interet == "tension": 
        tensions = [10, 20, 25, 30, 35, 40, 45, 50]  
        popt_moy, pcov_moy = curve_fit(droite, tensions, energies_moy)
        popt_max, pcov_max = curve_fit(droite, tensions, energies_max)
        x=np.linspace(10, 100, 1000)
        plt.plot(x, droite(x, popt_moy[0], popt_moy[1]), color="black", linestyle="--")
        plt.plot(x, droite(x, popt_max[0], popt_max[1]), color="black", linestyle="--")
        plt.plot(tensions, energies_moy, 'ro')
        plt.plot(tensions, energies_max, 'bo')
        plt.show()

    else:
        courants = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        popt_moy, pcov_moy = curve_fit(constante, courants, energies_moy)
        popt_max, pcov_max = curve_fit(constante, courants, energies_max)
        x=np.linspace(10, 100, 1000)
        plt.plot(x, constante(x, popt_moy[0]), color="black", linestyle="--")
        plt.plot(x, constante(x, popt_max[0]), color="black", linestyle="--")
        # plt.plot(courants, energies_moy, 'ro')
        plt.errorbar(courants, energies_moy, yerr=i_energies_moy, fmt='ro')
        plt.plot(courants, energies_max, 'bo')
        plt.show()

courant_ou_tension("courant")