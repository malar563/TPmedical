import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit




def find_energie_max(energies, counts, threshold=3):#3 pour tension, 8 pour courant
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

def parabole(x, a, h, k):
    return (a* ((x-h)**2)) +k


def courant_ou_tension(param_interet="tension", threshold=3):
    folder = f"spectres_bruts/filtration_csv/{param_interet}_variable"
    

    energies_moy = []
    i_energies_moy = []
    energies_max = []
    i_energies_moy = []
    list_counts = []
    files = os.listdir(folder)

    for file in files:
        print(file)
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

        i_energie_moy = np.sqrt(np.sum(counts*(energies-energie_moy)**2)/((np.sum(counts))**2))
        # i_energie_moy = np.sqrt(1/(np.sum(counts)**2))
        print(i_energie_moy)
        i_energies_moy.append(i_energie_moy)

        energie_max = find_energie_max(energies, counts, threshold)
        energies_max.append(energie_max)
        list_counts.append(np.sum(counts_s))

    
    

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
        droite_min_moy = (popt_moy[0]-(i_moy[0]))*tensions + (popt_moy[1]-i_moy[1])
        droite_max_moy = (popt_moy[0]+(i_moy[0]))*tensions + (popt_moy[1]+i_moy[1])
        droite_min_max = (popt_max[0]-(i_max[0]))*tensions + (popt_max[1]-i_max[1])
        droite_max_max = (popt_max[0]+(i_max[0]))*tensions + (popt_max[1]+i_max[1])
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



         
        popt, pcov = curve_fit(parabole, tensions, list_counts,  sigma=np.sqrt(list_counts))
        x=np.linspace(10, 50, 1000)
        plt.plot(x, parabole(x, popt[0], popt[1], popt[2]), color="red", linestyle="--")
        # plt.plot(courants, list_counts, 'ro')
        i = np.diag(pcov)**0.5
        print(i)
        droite_min = parabole(x, popt[0]-(i[0]), popt[1]+(i[1]), popt[2]-(i[2]))#()*x + (popt[1]-i[1])
        droite_max = parabole(x, popt[0]+(i[0]), popt[1]-(i[1]), popt[2]+(i[2]))
        # plt.plot(courants, energies_max, 'bo')
        plt.errorbar(tensions, list_counts, yerr=np.sqrt(list_counts), fmt='ro', ecolor = 'black', label = 'Nombre de comptes par seconde')
        plt.fill_between(x, droite_min, droite_max, alpha=0.2, color="red")
        plt.legend()
        plt.xlabel(r"Tension [kV]", fontsize=14)
        plt.text(15, 3000, fr"N = {popt[0]:.2f} (kV -  {popt[1]:.2f})$^2$ + {popt[2]:.2f}", fontsize=12)#
        plt.ylabel(r"Nombre de comptes par seconde", fontsize=14)
        plt.show()

    else:
        courants = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        courants = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        popt_moy, pcov_moy = curve_fit(constante, courants, energies_moy)
        popt_max, pcov_max = curve_fit(constante, courants, energies_max)

        x=np.linspace(10, 100, 1000)
        x=np.linspace(10, 90, 1000)
        plt.plot(x, constante(x, popt_moy[0]), color="blue", linestyle="--")
        plt.plot(x, constante(x, popt_max[0]), color="red", linestyle="--")
        print(pcov_moy[0]**0.5, pcov_max[0]**0.5)

        erreur_test = np.array(energies_moy) * 0.05
        erreur_test = i_energies_moy + (diff_energies*np.ones(len(i_energies_moy)))
        plt.errorbar(courants, energies_moy, yerr=4*erreur_test, fmt='bo', ecolor = 'black', label = 'Énergies moyennes')
        plt.fill_between(courants, popt_moy[0]-3*(pcov_moy[0]**0.5), popt_moy[0]+3*(pcov_moy[0]**0.5), alpha=0.2, color="blue")
        # plt.plot(courants, energies_max, 'bo')
        plt.errorbar(courants, energies_max, yerr=4*erreur_test, fmt='ro', ecolor = 'black', label = 'Énergies maximales')
        plt.fill_between(courants, popt_max[0]-3*(pcov_max[0]**0.5), popt_max[0]+3*(pcov_max[0]**0.5), alpha=0.2, color="red")
        plt.legend()
        plt.xlabel(r"Courant [$\mu$A]", fontsize=14)
        plt.text(10, 38, r"$E_{max}$"+ f' = ({popt_max[0]:.1f} ± {(pcov_max[0]**0.5)[0]:.1f}) keV', fontsize=12)#
        plt.text(10, 27, r"$E_{moy}$"+ f' = ({popt_moy[0]:.1f} ± {(pcov_moy[0]**0.5)[0]:.1f}) keV', fontsize=12)#
        plt.ylabel(r"Énergie [keV]", fontsize=14)
        plt.show()

 
        popt, pcov = curve_fit(droite, courants, list_counts,  sigma=np.sqrt(list_counts))
        x=np.linspace(10, 90, 1000)
        plt.plot(x, droite(x, popt[0], popt[1]), color="red", linestyle="--")
        # plt.plot(courants, list_counts, 'ro')
        i = np.diag(pcov)**0.5
        print(i)
        droite_min = (popt[0]-(i[0]))*courants + (popt[1]-i[1])
        droite_max = (popt[0]+(i[0]))*courants + (popt[1]+i[1])
        # plt.plot(courants, energies_max, 'bo')
        plt.errorbar(courants, list_counts, yerr=np.sqrt(list_counts), fmt='ro', ecolor = 'black', label = 'Nombre de comptes par seconde')
        plt.fill_between(courants, droite_min, droite_max, alpha=0.2, color="red")
        plt.legend()
        plt.xlabel(r"Courant [$\mu$A]", fontsize=14)
        plt.text(40, 5000, fr"N = {popt[0]:.2f} $\mu$A {popt[1]:.2f}", fontsize=12)#
        plt.ylabel(r"Nombre de comptes par seconde", fontsize=14)
        plt.show()


# incertitude du bruit de fond

courant_ou_tension("tension", threshold=3)
courant_ou_tension("courant", threshold=8)