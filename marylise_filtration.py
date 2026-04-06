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

# Calcul de RMSE
def rmse(y, y_fit):
    return np.sqrt(np.mean((y - y_fit)**2))

def incertitude_E_moy(list_counts, energies_canaux):
    # Moyenne pondérée = Somme(A*B)/Somme(B)
    i_list_counts = np.sqrt(list_counts)
    i_energies_canaux = np.diff(energies_canaux)[0]*np.ones(len(energies_canaux))

    # Incertitude A*B
    i_AB = (list_counts*energies_canaux)* np.sqrt((i_list_counts/list_counts)**2 + (i_energies_canaux/energies_canaux)**2)

    # Incertitude Somme(A*B)
    i_sommeAB = np.sum(i_AB**2)

    # Incertitude Somme(B)
    i_sommeB = np.sum(i_list_counts**2)

    return np.average(energies_canaux, weights=list_counts) * np.sqrt((i_sommeAB/np.sum(list_counts*energies_canaux))**2 + (i_sommeB/np.sum(list_counts)))


def courant_ou_tension(param_interet="tension", threshold=3):
    folder = f"spectres_bruts/filtration_csv/{param_interet}_variable"
    
    spectres = []
    noms_spectres = []
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

        spectres.append(counts_s)
        noms_spectres.append(file)

        energies = (0.017905141963904726*index) - 0.35239275479242

        diff_energies = np.mean(np.diff(energies))
        
        # plt.plot(energies, counts)
        # plt.show()

        energie_moy = np.average(energies, weights=counts_s)
        energies_moy.append(energie_moy)

        i_energie_moy = np.sqrt(np.sum(counts*(energies-energie_moy)**2)/((np.sum(counts))**2)) # rapport 2a j'avais ça
        i_energie_moy = incertitude_E_moy(counts_s, energies)
        # print(i_energie_moy)
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
        plt.legend(fontsize=14)
        plt.xlabel(r"Tension [kV]", fontsize=16)
        plt.text(10, 38, r"$E_{max}$"+ f" = {popt_max[0]:.2f} T {popt_max[1]:.2f}", fontsize=14)#
        plt.text(10, 34, r"$RMSE$"+ f" = {rmse(energies_max, droite(tensions, popt_max[0], popt_max[1])):.4f} keV", fontsize=14)#
        plt.text(30, 10, r"$E_{moy}$"+ f' = {popt_moy[0]:.2f} T +{popt_moy[1]:.2f}', fontsize=14)#
        plt.text(30, 6, r"$RMSE$"+ f" = {rmse(energies_moy, droite(tensions, popt_moy[0], popt_moy[1])):.4f} keV", fontsize=14)#
        plt.ylabel(r"Énergie [keV]", fontsize=16)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.tight_layout()
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
        plt.xlabel(r"Tension [kV]", fontsize=16)
        plt.text(10, 3000, fr"N = {popt[0]:.2f} (T -  {popt[1]:.2f})$^2$ + {popt[2]:.2f}", fontsize=14)#
        plt.text(10, 2600, r"$RMSE$"+ f" = {rmse(list_counts, parabole(tensions, popt[0], popt[1], popt[2])):.4f}", fontsize=14)# RÉSULTAT BIZARRE : À METTRE?
        plt.ylabel(r"$N$/$s$ [s$^{-1}$]", fontsize=16)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.tight_layout()
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
        plt.fill_between(courants, popt_moy[0]-1*(pcov_moy[0]**0.5), popt_moy[0]+1*(pcov_moy[0]**0.5), alpha=0.2, color="blue")
        # plt.plot(courants, energies_max, 'bo')
        plt.errorbar(courants, energies_max, yerr=4*erreur_test, fmt='ro', ecolor = 'black', label = 'Énergies maximales')
        plt.fill_between(courants, popt_max[0]-1*(pcov_max[0]**0.5), popt_max[0]+1*(pcov_max[0]**0.5), alpha=0.2, color="red")
        plt.legend(fontsize=14)
        plt.xlabel(r"Courant [$\mu$A]", fontsize=16)
        plt.text(10, 38, r"$E_{max}$"+ f' = ({popt_max[0]:.1f} ± {(pcov_max[0]**0.5)[0]:.1f}) keV', fontsize=14)#
        plt.text(10, 36, r"$RMSE$"+ f" = {rmse(energies_max, popt_max[0]*np.ones(len(energies_max))):.4f} keV", fontsize=14)#
        plt.text(10, 28, r"$E_{moy}$"+ f' = ({popt_moy[0]:.1f} ± {(pcov_moy[0]**0.5)[0]:.1f}) keV', fontsize=14)#
        plt.text(10, 26, r"$RMSE$"+ f" = {rmse(energies_moy, popt_moy[0]*np.ones(len(energies_moy))):.4f} keV", fontsize=14)#
        plt.ylabel(r"Énergie [keV]", fontsize=16)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.tight_layout()
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
        plt.legend(fontsize=14)
        plt.xlabel(r"Courant [$\mu$A]", fontsize=16)
        plt.text(40, 5000, fr"N = {popt[0]:.2f} I {popt[1]:.2f}", fontsize=14)#
        plt.text(40, 3500, r"$RMSE$"+ f" = {rmse(list_counts, droite(courants, popt[0], popt[1])):.4f}", fontsize=14) # RÉSULTAT BIZARRE : À METTRE?
        plt.ylabel(r"Nombre de comptes par seconde", fontsize=16)
        plt.show()
    return energies, spectres, noms_spectres


# incertitude du bruit de fond

energies, spectres, noms_spectres = courant_ou_tension("tension", threshold=3)
noms_spectres_reversed = list(reversed(noms_spectres))

for i, data in enumerate(reversed(spectres)):
    plt.plot(energies, data, label=fr"{noms_spectres_reversed[i][18:20]} kV")
plt.legend(fontsize=14)
plt.xlabel("Énergie [keV]",fontsize=16)
plt.ylabel(r"$N$/$s$ [s$^{-1}$]", fontsize=16)
plt.show()

energies, spectres, noms_spectres = courant_ou_tension("courant", threshold=8)
noms_spectres_reversed = list(reversed(noms_spectres))
for i, data in enumerate(reversed(spectres)):
    plt.plot(energies, data, label=fr"{noms_spectres_reversed[i][23:25]} $\mu$A")
plt.legend(fontsize=14)
plt.xlabel("Énergie [keV]", fontsize=16)
plt.ylabel(r"$N$/$s$ [s$^{-1}$]", fontsize=16)
plt.show()