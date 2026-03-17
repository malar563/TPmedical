import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

folder = "spectres_bruts/filtration_csv/tension_variable"

files = os.listdir(folder)

for file in files:
    df = pd.read_csv(os.path.join(folder, file))
    time = float(df.columns[0]) 

    index = df.index.to_numpy()
    counts = (df[df.columns[0]]/time).to_numpy()

    energies = (0.017905141963904726*index) - 0.35239275479242
    
    energie_moy = np.average(energies, weights=counts)
    print(energie_moy)
    
    # COMMENT TROUVER ÉNERGIE MAX


    # plt.plot(energies, counts)
    # plt.show()