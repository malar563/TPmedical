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

    # METTRE DROITE D'ÉTALONNAGE
    energie_moy = np.average(index, weights=counts)
    # COMMENT TROUVER ÉNERGIE MAX


    plt.plot(index, counts)
    plt.show()