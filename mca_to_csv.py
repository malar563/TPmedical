


import pandas as pd
import os

folder_initial = r"spectres_bruts/fluorescence"
folder_final = r"spectres_bruts/fluorescence_csv"

# Créer le dossier de destination s'il n'existe pas
if not os.path.exists(folder_final):
    os.makedirs(folder_final)

files = os.listdir(folder_initial)

for file in files:
    real_time_value = "Counts"  # Valeur par défaut si non trouvé
    data = []
    
    with open(os.path.join(folder_initial, file), "r") as f:
        lines = f.readlines()

    # Extraction du Real Time et des données
    in_data_section = False
    
    for line in lines:
        clean_line = line.strip()
        
        # Chercher la valeur de REAL_TIME
        if "REAL_TIME -" in clean_line:
            try:
                # On sépare par le tiret et on prend la partie numérique
                real_time_value = clean_line.split("-")[1].strip()
            except IndexError:
                pass
        
        # Détecter le début de la section DATA pour éviter de convertir des paramètres de configuration par erreur
        if "<<DATA>>" in clean_line:
            in_data_section = True
            continue
            
        # Extraire les valeurs numériques uniquement dans la section DATA
        if in_data_section:
            try:
                value = float(clean_line)
                data.append(value)
            except ValueError:
                pass

    # Création du DataFrame avec le real_time comme titre de colonne
    df = pd.DataFrame(data, columns=[real_time_value])
    
    # Sauvegarde
    csv_file_name = os.path.join(folder_final, os.path.splitext(file)[0]) + ".csv"
    df.to_csv(csv_file_name, index=False)

print("Terminé!!")