from math import ceil, floor
import numpy as np
from numpy import mean, amax, amin
from scipy.io import loadmat
from matplotlib import pyplot as plt
import time
import os
import queue
import threading
import multiprocessing as mp

def normalize(signal):
    """ Normalise la matrice en entrée en enlevant la moyenne et en mettant à l'échelle entre -1 et 1 """
    result = signal - mean(signal)
    result *= 1/max(abs(amin(result)), amax(result))
    return result

def calculer_limites(longueur_signal, largeur_fenetre, compression, deplacement):
    """ Calcule les limites de la zone d'intérêt pour la corrélation"""
    deplacement_maximum = int(longueur_signal * compression)
    centre = deplacement + (largeur_fenetre // 2)
    centre_attendu = ((1 - compression) * centre) + deplacement_maximum
    largeur_recherche = largeur_fenetre + 25

    debut = floor(centre_attendu - (largeur_recherche/2))
    if debut < (longueur_signal * compression):
        debut = int(longueur_signal * compression)

    fin = ceil(centre_attendu + (largeur_recherche/2))
    if fin > longueur_signal:
        fin = longueur_signal

    return (debut, fin)

def correlate(signal1, signal2, largeur_fenetre, compression):
    """ Retourne la corrélation du signal 1 avec le signal 2 """
    deplacement1 = 0
    longueur_signal = len(signal1)
    correlation = []
    
    while deplacement1+largeur_fenetre < len(signal1):
        coefficients = []
        # Fenêtrage du signal original
        debut, fin = calculer_limites(longueur_signal, largeur_fenetre, compression, deplacement1)
        # Normalisation
        window1 = normalize(signal1[deplacement1:deplacement1+largeur_fenetre])

        deplacement2 = debut

        while deplacement2+largeur_fenetre < fin:
            # Fenetrage du signal compressé
            window2 = normalize(signal2[deplacement2:deplacement2+largeur_fenetre])
            # Corrélation (produit point-par-point)
            somme = np.sum(window1 * window2)
            coefficients.append(somme)
            deplacement2 += 1
        
        # Calcule le déplacement (garde l'index de la corrélation la plus élevée)
        correlation.append((np.argmax(coefficients) + debut) - deplacement1)
        deplacement1 += 1

    return correlation

def calculer_colonne_deformation(colonne_initiale, colonne_compressee, largeur_fenetre):
    """ Calcule la déformation de la colonne en fonction des colonnes initiales et compressee """
    # Calcul du champ de déplacement
    correlation = correlate(colonne_initiale, colonne_compressee, largeur_fenetre, compression)
    longueur = len(correlation) # longeur de la corrélation

    polynome_deplacement = np.polyfit(np.arange(longueur), correlation, 10) # Approximation polynomiale du champ de déplacement
    polynome_deformation = np.polyder(polynome_deplacement) # Dérivation du champ de déplacement -> champ de déformation

    return np.polyval(polynome_deformation, np.arange(longueur)) # Évaluation du polynôme du champ de déformation
    
def processus_calcul_deformation(colonne_depart, colonne_fin, donnees_initiales, donnees_compressees, largeur_fenetre, queue):
    """ Code du processus pour calculer le champ de déformation """
    for x in range(colonne_depart, colonne_fin):
        # Calcule la déformation pour cette colonne
        resultat = calculer_colonne_deformation(donnees_initiales[:,x], donnees_compressees[:,x], largeur_fenetre)
        # Ajoute à la queue de résultat
        queue.put((x, resultat))

def afficher_temps(depart, maintenant, colonnes):
    """ Affiche le progrès de la tâche et l'estimation du temps restant """
    progres = colonnes/1000
    ecoule = time.time() - depart
    if progres == 0:
        restant = float('inf')
    else:
        restant = (ecoule/progres) - ecoule

    print(f"Colonnes: {colonnes}/1000 ({progres * 100:2.1f}%), Temps: {formatter_temps(ecoule)} (≈{formatter_temps(restant)} restants)")
    
def formatter_temps(secondes):
    """ Formatte le nombre de secondes en une chaine du type 1h, 2m, 30s """    
    if secondes > 60:
        minutes = int(secondes // 60)
        secondes = int(secondes % 60)

        if minutes > 60:
            heures = int(minutes // 60)
            minutes = int(minutes % 60)
            return f"{heures}h {minutes}m {secondes}s"
        else:
            return f"{minutes}m {secondes}s"
    else:
        return f"{int(secondes)}s"

# Charge les images
ut_init = loadmat("UT_INIT.mat")["IMAGE_INIT"]
ut_25pc = loadmat("UT_2,5pc.mat")["IMAGE_MOD"]
ut_50pc = loadmat("UT_5pc.mat")["IMAGE_MOD"]
ut_75pc = loadmat("UT_7,5pc.mat")["IMAGE_MOD"]

# Organise les images commpressées
images = [
    {"compression": 2.5 / 100, "data": ut_25pc},
    {"compression": 5 / 100, "data": ut_50pc},
    {"compression": 7.5 / 100, "data": ut_75pc},
]

# colonnes = [200, 350, 500, 650, 800] # Colonnes d'intérêt
largeur_fenetre = 128 # Largeur de la fenêtre de corrélation

nombre_threads = 11 # Nombre de processus de calcul
queue = mp.Queue() # Queue de résultats des processus

barre_de_couleur_affichee = False # Si la barre de couleur a déjà été affichée

for image in images:
    n_colonnes = 0
    compression = image["compression"]
    donnees = image["data"]
    image_deformation = np.zeros(shape=(1000-largeur_fenetre, 1000))

    for x in range(0,nombre_threads):
        # Déterminer la colonne de départ et de fin
        colonne_depart = int(x*(1000 / nombre_threads))
        colonne_fin = int((x+1)*(1000 / nombre_threads))

        # Démarrer le processus
        processus = mp.Process(target=processus_calcul_deformation, args=(colonne_depart, colonne_fin, ut_init, donnees, largeur_fenetre, queue))
        processus.start()

    depart = time.time()
    
    while n_colonnes < 1000:
        # Récupére le résultat de la queue
        colonne, resultat = queue.get()
        # Ajoute la colonne à l,image
        image_deformation[:,colonne] = resultat

        # Affiche le progrès
        n_colonnes += 1
        afficher_temps(depart, time.time(), n_colonnes)

    # Affiche l'image finale
    plt.imshow(image_deformation)
    plt.title(f"Champ de déformation avec compression à {compression * 100:2.1f}%")
    plt.xlabel("Position en x (px)")
    plt.ylabel("Position en y (px)")

    if not barre_de_couleur_affichee:
        colorbar = plt.colorbar()
        colorbar.set_label("Rigidité")
        barre_de_couleur_affichee = True

    # Change les limites du colormap pour mieux voir les masses
    cmap_min = np.amin(image_deformation[30:np.shape(image_deformation)[0] - 30,:])
    cmap_max = np.amax(image_deformation[30:np.shape(image_deformation)[0] - 30,:])
    plt.clim(cmap_min, cmap_max)

    # Sauvegarde la figure
    plt.savefig(f"figures/champ-deformation-{int(compression * 1000)}pc.pdf")

os.system("aplay /home/tommy/Downloads/saria_s_song_lost_woods_metal_rock_remix_legend_of_zelda_ocarina_of_time_theonlydeeralive_161050648240297768.wav")