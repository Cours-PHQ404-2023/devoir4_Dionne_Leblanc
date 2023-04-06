import csv

import numba as nb
import numpy as np

"""
Ce document contient les classes et méthodes requises pour effectuer des
simulations du modèle d'Ising.
"""

@nb.njit
def ising_aleatoire(temperature, taille):
    """ Génère une grille aléatoire de spins.

    Arguments
    ---------
    temperature : Température du système.
    taille : La grille a une dimension taille x taille.
    """
    # On initialise aléatoirement des spins de valeurs -1 ou +1
    # sur un grille de dimension taille x taille.
    spins = np.random.randint(0, 2, (taille, taille))
    spins = 2 * spins - 1
    return Ising(temperature, spins)


# Numba permet de compiler la classe pour qu'elle
# soit plus rapide. Il faut attention car certaines
# opérations ne sont plus permises.
@nb.experimental.jitclass([
    ("temperature", nb.float64),
    ("spins", nb.int64[:, :]),
    ("taille", nb.uint64),
    ("energie", nb.int64),
])
class Ising:
    """ Modèle de Ising paramagnétique en 2 dimensions.

    Représente une grille de spins classiques avec un couplage J = +1 entre
    les premiers voisins.

    Arguments
    ---------
    temperature : Température du système.
    spins : Tableau carré des valeurs de spins
    """

    def __init__(self, temperature, spins):
        self.temperature = temperature
        self.spins = spins
        self.taille = np.shape(spins)[0]
        self.energie = self.calcule_energie()

    def difference_energie(self, x, y):
        """Retourne la différence d'énergie comme si le spin à la position (x, y)
        était inversé.
        """
        # Énergie avant le flip
        energie_pre_flip = self.calcule_energie()

        # calcul de l'énergie avec inversion du spin à (x,y)
        self.spins[x,y] *= -1
        energie_post_flip = self.calcule_energie()
        self.spins[x,y] *= -1

        return energie_post_flip - energie_pre_flip

    def iteration_aleatoire(self):
        """Renverse un spin aléatoire avec probabilité ~ e^(-ΔE / T).

        Cette fonction met à jour la grille avec la nouvelle valeur de spin
        """
        random_float = np.random.random() # retourne une valeur aléatoire uniforme comprise dans [0.0, 1.0)
        # coordonnées aléatoires
        random_x_coord = np.random.randint(self.taille)
        random_y_coord = np.random.randint(self.taille)

        Delta_E = self.difference_energie(random_x_coord, random_y_coord)

        if random_float < np.exp(-Delta_E/self.temperature): # flip avec probabilité exp(-Delta_E/T)
            self.spins[random_x_coord, random_y_coord] *= -1

    def simulation(self, nombre_iterations):
        """Simule le système en effectuant des itérations aléatoires.
        """
        for _ in range(nombre_iterations):
            self.iteration_aleatoire()

    def calcule_energie(self):
        """Retourne l'énergie actuelle de la grille de spins."""
        energie = 0
        n = self.taille
        for x in range(n):
            for y in range(n):
                energie -= self.spins[x, y] * self.spins[(x + 1) % n, y]
                energie -= self.spins[x, y] * self.spins[x, (y + 1) % n]
        return energie

    def calcule_aimantation(self):
        """Retourne l'aimantation actuelle de la grille de spins."""
        return np.sum(self.spins)

class Observable:
    """Utilise la méthode du binning pour calculer des statistiques
    pour un observable.

    Arguments
    ---------
    nombre_niveaux : Le nombre de niveaux pour l'algorithme. Le nombre
                     de mesures est exponentiel selon le nombre de niveaux.
    """

    def __init__(self, nombre_niveaux):
        self.nombre_niveaux = nombre_niveaux

        # Les statistiques pour chaque niveau
        self.nombre_valeurs = np.zeros(nombre_niveaux + 1, dtype=np.int64)
        self.sommes = np.zeros(nombre_niveaux + 1)
        self.sommes_carres = np.zeros(nombre_niveaux + 1)

        # La dernière valeur ajoutée à chaque niveau.
        self.valeurs_precedentes = np.zeros(nombre_niveaux + 1)

        # Le niveau que nous allons utiliser.
        # La différence de 6 donne de bons résultats.
        # Voir les notes de cours pour plus de détails.
        self.niveau_erreur = self.nombre_niveaux - 6

        # Les erreurs par niveau de binning
        self.erreurs = np.zeros(self.niveau_erreur + 1)

    def ajout_mesure(self, valeur, niveau=0):
        """Ajoute une mesure.

        Arguments
        ---------
        valeur : Valeur de la mesure.
        niveau : Niveau à lequel ajouter la mesure. Par défaut,
                 le niveau doit toujours être 0. Les autres niveaux
                 sont seulement utilisé pour la récursion.
        """
        self.nombre_valeurs[niveau] += 1
        self.sommes[niveau] += valeur
        self.sommes_carres[niveau] += valeur*valeur
        # Si un nombre pair de valeurs a été ajouté,
        # on peut faire une simplification.
        if self.nombre_valeurs[niveau] % 2 == 0:
            moyenne = (valeur + self.valeurs_precedentes[niveau]) / 2
            self.ajout_mesure(moyenne, niveau + 1)
        else:
            self.valeurs_precedentes[niveau] = valeur

    def est_rempli(self):
        """Retourne vrai si le binnage est complété."""
        return self.nombre_valeurs[0] == 2**self.nombre_niveaux

    def update_erreurs(self):
        """Retourne l'erreur sur le mesure moyenne de l'observable.

        Le dernier niveau doit être rempli avant d'utiliser cette fonction.
        """
        for niveau in range(self.niveau_erreur + 1):
            self.erreurs[niveau] = np.sqrt(
                (
                    self.sommes_carres[niveau]
                    - self.sommes[niveau]**2 / self.nombre_valeurs[niveau]
                ) / (
                    self.nombre_valeurs[niveau]
                    * (self.nombre_valeurs[niveau] - 1)
                )
            )

    def erreur(self):
        """meilleure estimation de l'erreur
        """
        return self.erreurs[self.niveau_erreur]

    def temps_correlation(self):
        """Retourne le temps de corrélation. Basé sur (16.39) des notes à
        David S."""
        # calcul du ratio entre l'erreur estimée initiale et la meilleure estimation
        ratio_des_erreurs = self.erreurs[self.niveau_erreur]/self.erreurs[0]
        return (ratio_des_erreurs*ratio_des_erreurs - 1)/2

    def moyenne(self):
        """Retourne la moyenne des mesures."""
        return self.sommes[0]/self.nombre_valeurs[0] # la moyenne arithmétique des mesures


def etape_monte_carlo(Grille, iter_intermesure, iter_thermalisation, niveaux_binning, update_status_interval=5000):
    """Cette fonction effectue la précdure de Monte-Carlo pour une grille donnée.

    Paramètres
    ----------
    Grille: ising.Ising
        Grille de spin (avant la précdure).
    iter_intermesure: int
        Nombre d'itérations entre les mesures.
    iter_thermalisation: int
        Nombre d'itérations pour réchauffer le système.
    niveaux_binning: int
        L'exposant du nombre de mesure à effectuer (2^niveaux).
    updata_status: int
        Nombre d'itération avant d'afficher un message pendant la procédure.
        Sert à garder l'utilisateur au courant du progrès.

    Retourne
    --------
    Grille: ising.Ising
        Grille de spin (après la précdure).
    Aimantation: float
        Aimantation de la grille.
    Énergie: float
        Énergie de la grille.
    """

    # initialization des observables
    Aimantation = Observable(niveaux_binning)
    Energie = Observable(niveaux_binning)

    # Thermalisation de la grille de spins
    print("Thermalisation")
    Grille.simulation(iter_thermalisation)

    print("Collecte des mesures")
    # remplissage des listes de binning (pas besoin de self.est_rempli...)
    for i in range(2**niveaux_binning):
        if i % update_status_interval == 0: # faire un printout régulier pour les impatients!
            print(f"Iteration {i}")

        # brouillage de la grille entre les mesures
        Grille.simulation(iter_intermesure)

        # Calcul des valeurs actuelles des opérateurs
        aimantation_courante = Grille.calcule_aimantation()
        energie_courante = Grille.calcule_energie()

        # Ajouter les valeurs courantes au observables
        Aimantation.ajout_mesure(aimantation_courante)
        Energie.ajout_mesure(energie_courante)

    return Grille, Aimantation, Energie


def initialiser_fichier_resultats(nom_fichier):
    """Crée le fichier qui servira à stocker les résultats.
    """
    with open(nom_fichier, 'w+') as f:
        writer = csv.writer(f) # objet writer
        writer.writerow([
            "temperature",
            "moyenne_aimantation",
            "erreur_aimantation",
            "t_corr_aimantation",
            "moyenne_energie",
            "erreur_energie",
            "t_corr_energie"
            ]) # les noms des colonnes


def ecrire_resultats(nom_fichier,
                     temperature,
                     moyenne_aimantation,
                     erreur_aimantation,
                     t_corr_aimantation,
                     moyenne_energie,
                     erreur_energie,
                     t_corr_energie):
    """Cette fonction permet d'écrire les résultats dans un fichier. Ces résultat
    sont écris en grille de valeurs assosicées par rangées

    Paramètres
    ----------
    nom_fichier: str
        Nom du fichier dans lequel écrire.
    temperature: float
        Température du système de spins.
    moyenne_aimantation: float
        Moyenne de l'aimantation du système.
    erreur_aimantation: float
        Erreur sur la moyenne de l'aimantation.
    t_corr_aimantation: float
        Temps de corrélation sur l'aimantation.
    moyenne_energie: float
        Moyenne de l'énergie du système.
    erreur_energie: float
        Erreur sur la moyenne
    t_corr_energie: float

    """
    with open(nom_fichier, 'a') as f:
        writer = csv.writer(f) # objet writer
        writer.writerow([
            temperature,
            moyenne_aimantation,
            erreur_aimantation,
            t_corr_aimantation,
            moyenne_energie,
            erreur_energie,
            t_corr_energie
            ]) # les noms des colonnes


def simuler(temperature_ini,
            temperature_fin,
            pas_temperature,
            nom_fichier="data_monte_carlo_ising.csv",
            taille_grille=32,
            iter_intermesure=1e3,
            iter_thermalisation=1e6,
            niveaux_binning=16):
    """ Effectue les simulations Monte-Carlo pour un intervalle de températures.

    Paramètres
    ----------
    temperature_ini: float
        Température initiale des simulations.
    temperature_fin: float
        Température de fin.
    pas_temperature: float
        Intervalle en chaque température de la simulation.
    nom_fichier="data_monte_carlo_ising.csv": str
        Nom du fichier où enregistrer les résultats.
    taille_grille: int
        Nombre de spin dans un côté de la grille représentant notre système
        (32 par défaut).
    iter_intermesure: int
        Nombre d'itération entre les mesures (1 000 par défaut).
    iter_thermalisation:
        Nombre d'itération pour effectuer la thermalisation
        (1 000 000 par défaut).
    niveaux_binning=16:
        Puissance de 2 du nombre de mesure (16 par défaut).
    """
    # liste des temperatures à simuler
    liste_temperatures = np.arange(temperature_ini, temperature_fin, pas_temperature)

    # initialisation de la grille de spins
    Grille = ising_aleatoire(temperature_ini, taille_grille)

    # initialisation du fichier de data
    initialiser_fichier_resultats(nom_fichier)

    # Execution de la simulation pour les températures spécifiées
    for temperature in liste_temperatures:
        print(f"--- Simulation à T={temperature} en cours ---")
        Grille.temperature = temperature # mise à jour de la température de la grille

        # Génération des deux observables 'à jour' et récupération de la grille thermalisée à la temp. courante
        Grille, Aimantation, Energie = etape_monte_carlo(Grille, iter_intermesure, iter_thermalisation, niveaux_binning)

        # Calcul des erreurs à chaque niveau
        Aimantation.update_erreurs()
        Energie.update_erreurs()

        # Chercher les valeurs importantes
        moyenne_aimantation = Aimantation.moyenne()
        erreur_aimantation = Aimantation.erreur()
        temps_correlation_aimantation = Aimantation.temps_correlation()

        moyenne_energie = Energie.moyenne()
        erreur_energie = Energie.erreur()
        temps_correlation_energie = Energie.temps_correlation()

        ecrire_resultats(nom_fichier,
                    temperature,
                    moyenne_aimantation,
                    erreur_aimantation,
                    temps_correlation_aimantation,
                    moyenne_energie,
                    erreur_energie,
                    temps_correlation_energie
                )


if __name__ == "__main__":
    simuler(3., 2. ,-0.1, niveaux_binning=10)







