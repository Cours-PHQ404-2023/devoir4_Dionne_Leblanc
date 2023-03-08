import numpy as np
import numba as nb

"""
::: À faire :::
-> Fonction qui écrit la shit intéressante dans un CSV
-> Fonction qui run la simulation pour des températures diff avec les incr.
-> Rendre Observable numbafied aussi??s
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
        self.nombre_valeurs = np.zeros(nombre_niveaux + 1, int)
        self.sommes = np.zeros(nombre_niveaux + 1)
        self.sommes_carres = np.zeros(nombre_niveaux + 1)

        # La dernière valeur ajoutée à chaque niveau.
        self.valeurs_precedentes = np.zeros(nombre_niveaux + 1)

        # Le niveau que nous allons utiliser.
        # La différence de 6 donne de bons résultats.
        # Voir les notes de cours pour plus de détails.
        self.niveau_erreur = self.nombre_niveaux - 6

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

    def erreur(self):
        """Retourne l'erreur sur le mesure moyenne de l'observable.

        Le dernier niveau doit être rempli avant d'utiliser cette fonction.
        """
        erreurs = np.zeros(self.nombre_niveaux + 1)
        for niveau in range(self.niveau_erreur + 1):
            erreurs[niveau] = np.sqrt(
                (
                    self.sommes_carres[niveau]
                    - self.sommes[niveau]**2 / self.nombre_valeurs[niveau]
                ) / (
                    self.nombre_valeurs[niveau]
                    * (self.nombre_valeurs[niveau] - 1)
                )
            )
        return erreurs[self.niveau_erreur]

    def temps_correlation(self, erreurs):
        """Retourne le temps de corrélation. Basé sur (16.39) des notes à David S."""
        ### NOTE : je n'ai pas compris l'indice
        # calcul du ratio entre l'erreur estimée initiale et la meilleure estimation
        ratio_des_erreurs = erreurs[-1]/erreurs[0]
        return (ratio_des_erreurs*ratio_des_erreurs - 1)/2


    def moyenne(self):
        """Retourne la moyenne des mesures."""
        ### NOTE : Valider avec mon boiii
        return self.sommes[0]/self.nombre_valeurs[0] # la moyenne arithmétique des mesures


def etape_monte_carlo(Grille, iter_intermesure=1e3, iter_thermalisation=1e6, niveaux_binning=16):
    """Desc."""

    # initialization des observables
    Aimantation = Observable(niveaux_binning)
    Energie = Observable(niveaux_binning)

    # Thermalisation de la grille de spins
    print("Thermalisation")
    Grille.simulation(iter_thermalisation)

    print("Collecte des mesures")
    # remplissage des listes de binning
    for _ in range(2^niveaux_binning):
        # brouillage de la grille entre les mesures
        Grille.simulation(iter_intermesure)

        # Calcul des valeurs actuelles des opérateurs
        aimantation_courante = Grille.calcule_aimantation()
        energie_courante = Grille.calcule_energie()

        # Ajouter les valeurs courantes au observables
        Aimantation.ajout_mesure(aimantation_courante)
        Energie.ajout_mesure(energie_courante)

    return Grille, Aimantation, Energie


def ecrire_resultats():
    ...#faire
    return None


def simuler(temperature_ini, temperature_fin, pas_temperature, taille_grille=32, iter_intermesure=1e3, iter_thermalisation=1e6, niveaux_binning=16):
    """DESCRIPTION"""
    # liste des temperatures à simuler    
    liste_temperatures = np.arange(temperature_ini, temperature_fin, pas_temperature)

    # initialisation de la grille de spins
    Grille = ising_aleatoire(temperature_ini, taille_grille)

    # Execution de la simulation pour les températures spécifiées
    for temperature in liste_temperatures:
        print(f"--- Simulation à T={temperature} en cours ---")
        Grille.temperature = temperature # mise à jour de la température de la grille

        # Génération des deux observables 'à jour' et récupération de la grille thermalisée à la temp. courante
        Grille, Aimantation, Energie = etape_monte_carlo(Grille, iter_intermesure, iter_thermalisation, niveaux_binning)

        # Chercher les valeurs importantes
        moyenne_aimantation = Aimantation.moyenne()
        erreur_aimantation = Aimantation.erreur()[-1]
        temps_correlation_aimantation = Aimantation.temps_correlation()

        moyenne_energie = Energie.moyenne()
        erreur_energie = Energie.erreur()[-1]
        temps_correlation_energie = Energie.temps_correlation()

        ############################ METTRE LA FONCTION QUI RETOURNE LES RESULTATS
    








