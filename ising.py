import numpy as np
import numba as nb

"""
::: À faire :::
-> MONTE CARLO (au complet)

-> ~~Ising.difference_energie~~
-> ~~Ising.iteration_aleatoire~~
-> Ising.simulation
-> ~~Ising.aimantation~~

-> Observable.est_rempli
-> Observable.temps_correlation
-> Observable.moyenne
"""

@nb.jit(nopython=True)
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
        random_coords = np.random.randint(self.taille, size=2) # coordonnées aléatoires

        Delta_E = self.difference_energie(random_coords[0], random_coords[1])

        if random_float < np.exp(-Delta_E/self.temperature): # flip avec probabilité exp(-Delta_E/T)
            self.spins[random_coords[0], random_coords[1]] *= -1

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

    @property
    def aimantation(self):
        """Retourne l'aimantation actuelle de la grille de spins."""
        return np.sum(self.spins, axis=(0,1))


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
        ...

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

    def temps_correlation(self):
        """Retourne le temps de corrélation."""
        ...
        # Indice : Similaire à la fonction erreur

    def moyenne(self):
        """Retourne la moyenne des mesures."""
        ...
