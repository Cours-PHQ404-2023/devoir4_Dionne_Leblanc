<h1 align="center"> Devoir #4 Ising </h1>
<p align="center">
Théo Dionne et Jérôme Leblanc
</p>

## Structure

```bash
.
├── README.md
├── data_monte_carlo_ising_test.csv
├── devoir4_phq404.pdf
├── exemple_data_MC_ising.csv
├── graphiques.mplstyle
├── ising.py
├── pyproject.toml
└── rapport.ipynb
```

`rapport.ipynb` agit comme notre rapport. À l'intérieur s'y trouve le code pertinent et les figures. Le rapport utilise le code définit dans `ising.py` qui contient les fonctions implémentant la méthode de Monte-Carlo pour un système de spins dispoé dans une grille carrée. Les fichiers `.csv` contiennent les données de sorties des simulations qui sont utilisées pour l'analyse dans le rapport. Afin d'effectuer les-dites simulations, executer le fichier `ising.py`:

```bash
python ising.py
```
