# PARTIE 1 — Machine Learning classique

## Contenu

| Fichier | Rôle |
|---|---|
| `01_eda_preprocessing.ipynb` | **Étape 1** — EDA + Preprocessing complet du dataset LendingClub. Construit la variable cible TARGET, analyse les boxplots, calcule la matrice de corrélation, impute, écrête les outliers, crée les features dérivées, encode en One-Hot, et fait le split + scaling. Produit `../outputs/X_*.npy` et `y_*.npy`. |
| `02_modeles_ml.ipynb` | **Étape 2** — Entraîne et évalue trois modèles ML classiques sur les données préparées : Régression Logistique, Random Forest complet (80 features), et **Random Forest sans fuite** (37 features, démonstration méthodologique). Produit `../outputs/y_proba_*.npy` et `results_ml.csv`. |

## Ordre d'exécution

1. **D'abord** `01_eda_preprocessing.ipynb` (génère les .npy)
2. **Ensuite** `02_modeles_ml.ipynb` (utilise les .npy pour entraîner les modèles)

## Résultats principaux (Partie 1)

| Modèle | AUC | Accuracy | F1 | Temps |
|---|---|---|---|---|
| Régression Logistique | **0.7311** | 0.6672 | 0.4476 | 0.5 s |
| Random Forest (complet) | **0.7232** | 0.7834 | 0.3684 | 2.3 s |
| Random Forest (sans fuite) | **0.6941** | 0.7869 | 0.3043 | 2.6 s |

### Point méthodologique central de la Partie 1

Les variables `grade`, `sub_grade` et `int_rate` ne sont **pas** des données brutes mais **la décision de risque déjà prise par LendingClub**. Le Random Forest sans fuite démontre que cette « triche » gonfle l'AUC d'environ **0.03**. Voir section 5.4 du rapport final.

## Chemins relatifs

Les notebooks utilisent `../outputs/` (= `LIVRABLE_FINAL/outputs/`). Cela permet à la Partie 2 de réutiliser les mêmes données préparées.
