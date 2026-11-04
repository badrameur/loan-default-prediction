# Prédiction du défaut de remboursement de prêt

Projet académique de machine learning et deep learning réalisé à l’ENSA Tanger en 2025–2026. Il étudie la prédiction d’un défaut de remboursement à partir de données de prêt, compare des modèles classiques à un réseau de neurones et propose une application web FastAPI pour l’inférence.

## Équipe et contributions

- Abdourhman Mohammed Alhendi : exploration et préparation des données ;
- Bekkari Amr : modèles classiques (régression logistique et forêt aléatoire) ;
- Badr Ameur El Idrissi : deep learning, application et rapport.

## Contenu

- `Partie_1_Machine_Learning/` : notebooks d’exploration, préparation et modèles classiques ;
- `Partie_2_Deep_Learning_Application/` : notebooks de réseau de neurones et comparaison, puis application FastAPI ;
- `requirements.txt` : dépendances Python ;
- `RAPPORT_TECHNIQUE_COMPLET.md` : méthode et résultats détaillés.

## Reproduire

Utiliser Python 3.11 et installer les dépendances :

```bash
python -m pip install -r requirements.txt
```

Les données brutes, tableaux dérivés et modèles entraînés ne sont pas inclus dans ce dépôt. Les notebooks et `RAPPORT_TECHNIQUE_COMPLET.md` décrivent la préparation et l’entraînement. Les fichiers de modèles nécessaires à l’inférence doivent être générés avant de lancer l’application.

Ce projet est pédagogique. Ses prédictions ne doivent pas servir seules à une décision réelle d’octroi de crédit.
