# PARTIE 2 — Deep Learning + Application web

## Contenu

| Fichier / Dossier | Rôle |
|---|---|
| `03_deep_learning.ipynb` | **MLP PyTorch** — Architecture 80→64→32→1 (ReLU + BatchNorm + Dropout). Entraînement Adam + `BCEWithLogitsLoss(pos_weight=...)`, batch=256, EarlyStopping(patience=5). Produit `../outputs/y_proba_mlp.npy`, le modèle `mlp.pt`, et les courbes d'entraînement. |
| `04_comparison.ipynb` | **Comparaison finale** — Charge les probabilités des 4 modèles (LR, RF, RF lean, MLP), produit le tableau comparatif des métriques et les **courbes ROC superposées**. Conclusion sur Gunnarsson et al. 2021. |
| `webapp/` | **Application FastAPI complète** — Simulateur de poste de conseiller bancaire avec moteur de pricing, garde-fous métier, SHAP, et carte de comparaison live des 2 modèles RF. |

## Ordre d'exécution

1. Vérifier que la Partie 1 a été exécutée (`outputs/X_train.npy`, `y_proba_lr.npy`, `y_proba_rf.npy`, `y_proba_rf_lean.npy` doivent exister)
2. Lancer `03_deep_learning.ipynb` — entraîne le MLP et produit `y_proba_mlp.npy`
3. Lancer `04_comparison.ipynb` — produit le tableau et la figure ROC

## Application web — lancement

```bash
cd webapp
py -3.11 -m uvicorn app:app --port 8000
```

Puis ouvrir [http://localhost:8000](http://localhost:8000).

L'application charge `webapp/models/credit_pipeline.pkl` qui contient déjà tous les modèles entraînés (RF complet, RF sans fuite, LR, scalers, métadonnées). **Aucun réentraînement nécessaire pour la démo.**

## Structure de l'application web

```
webapp/
├── app.py                       # API FastAPI : routes /, /predict
├── train_pipeline.py            # Script d'entraînement (LR + RF + RF lean)
├── templates/
│   ├── index.html              # Formulaire conseiller (3 sections réalistes)
│   └── result.html             # Page résultat : offre, verdict, SHAP, comparaison
├── static/style.css            # Design palette navy/or, professionnel
└── models/credit_pipeline.pkl  # Bundle : modèles + scalers + métadonnées
```

## Résultat principal (MLP)

| Modèle | AUC | Accuracy | F1 | Temps |
|---|---|---|---|---|
| **MLP (Deep Learning)** | **0.7316** | 0.6788 | 0.4525 | 11.0 s |

## Conclusion de la Partie 2

- L'AUC du MLP (0.732) est essentiellement équivalent à celui de la Régression Logistique (0.731) — **écart 0.0005**.
- **Validation expérimentale de Gunnarsson et al. (2021)** : sur tabulaire, DL ≈ ML classique.
- En production, la Régression Logistique reste préférable (interprétable, 22× plus rapide).
- L'application web démontre concrètement la fuite méthodologique en affichant **les deux modèles RF** en parallèle.
