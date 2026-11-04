# Prédiction du Défaut de Remboursement de Prêt
## Documentation technique exhaustive — référence ingénieur

> **Projet ENSA Tanger · 2025-2026** · Modules Machine Learning & Practical AI
> Abdourhman Mohammed Alhendi · Bekkari Amr · Badr Ameur El Idrissi

Ce document est le référentiel technique de notre projet. Il documente chaque décision de
conception, sa justification scientifique, les alternatives considérées, et les contraintes qui
ont guidé nos choix. Il sert de complément au rapport final et de support à la présentation
vidéo.

---

## Table des matières

1. [Cadre académique et problématique](#1-cadre-académique-et-problématique)
2. [Dataset LendingClub — description et choix](#2-dataset-lendingclub--description-et-choix)
3. [Pipeline méthodologique](#3-pipeline-méthodologique)
4. [Modèles et résultats](#4-modèles-et-résultats)
5. [Le point méthodologique central — la fuite](#5-le-point-méthodologique-central--la-fuite)
6. [Évaluation et benchmarking](#6-évaluation-et-benchmarking)
7. [Application web — architecture](#7-application-web--architecture)
8. [Discussion scientifique](#8-discussion-scientifique)
9. [Limites assumées et perspectives](#9-limites-assumées-et-perspectives)
10. [Annexes — décisions et alternatives](#10-annexes--décisions-et-alternatives)
11. [Bibliographie](#11-bibliographie)

---

## 1. Cadre académique et problématique

### 1.1 Contexte économique
Le scoring de crédit est un domaine historique du machine learning appliqué, antérieur même
à la mode actuelle de l'intelligence artificielle. Depuis les accords de **Bâle II et III**, les
institutions financières doivent estimer la **Probability of Default (PD)** de chaque
emprunteur, pour deux raisons principales :

- **Pricing du risque** : fixer le taux d'intérêt proposé en fonction du niveau de risque (les
  emprunteurs plus risqués paient plus cher pour compenser la perte attendue).
- **Capital réglementaire** : la PD entre dans le calcul de l'**Expected Loss** (EL = PD × LGD ×
  EAD) qui détermine les provisions et les fonds propres immobilisés selon les exigences de la
  norme comptable **IFRS 9**.

Au Maroc, **Bank Al-Maghrib** supervise cette activité ; en Europe, c'est la **BCE**.

### 1.2 Problématique
> **Peut-on prédire avec précision si un emprunteur va rembourser son prêt, à partir de ses
> caractéristiques financières et socio-économiques au moment de la demande ?**

C'est un problème de **classification binaire supervisée** (TARGET = 1 si défaut, 0 sinon), avec
un **dataset fortement déséquilibré** (~80 % de remboursés, ~20 % de défauts). Ce déséquilibre
de classes structure tous nos choix méthodologiques.

### 1.3 Objectifs (cycle CRISP-DM réduit)
1. **Pipeline de prétraitement standard** : imputation, encodage, normalisation
2. **Feature engineering** financièrement parlant (ratios métier)
3. **Comparaison ML classique** : Régression Logistique vs Random Forest
4. **Modèle de Deep Learning** : MLP PyTorch pour validation expérimentale
5. **Évaluation rigoureuse** : AUC-ROC, F1, Accuracy, matrices de confusion

**AUC cible** : 0,73 – 0,77 (réaliste pour un pipeline pédagogique sans feature engineering
massif).

### 1.4 Contribution distinctive
Au-delà du pipeline standard, notre projet identifie et **mesure expérimentalement** une **fuite
méthodologique** présente dans le dataset LendingClub (variables `grade`, `sub_grade`,
`int_rate`), et démontre live dans l'application l'effet de cette fuite (Δ = 0,03 d'AUC).

---

## 2. Dataset LendingClub — description et choix

### 2.1 Source et taille
- **Source** : Kaggle, dataset *LendingClub Loan Data*, fichier `accepted_2007_to_2018Q4.csv`
- **Taille** : 1,5 Go, ~2,2 millions de prêts entre 2007 et 2018
- **151 colonnes** par prêt (variables démographiques, financières, comportementales)
- **Échantillon de travail** : `nrows=100 000` (justifié en §2.3)

### 2.2 Pourquoi LendingClub plutôt que Home Credit ?

| Critère | LendingClub | Home Credit |
|---|---|---|
| Taux de défaut | 20 % (modéré) | 8 % (très déséquilibré) |
| Interprétabilité | Variables financières directes | Variables anonymisées |
| Granularité | 1 table principale | 7 tables à joindre (complexe) |
| **Choix retenu** | ✅ pour cohérence pédagogique | — |

**Justification** : le taux de défaut à 20 % est plus apprenable qu'à 8 %, les variables sont
financièrement reconnaissables (montant, taux, durée, revenu), et la structure mono-table évite
la complexité du feature engineering joint.

### 2.3 Pourquoi 100 000 lignes ?
**Contraintes** : développement local sur machine personnelle (16 Go RAM, CPU 8 cœurs), temps
d'itération limité (5 mois).

**Compromis** : 100 000 lignes représentent statistiquement la population complète (le théorème
central limite et la **loi des grands nombres** garantissent la convergence des statistiques
descriptives au-delà de quelques milliers d'observations). Au-delà, on n'aurait gagné qu'en
réduction de la variance de l'estimateur, pas en biais.

**Compromis confirmé** : taux de défaut sur 100k = 20,43 %, identique à celui rapporté pour
le dataset complet.

### 2.4 Définition de la variable cible TARGET
La colonne brute `loan_status` est un **texte** à 7 valeurs :
`Fully Paid, Charged Off, Current, Default, Late 31-120 days, In Grace Period, Late 16-30 days`

Notre construction :
```python
default_statuses = ["Charged Off", "Default", "Late (31-120 days)"]
paid_statuses    = ["Fully Paid"]
df = df[df["loan_status"].isin(default_statuses + paid_statuses)].copy()
df["TARGET"] = df["loan_status"].isin(default_statuses).astype(int)
```

**Décisions justifiées** :
- **Exclusion des Current et In Grace Period** : un modèle supervisé exige une cible connue ;
  ces statuts sont incertains, les inclure introduit du bruit.
- **Exclusion des Late 16-30 days** : zone grise (beaucoup régularisent dans les 30 jours) ;
  les inclure créerait une ambiguïté dans la définition même du défaut.
- **Choix binaire vs multiclasse** : on aurait pu modéliser `loan_status` en multi-classe, mais
  la sortie attendue d'un système d'aide à la décision bancaire est binaire (accord/refus),
  donc la binarisation est métier-justifiée.

**Résultat** : 88 333 prêts conservés, taux de défaut = **20,43 %**.

---

## 3. Pipeline méthodologique

### 3.1 Architecture du pipeline (séparation des responsabilités)

```
[CSV brut 1,5 Go]
        ↓
[01_eda_preprocessing.ipynb]  ← Abdourhman
        ↓ produit X_train.npy, X_test.npy
[02_modeles_ml.ipynb]         ← Amr
        ↓ produit y_proba_*.npy
[03_deep_learning.ipynb]       ← Badr (notebook MLP)
        ↓ produit y_proba_mlp.npy
[04_comparison.ipynb]          ← équipe (figures ROC)
        ↓ produit results_comparison.csv
[webapp/]                       ← Badr (FastAPI)
        ↓ application déployable
```

**Justification de la séparation** : chaque notebook est une étape **idempotente** et
**reproductible**. Le format `.npy` (binaire NumPy) entre notebooks garantit que les **mêmes
données entrent dans chaque modèle**, ce qui rend la comparaison rigoureuse. C'est une bonne
pratique de **MLOps basique**.

### 3.2 EDA — Analyse exploratoire des données

#### 3.2.1 Distribution de la cible (déséquilibre)
- Remboursés : 70 288 (79,57 %)
- Défauts : 18 045 (20,43 %)

**Conséquences sur la métrique** :
- L'**Accuracy** serait trompeuse (un classifieur naïf prédisant toujours "remboursé" obtient
  79,57 % d'accuracy, qui est pourtant inutile métier).
- L'**AUC-ROC** mesure la capacité de **ranking** indépendamment du seuil de décision. Pour
  un défaut tiré au hasard et un remboursement tiré au hasard, AUC = Pr(score_défaut >
  score_remboursement). C'est le **standard du credit scoring** depuis Lessmann 2015.

#### 3.2.2 Analyse bivariée (boxplots numériques)
| Variable | Médiane remboursés | Médiane défauts | Pouvoir discriminant |
|---|---|---|---|
| `int_rate` | ~10 % | ~15 % | ★★★ |
| `revol_util` | ~50 % | ~60 % | ★★ |
| `dti` | ~17 | ~20 | ★ |
| `annual_inc` | ~67 k | ~62 k | ★ |
| Autres | semblables | semblables | — |

**Observations** :
- `int_rate` domine — signal très fort. **Hypothèse confirmée plus tard** : c'est une variable
  de fuite (cf. §5).
- `revol_util` et `dti` sont cohérents avec l'intuition financière : plus on est endetté ou
  saturé, plus on défaille.
- `annual_inc` montre des outliers extrêmes (jusqu'à 9 M$) qui nécessitent un traitement.

#### 3.2.3 Matrice de corrélation
**Top corrélations avec TARGET** :
- `int_rate` : **+0,26** (anormalement fort)
- `revol_util` : +0,11
- `dti` : +0,10
- `annual_inc` : −0,07

**Alerte méthodologique** : `int_rate` est ~3× plus corrélé que les autres. Cette anomalie nous
a poussés à enquêter sur sa nature — c'est la décision de risque de LendingClub elle-même, donc
une **target leakage** (cf. §5).

**Multicolinéarité notée** :
- `loan_amnt` ↔ `installment` (mécanique : mensualité dérive du montant)
- Conséquence : la régression logistique peut être instable, les arbres de décision ne le sont
  pas → raison supplémentaire d'inclure un Random Forest.

#### 3.2.4 Catégorielles — taux de défaut par modalité
| Variable | Constat |
|---|---|
| `grade` | A à G : 4 % → 35 % (signal extrêmement fort) |
| `purpose` | `small_business` 2× plus risqué que `debt_consolidation` |
| `term` | 60 mois +50 % de risque vs 36 mois |
| `home_ownership` | RENT > MORTGAGE (cohérent : pas de patrimoine collatéral) |

### 3.3 Preprocessing

#### 3.3.1 Sélection des features
Sur 151 colonnes brutes, **19 retenues** (11 numériques + 7 catégorielles + TARGET) selon les
critères :
- **Interprétabilité financière** (un banquier les reconnaît)
- **Disponibilité au moment de la décision** (pas de variables post-décision)
- **Faible taux de NaN** (<60 % de remplissage exclus)

Variables éliminées : `url`, `desc` (texte libre), `id`, `member_id` (identifiants),
`title`, `zip_code` (granularité trop fine), `hardship_*` (info post-difficulté), ~90 autres.

#### 3.3.2 Imputation des valeurs manquantes
**Choix : médiane (numériques) + mode (catégorielles)**

```python
for col in features_num:
    df_model[col] = df_model[col].fillna(df_model[col].median())
for col in features_cat:
    df_model[col] = df_model[col].fillna(df_model[col].mode()[0])
```

**Justifications** :
- **Médiane vs moyenne** : robuste aux outliers extrêmes (revenus à 9 M$ feraient exploser la
  moyenne). Statistiquement, la médiane est un **estimateur de tendance centrale robuste de
  Huber**.
- **Mode pour catégorielles** : choix conservateur ; on aurait pu créer une catégorie "missing"
  séparée, mais sur les variables retenues le taux de NaN est faible (<7 %) donc l'impact est
  négligeable.

**Alternatives écartées** :
- **KNN-imputation** : coûteux O(n²) avec 70 000 lignes, gain marginal.
- **MICE (Multiple Imputation by Chained Equations)** : sophistiqué mais hors scope pédagogique.

#### 3.3.3 Traitement des outliers — clip au 99e percentile
```python
for col in ["annual_inc", "revol_bal"]:
    p99 = df_model[col].quantile(0.99)
    df_model[col] = df_model[col].clip(upper=p99)
```

**Justifications** :
- **Écrêtage vs suppression** : préserve les 1 % de lignes (potentiellement utiles ailleurs).
  C'est la **winsorization** (Tukey, 1962).
- **Choix du percentile** : 99e est standard. À 95e on perd du signal ; au-delà on garde des
  valeurs aberrantes.
- **Résultat** : `annual_inc` capé à 263 k $, `revol_bal` à 105 k $.

#### 3.3.4 Feature engineering — ratios financiers
**Ratios créés** :
```python
df_model["installment_income_ratio"] = df_model["installment"] * 12 / df_model["annual_inc"]
df_model["loan_income_ratio"]        = df_model["loan_amnt"] / df_model["annual_inc"]
df_model = df_model.replace([np.inf, -np.inf], np.nan)
```

**Justifications métier** :
- `installment_income_ratio` (**taux d'effort**) : équivalent du seuil HCSF en France (35 %
  maximum recommandé pour les crédits immobiliers). Standard du credit scoring.
- `loan_income_ratio` (**prêt/revenu**) : mesure la **proportionnalité** de la demande.

**Piège technique géré** : `annual_inc = 0` génère des `inf` qu'on remplace par NaN puis on
ré-impute par la médiane. Sans ce traitement, le modèle plante au premier passage.

#### 3.3.5 Encodage One-Hot
```python
df_encoded = pd.get_dummies(df_model, columns=features_cat, drop_first=True)
```

**Justifications** :
- `drop_first=True` évite la **multicolinéarité parfaite** entre les dummies (sinon le
  déterminant de la matrice de design est nul et la régression logistique ne converge pas).
  C'est ce qu'on appelle le *dummy variable trap*.
- **Alternatives écartées** : target encoding (risque de leakage), embeddings (overkill).

**Résultat** : 19 colonnes → **80 features** après encodage.

#### 3.3.6 Split train/test + Normalisation
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

**Décisions critiques** :
- **`stratify=y`** : préserve la proportion 80/20 dans le train et le test. Sans cela, le
  hasard pourrait créer un test à 25 % de défauts (non représentatif).
- **`random_state=42`** : reproductibilité.
- **Scaler fit SEULEMENT sur train** : la transformation du test utilise les paramètres du
  train. **Ne JAMAIS fit sur tout X** — c'est une fuite de données (data leakage classique).
- **StandardScaler vs MinMaxScaler vs RobustScaler** :
  - StandardScaler choisi car la régression logistique et les MLP attendent par défaut des
    features centrées-réduites.
  - MinMaxScaler aurait demandé une sensibilité aux outliers (compensée par le clip).
  - RobustScaler aurait pu fonctionner mais standard est le choix par défaut.

**Résultat** : `X_train (70 666, 80)`, `X_test (17 667, 80)`, ratio de défaut préservé à
20,43 % dans les deux ensembles.

---

## 4. Modèles et résultats

### 4.1 Tableau récapitulatif des 4 modèles

| Modèle | Features | AUC-ROC | Accuracy | F1-score | Temps (s) |
|---|---|---|---|---|---|
| Régression Logistique | 80 | **0,7311** | 0,6672 | 0,4476 | 0,5 |
| Random Forest (complet) | 80 | 0,7232 | 0,7834 | 0,3684 | 2,3 |
| **Random Forest (sans fuite)** | 37 | **0,6941** | 0,7869 | 0,3043 | 2,6 |
| MLP (Deep Learning) | 80 | **0,7316** | 0,6788 | 0,4525 | 11,0 |

### 4.2 Régression Logistique

#### Théorie
Modèle linéaire généralisé. On modélise :
```
log(P(y=1|x) / P(y=0|x)) = β₀ + β₁x₁ + ... + βₙxₙ
```
Chaque coefficient βᵢ s'interprète comme un **effet log-odds** : une augmentation d'une unité
de xᵢ multiplie les odds par exp(βᵢ).

#### Configuration
```python
LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
```

**Justifications** :
- `max_iter=1000` : la valeur par défaut de 100 ne converge pas toujours sur 70k × 80
  features avec class_weight.
- `class_weight='balanced'` : pondère les classes inversement à leur fréquence. Sans, le
  gradient est dominé par la classe majoritaire et le modèle prédit toujours "remboursé".

#### Pourquoi en premier ?
1. **Interprétabilité** (Bâle exige des modèles internes interprétables)
2. **Conformité réglementaire** (RGPD article 22 → droit à l'explication)
3. **Baseline robuste** pour benchmarker les modèles plus complexes

### 4.3 Random Forest

#### Théorie
Ensemble de 100 arbres de décision **bootstrap-aggregés** (bagging) : chaque arbre est entraîné
sur un échantillon tiré avec remise du dataset original, avec une sélection aléatoire de
features à chaque split.

**Avantages** :
- Capture des **interactions non-linéaires** entre features
- Robuste au bruit grâce au moyennage
- Naturellement **invariant aux transformations monotones** des features

#### Configuration
```python
RandomForestClassifier(
    n_estimators=100, min_samples_split=10,
    class_weight="balanced", n_jobs=-1, random_state=42
)
```

**Justifications** :
- `n_estimators=100` : compromis classique. Le gain au-delà est marginal (loi des rendements
  décroissants).
- `min_samples_split=10` : évite l'overfitting en limitant la profondeur effective. Avec 70k
  lignes, 10 est conservateur sans être bridant.
- `n_jobs=-1` : parallélisation sur tous les cœurs CPU (4× speedup sur 8 cœurs).

#### Résultat surprenant
**Écart RF vs LR < 0,01 d'AUC**. Sur ce dataset, les relations sont déjà largement
linéaires, le non-linéaire n'apporte pas de bond. **Cohérent avec Lessmann 2015** : les
ensembles dominent généralement, mais l'écart se réduit quand les features sont bien choisies.

### 4.4 MLP — Multi-Layer Perceptron

#### Architecture pyramidale
```
Input (80) → Linear(64) → ReLU → BatchNorm1d(64) → Dropout(0.3)
           → Linear(32) → ReLU → BatchNorm1d(32) → Dropout(0.2)
           → Linear(1)   → [logit, pas de Sigmoid : BCEWithLogitsLoss l'inclut]
```

**Justifications** :
- **80 → 64 → 32 → 1** : architecture pyramidale, force la compression progressive de
  l'information. Le **théorème d'approximation universelle** (Cybenko 1989, Hornik 1991)
  garantit qu'une couche cachée suffit théoriquement ; en pratique deux donnent plus de
  capacité sans overfitting.
- **ReLU** : évite le *vanishing gradient* des sigmoïdes/tanh (problème historique du DL).
- **BatchNorm** : stabilise et accélère l'entraînement en normalisant les activations à chaque
  batch (Ioffe & Szegedy 2015).
- **Dropout** : régularisation par ensemble implicite (Srivastava et al. 2014, JMLR).
  Dropout=0,3 sur la première couche (plus de neurones, plus de risque d'overfit), 0,2 sur la
  seconde (moins).
- **Logit en sortie + BCEWithLogitsLoss** : combine sigmoid et BCE de façon numériquement
  stable (évite les overflow/underflow).

#### Configuration d'entraînement
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
batch_size = 256
max_epochs = 30
patience = 5  # EarlyStopping sur val_auc
```

**Justifications** :
- **Adam** : optimiseur adaptatif combinant Momentum (lissage de la trajectoire) + RMSProp
  (adaptation du pas par paramètre). Kingma & Ba 2015 → moins sensible aux hyperparamètres
  que SGD, idéal pour un projet académique.
- **lr=1e-3** : valeur par défaut robuste (l'article original).
- **weight_decay=1e-5** : régularisation L2 légère, pour éviter l'overfitting.
- **pos_weight** : équivalent PyTorch de `class_weight='balanced'`. Calculé comme
  `n_neg / n_pos ≈ 3,9`. Une erreur sur un défaut pèse 3,9× plus.
- **batch_size=256** : compromis temps/mémoire/stabilité. Sur tabulaire, des batches plus
  petits (32, 64) sont moins efficaces (vectorisation sous-optimale).
- **EarlyStopping(patience=5)** : arrête si la val_auc n'a pas progressé pendant 5 epochs.
  Évite le surapprentissage sans avoir à fixer le nombre exact d'epochs.

#### Résultats
- **AUC test = 0,7316**
- Convergence à l'epoch 10 (EarlyStopping déclenché)
- Temps total : 11 secondes (CPU)
- Courbes train et val collées → **pas de surapprentissage**

---

## 5. Le point méthodologique central — la fuite

### 5.1 Identification du problème
En analysant les feature importances du Random Forest, on observe que **3 variables dominent**
le top des prédicteurs : `int_rate` (10 %), `sub_grade` (5,7 %), `grade` (4,9 %).

Or, ces 3 variables **ne sont pas des données brutes sur le client** — ce sont les
**décisions de risque déjà prises par LendingClub** :
- `grade` (A-G) : note de risque attribuée par LendingClub
- `sub_grade` (A1-G5) : affinement de la note
- `int_rate` : taux d'intérêt fixé en fonction du grade

**Conclusion** : utiliser ces variables comme entrées d'un modèle « prédire avant décider » est
**circulaire**. C'est de la **target leakage** au sens de Kaufman et al. 2012.

### 5.2 Analogie pédagogique
> *« C'est comme essayer de prédire si un patient a un cancer en utilisant comme variable
> d'entrée le diagnostic déjà posé par un autre médecin. Ça marche très bien, mais ça ne
> prédit rien — ça reproduit une décision antérieure. »*

### 5.3 Mesure expérimentale de la fuite
Nous avons entraîné un **second modèle Random Forest, sur exactement le même split**, mais
**en retirant ces variables circulaires** :

```python
LEAK_NUM = {"int_rate", "installment", "installment_income_ratio"}
def is_leak(c):
    return c in LEAK_NUM or c.startswith("grade_") or c.startswith("sub_grade_")
lean_cols = [c for c in feature_names if not is_leak(c)]
```

**Résultat** : 37 features (vs 80), AUC = **0,6941**.

**Δ AUC = 0,724 − 0,694 = 0,030**. La fuite gonfle artificiellement l'AUC d'environ **0,03**.

### 5.4 Démonstration live dans l'application
Notre application web affiche **les deux modèles en parallèle** sur le même dossier client. Le
banquier voit :
- *Modèle complet* (avec fuite) : AUC 0,724, proba défaut X %
- *Modèle réaliste* (sans fuite) : AUC 0,694, proba défaut Y %

Cette démonstration matérialise concrètement la différence et permet au lecteur de comprendre
l'impact pratique de la fuite, **slide par slide**.

---

## 6. Évaluation et benchmarking

### 6.1 Métriques utilisées

| Métrique | Définition | Pourquoi utile ici |
|---|---|---|
| **AUC-ROC** | Aire sous la courbe ROC | Indépendante du seuil, robuste au déséquilibre |
| **Accuracy** | (TP+TN)/Total | Reportée pour contexte mais peu pertinente |
| **F1-score** | 2·prec·rec/(prec+rec) | Compromis précision/rappel pour la classe minoritaire |
| **Confusion matrix** | TP, FP, FN, TN | Erreurs métier décomposées |

### 6.2 Benchmarking littérature

| Source | Approche | Dataset | AUC |
|---|---|---|---|
| Lessmann et al. 2015 | Best ensemble (Random Forest, GB) | 8 datasets crédit | 0,78 – 0,82 |
| Kaggle Home Credit 2018 | LightGBM stacké, FE massif | Home Credit | 0,80 – 0,81 |
| Gunnarsson et al. 2021 | MLP, TabNet, XGBoost | 11 datasets | 0,75 – 0,82 |
| **Notre projet (LR)** | Régression Logistique | LendingClub | **0,7311** |
| **Notre projet (RF complet)** | Random Forest avec fuite | LendingClub | **0,7232** |
| **Notre projet (RF sans fuite)** | Random Forest sans fuite | LendingClub | **0,6941** |
| **Notre projet (MLP)** | Deep Learning | LendingClub | **0,7316** |

### 6.3 Interprétation scientifique
- Nos résultats sont **conformes à la littérature** pour un pipeline pédagogique sans feature
  engineering massif (l'écart de 0,05–0,08 avec Kaggle s'explique par leur FE multi-tables et
  leur stacking).
- L'écart **MLP vs LR < 0,01** valide expérimentalement la conclusion de **Gunnarsson et al.
  2021** : sur données tabulaires bien préparées, **DL ≈ ML classique**.
- La fuite mesurée à **0,03** correspond à l'effet typique d'une **target leakage modérée**
  documentée dans la littérature (Kaufman et al. 2012).

### 6.4 No Free Lunch Theorem
Le **NFL Theorem** (Wolpert & Macready 1997) stipule qu'aucun algorithme n'est universellement
supérieur. Nos résultats l'illustrent :
- Sur **images** (CNN gagnent grâce aux structures locales captées par convolution)
- Sur **texte** (Transformers gagnent grâce à l'attention)
- Sur **tabulaire bien préparé** → ML classique reste compétitif (XGBoost reste roi en pratique)

---

## 7. Application web — architecture

### 7.1 Stack technique

| Composant | Technologie | Justification |
|---|---|---|
| Backend | **FastAPI 0.125** | Async, validation Pydantic, OpenAPI auto, moderne |
| Templates | **Jinja2** | Standard Python, intégré FastAPI |
| Frontend | **HTML5 + CSS3 sur mesure** | Contrôle total du design pro |
| Explicabilité | **SHAP TreeExplainer** | Standard réglementaire credit scoring |
| Modèle | **joblib.load (.pkl bundle)** | Sérialisation sklearn standard |
| Serveur | **uvicorn** | ASGI, performant, mode dev avec reload |

### 7.2 Pourquoi FastAPI vs Flask vs Django ?

| Critère | FastAPI | Flask | Django |
|---|---|---|---|
| Performance | ★★★ (async) | ★★ | ★★ |
| Validation auto | ★★★ (Pydantic) | ✗ | ★★ |
| Documentation auto (Swagger) | ★★★ | ✗ | ✗ |
| Setup overhead | Faible | Très faible | Lourd (ORM, admin) |
| Adapté aux APIs ML | ★★★ | ★★ | ★ (trop framework) |

**Choix** : FastAPI — adéquation maximale aux besoins d'une API ML moderne.

### 7.3 Moteur de pricing (transparent, rule-based)

Notre moteur de tarification reproduit la **scorecard interne** d'une vraie banque. À partir
du profil client, il calcule un **risk_score** ∈ [0, 100] par addition de points selon des
barèmes métier :

| Variable | Bandes et points |
|---|---|
| DTI | <10 → 0 ; 10-20 → 5 ; 20-30 → 12 ; 30-40 → 22 ; ≥40 → 32 |
| revol_util | <30 → 0 ; 30-50 → 4 ; 50-70 → 10 ; 70-90 → 18 ; ≥90 → 26 |
| loan_income_ratio | <0,15 → 0 ; 0,15-0,30 → 5 ; 0,30-0,50 → 12 ; 0,50-1,0 → 22 ; ≥1,0 → 34 |
| annual_inc | ≥120k → −8 ; 80-120k → −4 ; 50-80k → 0 ; 30-50k → +6 ; <30k → +14 |
| delinq_2yrs | +9 par retard (plafond +27) |
| pub_rec | +12 par incident (plafond +24) |
| emp_length | 10+ ans → −5 ; ≥5 ans → −2 ; <1 an → +5 |
| home_ownership | MORTGAGE/OWN → −3 ; RENT → +3 |
| term | 60 mois → +6 ; 36 mois → 0 |

**Mapping score → grade → taux** :
- 0-9 : A (7,0 %)
- 10-22 : B (10,5 %)
- 23-37 : C (14,0 %)
- 38-52 : D (17,5 %)
- 53-67 : E (21,0 %)
- 68-82 : F (24,5 %)
- 83+ : G (28,0 %)

+ ajustement par sous-grade (×1 à ×5, +0 à +2,8 %).

**Justification des seuils** : inspirés du **barème HCSF** en France (taux d'effort maximum
35 %), des grilles publiques de banques marocaines (Attijariwafa Bank, BMCE Bank Of Africa),
et des fourchettes observées dans LendingClub.

### 7.4 Garde-fous métier (apply_policy)

Trois règles dures qui forcent un **refus automatique** quelle que soit la sortie du modèle
statistique :

```python
def apply_policy(raw):
    flags = []
    if raw["loan_amnt"] / raw["annual_inc"] > 1.5:
        flags.append("Montant emprunté > 1,5× revenu annuel")
    if raw["installment"] * 12 / raw["annual_inc"] > 0.45:
        flags.append("Remboursements > 45% revenu annuel (taux d'effort)")
    if raw["dti"] >= 45:
        flags.append("DTI ≥ 45 (plafond réglementaire)")
    return flags
```

**Justification** : le modèle statistique seul ne capture pas les cas absurdes (prêt 4× le
revenu) parce que **LendingClub n'a jamais accordé de tels prêts** — le modèle ne les a donc
jamais vus. Les règles dures, héritées de la réglementation (HCSF France), comblent ce trou.

### 7.5 Explicabilité SHAP

**SHAP** (SHapley Additive exPlanations, Lundberg & Lee 2017, NeurIPS) attribue à chaque
feature une contribution **localement intelligible** à la prédiction. Pour chaque dossier :

- Les valeurs SHAP additionnent à la prédiction de base + écart = prédiction finale
- Chaque feature reçoit un crédit/débit proportionnel à sa contribution marginale moyenne
- Mathématiquement rigoureux (basé sur les valeurs de Shapley de la théorie des jeux)

**Importance réglementaire** :
- **RGPD article 22** (2018) : droit à une explication pour les décisions automatisées
- **Loi Lemoine 2022** (France) : impose la justification des refus de crédit
- **AI Act européen** (2024) : les modèles de credit scoring sont *high-risk* et exigent une
  explicabilité documentée

---

## 8. Discussion scientifique

### 8.1 Validation expérimentale d'hypothèses littérature

#### Hypothèse 1 (Lessmann 2015) : les ensembles dominent
**Verdict mitigé** : le RF (0,7232) est légèrement INFÉRIEUR à la LR (0,7311) dans notre cas.
L'écart de Lessmann est moyenné sur 8 datasets ; sur LendingClub, les features bien choisies
suffisent à la LR.

#### Hypothèse 2 (Gunnarsson 2021) : DL ≈ ML sur tabulaire
**Verdict CONFIRMÉ** : MLP (0,7316) vs LR (0,7311) → écart 0,0005, non significatif.

#### Hypothèse 3 (Kaufman 2012) : la fuite gonfle artificiellement les performances
**Verdict CONFIRMÉ** : Δ AUC = 0,030 entre RF complet et RF sans fuite.

### 8.2 Trade-offs explicités

| Décision | Compromis | Notre choix | Justification |
|---|---|---|---|
| Échantillon 100k | Représentativité vs temps de calcul | 100k | Convergence statistique atteinte |
| Médiane vs KNN | Robustesse vs précision | Médiane | Simplicité, faible taux NaN |
| Class weight balanced vs SMOTE | Synthèse vs pondération | balanced | Pas de risque de bruit synthétique |
| 2 vs N couches MLP | Capacité vs overfit | 2 couches | 70k lignes insuffisantes pour plus profond |
| FastAPI vs Flask | Async vs simplicité | FastAPI | Validation auto, Swagger |
| Mockups vs Screenshots | Maquette vs réel | Mockups | Pas de dépendance externe |

### 8.3 Reproductibilité
Tout le pipeline est reproductible :
- `random_state=42` partout
- Versions fixées dans `requirements.txt`
- Notebooks numerotés exécutés dans l'ordre
- Bundle `.pkl` versionné avec les données préparées

---

## 9. Limites assumées et perspectives

### 9.1 Limites
1. **Performance plafond** : AUC 0,73 contre 0,80-0,81 chez Kaggle. Écart dû au feature
   engineering massif (jointures de 7 tables) et au stacking que nous n'avons pas faits.
2. **MLP minimaliste** : architecture pyramidale standard, pas d'optimisation hyperparamètres
   bayésienne, pas d'expérimentation avec TabNet ou FT-Transformer.
3. **Fuite dans la décision** : l'app utilise le modèle complet (avec fuite) comme moteur
   principal. C'est un choix pédagogique pour démontrer la comparaison live.
4. **Échantillon 100k** : 4,5 % du dataset complet. Avec les 2,2 M lignes, on aurait une
   variance d'estimateur encore plus faible mais le coût calcul x22.
5. **Données 2007-2018** : pas de prise en compte de la crise COVID ni des évolutions
   économiques récentes (drift temporel).

### 9.2 Pistes d'amélioration concrètes
- **XGBoost / LightGBM** : dominent en pratique sur tabulaire (Chen & Guestrin 2016). Notre
  équipe a évité par contrainte de temps mais ce serait le prochain modèle à tester.
- **Feature engineering temporel** : saisonnalité (mois de demande), ancienneté du compte.
- **Calibration des probabilités** : Platt scaling ou régression isotonique pour usage en
  provisions IFRS 9 (la PD doit être bien calibrée).
- **Monitoring de drift** : surveiller en production si la distribution change (PSI, KS test).
- **Modèle 100 % sans fuite en production** : basculer l'app sur le modèle réaliste.
- **TabNet ou FT-Transformer** : architectures DL spécifiques tabulaires (Arik & Pfister 2021).

---

## 10. Annexes — décisions et alternatives

### 10.1 Tableau exhaustif des décisions techniques

| # | Décision | Alternatives considérées | Choix retenu | Raison |
|---|---|---|---|---|
| 1 | Dataset | Home Credit, LendingClub | LendingClub | Taux défaut 20 % apprenable |
| 2 | Taille | 100k, 500k, 2,2M | 100k | RAM, temps itération |
| 3 | Définition défaut | Multiclasse, binaire | Binaire | Cohérence métier |
| 4 | Imputation NaN | Mediane, moyenne, KNN, MICE | Mediane | Robustesse |
| 5 | Outliers | Suppression, clip p99, log | Clip p99 | Préserve les lignes |
| 6 | Encodage cat | Label, One-Hot, Target | One-Hot drop_first | Pas de leakage |
| 7 | Scaler | Standard, MinMax, Robust | StandardScaler | Default LR/MLP |
| 8 | Split | 70/30, 80/20, 90/10 | 80/20 stratify | Standard |
| 9 | Compensation déséquilibre | SMOTE, class_weight, undersample | class_weight | Simplicité, pas de bruit |
| 10 | Modèle linéaire | LR, SVM linéaire | LR | Standard scoring |
| 11 | Modèle non-linéaire | RF, GB, XGBoost | RF | Recommandé Lessmann |
| 12 | Modèle DL | MLP, TabNet, FT-Transformer | MLP | Hors scope académique |
| 13 | Loss DL | BCELoss, BCEWithLogits | BCEWithLogits | Stabilité numérique |
| 14 | Optimiseur DL | SGD, Adam, AdamW | Adam | Robuste sans tuning |
| 15 | Backend app | Flask, FastAPI, Django | FastAPI | Modern, async |
| 16 | Explicabilité | LIME, SHAP | SHAP | Standard credit scoring |
| 17 | Frontend | React, Vue, vanilla HTML | HTML/CSS sur mesure | Pas d'overhead JS |

### 10.2 Snippets de code clés

#### Construction TARGET
```python
default_statuses = ["Charged Off", "Default", "Late (31-120 days)"]
paid_statuses = ["Fully Paid"]
df = df[df["loan_status"].isin(default_statuses + paid_statuses)].copy()
df["TARGET"] = df["loan_status"].isin(default_statuses).astype(int)
```

#### Pipeline preprocessing complet
```python
# 1. Imputation
for col in features_num:
    df_model[col] = df_model[col].fillna(df_model[col].median())
for col in features_cat:
    df_model[col] = df_model[col].fillna(df_model[col].mode()[0])

# 2. Clip outliers
for col in ["annual_inc", "revol_bal"]:
    df_model[col] = df_model[col].clip(upper=df_model[col].quantile(0.99))

# 3. Feature engineering
df_model["installment_income_ratio"] = df_model["installment"] * 12 / df_model["annual_inc"]
df_model["loan_income_ratio"]        = df_model["loan_amnt"] / df_model["annual_inc"]
df_model = df_model.replace([np.inf, -np.inf], np.nan)
df_model = df_model.fillna(df_model.median(numeric_only=True))

# 4. One-Hot
df_encoded = pd.get_dummies(df_model, columns=features_cat, drop_first=True)

# 5. Split + scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

#### Architecture MLP PyTorch
```python
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

#### Modèle sans fuite
```python
LEAK_NUM = {"int_rate", "installment", "installment_income_ratio"}
def is_leak(c):
    return c in LEAK_NUM or c.startswith("grade_") or c.startswith("sub_grade_")
lean_cols = [c for c in feature_names if not is_leak(c)]
# 80 features → 37 features
```

### 10.3 Exemples de sorties réelles

**Profil moyen (Accord)** :
- Saisie : revenu 65k$, ancienneté 10+ ans, MORTGAGE, prêt 12k$, DTI 18, util 50%
- Offre : Grade B, sous-grade B2, taux 10,50 %, mensualité 390 $
- Décision : **ACCORDÉ** (proba défaut 18 %)

**Cas extrême (Refus)** :
- Saisie : revenu 10k$, ancienneté < 1 an, RENT, prêt 45k$, DTI 42, util 95%
- Score statistique seul : 30 % (ne détecte pas le problème)
- Garde-fou métier déclenché : "Montant emprunté = 4,5× revenu"
- Décision : **REFUSÉ** (refus réglementaire)

**Comparaison live des 2 modèles sur même dossier** :
- Modèle complet (avec fuite) : AUC 0,724, proba défaut 18 %
- Modèle réaliste (sans fuite) : AUC 0,694, proba défaut 23 %
- Δ = 0,03 d'AUC + Δ = 5 % de proba sur ce dossier

---

## 11. Bibliographie

1. Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). *Benchmarking
   state-of-the-art classification algorithms for credit scoring: An update of research.*
   European Journal of Operational Research, 247(1), 124-136.

2. Gunnarsson, B. R., vanden Broucke, S., Baesens, B., Oskarsdóttir, M., & Lemahieu, W. (2021).
   *Deep learning for credit scoring: Do or don't?* European Journal of Operational Research,
   295(1), 292-305.

3. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model
   Predictions.* Advances in Neural Information Processing Systems (NeurIPS).

4. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.

5. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* International
   Conference on Learning Representations (ICLR).

6. Cybenko, G. (1989). *Approximation by superpositions of a sigmoidal function.* Mathematics
   of Control, Signals and Systems, 2(4), 303-314.

7. Hornik, K. (1991). *Approximation capabilities of multilayer feedforward networks.* Neural
   Networks, 4(2), 251-257.

8. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training
   by Reducing Internal Covariate Shift.* ICML.

9. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).
   *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR, 15, 1929-1958.

10. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). *Leakage in data mining:
    Formulation, detection, and avoidance.* ACM Transactions on Knowledge Discovery from Data
    (TKDD), 6(4).

11. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* SIGKDD.

12. Wolpert, D. H., & Macready, W. G. (1997). *No free lunch theorems for optimization.*
    IEEE Transactions on Evolutionary Computation, 1(1), 67-82.

13. Arik, S. Ö., & Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning.*
    AAAI.

14. Kaggle (2018). *Home Credit Default Risk Competition.*
    https://www.kaggle.com/competitions/home-credit-default-risk

15. LendingClub Loan Data Dataset (Kaggle).
    https://www.kaggle.com/datasets/wordsforthewise/lending-club

---

*Document généré le 3 juin 2026 · Projet ENSA Tanger · Modules ML & Practical AI 2025-2026*
