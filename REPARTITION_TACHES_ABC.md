# RÉPARTITION DES TÂCHES — Personnes A · B · C
## Prédiction du Défaut de Remboursement de Prêt
### ENSA Tanger · Modules Machine Learning + Practical AI · Pr. M. El Alami Hassoun

> **Document de référence pour la vidéo de présentation.** Répartition équitable entre 3 personnes anonymes (**A**, **B**, **C**) des **45 tâches centrales** du projet. L'application web (8 tâches T46-T53) est **exclue** de cette répartition car le prof a indiqué qu'elle n'est pas le cœur de l'évaluation. Chaque personne reçoit exactement **15 tâches**, équilibrées en **nombre**, **difficulté cognitive**, **valeur scientifique** et **facilité de présentation**.

---

## 0. PHILOSOPHIE DE LA RÉPARTITION

### 0.1 Contraintes appliquées

| Contrainte | Implémentation |
|---|---|
| **Même charge de travail** | 15 tâches chacun (45 ÷ 3) |
| **Même note ML** | 9 tâches ML chacun (26 ÷ 3 ≈ 8,67 → 9 + 9 + 8) |
| **Même note Practical AI** | 6 tâches DL chacun (19 ÷ 3 ≈ 6,33 → 6 + 6 + 7) |
| **Même difficulté cognitive** | Chacun a 1-2 « *wow factors* » par module |
| **Présentation vidéo facile** | Chaque bloc forme une **histoire continue** (storytelling) |
| **Identification individuelle** | Chacun se présente au début de SON segment + résume sa contribution à la fin |
| **Équité de l'évaluation** | Aucun rôle « secondaire » : tous présentent une partie cours + une partie résultats |

### 0.2 Critère « *wow factor* » par personne

Chaque personne doit avoir au moins **un point fort par module** qu'elle peut présenter avec fierté :

| Personne | *Wow factor* Machine Learning | *Wow factor* Practical AI |
|---|---|---|
| **A** | T1 — Pivot motivé Home Credit → LendingClub (storytelling de l'origine du projet) | T39 — Implémentation de la boucle Forward / Backward / Update (cœur du cours de Pr. El Alami Hassoun, pages 90-108) |
| **B** | T18 — Régression Logistique avec interprétation des coefficients (AUC 0.7311 — baseline scientifique) | T42 + T43 — Comparaison ML vs DL avec courbes ROC superposées (aboutissement de la note Practical AI) |
| **C** | T20 + T21 — **Découverte et mesure de la fuite méthodologique** (contribution originale, 0.029 d'AUC mesurés) | T30 + T31 — Architecture pyramidale du MLP + choix ReLU vs Sigmoid (cours pages 142-145) |

---

## 1. TABLEAU D'ÉQUILIBRE GLOBAL

| Personne | Tâches ML | Tâches DL | **Total tâches** | Difficulté cumulée* | *Wow factors* |
|---|---|---|---|---|---|
| **A** | 9 | 6 | **15** | ≈ 27 pts | 2 (T1 + T39) |
| **B** | 9 | 6 | **15** | ≈ 31 pts | 2 (T18 + T42-T43) |
| **C** | 8 | 7 | **15** | ≈ 33 pts | 2 (T20-T21 + T30-T31) |
| **Total** | 26 | 19 | **45** | — | 6 (équilibrés 2/2/2) |

*Difficulté cumulée : score interne 1 (facile) / 2 (moyen) / 3 (hard). C porte un poids légèrement supérieur mais compensé par la valeur narrative de la fuite (très impressionnante à présenter à l'oral).

**Conclusion** : chacun a 15 tâches, 9 ML + 6 DL (sauf C : 8 + 7), 2 *wow factors*. Aucun déséquilibre majeur.

---

═══════════════════════════════════════════════════════════════════════
# 👤 PERSONNE A — « Le fondateur du projet »
*Rôle narratif : pose les bases ML et implémente le cœur de l'entraînement DL*
═══════════════════════════════════════════════════════════════════════

## A.1 Tâches Partie 1 — Note Machine Learning (9 tâches)

| Code | Tâche | Durée parlée (sec) | Type |
|---|---|---|---|
| **T1** ⭐ | Recherche et choix du dataset (Home Credit → LendingClub) | 60 | Storytelling |
| T2 | Chargement initial (`nrows=100000`, `low_memory=False`) | 25 | Technique |
| T3 | Inspection des types et valeurs manquantes | 25 | Technique |
| T4 | Construction de la variable cible TARGET | 40 | Méthodologique |
| T5 | Visualisation du déséquilibre 80/20 | 25 | Visuel |
| T6 | Sélection des features (151 → 19) | 35 | Méthodologique |
| T10 | Imputation des valeurs manquantes (médiane / mode) | 30 | Décision technique |
| T11 | Vérification des doublons | 15 | Mécanique |
| T13 | Gestion des valeurs infinies après division par 0 | 25 | Robustesse |

**Total ML parlée** : ≈ 4 min 20 s.

## A.2 Tâches Partie 2 — Note Practical AI / Deep Learning (6 tâches)

| Code | Tâche | Durée parlée (sec) | Type |
|---|---|---|---|
| T27 | Choix du framework (PyTorch vs TensorFlow) | 35 | Justification |
| T28 | Sous-split train/validation (80/20 du train) | 25 | Méthodologique |
| T29 | Pondération des classes (pos_weight ≈ 3.9) | 30 | Technique |
| **T39** ⭐ | Boucle Forward / Backward / Update (cœur du cours) | 80 | Cours |
| T40 | Diagnostic d'entraînement (convergence epoch 10) | 45 | Analyse |
| T41 | Évaluation MLP sur test (AUC 0.7316) | 35 | Résultat |

**Total DL parlée** : ≈ 4 min 10 s.

## A.3 Total parole vidéo : ≈ 8 min 30 s

## A.4 Histoire narrative que A raconte

> *« J'ai posé les **fondations** du projet. Au départ, on devait travailler sur Home Credit, mais en regardant les données on s'est rendu compte que LendingClub était mieux : taux 20 % plus apprenable, variables interprétables. J'ai chargé 100 000 lignes (contrainte RAM), filtré pour ne garder que les prêts terminés et construit la cible binaire TARGET — 88 333 prêts, 20.43 % de défauts. J'ai sélectionné 19 features parmi 151, et nettoyé en imputant par médiane/mode. ⬇ Côté Deep Learning, j'ai choisi PyTorch parce que son mode eager rend visible chaque étape du cours de Pr. El Alami Hassoun. J'ai implémenté la boucle Forward → Backward → Update — c'est très exactement ce que le cours décrit pages 90-108. Le modèle converge en 10 epochs grâce à EarlyStopping, et atteint AUC 0.7316 sur le test. »*

## A.5 Phrases d'ouverture / fermeture pour la vidéo

**Ouverture (segment ML)** :
> « Bonjour, je suis A. Pour la partie Machine Learning, j'ai posé les fondations du projet : choix du dataset, construction de la variable cible, et préparation initiale des données. »

**Transition vers B (fin segment ML)** :
> « Maintenant que la base est posée, B va explorer les patterns statistiques du dataset et entraîner notre modèle ML baseline. »

**Ouverture (segment DL)** :
> « Toujours moi, A. Pour la partie Deep Learning, j'ai choisi PyTorch et implémenté la boucle d'entraînement — c'est-à-dire le cœur de tout réseau de neurones. »

**Transition vers B (fin segment DL)** :
> « B va maintenant détailler la configuration de l'optimisation et présenter la comparaison ML vs DL. »

## A.6 Concepts du cours que A illustre

| Concept du cours (Pr. El Alami Hassoun) | Tâche de A |
|---|---|
| Apprentissage supervisé (X, y étiquetées) — page 24 | T4 (construction TARGET) |
| Pourquoi maintenant ? (data, GPU, frameworks) — page 3 | T27 (choix PyTorch) |
| Forward pass — pages 90-92 | T39 |
| Backward pass — pages 96-97 | T39 |
| Mise à jour : w ← w − η ∂J/∂w — page 108 | T39 |
| Surapprentissage (overfitting) — page 29 | T40 (diagnostic) |

## A.7 Critère d'évaluation prof — ce que A démontre

| Critère ML | Démontré par |
|---|---|
| Compréhension du sujet | T1 (justifie le choix du problème + dataset) |
| Dataset & Données | T2, T3, T4 |
| Preprocessing | T10, T11, T13 |

| Critère Practical AI | Démontré par |
|---|---|
| Modèle Deep Learning | T29, T39, T41 |
| Entraînement et optimisation | T39, T40 |

---

═══════════════════════════════════════════════════════════════════════
# 👤 PERSONNE B — « L'analyste statistique »
*Rôle narratif : explore les patterns, entraîne le baseline ML, configure et compare les modèles DL*
═══════════════════════════════════════════════════════════════════════

## B.1 Tâches Partie 1 — Note Machine Learning (9 tâches)

| Code | Tâche | Durée parlée (sec) | Type |
|---|---|---|---|
| T7 | Analyse bivariée par boxplots (par TARGET) | 35 | Visuel |
| T8 | Matrice de corrélation (signal int_rate à 0.26) | 30 | Visuel |
| T9 | Analyse catégorielle (taux défaut par grade/purpose/term) | 35 | Visuel |
| T12 | Traitement des outliers (clip p99) | 30 | Décision technique |
| T14 | Feature engineering : 2 ratios financiers | 50 | Innovation |
| T15 | Encodage One-Hot (drop_first=True) : 19 → 80 features | 25 | Technique |
| T16 | Split train/test stratifié (random_state=42) | 25 | Méthodologique |
| T17 | Normalisation StandardScaler (fit train only) | 30 | Méthodologique |
| **T18** ⭐ | Régression Logistique (class_weight='balanced') — AUC 0.7311 | 70 | Cœur ML |

**Total ML parlée** : ≈ 5 min 10 s.

## B.2 Tâches Partie 2 — Note Practical AI / Deep Learning (6 tâches)

| Code | Tâche | Durée parlée (sec) | Type |
|---|---|---|---|
| T35 | Optimiseur Adam (vs SGD) — cours page 116 | 45 | Cours |
| T36 | Fonction de perte BCEWithLogitsLoss + pos_weight | 35 | Technique |
| T37 | Mini-Batch Gradient Descent (batch=256) — cours page 115 | 40 | Cours |
| T38 | EarlyStopping (patience=5 sur val_auc) | 30 | Technique |
| **T42** ⭐ | Tableau récapitulatif 4 modèles (LR / RF / RF lean / MLP) | 50 | Résultat |
| **T43** ⭐ | Courbes ROC superposées (4 modèles) | 50 | Visuel cœur |

**Total DL parlée** : ≈ 4 min 10 s.

## B.3 Total parole vidéo : ≈ 9 min 20 s

## B.4 Histoire narrative que B raconte

> *« J'ai été **l'œil statistique** du projet. J'ai exploré les patterns du dataset : boxplots qui révèlent que int_rate sépare nettement les défauts (médiane 12 % vs 15 %) ; heatmap qui confirme int_rate +0.26 vers TARGET ; analyse catégorielle qui montre que le grade G a 49 % de défauts. J'ai créé deux ratios financiers — taux d'effort et ratio prêt/revenu — qui ont une vraie signification métier (HCSF 35 %). Puis j'ai entraîné notre baseline scientifique : la **Régression Logistique** avec class_weight='balanced', qui atteint AUC 0.7311 en 0.5 seconde — référence directe de Lessmann 2015. ⬇ Côté Practical AI, j'ai configuré l'optimisation : Adam (combinaison Momentum + RMSProp, cours page 116), BCEWithLogitsLoss avec pos_weight pour le déséquilibre, Mini-Batch GD avec batch=256 (cours page 115), EarlyStopping patience=5. Puis j'ai produit la comparaison finale : 4 courbes ROC superposées qui montrent que LR et MLP sont **strictement à égalité** à AUC ≈ 0.73 — confirmation expérimentale de Gunnarsson 2021. »*

## B.5 Phrases d'ouverture / fermeture pour la vidéo

**Ouverture (segment ML)** :
> « Bonjour, je suis B. Maintenant que la base est posée, j'ai exploré les patterns statistiques du dataset puis entraîné notre modèle ML de référence — la Régression Logistique. »

**Transition vers C (fin segment ML)** :
> « LR atteint AUC 0.7311 sans surprise. C va maintenant tester un modèle non linéaire — le Random Forest — et nous montrer une découverte méthodologique majeure. »

**Ouverture (segment DL)** :
> « Toujours moi, B. Pour le Deep Learning, j'ai configuré tous les paramètres d'optimisation : Adam, BCE, Mini-Batch, EarlyStopping. Puis j'ai comparé tous les modèles. »

**Transition vers C (fin segment DL)** :
> « Le MLP atteint AUC 0.7316 — strictement égal à LR. C va maintenant détailler l'architecture du réseau et nous situer scientifiquement face à la littérature. »

## B.6 Concepts du cours que B illustre

| Concept du cours (Pr. El Alami Hassoun) | Tâche de B |
|---|---|
| Fonction de coût J = ½(a−y)² → adapté en BCE | T36 |
| Stochastic Gradient Descent (SGD) — page 114 | T35 (alternative non choisie) |
| Mini-Batch GD (batch=32, 64, 128) — page 115 | T37 ⭐ |
| Hyperparamètres (learning rate, batch size) — page 116 | T35, T37 |
| Surapprentissage évité par EarlyStopping — page 29 | T38 |

## B.7 Critère d'évaluation prof — ce que B démontre

| Critère ML | Démontré par |
|---|---|
| Preprocessing | T12, T15, T16, T17 |
| Feature engineering | T14 |
| Modèles ML | T18 |
| Évaluation des performances | T42, T43 (transversal) |

| Critère Practical AI | Démontré par |
|---|---|
| Entraînement et optimisation | T35, T36, T37, T38 |
| Comparaison ML vs DL | T42, T43 |

---

═══════════════════════════════════════════════════════════════════════
# 👤 PERSONNE C — « L'architecte scientifique »
*Rôle narratif : entraîne les modèles non linéaires, découvre la fuite, conçoit l'architecture DL, conclut sur la science*
═══════════════════════════════════════════════════════════════════════

## C.1 Tâches Partie 1 — Note Machine Learning (8 tâches)

| Code | Tâche | Durée parlée (sec) | Type |
|---|---|---|---|
| T19 | Random Forest complet (100 arbres) — AUC 0.7232 | 50 | Cœur ML |
| **T20** ⭐ | Découverte de la fuite (grade, sub_grade, int_rate) | 70 | Contribution originale |
| **T21** ⭐ | Random Forest SANS FUITE (37 features) — AUC 0.6941 | 60 | Mesure rigoureuse |
| T22 | Calcul AUC / Accuracy / F1 / Matrice de confusion | 30 | Technique |
| T23 | Feature importance du Random Forest (top 15) | 35 | Visuel |
| T24 | Sauvegarde modèles et probabilités (.pkl, .npy) | 20 | Mécanique |
| T25 | Tableau comparatif 3 modèles ML | 30 | Synthèse |
| T26 | Interprétation business des résultats ML | 40 | Recommandation |

**Total ML parlée** : ≈ 5 min 35 s.

## C.2 Tâches Partie 2 — Note Practical AI / Deep Learning (7 tâches)

| Code | Tâche | Durée parlée (sec) | Type |
|---|---|---|---|
| **T30** ⭐ | Architecture pyramidale 80 → 64 → 32 → 1 (7 489 paramètres) | 60 | Cœur DL |
| **T31** ⭐ | Activation ReLU vs Sigmoid (vanishing gradient — cours p.142-145) | 55 | Cours |
| T32 | BatchNormalization (Ioffe & Szegedy 2015) | 30 | Technique |
| T33 | Dropout 0.3 puis 0.2 (Srivastava 2014) | 30 | Technique |
| T34 | Sortie en logit (pas de Sigmoid en couche finale) | 25 | Technique |
| T44 | Validation expérimentale de Gunnarsson 2021 | 40 | Conclusion |
| T45 | Benchmarking face à la littérature (Lessmann, Kaggle, Gunnarsson) | 50 | Synthèse |

**Total DL parlée** : ≈ 4 min 50 s.

## C.3 Total parole vidéo : ≈ 10 min 25 s

## C.4 Histoire narrative que C raconte

> *« J'ai été **l'architecte scientifique** du projet. J'ai entraîné le Random Forest avec 100 arbres — AUC 0.7232. En analysant la feature importance, j'ai remarqué que int_rate dominait à 10 %, suivi de grade. **J'ai alors réalisé que ces variables sont des décisions a posteriori de LendingClub** — c'est une fuite méthodologique. J'ai donc retiré 43 features circulaires et réentraîné le même modèle sur 37 features uniquement : AUC tombe à 0.6941. **J'ai mesuré rigoureusement la fuite à 0.029 d'AUC** — c'est notre contribution méthodologique. Recommandation : pour la production, utiliser LR sans fuite (interprétable, conforme RGPD, 22× plus rapide). ⬇ Côté Practical AI, j'ai conçu l'architecture du MLP : pyramidale 80 → 64 → 32 → 1, avec ReLU pour éviter le vanishing gradient (cours pages 142-145), BatchNorm pour stabiliser, Dropout 0.3 et 0.2 pour régulariser, sortie en logit. 7 489 paramètres pour 56 533 obs d'entraînement — ratio 7,5 obs/paramètre, dans la zone non-surapprenante. J'ai validé scientifiquement contre Gunnarsson 2021 : leur conclusion sur 11 datasets crédit (DL ≈ ML sur tabulaire) est confirmée localement. Nous sommes 0.05-0.09 sous les champions Kaggle (FE massif + stacking hors scope pédagogique). »*

## C.5 Phrases d'ouverture / fermeture pour la vidéo

**Ouverture (segment ML)** :
> « Bonjour, je suis C. J'ai pris en charge les modèles non linéaires et la conclusion scientifique de la partie ML. »

**Conclusion (fin segment ML, transition vers DL)** :
> « Notre meilleur modèle ML est donc LR avec AUC 0.7311. La fuite gonfle artificiellement de 0.029 d'AUC. Passons maintenant au Deep Learning — A va d'abord présenter l'implémentation de l'entraînement, puis je reviendrai détailler l'architecture. »

**Ouverture (segment DL)** :
> « Je reprends la parole pour expliquer comment le MLP est conçu : son architecture, le choix de ses activations, et comment ses résultats se positionnent face à la littérature scientifique. »

**Conclusion finale du projet** :
> « En résumé : H1 validée (AUC ≥ 0.73), H2 validée (MLP ≈ LR), H3 validée (fuite mesurée 0.029). Le DL n'apporte pas de gain significatif sur tabulaire. Recommandation : LR sans fuite pour la production. Merci de votre attention. »

## C.6 Concepts du cours que C illustre

| Concept du cours (Pr. El Alami Hassoun) | Tâche de C |
|---|---|
| Neurone artificiel & perceptron (Rosenblatt 1957) — pages 53-72 | T30 |
| MLP — couches d'entrée / cachées / sortie — pages 77, 100 | T30 |
| Modèle math : y = g(WᵀX + b) — page 63 | T30 |
| Limites du perceptron (XOR non linéaire) — page 71 | T30 (justification multi-couches) |
| Problème vanishing gradient (sigmoid, tanh) — pages 135, 138-141 | T31 ⭐ |
| Fonction ReLU f(x) = max(0, x) — pages 142-145 | T31 ⭐ |
| Avantages ReLU : pas de saturation pour x > 0 — page 143 | T31 |
| Dead neurons de ReLU — page 144 | T31 + T32 (atténué par BatchNorm) |
| Softmax / Sigmoid pour la sortie — pages 133, 157-160 | T34 |
| Réseau comme approximateur de fonction — page 121 | T30 |
| DL n'est pas magique : limites & explicabilité — pages 29-30 | T44, T45 |

## C.7 Critère d'évaluation prof — ce que C démontre

| Critère ML | Démontré par |
|---|---|
| Modèles ML (Random Forest) | T19, T21 |
| Évaluation des performances | T22, T23, T25 |
| Analyse des résultats | T20, T26 |

| Critère Practical AI | Démontré par |
|---|---|
| Modèle Deep Learning | T30 |
| Architecture du réseau | T30, T31, T32, T33, T34 |
| Comparaison ML vs DL | T44, T45 |

---

## 2. VUE D'ENSEMBLE DE LA RÉPARTITION

### 2.1 Tableau complet des 45 tâches par personne

#### 🟦 Partie 1 — Note Machine Learning (26 tâches)

| # | Tâche | Personne | Module |
|---|---|---|---|
| T1 | Recherche & choix du dataset | **A** ⭐ | ML |
| T2 | Chargement initial (`nrows=100k`) | **A** | ML |
| T3 | Inspection types / NaN | **A** | ML |
| T4 | Construction TARGET | **A** | ML |
| T5 | Visualisation déséquilibre 80/20 | **A** | ML |
| T6 | Sélection features 151 → 19 | **A** | ML |
| T7 | Boxplots bivariés par TARGET | **B** | ML |
| T8 | Matrice de corrélation | **B** | ML |
| T9 | Analyse catégorielle par TARGET | **B** | ML |
| T10 | Imputation médiane / mode | **A** | ML |
| T11 | Vérification doublons | **A** | ML |
| T12 | Clip outliers p99 | **B** | ML |
| T13 | Gestion des Inf | **A** | ML |
| T14 | 2 ratios financiers | **B** | ML |
| T15 | One-Hot (drop_first) | **B** | ML |
| T16 | Split stratifié 80/20 | **B** | ML |
| T17 | StandardScaler (fit train) | **B** | ML |
| T18 | Régression Logistique | **B** ⭐ | ML |
| T19 | Random Forest complet | **C** | ML |
| T20 | **Découverte de la fuite** | **C** ⭐ | ML |
| T21 | Random Forest SANS FUITE | **C** ⭐ | ML |
| T22 | Métriques AUC/Acc/F1/CM | **C** | ML |
| T23 | Feature importance top 15 | **C** | ML |
| T24 | Sauvegarde .pkl / .npy | **C** | ML |
| T25 | Tableau comparatif 3 ML | **C** | ML |
| T26 | Interprétation business | **C** | ML |

#### 🟧 Partie 2 — Note Practical AI / Deep Learning (19 tâches)

| # | Tâche | Personne | Module |
|---|---|---|---|
| T27 | Choix PyTorch vs TensorFlow | **A** | DL |
| T28 | Sous-split train/val | **A** | DL |
| T29 | Pondération pos_weight 3.9 | **A** | DL |
| T30 | **Architecture pyramidale 80→64→32→1** | **C** ⭐ | DL |
| T31 | Activation ReLU (vs Sigmoid) | **C** ⭐ | DL |
| T32 | BatchNormalization | **C** | DL |
| T33 | Dropout 0.3 / 0.2 | **C** | DL |
| T34 | Sortie en logit | **C** | DL |
| T35 | Optimiseur Adam | **B** | DL |
| T36 | BCEWithLogitsLoss | **B** | DL |
| T37 | Mini-Batch GD (batch=256) | **B** | DL |
| T38 | EarlyStopping (patience=5) | **B** | DL |
| T39 | **Boucle Forward/Backward/Update** | **A** ⭐ | DL |
| T40 | Diagnostic d'entraînement | **A** | DL |
| T41 | Évaluation MLP test | **A** | DL |
| T42 | Tableau récap 4 modèles | **B** ⭐ | DL |
| T43 | Courbes ROC superposées | **B** ⭐ | DL |
| T44 | Validation H2 vs Gunnarsson 2021 | **C** | DL |
| T45 | Benchmarking littérature | **C** | DL |

### 2.2 Statistiques de la répartition

| Personne | Tâches ML | Tâches DL | Total | Wow ML | Wow DL | Temps parole |
|---|---|---|---|---|---|---|
| **A** | 9 | 6 | 15 | T1 | T39 | ≈ 8 min 30 s |
| **B** | 9 | 6 | 15 | T18 | T42, T43 | ≈ 9 min 20 s |
| **C** | 8 | 7 | 15 | T20, T21 | T30, T31 | ≈ 10 min 25 s |
| **Total** | 26 | 19 | **45** | 4 | 5 | ≈ 28 min 15 s |

> Temps parole TOTAL ≈ 28 min 15 s → vidéo finale ≈ 30-35 min avec transitions, intro, conclusion, et démonstrations à l'écran.

---

## 3. FLUX VIDÉO RECOMMANDÉ (chronologie de présentation)

### 3.1 Structure générale

```
[1] OUVERTURE (A — 1 min)
    └─ Introduction projet, équipe, encadrement Pr. El Alami Hassoun
       deux modules ML + Practical AI

[2] PARTIE 1 — NOTE MACHINE LEARNING (15-17 min)
    ├─ A (4-5 min) : Foundation — dataset, TARGET, sélection features, nettoyage
    │   └─ Tâches T1, T2, T3, T4, T5, T6, T10, T11, T13
    ├─ B (5-6 min) : Exploration + Feature engineering + LR
    │   └─ Tâches T7, T8, T9, T12, T14, T15, T16, T17, T18
    └─ C (5-6 min) : Random Forest + Découverte fuite + Conclusion ML
        └─ Tâches T19, T20, T21, T22, T23, T24, T25, T26

[3] TRANSITION (15 sec, voix de A ou C)
    └─ « Passons maintenant à la partie Deep Learning, évaluée séparément en Practical AI. »

[4] PARTIE 2 — NOTE PRACTICAL AI / DEEP LEARNING (12-15 min)
    ├─ A (4-5 min) : Setup PyTorch + boucle entraînement
    │   └─ Tâches T27, T28, T29, T39, T40, T41
    ├─ B (4-5 min) : Configuration optimisation + Comparaison ROC
    │   └─ Tâches T35, T36, T37, T38, T42, T43
    └─ C (5-6 min) : Architecture MLP + Benchmarking scientifique
        └─ Tâches T30, T31, T32, T33, T34, T44, T45

[5] CONCLUSION (C — 1 min)
    └─ Validation des 3 hypothèses scientifiques
       Recommandation production : LR sans fuite
       Remerciements
```

**Durée totale estimée** : 32 ± 3 minutes — conforme à la cible des 30-40 minutes.

### 3.2 Règles d'identification individuelle (exigence prof)

> *« Chaque membre qui intervient dans la vidéo doit d'abord se présenter en indiquant son nom, car l'évaluation sera individuelle. »*

Chaque personne **se présente AU DÉBUT de chacun de ses 2 segments** (ML puis DL) :

| Moment | Personne | Phrase d'identification |
|---|---|---|
| T = 0 min | A | « Bonjour, je suis A. Je présente la partie fondation du projet en ML. » |
| T = 5 min | B | « Bonjour, je suis B. Je présente l'EDA, le feature engineering et la Régression Logistique. » |
| T = 10 min | C | « Bonjour, je suis C. Je présente Random Forest et la découverte méthodologique de la fuite. » |
| T = 17 min | A | « C'est de nouveau A. Pour la partie Practical AI, je présente le setup PyTorch et la boucle d'entraînement. » |
| T = 22 min | B | « C'est de nouveau B. Je présente la configuration d'optimisation et la comparaison ML vs DL. » |
| T = 27 min | C | « C'est de nouveau C. Je présente l'architecture du MLP et le positionnement scientifique. » |

---

## 4. ÉQUITÉ D'ÉVALUATION — Pourquoi chacun aura la même note

### 4.1 Critères ML couverts par chaque personne

| Critère ML | A | B | C |
|---|---|---|---|
| Compréhension du sujet | ✅ T1 | ✅ contexte | ✅ T26 |
| Dataset & Données | ✅ T2-T6 | ✅ T7-T9 EDA | ✅ T23 feat imp |
| Preprocessing | ✅ T10-T13 | ✅ T12, T15-T17 | (transversal) |
| Feature engineering | — | ✅ T14 ratios | (interp) |
| Modèles ML | ✅ T29 pondération | ✅ T18 LR | ✅ T19, T21 RF |
| Évaluation performances | ✅ T41 | ✅ T42, T43 | ✅ T22, T23, T25 |
| Analyse des résultats | ✅ T40 diagnostic | ✅ T42 comparaison | ✅ T20, T26 |

→ **Chaque personne couvre les 7 critères ML**. Aucun déséquilibre majeur.

### 4.2 Critères Practical AI couverts par chaque personne

| Critère Practical AI | A | B | C |
|---|---|---|---|
| Modèle Deep Learning | ✅ T29, T39, T41 | ✅ T35-T38 | ✅ T30 |
| Architecture du réseau | (T39 implémente) | (config training) | ✅ T30-T34 ⭐ |
| Entraînement et optimisation | ✅ T39, T40 | ✅ T35-T38 ⭐ | ✅ T32, T33 |
| Comparaison ML vs DL | ✅ T41 contribue | ✅ T42, T43 ⭐ | ✅ T44, T45 ⭐ |
| Partie pratique / intelligente | (la fuite mesurée client par client) | (visualisation comparative) | (validation littérature) |

→ **Chaque personne couvre les 5 critères Practical AI**. Note attendue : équivalente entre les 3.

### 4.3 Diagnostic anti-déséquilibre

| Risque | Mitigation appliquée |
|---|---|
| « Une personne fait que de l'EDA » | A et B ont chacun des tâches ML + DL ; pas de spécialisation extrême |
| « Une personne est noyée dans le code mécanique » | Chaque personne a 1-2 *wow factors* visibles |
| « Une personne ne parle pas du cours » | Chaque personne illustre ≥ 5 concepts du cours Pr. El Alami Hassoun |
| « La fuite vole le show » | C, qui présente la fuite, n'a que 8 tâches ML (vs 9 pour A et B) pour compenser |
| « ROC est l'image finale » | B présente ROC mais A clôt l'évaluation MLP (T41) et C clôt la conclusion (T45) |

---

## 5. CHECKLIST DE PRÉPARATION VIDÉO

### 5.1 Pour chaque personne avant l'enregistrement

- [ ] Préparer **ses 15 slides** sur le PowerPoint correspondant
- [ ] Lire son **script narratif** (sections A.4, B.4, C.4) 3 fois à voix haute
- [ ] Mémoriser ses **2 phrases d'ouverture/fermeture** (sections X.5)
- [ ] Vérifier qu'elle **prononce clairement son nom** au début de chaque segment
- [ ] Préparer **2 ou 3 chiffres clés** à citer (AUC, % défaut, pages cours)
- [ ] Tester son **microphone** + connexion réseau

### 5.2 Coordination commune

- [ ] Définir l'**ordre fixe** : A → B → C en partie 1, puis A → B → C en partie 2
- [ ] Définir le **logiciel de capture** : OBS Studio, Loom, Microsoft PowerPoint screen recording, etc.
- [ ] Tester les **transitions** entre les segments avec une coupe propre
- [ ] Vérifier que la vidéo finale **ne dépasse pas 40 min** (limite usuelle)
- [ ] Enregistrer le **fichier final en MP4** 1080p
- [ ] Vérifier que **chaque voix est clairement identifiable**

### 5.3 Risques à éviter

| Risque | Solution |
|---|---|
| Une personne oublie de se présenter | Mettre un slide « VOIX : X » avant chaque segment |
| Difficile d'identifier qui parle | Ajouter un sous-titre au bas du slide avec la lettre A/B/C ou le nom |
| Une personne parle 15 min, l'autre 5 min | Suivre les durées du tableau (A 8'30, B 9'20, C 10'25) |
| Un segment manque de profondeur | Chaque tâche doit être présentée AVEC son output mesuré (chiffres) |

---

## 6. RÉFÉRENCE AUX TÂCHES DÉTAILLÉES

Pour le **contenu détaillé** de chaque tâche (Contexte / Décision / Implémentation / Output / Cas réel), se référer au document maître :

📄 **`LIVRABLE_FINAL/CORE_PROJET_TACHES.md`** — 53 tâches documentées avec les 5 composants chacune.

Ce document de répartition (présent) sert de **plan de présentation vidéo** et de **carnet de bord individuel**. Le CORE sert de **manuel de référence technique** lors de l'enregistrement.

---

## 7. TABLEAU DE VALIDATION FINALE

| Critère du prof | A | B | C | Validé ? |
|---|---|---|---|---|
| Même nombre de tâches | 15 | 15 | 15 | ✅ |
| Même nombre tâches ML | 9 | 9 | 8 | ≈ |
| Même nombre tâches DL | 6 | 6 | 7 | ≈ |
| Wow factor ML par personne | T1 | T18 | T20-T21 | ✅ |
| Wow factor DL par personne | T39 | T42-T43 | T30-T31 | ✅ |
| Identification individuelle | 2× | 2× | 2× | ✅ |
| Concepts cours par personne | 6 | 5 | 11 | ✅ |
| Tous les critères ML couverts | ✅ | ✅ | ✅ | ✅ |
| Tous les critères Practical AI couverts | ✅ | ✅ | ✅ | ✅ |
| Temps de parole équilibré (±2 min) | 8'30 | 9'20 | 10'25 | ✅ |
| Difficulté cognitive comparable | 27 | 31 | 33 | ≈ |
| Application web exclue | — | — | — | ✅ |

**Verdict** : répartition équitable. Les 3 personnes auront des notes attendues comparables dans les deux modules.

---

*Document généré le 3 juin 2026 — ENSA Tanger — Modules Machine Learning + Practical AI — Encadré par Pr. M. El Alami Hassoun.*
*La répartition reste anonyme (A, B, C). L'attribution aux trois membres réels (Abdou, Amr, Badr) reste à effectuer par l'équipe selon les préférences personnelles.*
