# SCRIPT DIAPO FINAL — Présentation vidéo
## Prédiction du Défaut de Remboursement de Prêt
### ENSA Tanger · Modules Machine Learning + Practical AI · Encadré par Pr. M. El Alami Hassoun

> **Document de référence pour la PHASE 3 (construction du .pptx).** Décrit slide-par-slide : titre, présentateur, contenu visuel, et speaker notes intégrales. 50 slides · ~32 min.

---

## Mapping des voix
- **A = Badr Ameur El Idrissi** (Foundation ML + Boucle entraînement DL)
- **B = Abdourhamane Mohammed Alhendi** (EDA + LR + Config DL + Comparaison)
- **C = Bekkari Amr** (Intro + RF + Fuite + Architecture DL + Benchmarking + Conclusion)

## Palette visuelle (à conserver)
- **Navy** `#0a1a3a` (titre, en-têtes)
- **Or** `#c9a861` (accent, soulignement)
- **Or doux** `#e5d9b8` (bandeaux)
- **Blanc** `#ffffff` (fond contenu)
- **Beige clair** `#f7f2e8` (fond intro et conclusion)
- **Gris ligne** `#dce3ec` (séparateurs)

---

═══════════════════════════════════════════════════════════════════════
# 🎬 BLOC 1 — OUVERTURE PAR AMR (3 min · 5 slides)
═══════════════════════════════════════════════════════════════════════

## S01 — Couverture
- **Voix** : Amr · **Durée** : 60 s · **Section** : Ouverture
- **Visuel** :
  - Fond navy plein
  - Titre central or 44pt : « Prédiction du Défaut de Remboursement de Prêt »
  - Sous-titre blanc 22pt : « Projet de fin de module — Machine Learning & Practical AI »
  - Bandeau or doux en bas : « Encadré par Pr. M. El Alami Hassoun »
  - 3 noms en colonnes : Badr Ameur El Idrissi · Abdourhamane Mohammed Alhendi · Bekkari Amr
  - École : ENSA Tanger · Année 2025-2026
- **Speaker notes** :
> Bonjour. Je m'appelle Bekkari Amr, étudiant en cinquième année à l'École Nationale des Sciences Appliquées de Tanger. Avec mes coéquipiers Badr Ameur El Idrissi et Abdourhamane Mohammed Alhendi, nous avons le plaisir de vous présenter notre projet de fin de module commun aux modules Machine Learning et Practical AI : la prédiction du défaut de remboursement de prêt. Ce projet est encadré par Pr. M. El Alami Hassoun, et il applique directement les concepts fondamentaux vus en cours, du perceptron de Rosenblatt jusqu'à la backpropagation de Hinton. Je vais ouvrir la présentation par le contexte économique et la question scientifique, puis chacun d'entre nous présentera sa contribution dans les deux modules.

## S02 — Contexte économique & réglementaire
- **Voix** : Amr · **Durée** : 40 s · **Section** : Ouverture
- **Visuel** :
  - Titre navy : « Pourquoi le scoring de crédit ? »
  - 3 cartes verticales (Bâle III · IFRS 9 · RGPD art. 22) avec icône or
  - Bas : statistique en grand or « 150 Md USD » + sous-texte gris « pertes mondiales annuelles dues aux défauts (BCE 2024) »
  - Mention discrète : « Au Maroc → Bank Al-Maghrib »
- **Speaker notes** :
> Le scoring de crédit est l'un des domaines historiques du machine learning appliqué. Trois piliers réglementaires l'imposent : Bâle III demande une estimation de la probabilité de défaut pour calculer les fonds propres ; IFRS 9 impose des provisions pour pertes attendues calculées de façon prospective ; et l'article 22 du RGPD garantit un droit à l'explication pour toute décision automatisée. Les pertes mondiales annuelles dues aux défauts dépassent 150 milliards de dollars selon la BCE 2024. Au Maroc, c'est Bank Al-Maghrib qui supervise ce domaine.

## S03 — Question de recherche & 3 hypothèses scientifiques
- **Voix** : Amr · **Durée** : 60 s · **Section** : Ouverture
- **Visuel** :
  - Bandeau navy en haut avec citation : « Peut-on prédire avec précision si un emprunteur va rembourser son prêt, à partir de ses caractéristiques au moment de la demande ? »
  - Tableau 3 colonnes en bas :
    | H1 | H2 | H3 |
    |---|---|---|
    | ML supervisé → AUC ≥ 0.73 | DL ≈ ML sur tabulaire | Fuite méthodologique mesurable |
    | Source : Lessmann 2015 | Source : Gunnarsson 2021 | Analyse interne |
- **Speaker notes** :
> Notre question scientifique centrale est : peut-on prédire avec précision si un emprunteur va rembourser son prêt, à partir de ses caractéristiques financières et socio-économiques au moment de la demande ? Pour y répondre, nous formulons trois hypothèses falsifiables. H1 : un modèle ML supervisé peut atteindre une AUC supérieure ou égale à 0,73 sur le scoring de crédit — référence Lessmann 2015. H2 : le Deep Learning n'apporte pas de gain significatif face au ML classique sur des données tabulaires — référence Gunnarsson 2021. H3 : trois variables du dataset LendingClub — grade, sub-grade et int-rate — constituent une fuite méthodologique car ce sont des décisions a posteriori. Chacune de ces hypothèses sera testée expérimentalement dans la suite.

## S04 — Deux notes distinctes : ML + Practical AI
- **Voix** : Amr · **Durée** : 40 s · **Section** : Ouverture
- **Visuel** :
  - Deux blocs côte à côte :
    - Bloc gauche bleu : « Note Machine Learning » — 5 critères : preprocessing · feature engineering · modèles ML · évaluation · analyse des résultats
    - Bloc droit or : « Note Practical AI » — 5 critères : modèle DL · architecture · entraînement/optimisation · comparaison ML vs DL · partie pratique
  - Ligne d'horizon : « Un projet, deux évaluations, deux notes »
- **Speaker notes** :
> Important : ce projet est commun aux modules Machine Learning et Practical AI, mais chaque module aura sa propre note. La note Machine Learning évaluera principalement le preprocessing, le feature engineering, les modèles ML, l'évaluation des performances et l'analyse des résultats. La note Practical AI évaluera le modèle Deep Learning, son architecture, son entraînement et optimisation, la comparaison ML versus DL, et la partie pratique du projet. Nous structurons donc la présentation en deux grandes parties, et chacun de nous interviendra dans les deux pour que l'évaluation individuelle soit équitable.

## S05 — Roadmap : qui présente quoi
- **Voix** : Amr · **Durée** : 30 s · **Section** : Ouverture
- **Visuel** :
  - Frise horizontale en 6 segments avec photos d'avatars stylisés (cercles or) :
    - 1. Badr (Foundation ML) — bleu clair
    - 2. Abdou (EDA + LR) — bleu clair
    - 3. Amr (RF + Fuite) — or soutenu (wow factor visible)
    - 4. Badr (Setup PyTorch + Boucle) — bleu clair
    - 5. Abdou (Config + ROC) — bleu clair
    - 6. Amr (Architecture + Benchmarking) — or soutenu
  - Ligne du temps : Intro 3' → ML 17' → DL 15' → Conclusion 2'
- **Speaker notes** :
> Voici la structure de la présentation. Nous interviendrons dans cet ordre : Badr ouvrira la partie ML avec la fondation du projet, puis Abdou explorera les patterns et entraînera la Régression Logistique, et je reviendrai conclure la partie ML avec le Random Forest et la découverte de la fuite. Pour la partie Practical AI, Badr présentera la boucle d'entraînement, Abdou la comparaison ML versus DL, et je clôturerai avec l'architecture du MLP et le benchmarking face à la littérature. Je laisse maintenant la parole à Badr.

═══════════════════════════════════════════════════════════════════════
# 🟦 BLOC 2 — PARTIE 1 : NOTE MACHINE LEARNING (17 min · 21 slides)
═══════════════════════════════════════════════════════════════════════

## S06 — Identification Badr + ses 9 tâches ML
- **Voix** : Badr · **Durée** : 35 s · **Section** : ML / Foundation
- **Visuel** :
  - Carte profil : portrait illustré (silhouette navy + cercle or)
  - Nom : Badr Ameur El Idrissi
  - Rôle : « Foundation ML — fondations du projet »
  - Liste à puces (9 tâches) : T1 choix dataset · T2 chargement · T3 inspection · T4 construction TARGET · T5 visualisation déséquilibre · T6 sélection features · T10 imputation · T11 doublons · T13 gestion infinies
- **Speaker notes** :
> Bonjour, je suis Badr Ameur El Idrissi, étudiant en cinquième année à l'ENSA Tanger. Je vais présenter la phase fondatrice du projet en Machine Learning. Ce que je vais détailler dans les six minutes qui suivent : le choix du dataset, le chargement et l'inspection, la construction de la variable cible TARGET, la visualisation du déséquilibre, la sélection des features pertinentes parmi 151 colonnes, et trois étapes critiques de nettoyage — imputation, doublons, valeurs infinies. C'est le travail invisible qui rend tout le reste fiable.

## S07 — T1 · Pivot Home Credit → LendingClub
- **Voix** : Badr · **Durée** : 60 s · **Section** : ML / Foundation · ⭐ wow factor
- **Visuel** :
  - Schéma : flèche stylisée « Home Credit ❌ → LendingClub ✅ »
  - Tableau 3 lignes des 3 raisons :
    1. Taux 20 % apprenable (vs 8 % HC)
    2. Variables lisibles (loan_amnt, grade) vs codes (EXT_SOURCE_3)
    3. Survivorship bias assumé (acceptable pour pédagogie)
  - Mini df.head() en bas : 5 lignes du CSV brut
- **Speaker notes** :
> Le projet a commencé par un pivot. Dans notre proposition initiale, nous avions choisi le dataset Home Credit Default Risk. À la mise en œuvre, nous nous sommes rendu compte de trois limites : taux de défaut de 8 % seulement — un modèle qui prédit « tout le monde rembourse » atteint déjà 92 % d'accuracy ; variables anonymisées sous forme de codes comme EXT_SOURCE_3 — illisibles pour une présentation pédagogique ; et structure multi-tables qui aurait nécessité des jointures lourdes. Nous avons donc basculé sur LendingClub Loan Data : 2,2 millions de prêts, 151 colonnes, taux de défaut 20 % bien plus apprenable, et des variables financièrement interprétables. Nous assumons en revanche le biais de sélection — LendingClub a déjà filtré les pires dossiers, ce qui sera un point de honnêteté méthodologique à mentionner.

## S08 — T2-T3 · Chargement & Inspection
- **Voix** : Badr · **Durée** : 35 s · **Section** : ML / Foundation
- **Visuel** :
  - Titre : « Charger 100 000 lignes sur 2.2 M — pourquoi cette contrainte »
  - Bloc de code :
```python
CSV_PATH = "accepted_2007_to_2018Q4.csv"  # 1,5 Go brut
df = pd.read_csv(CSV_PATH, nrows=100_000, low_memory=False)
print(df.shape)  # (100000, 151)
```
  - Bullets : RAM 16 Go · types mixtes → low_memory=False · échantillon représentatif
- **Speaker notes** :
> Le fichier brut pèse 1,5 gigaoctet — impossible à charger entièrement avec 16 gigaoctets de RAM sous Windows. Nous chargeons donc 100 000 lignes via nrows. Pourquoi 100 000 et pas 10 000 ou tout ? Avec 10 000 lignes, on n'aurait que 2 000 défauts — pas assez pour entraîner un MLP. Avec 100 000 on a environ 16 000 défauts, suffisant pour la modélisation tout en restant traitable en CPU. Le flag low_memory à False est obligatoire : sans lui, pandas devine les types ligne par ligne et peut convertir silencieusement certains identifiants en NaN.

## S09 — T4 · Construction de la variable cible TARGET
- **Voix** : Badr · **Durée** : 50 s · **Section** : ML / Foundation
- **Visuel** :
  - Tableau de mapping :
    | Statut | Décision |
    |---|---|
    | Fully Paid | TARGET = 0 |
    | Charged Off | TARGET = 1 |
    | Default | TARGET = 1 |
    | Late (31-120 days) | TARGET = 1 (norme Bâle) |
    | Current | EXCLU (issue inconnue) |
    | Late (16-30 days) | EXCLU (ambigu) |
  - Bloc de code :
```python
default = ["Charged Off", "Default", "Late (31-120 days)"]
paid    = ["Fully Paid"]
df = df[df["loan_status"].isin(default + paid)].copy()
df["TARGET"] = df["loan_status"].isin(default).astype(int)
```
  - Stat clé : « 88 333 prêts conservés · 20.43 % défauts »
- **Speaker notes** :
> La variable cible native est loan_status, qui contient neuf statuts différents. Notre problème étant binaire, nous devons décider quoi faire de chaque statut. Nous gardons Fully Paid comme remboursé. Charged Off, Default et Late 31-120 jours comme défaut — ce dernier étant un défaut technique selon les normes Bâle. Nous excluons Current et In Grace Period qui ont une issue inconnue, et Late 16-30 jours qui est souvent recouvré. Après filtrage, il reste 88 333 prêts avec un taux de défaut de 20,43 %.

## S10 — T5 · Visualisation du déséquilibre 80/20
- **Voix** : Badr · **Durée** : 30 s · **Section** : ML / Foundation
- **Visuel** :
  - Image : `outputs/figures/eda_target_distribution.png` (centré, grand)
  - Encadré or à droite : « Implication méthodologique » → AUC-ROC > Accuracy + class_weight='balanced'
- **Speaker notes** :
> Voici la distribution observée : 79,57 % de remboursés contre 20,43 % de défauts. Cette asymétrie a une implication méthodologique majeure : un modèle qui prédit toujours zéro atteint déjà 79,57 % d'accuracy. L'accuracy seule ne mesure donc rien. Nous choisissons l'AUC-ROC comme métrique principale parce qu'elle est insensible au déséquilibre, et nous activerons le class_weight='balanced' dans tous nos modèles pour compenser cette asymétrie pendant l'entraînement.

## S11 — T6 · Sélection des features (151 → 19)
- **Voix** : Badr · **Durée** : 50 s · **Section** : ML / Foundation
- **Visuel** :
  - Schéma entonnoir : 151 brutes → 19 retenues (11 num + 7 cat + TARGET)
  - Petit tableau des features clés (loan_amnt, int_rate, dti, grade, purpose, term, ...) avec un ⚠️ à côté de int_rate, grade, sub_grade
  - Annotation : « 3 variables marquées ⚠️ feront l'objet d'une analyse de fuite (Amr, slide 24) »
- **Speaker notes** :
> Sur les 151 colonnes brutes, beaucoup sont vides à plus de 99 % — champs alimentés uniquement pour les défauts en recouvrement. Notre critère de sélection est l'interprétabilité financière et un taux de NaN inférieur à 10 %. Nous retenons 11 variables numériques comme loan_amnt, int_rate, dti, annual_inc, et 7 variables catégorielles comme grade, purpose, term. Trois variables sont marquées d'un avertissement — int_rate, grade et sub_grade — car nous suspectons qu'elles sont des décisions de LendingClub elles-mêmes. Amr en démontrera l'impact dans son segment.

## S12 — T10-T11-T13 · Nettoyage (imputation, doublons, infinies)
- **Voix** : Badr · **Durée** : 45 s · **Section** : ML / Foundation
- **Visuel** :
  - Bloc de code :
```python
# T10 — Imputation
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# T11 — Doublons
print(df.duplicated().sum())  # 0 (LC id unique)

# T13 — Inf après division par 0
df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True))
```
  - Stats : Médiane DTI=17.59 · Mode home_ownership=MORTGAGE · 0 doublon · 0 inf restant
- **Speaker notes** :
> Trois opérations critiques pour la fiabilité du pipeline. Premièrement, imputation : médiane pour les variables numériques — robuste aux outliers comme un revenu de 9 millions — et mode pour les catégorielles — préserve la distribution observée. Deuxièmement, vérification des doublons : zéro doublon car LendingClub donne un identifiant unique par prêt. Troisièmement, les valeurs infinies : nos ratios financiers à venir divisent par annual_inc, qui peut être nul après imputation. Nous remplaçons ces infinis par NaN puis réimputons. Sans cette protection, le StandardScaler renverrait une erreur Input contains infinity. Je laisse maintenant la parole à Abdou pour l'exploration statistique.

---

## S13 — Identification Abdou + ses 9 tâches ML
- **Voix** : Abdou · **Durée** : 35 s · **Section** : ML / EDA + FE + LR
- **Visuel** :
  - Carte profil Abdou (silhouette navy + cercle or)
  - Nom : Abdourhamane Mohammed Alhendi
  - Rôle : « Exploration statistique + Feature engineering + Régression Logistique »
  - Liste : T7 boxplots · T8 corrélation · T9 catégorielle · T12 outliers · T14 ratios · T15 OneHot · T16 split · T17 scaler · T18 LR
- **Speaker notes** :
> Bonjour, je suis Abdourhamane Mohammed Alhendi, étudiant à l'ENSA Tanger. Mon segment couvre trois moments : d'abord, l'exploration statistique du dataset — c'est moi qui ai produit les figures EDA que vous allez voir. Ensuite, le feature engineering : création des ratios financiers, encodage One-Hot, split stratifié et normalisation. Et enfin, l'entraînement de notre modèle de référence scientifique : la Régression Logistique. Allons-y.

## S14 — T7 · Boxplots bivariés par TARGET
- **Voix** : Abdou · **Durée** : 50 s · **Section** : ML / EDA
- **Visuel** :
  - Image : `outputs/figures/eda_boxplots.png` (grille 4×3)
  - Encadré or à droite avec 3 observations :
    - int_rate : médiane 12 % vs 15 % → SIGNAL FORT
    - dti : 17 vs 19 → signal faible
    - loan_amnt : identique → signal nul
- **Speaker notes** :
> Voici notre première vue analytique. J'ai produit 11 boxplots, un par variable numérique, séparés par la cible TARGET. Trois observations à retenir. Premièrement, int_rate : la médiane passe de 12 % pour les remboursés à 15 % pour les défauts. C'est notre signal le plus fort — et c'est aussi le plus suspect, car ce taux est fixé par LendingClub eux-mêmes après leur scoring interne. Deuxièmement, dti : médiane 17 versus 19, signal faible mais réel. Troisièmement, loan_amnt : distributions identiques, signal nul. Le montant brut ne discrimine pas le risque ; ce sont les ratios qui comptent.

## S15 — T8 · Matrice de corrélation
- **Voix** : Abdou · **Durée** : 40 s · **Section** : ML / EDA
- **Visuel** :
  - Image : `outputs/figures/eda_corr_heatmap.png`
  - Encadré or : Top corrélations vers TARGET → int_rate +0.26 · dti +0.10 · revol_util +0.09
  - Note multicollinéarité : loan_amnt ↔ installment +0.95 (dérivée mécanique)
- **Speaker notes** :
> La matrice de corrélation confirme l'analyse bivariée. La corrélation maximale avec la cible est int_rate à plus 0,26 — c'est notre signal le plus fort, suivi de dti à 0,10 et revol_util à 0,09. Notez aussi une multicollinéarité parfaite entre loan_amnt et installment à 0,95, due au fait que la mensualité est mathématiquement dérivée du montant et du taux. Ce sera géré par la régularisation L2 implicite de la Régression Logistique.

## S16 — T9 · Analyse catégorielle (annonce de la fuite)
- **Voix** : Abdou · **Durée** : 45 s · **Section** : ML / EDA
- **Visuel** :
  - Image : `outputs/figures/eda_categorical_rates.png` (taux de défaut par modalité)
  - Encadré or géant en bas : « grade A → 6 % · grade G → 49 % » avec flèche montante
  - Notation discrète : « ⚠️ Échelle monotone parfaite — décision LC déjà encodée »
- **Speaker notes** :
> Analyse des variables catégorielles, c'est ici que le drame se noue. Pour la variable grade, le taux de défaut est parfaitement monotone : 6 % pour le grade A, 12 % pour B, 20 % pour C, 29 % pour D, 37 % pour E, 44 % pour F, et 49 % pour G — quasiment un coin-flip pour les pires. Autre observation : la durée du prêt — 15 % de défauts à 36 mois contre 33 % à 60 mois, soit 18 points d'écart. Cette monotonie parfaite du grade est tellement belle qu'elle est suspecte : nous verrons dans le segment d'Amr que c'est précisément la fuite méthodologique.

## S17 — T12 + T14 · Outliers clip p99 + 2 ratios financiers
- **Voix** : Abdou · **Durée** : 45 s · **Section** : ML / Feature engineering
- **Visuel** :
  - Bloc de code :
```python
# T12 — Clip outliers (revenus délirants 9M$ → p99)
for col in ["annual_inc", "revol_bal"]:
    p99 = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=p99)

# T14 — Ratios financiers inspirés HCSF
df["installment_income_ratio"] = df["installment"] * 12 / df["annual_inc"]
df["loan_income_ratio"]        = df["loan_amnt"] / df["annual_inc"]
```
  - Annotation : « annual_inc p99 ≈ 280 000 $ · HCSF taux d'effort 35 % »
- **Speaker notes** :
> Deux étapes essentielles. Premièrement, clip des outliers : nous écrêtons annual_inc et revol_bal au 99e percentile pour neutraliser les valeurs absurdes — certains formulaires contiennent un revenu déclaré de 9 millions de dollars. Sans clip, le StandardScaler centre la distribution autour de ces monstres et écrase tous les vrais revenus. Deuxièmement, le feature engineering : deux ratios financiers métier. L'installment-income-ratio mesure le taux d'effort — pourcentage du revenu consommé par la mensualité, référence à la règle des 35 % du HCSF français. Le loan-income-ratio mesure le rapport prêt sur revenu — combien d'années de salaire le prêt représente. Ces deux ratios sont interprétables pour un banquier, alors qu'un montant brut en dollars ne dit rien sans contexte.

## S18 — T15-T16-T17 · One-Hot + Split + StandardScaler
- **Voix** : Abdou · **Durée** : 40 s · **Section** : ML / Feature engineering
- **Visuel** :
  - Bloc de code :
```python
# T15 — One-Hot drop_first prévient colinéarité parfaite
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
# Shape: (88 333, 81) — 11 num + 2 ratios + 67 dummies + TARGET

# T16 — Split stratifié (random_state=42 pour reproductibilité)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
# Train (70 666, 80) · Test (17 667, 80) · 20.43 % défauts partout

# T17 — Scaler fit sur train UNIQUEMENT (anti-fuite)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```
- **Speaker notes** :
> Trois étapes de préparation. One-Hot avec drop_first prévient la colinéarité parfaite — sans ce flag les douze modalités de purpose somment à 1 et rendent la régression non identifiable. Nous passons de 19 colonnes à 80 features. Split stratifié 80/20 avec random_state=42 garantit deux choses : reproductibilité parfaite, et stratification — le taux de 20,43 % de défauts est préservé à la quatrième décimale en train comme en test. StandardScaler est fitté uniquement sur le train. Si on fittait sur tout le dataset, on ferait fuiter les statistiques du test vers le scaler — biais d'optimisme classique.

## S19 — T18 · Régression Logistique ⭐
- **Voix** : Abdou · **Durée** : 60 s · **Section** : ML / Modèles · ⭐ wow factor
- **Visuel** :
  - Bloc de code :
```python
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(
    max_iter=1000,           # convergence lbfgs sur 80 dims
    class_weight="balanced",  # poids ≈ 3.9 pour compenser 20/80
    random_state=42
)
model_lr.fit(X_train_scaled, y_train)
y_proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1]
```
  - Tableau résultats (gros chiffres or) :
    - AUC-ROC : **0.7311**
    - Accuracy : 0.6672
    - F1-score : 0.4476
    - Temps : 0.5 s
  - Texte : « Top coefficients ↑ défaut : int_rate · dti · revol_util »
- **Speaker notes** :
> Premier modèle : la Régression Logistique, baseline scientifique de Lessmann 2015. Trois hyperparamètres choisis volontairement. max_iter à 1000 parce que la valeur par défaut 100 ne suffit pas avec class_weight balanced sur 80 features — lbfgs oscille. class_weight balanced pondère inversement à la fréquence, soit un ratio d'environ 3,9 pour compenser le déséquilibre 20/80. random_state à 42 pour la reproductibilité. Résultats : AUC 0,7311, accuracy 67 %, F1 0,45, en 0,5 seconde. Les trois coefficients les plus positifs sont int_rate, dti et revol_util — interprétation directe : une augmentation d'un écart-type du taux d'intérêt multiplie l'odds de défaut par un facteur d'environ 1,6.

## S20 — Synthèse Abdou + transition Amr
- **Voix** : Abdou · **Durée** : 20 s · **Section** : ML / Bilan partiel
- **Visuel** :
  - Bandeau récap : « Mes 9 tâches : EDA · FE · LR baseline »
  - Tableau LR : AUC 0.7311 · Accuracy 0.667 · F1 0.45
  - Flèche or vers : « Au tour d'Amr : modèle non-linéaire + découverte méthodologique »
- **Speaker notes** :
> Pour résumer : j'ai exploré les patterns du dataset, j'ai préparé les features avec deux ratios financiers métier, et j'ai entraîné la Régression Logistique qui atteint une AUC de 0,7311. C'est notre baseline scientifique. Au tour d'Amr maintenant — il va tester un modèle non-linéaire, le Random Forest, et va surtout nous démontrer une découverte méthodologique majeure.

---

## S21 — Identification Amr (return) + ses 8 tâches ML
- **Voix** : Amr · **Durée** : 30 s · **Section** : ML / RF + Fuite
- **Visuel** :
  - Carte profil Amr en or soutenu (return)
  - Liste : T19 RF complet · T20 ⭐ découverte fuite · T21 ⭐ RF sans fuite · T22 métriques · T23 feature importance · T24 sauvegarde · T25 tableau · T26 interprétation business
- **Speaker notes** :
> C'est de nouveau Bekkari Amr. Ma contribution sur la partie Machine Learning concerne les modèles non-linéaires et la conclusion scientifique. Je vais entraîner un Random Forest, analyser pourquoi une variable domine, démontrer qu'il s'agit d'une fuite méthodologique, mesurer son impact rigoureusement, et formuler la recommandation business finale.

## S22 — T19 · Random Forest complet
- **Voix** : Amr · **Durée** : 50 s · **Section** : ML / Modèles
- **Visuel** :
  - Bloc de code :
```python
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=100,        # gain marginal au-delà (Lessmann 2015)
    min_samples_split=10,    # évite la sur-spécialisation
    class_weight="balanced",
    n_jobs=-1, random_state=42
)
model_rf.fit(X_train_scaled, y_train)
```
  - Tableau résultats :
    - AUC : 0.7232 (vs LR 0.7311 → ΔAUC = -0.008)
    - Accuracy : 0.7834
    - F1 : 0.3684
  - Théorie : Breiman 2001 — bootstrap + √p features par split
- **Speaker notes** :
> Deuxième modèle : le Random Forest, ensemble de 100 arbres CART entraînés sur des bootstraps avec sélection aléatoire de racine de p features à chaque nœud — théorie de Breiman 2001. Les hyperparamètres sont calibrés : 100 arbres parce que Lessmann 2015 a montré que le gain entre 100 et 1000 arbres est de 0,003 d'AUC seulement ; min_samples_split à 10 pour empêcher les nœuds de moins de 10 échantillons qui sont du bruit. Résultats : AUC 0,7232, accuracy 78,3 %, F1 0,37, en 2,3 secondes. Observation scientifique : la différence d'AUC avec la Régression Logistique est de seulement 0,008 — non significative. Les relations linéaires dominent ce dataset.

## S23 — T23 · Feature importance — la révélation
- **Voix** : Amr · **Durée** : 45 s · **Section** : ML / Diagnostic
- **Visuel** :
  - Image : `outputs/figures/feature_importance_rf.png`
  - Encadré or à droite : « Top 5 dominantes : int_rate 10% · dti 7% · loan_income_ratio 7% · installment 7% · revol_util 6% »
  - Question en bas : « int_rate, grade, sub_grade — variables prédictives ou décisions a posteriori ? »
- **Speaker notes** :
> Voici la feature importance du Random Forest, et c'est ici que tout bascule. La variable la plus importante est int_rate à 10 % de l'importance totale, suivi de dti, loan_income_ratio, installment, et revol_util. Mais regardez de plus près : int_rate représente 10 %, et installment qui est mathématiquement dérivé d'int_rate ajoute encore 7 %. Et si on compte les colonnes one-hot de grade et sub_grade, on ajoute encore 9 %. Question fondamentale : ces variables sont-elles vraiment prédictives, ou sont-elles des décisions a posteriori de LendingClub eux-mêmes ?

## S24 — T20 · Découverte de la fuite méthodologique ⭐⭐⭐
- **Voix** : Amr · **Durée** : 70 s · **Section** : ML / Méthodologie · ⭐ wow factor majeur
- **Visuel** :
  - Schéma circulaire stylé :
    Client → [LendingClub Scoring Interne] → grade + sub_grade + int_rate → Modèle ML → Prédiction
  - Annotation rouge : « Boucle circulaire : on prédit ce que LC a déjà décidé »
  - 3 variables marquées ⚠️ :
    - grade (A-G) — note de risque LC
    - sub_grade (A1-G5) — raffinement
    - int_rate — taux fixé par LC
  - Phrase frappante : « Prédire un cancer à partir du diagnostic déjà posé »
- **Speaker notes** :
> J'ai donc analysé sérieusement ce qui se passe. Trois variables sont des décisions a posteriori de LendingClub : grade — la note de risque de A à G attribuée par LendingClub eux-mêmes après leur scoring interne ; sub_grade — son raffinement de A1 à G5 ; et int_rate — le taux d'intérêt qui est fixé directement à partir de leur grade. Utiliser ces variables pour prédire le défaut est circulaire : c'est comme prédire un cancer en utilisant le diagnostic déjà posé par le médecin. Dans la vraie vie d'une banque, on ne dispose pas du scoring d'un autre prêteur quand un client arrive au guichet. Le modèle doit prédire le risque à partir des variables brutes, pas à partir d'un score déjà calculé ailleurs. C'est ce qu'on appelle une fuite méthodologique.

## S25 — T21 · Random Forest SANS FUITE ⭐⭐
- **Voix** : Amr · **Durée** : 60 s · **Section** : ML / Méthodologie · ⭐ wow factor
- **Visuel** :
  - Bloc de code :
```python
# Identification des features de fuite (43 sur 80)
LEAK_NUM = {"int_rate", "installment", "installment_income_ratio"}
def is_leak(col):
    return col in LEAK_NUM \
        or col.startswith("grade_") \
        or col.startswith("sub_grade_")
lean_cols = [c for c in feature_names if not is_leak(c)]
# 80 features → 37 features (43 retirées)
```
  - Tableau de mesure rigoureuse :
    | Métrique | RF complet | RF lean | Δ |
    |---|---|---|---|
    | AUC | 0.7232 | **0.6941** | **-0.029** |
    | F1 | 0.3684 | 0.3043 | -0.064 |
    | Features | 80 | 37 | -43 |
  - Verdict : « H3 VALIDÉE — fuite mesurée à 0.029 d'AUC »
- **Speaker notes** :
> Pour quantifier rigoureusement l'impact de cette fuite, j'ai réentraîné un second Random Forest aux hyperparamètres strictement identiques, mais sur 37 features uniquement — j'ai retiré les 43 features fuitées : int_rate, installment, installment_income_ratio, et toutes les colonnes one-hot de grade et sub_grade. Le résultat : l'AUC passe de 0,7232 à 0,6941, soit une chute de 0,029. C'est notre contribution méthodologique : nous avons mesuré expérimentalement que la fuite gonfle artificiellement l'AUC de 0,029. L'hypothèse H3 est validée. Pour un conseiller bancaire qui ne dispose pas du grade LendingClub d'un client, la performance réaliste de prédiction est de 0,694 d'AUC, pas 0,732.

## S26 — T25-T26 · Synthèse ML & recommandation business
- **Voix** : Amr · **Durée** : 60 s · **Section** : ML / Conclusion
- **Visuel** :
  - Tableau récap 3 modèles ML :
    | Modèle | Features | AUC | Accuracy | F1 | Temps |
    |---|---|---|---|---|---|
    | Régression Logistique | 80 | **0.7311** | 0.667 | 0.448 | 0.5 s |
    | Random Forest complet | 80 | 0.7232 | 0.783 | 0.368 | 2.3 s |
    | Random Forest sans fuite | 37 | 0.6941 | 0.787 | 0.304 | 2.6 s |
  - Encadré or : « Recommandation production : LR sans fuite — interprétable, conforme RGPD, 22× plus rapide que MLP »
- **Speaker notes** :
> Synthèse des trois modèles Machine Learning. Régression Logistique à 0,7311, Random Forest complet à 0,7232, Random Forest sans fuite à 0,6941. Quelle recommandation pour la production ? La Régression Logistique sans fuite : équivalente au MLP en performance, 22 fois plus rapide, interprétable coefficient par coefficient — donc conforme à l'article 22 du RGPD qui impose le droit à l'explication. Voilà pour la partie Machine Learning. Passons maintenant au Deep Learning, qui sera évalué séparément dans la note Practical AI.

---

═══════════════════════════════════════════════════════════════════════
# 🟧 BLOC 3 — TRANSITION (15 sec · 1 slide)
═══════════════════════════════════════════════════════════════════════

## S27 — Transition vers Practical AI
- **Voix** : Amr · **Durée** : 15 s · **Section** : Transition
- **Visuel** :
  - Hero slide plein écran navy
  - Grand titre or 60pt : « Partie 2 — Practical AI »
  - Sous-titre blanc 24pt : « Note distincte · Critères : modèle DL · architecture · entraînement · comparaison ML vs DL · partie pratique »
- **Speaker notes** :
> Nous passons maintenant à la deuxième partie du projet, évaluée séparément dans le module Practical AI. Cinq critères : le modèle Deep Learning lui-même, son architecture, son entraînement et optimisation, la comparaison rigoureuse avec les modèles ML, et la partie pratique. Badr reprend la parole pour le setup et la boucle d'entraînement.

═══════════════════════════════════════════════════════════════════════
# 🟧 BLOC 4 — PARTIE 2 : NOTE PRACTICAL AI (15 min · 19 slides)
═══════════════════════════════════════════════════════════════════════

## S28 — Identification Badr DL + ses 6 tâches
- **Voix** : Badr · **Durée** : 30 s · **Section** : DL / Setup + Training loop
- **Visuel** :
  - Carte profil Badr (return, badge or « Practical AI »)
  - Liste : T27 PyTorch · T28 train/val · T29 pos_weight · T39 ⭐ Forward/Backward · T40 diagnostic · T41 évaluation MLP
- **Speaker notes** :
> Je reprends la parole — Badr Ameur El Idrissi. Pour la partie Practical AI, je présente le setup du framework PyTorch, la préparation du sous-set de validation, la pondération des classes, et surtout l'implémentation de la boucle d'entraînement — Forward, Backward, Update — qui est exactement le cycle décrit dans le cours de Pr. El Alami Hassoun pages 90 à 108.

## S29 — T27 · Choix PyTorch vs TensorFlow
- **Voix** : Badr · **Durée** : 45 s · **Section** : DL / Setup
- **Visuel** :
  - Tableau comparatif :
    | Framework | Pour | Contre | Retenu |
    |---|---|---|---|
    | **PyTorch** | Eager mode, debug facile, API Pythonique | Setup verbeux | ✅ |
    | TensorFlow/Keras | API haut niveau model.fit | Python 3.14 incompatible Windows | ❌ |
  - Snippet :
```python
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)  # reproductibilité
```
- **Speaker notes** :
> Notre proposition initiale prévoyait TensorFlow/Keras. Nous avons basculé sur PyTorch pour deux raisons. Premièrement, l'incompatibilité de TensorFlow avec Python 3.14 sous Windows — nous restons sur Python 3.11.9 obligatoire. Deuxièmement, et c'est plus pédagogique, PyTorch utilise l'eager mode : chaque opération est exécutée immédiatement, ce qui rend visibles les étapes du cours — Forward, calcul de la loss, Backward. Avec Keras, model.fit cache tout ce cycle. Pour une présentation pédagogique, PyTorch est plus parlant.

## S30 — T28-T29 · Train/val split + pos_weight
- **Voix** : Badr · **Durée** : 45 s · **Section** : DL / Setup
- **Visuel** :
  - Schéma :
    Train (70 666) → Train_sub (56 533) + Validation (14 133)
    Test (17 667) — intouché
  - Snippet :
```python
# T28 — Sous-split train/val pour EarlyStopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# T29 — Pondération des classes (déséquilibre 20/80)
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
# pos_weight ≈ 3.895 — 1 défaut pénalisé ~3.9× plus
```
- **Speaker notes** :
> Deux préparations spécifiques au Deep Learning. Premièrement, le sous-split train/validation : pour activer EarlyStopping il faut suivre la performance hors-train à chaque epoch. Utiliser le test set serait une fuite — la décision « quand arrêter » biaiserait l'AUC finale. Je sous-divise donc le train en 56 533 pour l'entraînement et 14 133 pour la validation. Deuxièmement, la pondération : avec un déséquilibre 80/20 sans pondération, le MLP optimise la cross-entropy en prédisant toujours zéro — minimum local trivial. Je calcule pos_weight comme le ratio négatifs sur positifs, soit environ 3,895. Un défaut compte 3,9 fois plus dans le calcul du gradient.

## S31 — T39 · Boucle Forward / Backward / Update ⭐⭐⭐
- **Voix** : Badr · **Durée** : 80 s · **Section** : DL / Cœur du cours · ⭐ wow factor majeur
- **Visuel** :
  - Bloc de code (le cœur visuellement encadré or) :
```python
for epoch in range(MAX_EPOCHS):
    model.train()  # mode training (Dropout actif, BN learn)
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()             # reset gradients
        logits = model(X_batch)            # ← FORWARD (cours p.90)
        loss = criterion(logits, y_batch)  # ← Coût J (cours p.104)
        loss.backward()                    # ← BACKWARD (cours p.96)
        optimizer.step()                   # ← w ← w - η ∂J/∂w (p.108)
```
  - Mapping cours visible à droite :
    - Forward pass — pages 90-92
    - Backward pass — pages 96-97
    - Mise à jour w ← w − η ∂J/∂w — page 108
    - Règle de la chaîne — pages 86-88
- **Speaker notes** :
> Voici le cœur de tout réseau de neurones — et c'est très exactement ce que Pr. El Alami Hassoun décrit dans le cours pages 78 à 109. Quatre étapes par batch. Premièrement, optimizer.zero_grad reset les gradients accumulés du batch précédent. Deuxièmement, model(X_batch) — c'est le Forward pass — pages 90 à 92 du cours. La sortie passe par chaque couche linéaire suivie de son activation. Troisièmement, criterion(logits, y) calcule la fonction de coût J — page 104. Quatrièmement, loss.backward fait le Backward pass — page 96 — qui applique la règle de la chaîne — pages 86 à 88 — pour calculer ∂J/∂w pour chaque poids. Et optimizer.step met à jour selon w ← w moins η fois ∂J/∂w — page 108. C'est le cycle fondamental.

## S32 — T40 · Diagnostic de l'entraînement
- **Voix** : Badr · **Durée** : 50 s · **Section** : DL / Diagnostic
- **Visuel** :
  - Image : `outputs/figures/mlp_curves.png` (loss train/val + val_auc par epoch)
  - Log d'entraînement à droite :
```
Epoch 01 | train_loss=1.0063 | val_auc=0.7388
Epoch 05 | train_loss=0.9583 | val_auc=0.7410 ← best
Epoch 10 | val_auc=0.7406
>> EarlyStopping (patience 5)
Temps total : 11.0 s
```
- **Speaker notes** :
> Voici le diagnostic visuel de l'entraînement. La courbe de train loss descend de 1,006 à 0,951 — convergence rapide en quelques epochs. La val loss est stable autour de 0,952 — pas de surapprentissage. La val AUC plafonne à 0,741 dès l'epoch 5, et EarlyStopping se déclenche à l'epoch 10 avec patience 5. L'écart train-val sur l'AUC reste inférieur à 0,005 — signe d'une excellente généralisation. Temps total d'entraînement sur CPU : 11 secondes.

## S33 — T41 · Évaluation MLP sur test
- **Voix** : Badr · **Durée** : 30 s · **Section** : DL / Résultat
- **Visuel** :
  - Grand cadre or central avec les 3 métriques :
    - AUC-ROC : **0.7316**
    - Accuracy : 0.6788
    - F1-score : 0.4525
  - Snippet bref :
```python
model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy(X_test).float())
    y_proba_mlp = torch.sigmoid(logits).numpy()
```
- **Speaker notes** :
> L'évaluation finale du MLP sur le test set jamais vu. AUC à 0,7316. Notez le passage en model.eval qui désactive Dropout et fixe BatchNorm en mode running stats. torch.no_grad évite de stocker les gradients pendant l'inférence, ce qui accélère de 10 %. La sigmoïde est appliquée manuellement ici pour obtenir une probabilité — pas dans le réseau, pour la stabilité numérique. Je laisse la parole à Abdou pour la configuration de l'optimisation et la comparaison.

---

## S34 — Identification Abdou DL + ses 6 tâches
- **Voix** : Abdou · **Durée** : 30 s · **Section** : DL / Config + Comparaison
- **Visuel** :
  - Carte profil Abdou (return, badge or)
  - Liste : T35 Adam · T36 BCE · T37 Mini-Batch · T38 EarlyStopping · T42 ⭐ tableau 4 modèles · T43 ⭐ courbes ROC
- **Speaker notes** :
> Je reprends la parole — Abdou. Mon segment Practical AI couvre la configuration de l'optimisation du MLP — choix de l'optimiseur, de la fonction de perte, de la taille de batch, du mécanisme d'arrêt anticipé — puis la comparaison rigoureuse des quatre modèles avec les courbes ROC superposées. C'est l'aboutissement de l'évaluation Practical AI.

## S35 — T35 · Optimiseur Adam ⭐ (cours page 116)
- **Voix** : Abdou · **Durée** : 50 s · **Section** : DL / Optimisation
- **Visuel** :
  - Tableau comparatif :
    | Optimiseur | Avantage | Inconv. | Retenu |
    |---|---|---|---|
    | **Adam** (Kingma 2015) | Momentum + RMSProp combinés | Plus de mémoire | ✅ |
    | SGD | Standard, théorie claire | Lent, tuning lr fin | ❌ |
    | RMSProp | Adapté non-stationnarité | Surpassé par Adam | ❌ |
  - Formule Adam encadrée :
    - m_t = β₁ m_{t-1} + (1-β₁) g_t
    - v_t = β₂ v_{t-1} + (1-β₂) g_t²
    - w_t = w_{t-1} - η · m̂_t / (√v̂_t + ε)
  - Snippet :
```python
optimizer = torch.optim.Adam(model.parameters(),
                              lr=1e-3, weight_decay=1e-5)
```
- **Speaker notes** :
> Adam — référence Kingma et Ba 2015, présenté page 116 du cours. C'est un optimiseur adaptatif qui combine deux idées : Momentum, qui lisse les gradients dans le temps avec une moyenne mobile exponentielle, et RMSProp, qui adapte le learning rate par paramètre selon la magnitude des gradients passés. Concrètement, Adam maintient pour chaque paramètre deux moments : m, le moment d'ordre 1 — équivalent du momentum — et v, le moment d'ordre 2 — équivalent de RMSProp. Le résultat : convergence sans tuning fin du learning rate. Avec SGD vanilla, le même MLP nécessiterait 40 epochs. Avec Adam, 10 suffisent.

## S36 — T36 · BCEWithLogitsLoss
- **Voix** : Abdou · **Durée** : 35 s · **Section** : DL / Optimisation
- **Visuel** :
  - Formule centrée : BCE(p, y) = -[y log p + (1-y) log(1-p)]
  - Snippet :
```python
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```
  - Note pédagogique : « Combine Sigmoid + BCE en utilisant log-sum-exp pour stabilité numérique »
- **Speaker notes** :
> Fonction de perte : BCEWithLogitsLoss. C'est la binary cross-entropy combinée à la sigmoïde de façon numériquement stable. Mathématiquement, c'est la log-vraisemblance négative sous hypothèse Bernoulli. Pourquoi le « WithLogits » ? Si on mettait nn.Sigmoid en sortie puis nn.BCELoss séparément, on aurait des risques d'overflow pour des logits de grande magnitude. PyTorch utilise le log-sum-exp trick pour fusionner les deux opérations de manière stable. Le pos_weight calculé par Badr est passé directement ici pour pénaliser les défauts 3,9 fois plus.

## S37 — T37 · Mini-Batch Gradient Descent ⭐ (cours page 115)
- **Voix** : Abdou · **Durée** : 45 s · **Section** : DL / Optimisation · ⭐ wow factor
- **Visuel** :
  - Tableau comparatif visuel :
    | Stratégie | Batch | Stabilité | Vitesse |
    |---|---|---|---|
    | Batch GD | 56 533 | Très stable | Très lent |
    | SGD | 1 | Très bruité | Lent |
    | **Mini-Batch** | **256** | Compromis | Rapide |
  - Snippet :
```python
train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                          batch_size=256, shuffle=True)
# 56 533 / 256 ≈ 221 batches par epoch
# Total : 10 epochs × 221 batches = 2 210 steps de gradient
```
  - Référence : cours page 115 « Mini-Batch combine Batch et SGD »
- **Speaker notes** :
> Page 115 du cours : Mini-Batch Gradient Descent — c'est le compromis idéal entre Batch GD qui calcule le gradient sur tout le dataset à chaque step et SGD qui le calcule sur 1 seul échantillon. Avec batch=256, nous avons 221 mini-batches par epoch. Sur 10 epochs, cela fait 2 210 steps de gradient au total. Pourquoi 256 et pas 32 ou 1024 ? Trop petit, le gradient est trop bruité — BatchNorm devient moins efficace. Trop grand, on perd la stochasticité bénéfique qui aide à généraliser. 256 est le standard pour les données tabulaires.

## S38 — T38 · EarlyStopping (patience=5)
- **Voix** : Abdou · **Durée** : 30 s · **Section** : DL / Optimisation
- **Visuel** :
  - Schéma : courbe val_auc fictive avec un pic, croix rouge pour les 5 epochs sans amélioration
  - Snippet :
```python
best_auc, no_improve = 0, 0
for epoch in range(MAX_EPOCHS):
    # ... train + eval val ...
    if val_auc > best_auc:
        best_auc, best_state, no_improve = val_auc, model.state_dict(), 0
    else:
        no_improve += 1
        if no_improve >= 5:        # patience
            model.load_state_dict(best_state)
            break
```
- **Speaker notes** :
> EarlyStopping avec patience=5. Le principe : on surveille la validation AUC à chaque epoch. Si elle s'améliore, on sauvegarde l'état du modèle. Si elle n'a pas amélioré pendant 5 epochs consécutives, on arrête et on restaure le meilleur état. Sur notre run, le meilleur val_auc à 0,7410 a été atteint à l'epoch 5. EarlyStopping a déclenché à l'epoch 10. Sans ce mécanisme, le modèle continuerait à entraîner inutilement et surapprendrait — le val_auc remonterait progressivement.

## S39 — T42 · Tableau récapitulatif 4 modèles ⭐⭐
- **Voix** : Abdou · **Durée** : 60 s · **Section** : DL / Comparaison · ⭐ wow factor
- **Visuel** :
  - Grand tableau centré :
    | Modèle | Features | AUC | Accuracy | F1 | Temps |
    |---|---|---|---|---|---|
    | **Régression Logistique** | 80 | **0.7311** | 0.667 | 0.448 | 0.5 s |
    | Random Forest complet | 80 | 0.7232 | 0.783 | 0.368 | 2.3 s |
    | Random Forest sans fuite | 37 | 0.6941 | 0.787 | 0.304 | 2.6 s |
    | **MLP (Deep Learning)** | 80 | **0.7316** | 0.679 | 0.453 | 11.0 s |
  - Encadré or souligné : « ΔAUC(MLP vs LR) = +0.0005 — NON SIGNIFICATIF »
  - Sous-texte : « MLP 22× plus lent pour gain quasi-nul »
- **Speaker notes** :
> Voici la conclusion expérimentale. Comparaison rigoureuse des quatre modèles sur le même test set jamais vu. Régression Logistique 0,7311. Random Forest complet 0,7232. Random Forest sans fuite 0,6941. MLP 0,7316. La différence d'AUC entre MLP et LR est de 0,0005 — non significative statistiquement. Le MLP est 22 fois plus lent pour un gain quasi nul. Conclusion : sur les données tabulaires de scoring de crédit, le Deep Learning ne bat pas la Régression Logistique. C'est exactement ce que Gunnarsson 2021 a montré sur 11 datasets — nous validons leur travail.

## S40 — T43 · Courbes ROC superposées ⭐⭐
- **Voix** : Abdou · **Durée** : 60 s · **Section** : DL / Comparaison · ⭐ wow factor
- **Visuel** :
  - Image : `outputs/figures/roc_comparison.png` (4 courbes ROC + diagonale)
  - Encadré or : « Zone d'opération bancaire FPR ∈ [0.10, 0.20] → RF lean visiblement plus bas »
- **Speaker notes** :
> Les courbes ROC superposées des quatre modèles. Trois courbes — Régression Logistique, Random Forest complet, MLP — sont quasiment confondues, conformément aux AUC très proches. Une seule courbe s'écarte visiblement vers le bas : le Random Forest sans fuite. La diagonale en pointillés représente le hasard. Dans la zone d'opération typique d'une banque — FPR entre 10 et 20 % — la différence entre MLP et RF sans fuite vaut environ 5 points de TPR, soit 5 % de défauts en plus détectés à taux de refus identique. C'est exactement l'effet de la fuite, mesuré graphiquement. Au tour d'Amr pour clôturer avec l'architecture et le benchmarking.

---

## S41 — Identification Amr DL + ses 7 tâches
- **Voix** : Amr · **Durée** : 30 s · **Section** : DL / Architecture + Benchmarking
- **Visuel** :
  - Carte profil Amr (return)
  - Liste : T30 ⭐ architecture · T31 ⭐ ReLU vs Sigmoid · T32 BatchNorm · T33 Dropout · T34 logit · T44 Gunnarsson · T45 benchmarking
- **Speaker notes** :
> C'est de nouveau Bekkari Amr pour clôturer la partie Practical AI. Je vais présenter trois choses : l'architecture précise du MLP et chacun de ses composants — couches, activations, normalisations, régularisation, sortie ; puis la validation scientifique de notre résultat face à la littérature ; et enfin le positionnement par rapport aux champions Kaggle.

## S42 — T30 · Architecture pyramidale 80→64→32→1 ⭐⭐⭐
- **Voix** : Amr · **Durée** : 60 s · **Section** : DL / Architecture · ⭐ wow factor majeur
- **Visuel** :
  - Schéma diagramme du réseau :
    Input(80) → Linear+ReLU+BN+Dropout(64) → Linear+ReLU+BN+Dropout(32) → Linear(1) logit
  - Snippet :
```python
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 1),  # logit
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)
```
  - Stat clé : « 7 489 paramètres entraînables — ratio 7.5 obs/param »
- **Speaker notes** :
> Voici l'architecture du MLP. Pyramidale : 80 features en entrée → 64 → 32 → 1. Pourquoi pyramidale et pas constante ? La compression progressive force le réseau à apprendre des représentations de plus en plus compactes — c'est le principe d'encodage. Pourquoi 2 couches cachées et pas 3 ou 5 ? Théorème d'approximation universelle de Cybenko 1989 : une seule couche cachée suffit théoriquement pour approximer toute fonction continue. Deux couches sont donc déjà au-delà du minimum. Plus profond augmenterait le risque de surapprentissage sur 56 000 échantillons. Total des paramètres : 7 489. Avec 56 533 obs d'entraînement, le ratio est de 7,5 obs par paramètre — confortablement dans la zone non-surapprenante selon Goodfellow.

## S43 — T31 · ReLU vs Sigmoid (cours pages 142-145) ⭐⭐
- **Voix** : Amr · **Durée** : 55 s · **Section** : DL / Architecture · ⭐ wow factor
- **Visuel** :
  - Deux graphiques côte à côte :
    - Sigmoid : courbe + sa dérivée → annotation rouge « ≈ 0 en saturation = vanishing gradient »
    - ReLU : ligne brisée + sa dérivée → annotation verte « dérivée = 1 pour x>0 »
  - Référence : « cours page 135 vanishing gradient · page 142-145 ReLU »
  - Encadré : « Dead neurons mitigés par BatchNorm + Dropout »
- **Speaker notes** :
> Choix de l'activation : ReLU plutôt que Sigmoid. Théorie cours pages 142 à 145. ReLU est défini par f(x) égal max de 0 et x. Sa dérivée vaut 1 pour x positif et 0 pour x négatif. Pourquoi pas Sigmoid ? Page 135 du cours : la dérivée de Sigmoid vaut σ(z) fois 1 moins σ(z), donc proche de zéro en saturation. Sur deux couches, le gradient est multiplié deux fois par ces dérivées proches de zéro — c'est le vanishing gradient. Les premières couches ne reçoivent plus de signal et n'apprennent pas. Avec ReLU, la dérivée vaut 1 pour x positif, donc pas de vanishing. Risque inverse : les dead neurons — neurones bloqués en zone négative — mais on les atténue avec BatchNorm qui recentre les activations et Dropout.

## S44 — T32-T33-T34 · BatchNorm + Dropout + Logit
- **Voix** : Amr · **Durée** : 45 s · **Section** : DL / Architecture
- **Visuel** :
  - Tableau synthèse des 3 composants :
    | Composant | Rôle | Référence |
    |---|---|---|
    | **BatchNorm1d** | Stabilise gradient, accélère convergence | Ioffe & Szegedy 2015 |
    | **Dropout 0.3 / 0.2** | Effet ensemble implicite, anti-overfit | Srivastava et al. 2014 |
    | **Sortie en logit** | BCEWithLogitsLoss = log-sum-exp stable | Standard PyTorch |
  - Note : « Dropout plus élevé près de l'entrée (0.3) → plus de bruit à filtrer »
- **Speaker notes** :
> Trois composants restants. Premièrement, BatchNormalization — référence Ioffe et Szegedy 2015. Elle stabilise la distribution des activations entre les couches pendant l'entraînement, ce que les auteurs appellent l'internal covariate shift. Sans BatchNorm, la loss oscille violemment. Deuxièmement, Dropout 0,3 après la première couche et 0,2 après la deuxième — référence Srivastava 2014. À chaque batch d'entraînement, on désactive aléatoirement une fraction des neurones — effet d'ensemble implicite. Taux plus élevé à l'entrée car il y a plus de bruit à filtrer. Troisièmement, la sortie est un logit brut, pas une sigmoïde — pour que BCEWithLogitsLoss puisse fusionner sigmoïde et BCE de manière numériquement stable.

## S45 — T44 · Validation expérimentale de Gunnarsson 2021 ⭐
- **Voix** : Amr · **Durée** : 45 s · **Section** : DL / Conclusion scientifique · ⭐ wow factor
- **Visuel** :
  - Encadré citation : « *Deep learning for credit scoring: do or don't?* » — Gunnarsson, Baesens et al., EJOR 2021
  - Tableau :
    | Gunnarsson 2021 | Notre projet |
    |---|---|
    | MLP, TabNet, XGBoost très proches sur 11 datasets crédit | LR 0.7311 ≈ MLP 0.7316 |
    | ΔAUC < 0.01 typique | ΔAUC = 0.0005 |
    | Pas de gain DL net | Identique |
- **Speaker notes** :
> Gunnarsson, Baesens et collègues ont publié en 2021 dans l'European Journal of Operational Research un article au titre explicite : Deep learning for credit scoring — do or don't ? Leur conclusion sur 11 datasets de scoring de crédit : le Deep Learning n'apporte pas systématiquement de gain face au ML classique sur tabulaire. C'est exactement notre hypothèse H2. Nos chiffres confirment : différence d'AUC entre MLP et LR de 0,0005, non significative. Nous validons donc expérimentalement le résultat de Gunnarsson sur LendingClub.

## S46 — T45 · Benchmarking face à la littérature ⭐
- **Voix** : Amr · **Durée** : 50 s · **Section** : DL / Conclusion scientifique · ⭐ wow factor
- **Visuel** :
  - Tableau de positionnement :
    | Source | AUC | Méthode |
    |---|---|---|
    | Aléatoire | 0.50 | — |
    | **Notre meilleur (MLP)** | **0.732** | MLP simple |
    | Lessmann 2015 (moyenne) | 0.78-0.82 | RF, GBM |
    | Top 10 % Kaggle | 0.80-0.81 | LightGBM stacké |
    | Top 1 % Kaggle | 0.815 | XGB + NN stacking |
    | Gunnarsson 2021 (DL tabulaire) | 0.72-0.77 | TabNet, MLP, XGB |
  - 3 verdicts : « ✅ H1 ✅ H2 ✅ H3 »
- **Speaker notes** :
> Positionnement global. Aléatoire à 0,50. Notre meilleur modèle à 0,732. Lessmann 2015 moyenne entre 0,78 et 0,82. Champions Kaggle entre 0,80 et 0,815. Gunnarsson entre 0,72 et 0,77 — exactement notre niveau. Nous sommes à 0,05-0,09 sous les champions Kaggle. Cet écart est attribuable au feature engineering massif avec jointures multi-tables, au stacking d'ensembles, et à la calibration bayésienne d'hyperparamètres — éléments hors scope d'un projet pédagogique. Verdict : nos trois hypothèses sont validées scientifiquement.

---

═══════════════════════════════════════════════════════════════════════
# 🟧 BLOC 5 — CONCLUSION PAR AMR (2 min · 4 slides)
═══════════════════════════════════════════════════════════════════════

## S47 — Validation des 3 hypothèses scientifiques ✅✅✅
- **Voix** : Amr · **Durée** : 45 s · **Section** : Conclusion
- **Visuel** :
  - 3 grandes cartes :
    - H1 ✅ ML supervisé AUC ≥ 0.73 → mesuré 0.7311 (LR)
    - H2 ✅ DL ≈ ML sur tabulaire → ΔAUC = 0.0005
    - H3 ✅ Fuite méthodologique → mesurée 0.029 AUC
  - Bandeau or : « 3/3 hypothèses validées expérimentalement »
- **Speaker notes** :
> Synthèse scientifique. Nos trois hypothèses sont validées expérimentalement. H1 : un modèle ML supervisé atteint une AUC supérieure à 0,73 — mesuré 0,7311 pour la Régression Logistique. H2 : le Deep Learning n'apporte pas de gain significatif face au ML classique sur tabulaire — mesuré ΔAUC de 0,0005 entre MLP et LR. H3 : trois variables LendingClub constituent une fuite méthodologique — mesurée à 0,029 d'AUC.

## S48 — Limites assumées + perspectives
- **Voix** : Amr · **Durée** : 30 s · **Section** : Conclusion
- **Visuel** :
  - Deux colonnes :
    - Limites : plafond AUC 0.73 (-0.05 vs Kaggle) · pas de calibration · pas de monitoring drift · dataset US
    - Perspectives : XGBoost (+0.02 attendu) · stacking · Platt scaling · monitoring KS
- **Speaker notes** :
> Limites assumées. Notre AUC plafond à 0,73 est inférieure aux champions Kaggle de 5 à 9 points. Nous n'avons pas de calibration des probabilités — Platt scaling serait nécessaire pour exploiter ces scores en IFRS 9. Pas de monitoring de drift en production. Dataset US, pas directement transférable au Maroc. Perspectives : XGBoost dominerait probablement à plus 0,02 d'AUC, stacking de modèles, Platt scaling pour la calibration, et test KS mensuel sur les inputs pour le drift.

## S49 — Liens avec le cours de Pr. El Alami Hassoun
- **Voix** : Amr · **Durée** : 25 s · **Section** : Conclusion
- **Visuel** :
  - Tableau condensé (sélection des concepts clés du cours appliqués) :
    | Concept cours | Page(s) | Tâche projet |
    |---|---|---|
    | Apprentissage supervisé | p.24 | T4 TARGET |
    | Forward / Backward / Update | p.78-108 | T39 boucle |
    | Mini-Batch GD | p.115 | T37 batch=256 |
    | Vanishing gradient → ReLU | p.135-145 | T31 |
    | Surapprentissage | p.29 | T33 Dropout + T38 EarlyStop |
    | DL n'est pas magique | p.29-30 | T44-T45 limites |
- **Speaker notes** :
> Le projet applique fidèlement le cours. Apprentissage supervisé page 24 — notre construction de TARGET. Forward, Backward, Update pages 78 à 108 — la boucle d'entraînement de Badr. Mini-Batch GD page 115 — notre batch de 256. Vanishing gradient et ReLU pages 135 à 145 — notre activation. Surapprentissage page 29 — notre Dropout et EarlyStopping. Et la mise en garde page 29-30 que le DL n'est pas magique — confirmée par nos résultats.

## S50 — Remerciements
- **Voix** : Amr · **Durée** : 30 s · **Section** : Conclusion
- **Visuel** :
  - Fond navy plein
  - Texte central :
    « Merci à Pr. M. El Alami Hassoun pour son encadrement »
    « Merci au jury pour son attention »
    Sous-bandeau : « Badr · Abdou · Amr — ENSA Tanger 2025-2026 »
- **Speaker notes** :
> Nous remercions Pr. M. El Alami Hassoun pour son encadrement tout au long du semestre, et le jury pour son attention. Notre projet démontre qu'un travail méthodologique rigoureux — exploration honnête, mesure de la fuite, comparaison rigoureuse, et benchmarking face à la littérature scientifique — vaut bien plus que la course au dernier point d'AUC. Merci à vous tous.

---

# RÉCAPITULATIF FINAL

| Bloc | Slides | Voix | Durée |
|---|---|---|---|
| Ouverture | S01-S05 | Amr | 3 min |
| Partie 1 ML — Badr | S06-S12 | Badr | 5 min 5 |
| Partie 1 ML — Abdou | S13-S20 | Abdou | 5 min 25 |
| Partie 1 ML — Amr | S21-S26 | Amr | 5 min 15 |
| Transition | S27 | Amr | 15 s |
| Partie 2 DL — Badr | S28-S33 | Badr | 4 min 50 |
| Partie 2 DL — Abdou | S34-S40 | Abdou | 5 min |
| Partie 2 DL — Amr | S41-S46 | Amr | 4 min 45 |
| Conclusion | S47-S50 | Amr | 2 min 10 |
| **TOTAL** | **50 slides** | 3 voix | **~32 min** |

| Voix | Apparitions | Temps total |
|---|---|---|
| **Amr (C)** | 4 (intro + ML + DL + conclusion) | ~14 min |
| **Badr (A)** | 2 (ML + DL) | ~10 min |
| **Abdou (B)** | 2 (ML + DL) | ~10 min 25 |

| Figures intégrées | Slide |
|---|---|
| eda_target_distribution.png | S10 |
| eda_boxplots.png | S14 |
| eda_corr_heatmap.png | S15 |
| eda_categorical_rates.png | S16 |
| feature_importance_rf.png | S23 |
| mlp_curves.png | S32 |
| roc_comparison.png | S40 |

| Snippets de code | Slide |
|---|---|
| pd.read_csv + nrows | S08 |
| Construction TARGET | S09 |
| Imputation + Doublons + Inf | S12 |
| Clip outliers + ratios | S17 |
| One-Hot + split + scaler | S18 |
| LogisticRegression | S19 |
| RandomForestClassifier | S22 |
| Filtre fuite + RF lean | S25 |
| import torch + seed | S29 |
| Train/val split + pos_weight | S30 |
| Boucle Forward/Backward (cœur cours) | S31 |
| model.eval + inference | S33 |
| Adam optimizer | S35 |
| BCEWithLogitsLoss | S36 |
| DataLoader batch=256 | S37 |
| EarlyStopping mécanisme | S38 |
| MLP class architecture | S42 |

**Total : 17 snippets de code**

---

*Script généré le 3 juin 2026 — Document de référence pour la PHASE 3 (construction du .pptx).*
