# Script à lire — Slide bonus « Interface bancaire ATLAS BANK »

> **Présentateur** : Bekkari Amr
> **Durée** : ~75 secondes
> **Position recommandée** : après S46 (benchmarking littérature), juste avant S47 (validation des hypothèses)
> **Fichier slide** : `LIVRABLE_FINAL/Slide_Bonus_Interface_Bancaire.pptx`

---

## 🎙️ Script complet (à lire à voix haute)

> *« C'est Bekkari Amr qui reprend la parole pour une démonstration de déployabilité. Nous avons aussi développé une preuve de concept — une interface web qui simule un véritable outil bancaire de scoring de crédit, baptisée **ATLAS BANK**. L'objectif est de montrer que notre modèle n'est pas seulement un notebook — il est déployable demain matin dans une agence.*
>
> *Voici un dossier réel. Une cliente nommée Sophie, 37 ans, salariée en CDI avec 10 ans et plus d'ancienneté, propriétaire avec un crédit immobilier. Elle demande **12 000 dollars sur 36 mois** pour consolider ses dettes. Revenu annuel **65 000 dollars**, taux d'endettement maîtrisé à **18 %**, utilisation crédit revolving à 50 %, aucun incident de paiement sur les deux dernières années.*
>
> *Le conseiller clique sur Analyser. Le pipeline serveur reproduit le preprocessing — imputation, ratios, one-hot, scaling — puis passe par notre **Random Forest sans fuite**. La probabilité de défaut calculée est de **18,2 %**. Le moteur de pricing transparent attribue le grade **B sous-grade B2**, soit un taux de **11,20 %** et une mensualité de **393 dollars**.*
>
> *Mais le vrai apport est dans le panneau du milieu : l'**explicabilité SHAP**. Le système identifie pour ce dossier précis les trois facteurs qui ont pesé sur la décision : le DTI à 18 % monte le risque de 0,022 points, l'utilisation revolving à 50 % de 0,015, et l'ancienneté de 10+ ans baisse le risque de 0,018. Ces chiffres ne sont pas inventés — ce sont les valeurs de Shapley calculées en temps réel sur ce dossier. C'est ce qui rend notre système **conforme à l'article 22 du RGPD** qui impose le droit à l'explication.*
>
> *Enfin, en bas, les **garde-fous métier** sont vérifiés : le ratio prêt sur revenu reste sous le plafond de 1,5×, le taux d'effort sous les 45 % du HCSF, et le DTI sous le plafond Bâle. Verdict : **ACCORDÉ**.*
>
> *C'est cette chaîne complète — modèle scientifique sans fuite, explicabilité réglementaire, garde-fous métier, pricing transparent — qui rend notre projet **déployable dans une vraie banque**, pas seulement présentable en soutenance. »*

---

## 📌 Version courte (45 sec) — si pression temps

> *« Voici une démonstration de la déployabilité de notre modèle. Nous avons développé ATLAS BANK, une preuve de concept d'outil bancaire. Une cliente demande 12 000 dollars sur 36 mois — revenu 65 000 dollars, DTI 18 %. Le serveur calcule en temps réel : grade B2, taux 11,20 %, mensualité 393 dollars, probabilité de défaut 18,2 %. L'explicabilité SHAP affiche les trois facteurs clés (DTI, utilisation revolving, ancienneté), conforme RGPD article 22. Les garde-fous métier (HCSF, Bâle) sont vérifiés. Verdict : ACCORDÉ. Notre modèle n'est pas qu'un notebook — il est prêt à entrer en agence. »*

---

## 🎯 Points clés à insister oralement

| Point | Pourquoi c'est important |
|---|---|
| **Random Forest sans fuite** | Cohérence avec la découverte méthodologique (slide 24-25) — on déploie le modèle réaliste |
| **18,2 % de proba** | Chiffre concret, mesurable, traçable |
| **Grade B2 + taux 11,20 %** | Montre que le pricing est transparent et auditable |
| **SHAP en temps réel** | Démonstration RGPD art. 22 — droit à l'explication |
| **Garde-fous HCSF/Bâle** | Conformité réglementaire concrète |
| **« déployable demain matin »** | Argument de clôture fort pour la note Practical AI |

---

## 🎬 Transitions

**Phrase d'entrée** (après slide 46 sur le benchmarking) :
> « Avant de conclure, je veux montrer une chose : notre projet n'est pas qu'une étude scientifique — il est déployable. »

**Phrase de sortie** (vers slide 47 conclusion) :
> « Voilà, notre projet est validé scientifiquement et déployable opérationnellement. Passons maintenant à la synthèse finale des 3 hypothèses. »

---

## 📂 Fichiers liés

- **Slide PowerPoint** : `LIVRABLE_FINAL/Slide_Bonus_Interface_Bancaire.pptx`
- **Script (ce document)** : `LIVRABLE_FINAL/Script_Slide_Bonus_Interface_Bancaire.md`

---

## ⚙️ Comment intégrer la slide dans le deck principal

**Option 1 — Présentation séparée** :
Ouvrir les 2 fichiers .pptx côte à côte, présenter la bonus juste avant le slide 47 du deck principal.

**Option 2 — Insertion dans le deck principal** :
1. Ouvrir `Presentation_Video_Projet_FINAL_v2.pptx` dans PowerPoint
2. Ouvrir `Slide_Bonus_Interface_Bancaire.pptx`
3. Clic droit sur la slide bonus → Copier
4. Dans le deck principal, clic droit entre les slides 46 et 47 → Coller (Conserver le format source)
5. Le deck passera à 51 slides au lieu de 50

---

*Document généré le 4 juin 2026 — ENSA Tanger — Slide bonus pour démontrer la déployabilité bancaire du projet.*
