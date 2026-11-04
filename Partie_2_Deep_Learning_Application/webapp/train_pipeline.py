"""
train_pipeline.py
-----------------
Ré-entraîne les modèles (Logistic Regression + Random Forest) sur le dataset
LendingClub en reproduisant EXACTEMENT le preprocessing du notebook 01_eda_lendingclub2,
puis sauvegarde un bundle complet pour l'application web (FastAPI).

Le bundle contient tout ce qu'il faut pour transformer une saisie brute du banquier
en les 80 features attendues par le modèle :
    - modèles entraînés (rf, lr)
    - scaler
    - liste ordonnée des features one-hot (feature_names)
    - listes des colonnes numériques / catégorielles
    - modalités possibles de chaque catégorielle (pour le formulaire)
    - médianes (imputation des champs vides)
    - bornes de clip (annual_inc, revol_bal)

Lancer avec :  py -3.11 train_pipeline.py
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------
CSV_PATH = r"C:\Users\USER\OneDrive\Pièces jointes\lablundi\mlmd\accepted_2007_to_2018Q4.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(HERE, "models", "credit_pipeline.pkl")

# --------------------------------------------------------------------------
# Features (identiques au notebook EDA)
# --------------------------------------------------------------------------
FEATURES_NUM = [
    "loan_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc",
]
FEATURES_CAT = [
    "term", "grade", "sub_grade", "emp_length",
    "home_ownership", "verification_status", "purpose",
]
DEFAULT_STATUSES = ["Charged Off", "Default", "Late (31-120 days)"]
PAID_STATUSES = ["Fully Paid"]
CLIP_COLS = ["annual_inc", "revol_bal"]


def main():
    print("=" * 60)
    print("ENTRAÎNEMENT DU PIPELINE COMPLET (pour l'app web)")
    print("=" * 60)

    # 1. Chargement -------------------------------------------------------
    print("\n[1/7] Chargement du CSV (nrows=100000)...")
    df = pd.read_csv(CSV_PATH, nrows=100000, low_memory=False)
    print(f"  -> {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # 2. Filtrage + TARGET -----------------------------------------------
    print("\n[2/7] Filtrage des prêts terminés + création TARGET...")
    df = df[df["loan_status"].isin(DEFAULT_STATUSES + PAID_STATUSES)].copy()
    df["TARGET"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    print(f"  -> {df.shape[0]} lignes | taux de défaut : {df['TARGET'].mean():.2%}")

    # 3. Sélection des features ------------------------------------------
    df_model = df[FEATURES_NUM + FEATURES_CAT + ["TARGET"]].copy()

    # 4. Imputation -------------------------------------------------------
    print("\n[3/7] Imputation (médiane / mode)...")
    medians = {}
    for col in FEATURES_NUM:
        medians[col] = float(df_model[col].median())
        df_model[col] = df_model[col].fillna(medians[col])
    modes = {}
    for col in FEATURES_CAT:
        modes[col] = df_model[col].mode()[0]
        df_model[col] = df_model[col].fillna(modes[col])

    # Modalités possibles (pour le formulaire) AVANT encodage
    cat_values = {col: sorted(df_model[col].dropna().unique().tolist())
                  for col in FEATURES_CAT}

    # 5. Doublons + clip outliers ----------------------------------------
    df_model = df_model.drop_duplicates().reset_index(drop=True)
    clip_bounds = {}
    for col in CLIP_COLS:
        p99 = float(df_model[col].quantile(0.99))
        clip_bounds[col] = p99
        df_model[col] = df_model[col].clip(upper=p99)
    print(f"  -> bornes de clip : {clip_bounds}")

    # 6. Feature engineering (ratios) ------------------------------------
    print("\n[4/7] Feature engineering (ratios financiers)...")
    df_model["installment_income_ratio"] = df_model["installment"] * 12 / df_model["annual_inc"]
    df_model["loan_income_ratio"] = df_model["loan_amnt"] / df_model["annual_inc"]
    df_model = df_model.replace([np.inf, -np.inf], np.nan)
    df_model = df_model.fillna(df_model.median(numeric_only=True))

    # médianes des ratios (pour imputation côté app)
    medians["installment_income_ratio"] = float(df_model["installment_income_ratio"].median())
    medians["loan_income_ratio"] = float(df_model["loan_income_ratio"].median())

    # 7. One-hot + split + scaling ---------------------------------------
    print("\n[5/7] Encodage One-Hot + split + normalisation...")
    df_encoded = pd.get_dummies(df_model, columns=FEATURES_CAT, drop_first=True)
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    X = df_encoded.drop("TARGET", axis=1)
    y = df_encoded["TARGET"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"  -> {len(feature_names)} features | train {X_train_s.shape}")

    # 8. Entraînement -----------------------------------------------------
    print("\n[6/7] Entraînement des modèles...")
    t0 = time.time()
    model_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model_lr.fit(X_train_s, y_train)
    auc_lr = roc_auc_score(y_test, model_lr.predict_proba(X_test_s)[:, 1])
    print(f"  LogReg       -> AUC {auc_lr:.4f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    model_rf = RandomForestClassifier(
        n_estimators=100, min_samples_split=10,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    model_rf.fit(X_train_s, y_train)
    proba_rf = model_rf.predict_proba(X_test_s)[:, 1]
    auc_rf = roc_auc_score(y_test, proba_rf)
    acc_rf = accuracy_score(y_test, model_rf.predict(X_test_s))
    f1_rf = f1_score(y_test, model_rf.predict(X_test_s))
    print(f"  RandomForest -> AUC {auc_rf:.4f} ({time.time()-t0:.1f}s)")

    # 8b. Modèle "sans fuite" (réaliste) ---------------------------------
    # On retire les variables qui sont la décision de risque de LendingClub
    # (grade, sub_grade, int_rate) + ce qui en dérive (installment et son ratio).
    # Entraîné sur le MÊME split, restreint aux colonnes non-fuite, pour une
    # comparaison équitable de l'AUC avec / sans fuite.
    LEAK_NUM = {"int_rate", "installment", "installment_income_ratio"}

    def _is_leak(col):
        return col in LEAK_NUM or col.startswith("grade_") or col.startswith("sub_grade_")

    lean_cols = [c for c in feature_names if not _is_leak(c)]
    scaler_lean = StandardScaler()
    Xtr_lean = scaler_lean.fit_transform(X_train[lean_cols])
    Xte_lean = scaler_lean.transform(X_test[lean_cols])

    t0 = time.time()
    model_rf_lean = RandomForestClassifier(
        n_estimators=100, min_samples_split=10,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    model_rf_lean.fit(Xtr_lean, y_train)
    auc_rf_lean = roc_auc_score(y_test, model_rf_lean.predict_proba(Xte_lean)[:, 1])
    print(f"  RandomForest SANS FUITE -> AUC {auc_rf_lean:.4f} "
          f"({len(lean_cols)} features, {time.time()-t0:.1f}s)")

    # 9. Sauvegarde du bundle --------------------------------------------
    print("\n[7/7] Sauvegarde du bundle...")
    bundle = {
        "model_rf": model_rf,
        "model_lr": model_lr,
        "scaler": scaler,
        "feature_names": feature_names,
        # modèle "sans fuite" (réaliste) — comparaison
        "model_rf_lean": model_rf_lean,
        "scaler_lean": scaler_lean,
        "feature_names_lean": lean_cols,
        "features_num": FEATURES_NUM,
        "features_cat": FEATURES_CAT,
        "cat_values": cat_values,
        "medians": medians,
        "modes": modes,
        "clip_bounds": clip_bounds,
        "metrics": {
            "rf": {"auc": auc_rf, "accuracy": acc_rf, "f1": f1_rf},
            "rf_lean": {"auc": auc_rf_lean},
            "lr": {"auc": auc_lr},
        },
        "default_rate": float(y.mean()),
        # médiane de référence des features brutes (pour comparer un client à la "norme")
        "ref_stats": {col: {
            "median": float(df_model[col].median()),
            "q25": float(df_model[col].quantile(0.25)),
            "q75": float(df_model[col].quantile(0.75)),
        } for col in FEATURES_NUM + ["installment_income_ratio", "loan_income_ratio"]},
    }
    os.makedirs(os.path.dirname(BUNDLE_PATH), exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"  -> bundle sauvegardé : {BUNDLE_PATH}")
    print(f"\nOK. RF complet AUC={auc_rf:.4f} ({len(feature_names)} feat.) | "
          f"RF sans fuite AUC={auc_rf_lean:.4f} ({len(lean_cols)} feat.) | "
          f"taux défaut réf={y.mean():.2%}")


if __name__ == "__main__":
    main()
