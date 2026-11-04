"""
app.py — Simulateur de décision de crédit (FastAPI)
---------------------------------------------------
Interface "banquier" : le conseiller saisit les caractéristiques d'un client,
le modèle (Random Forest) prédit la probabilité de défaut, et l'application
JUSTIFIE la décision via SHAP (contribution de chaque caractéristique) + une
synthèse en langage clair (points forts / points d'attention).

Lancer avec :
    py -3.11 -m uvicorn app:app --reload --port 8000
puis ouvrir http://localhost:8000
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --------------------------------------------------------------------------
# Chargement du bundle
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = joblib.load(os.path.join(HERE, "models", "credit_pipeline.pkl"))

MODEL_RF = BUNDLE["model_rf"]
MODEL_LR = BUNDLE["model_lr"]
SCALER = BUNDLE["scaler"]
FEATURE_NAMES = BUNDLE["feature_names"]
# Modèle "sans fuite" (réaliste) — sert uniquement à la comparaison
MODEL_RF_LEAN = BUNDLE["model_rf_lean"]
SCALER_LEAN = BUNDLE["scaler_lean"]
FEATURE_NAMES_LEAN = BUNDLE["feature_names_lean"]
FEATURES_NUM = BUNDLE["features_num"]
FEATURES_CAT = BUNDLE["features_cat"]
CAT_VALUES = BUNDLE["cat_values"]
MEDIANS = BUNDLE["medians"]
CLIP_BOUNDS = BUNDLE["clip_bounds"]
REF_STATS = BUNDLE["ref_stats"]
METRICS = BUNDLE["metrics"]
DEFAULT_RATE = BUNDLE["default_rate"]

# Explainer SHAP (créé une fois au démarrage)
EXPLAINER = shap.TreeExplainer(MODEL_RF)

# Libellés lisibles (FR) pour l'affichage
LABELS = {
    "loan_amnt": "Montant du prêt",
    "int_rate": "Taux d'intérêt",
    "installment": "Mensualité",
    "annual_inc": "Revenu annuel",
    "dti": "Ratio dette/revenu (DTI)",
    "delinq_2yrs": "Retards de paiement (2 ans)",
    "open_acc": "Comptes de crédit ouverts",
    "pub_rec": "Incidents publics (faillites…)",
    "revol_bal": "Solde de crédit renouvelable",
    "revol_util": "Taux d'utilisation du crédit",
    "total_acc": "Nombre total de comptes",
    "term": "Durée du prêt",
    "grade": "Grade LendingClub",
    "sub_grade": "Sous-grade",
    "emp_length": "Ancienneté professionnelle",
    "home_ownership": "Statut de logement",
    "verification_status": "Vérification du revenu",
    "purpose": "Objet du prêt",
    "installment_income_ratio": "Charge mensuelle / revenu",
    "loan_income_ratio": "Montant prêt / revenu",
}

app = FastAPI(title="Simulateur de Crédit — ENSA Tanger")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def origin_variable(feature_name: str) -> str:
    """Retrouve la variable d'origine d'une colonne one-hot (ex: grade_B -> grade)."""
    # tester les catégorielles par longueur de préfixe décroissante (sub_grade avant grade)
    for cat in sorted(FEATURES_CAT, key=len, reverse=True):
        if feature_name.startswith(cat + "_"):
            return cat
    return feature_name


def build_feature_vector(raw: dict, feature_names=FEATURE_NAMES) -> pd.DataFrame:
    """Transforme une saisie brute en DataFrame d'1 ligne aligné sur `feature_names`.

    Par défaut : les 80 features du modèle complet. En passant FEATURE_NAMES_LEAN,
    on obtient le vecteur du modèle sans fuite (les colonnes de fuite absentes de la
    liste sont simplement ignorées par le reindex)."""
    df = pd.DataFrame([raw])

    # clip outliers (mêmes bornes qu'à l'entraînement)
    for col, ub in CLIP_BOUNDS.items():
        df[col] = df[col].clip(upper=ub)

    # ratios financiers
    df["installment_income_ratio"] = df["installment"] * 12 / df["annual_inc"]
    df["loan_income_ratio"] = df["loan_amnt"] / df["annual_inc"]
    df = df.replace([np.inf, -np.inf], np.nan)

    # imputation des éventuels NaN par les médianes d'entraînement
    for col, med in MEDIANS.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)

    # one-hot SANS drop_first, puis reindex sur les colonnes demandées
    df_enc = pd.get_dummies(df, columns=FEATURES_CAT)
    df_enc = df_enc.reindex(columns=feature_names, fill_value=0)
    return df_enc


def compute_shap(x_scaled: np.ndarray):
    """Retourne (base_value, shap_pour_classe_defaut[80])."""
    sv = EXPLAINER.shap_values(x_scaled)
    if isinstance(sv, list):                 # [classe0, classe1]
        shap_def = np.array(sv[1])[0]
    else:
        sv = np.array(sv)
        if sv.ndim == 3:                     # (1, 80, 2)
            shap_def = sv[0, :, 1]
        else:                                # (1, 80)
            shap_def = sv[0]

    ev = EXPLAINER.expected_value
    if isinstance(ev, (list, np.ndarray)) and np.ndim(ev) > 0:
        base = float(ev[1]) if len(np.atleast_1d(ev)) > 1 else float(np.atleast_1d(ev)[0])
    else:
        base = float(ev)
    return base, shap_def


def aggregate_contributions(shap_def: np.ndarray):
    """Agrège les valeurs SHAP par variable d'origine et trie par impact."""
    agg = {}
    for fname, val in zip(FEATURE_NAMES, shap_def):
        var = origin_variable(fname)
        agg[var] = agg.get(var, 0.0) + float(val)
    items = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)
    contribs = []
    for var, val in items:
        contribs.append({
            "variable": var,
            "label": LABELS.get(var, var),
            "value": val,                       # >0 augmente le risque, <0 le diminue
            "direction": "risque" if val > 0 else "favorable",
        })
    return contribs


def make_narrative(raw: dict, contribs: list):
    """Synthèse en langage clair : points forts (favorables) et points d'attention."""
    strengths, warnings = [], []

    def ref(col, key="median"):
        return REF_STATS.get(col, {}).get(key, None)

    # Règles métier lisibles, alignées sur l'EDA (int_rate, dti, revol_util, ratios...)
    if raw["int_rate"] >= (ref("int_rate", "q75") or 14):
        warnings.append(f"Taux d'intérêt élevé ({raw['int_rate']:.1f} %) — signe d'un profil jugé risqué par le marché.")
    elif raw["int_rate"] <= (ref("int_rate", "q25") or 8):
        strengths.append(f"Taux d'intérêt bas ({raw['int_rate']:.1f} %) — profil bien noté.")

    if raw["dti"] >= 30:
        warnings.append(f"Endettement élevé (DTI = {raw['dti']:.1f}) — proche ou au-delà du seuil prudentiel.")
    elif raw["dti"] <= (ref("dti", "q25") or 12):
        strengths.append(f"Faible endettement (DTI = {raw['dti']:.1f}).")

    if raw["revol_util"] >= 70:
        warnings.append(f"Forte utilisation du crédit renouvelable ({raw['revol_util']:.0f} %).")
    elif raw["revol_util"] <= 30:
        strengths.append(f"Utilisation maîtrisée du crédit renouvelable ({raw['revol_util']:.0f} %).")

    inst_ratio = raw["installment"] * 12 / raw["annual_inc"] if raw["annual_inc"] else 0
    if inst_ratio >= 0.15:
        warnings.append(f"Charge de remboursement lourde ({inst_ratio:.0%} du revenu annuel).")
    elif inst_ratio <= 0.05:
        strengths.append(f"Charge de remboursement légère ({inst_ratio:.0%} du revenu annuel).")

    loan_ratio = raw["loan_amnt"] / raw["annual_inc"] if raw["annual_inc"] else 0
    if loan_ratio >= 0.4:
        warnings.append(f"Montant emprunté important ({loan_ratio:.1f}× le revenu annuel).")

    if raw["annual_inc"] <= (ref("annual_inc", "q25") or 46000):
        warnings.append(f"Revenu annuel modeste ({raw['annual_inc']:,.0f} $).".replace(",", " "))
    elif raw["annual_inc"] >= (ref("annual_inc", "q75") or 92000):
        strengths.append(f"Revenu annuel confortable ({raw['annual_inc']:,.0f} $).".replace(",", " "))

    if raw["delinq_2yrs"] >= 1:
        warnings.append(f"{int(raw['delinq_2yrs'])} retard(s) de paiement sur les 2 dernières années.")
    if raw["pub_rec"] >= 1:
        warnings.append(f"{int(raw['pub_rec'])} incident(s) public(s) au dossier.")

    if raw["grade"] in ("A", "B"):
        strengths.append(f"Bon grade LendingClub ({raw['grade']}).")
    elif raw["grade"] in ("E", "F", "G"):
        warnings.append(f"Grade LendingClub faible ({raw['grade']}).")

    if str(raw["emp_length"]).startswith("10"):
        strengths.append("Ancienneté professionnelle élevée (10+ ans).")

    if not strengths:
        strengths.append("Aucun indicateur franchement favorable identifié.")
    if not warnings:
        warnings.append("Aucun point d'attention majeur identifié.")
    return strengths, warnings


def decision_from_proba(p_default: float):
    """Décision + style selon la probabilité de défaut (%)."""
    if p_default < 30:
        return "ACCORDÉ", "approved", "Risque faible — le crédit peut être accordé."
    elif p_default < 50:
        return "À ÉTUDIER", "warning", "Risque modéré — dossier à examiner au cas par cas."
    else:
        return "REFUSÉ", "rejected", "Risque élevé — le crédit est déconseillé."


def apply_policy(raw: dict):
    """Règles d'octroi métier (garde-fous d'affordabilité).

    Le score statistique ne suffit pas : un prêt peut être insoutenable même si le
    modèle le note "faible risque" (car les vraies données LendingClub ne contiennent
    pas de prêts manifestement non finançables). Ces règles dures, comme dans une vraie
    banque, déclenchent un refus automatique. Retourne la liste des motifs.
    """
    flags = []
    inc = raw["annual_inc"]
    if inc <= 0:
        return ["Revenu annuel nul ou non renseigné — dossier inéligible."]

    loan_inc = raw["loan_amnt"] / inc
    pay_inc = raw["installment"] * 12 / inc

    if loan_inc > 1.5:
        flags.append(f"Le montant emprunté représente {loan_inc:.1f}× le revenu annuel "
                     f"(plafond interne : 1,5×).")
    if pay_inc > 0.45:
        flags.append(f"Les remboursements représentent {pay_inc:.0%} du revenu annuel brut "
                     f"(plafond d'effort : 45%).")
    if raw["dti"] >= 45:
        flags.append(f"Taux d'endettement (DTI) de {raw['dti']:.0f} ≥ 45 "
                     f"(plafond réglementaire).")
    return flags


# --------------------------------------------------------------------------
# Moteur de tarification du risque (underwriting)
#
# Dans une vraie banque, le grade de risque et le taux d'intérêt ne sont PAS
# saisis par le conseiller : ce sont des SORTIES, attribuées par la banque à
# partir du profil de l'emprunteur. Le modèle gelé exige pourtant grade /
# sub_grade / int_rate / installment en entrée — ce moteur les dérive de façon
# transparente avant de nourrir le modèle.
# --------------------------------------------------------------------------
GRADE_BASE_RATE = {"A": 7.0, "B": 10.5, "C": 14.0, "D": 17.5,
                   "E": 21.0, "F": 24.5, "G": 28.0}
GRADE_BANDS = [(9, "A"), (22, "B"), (37, "C"), (52, "D"),
               (67, "E"), (82, "F"), (100, "G")]
GRADE_RANGE = {"A": (0, 9), "B": (10, 22), "C": (23, 37), "D": (38, 52),
               "E": (53, 67), "F": (68, 82), "G": (83, 100)}


def amortize(principal: float, annual_rate_pct: float, n_months: int) -> float:
    """Mensualité d'un prêt amortissable : M = P·i(1+i)^n / ((1+i)^n − 1)."""
    if n_months <= 0:
        return 0.0
    i = (annual_rate_pct / 100) / 12
    if i == 0:
        return principal / n_months
    f = (1 + i) ** n_months
    return principal * i * f / (f - 1)


def _term_months(term: str) -> int:
    """' 36 months' / ' 60 months' -> 36 / 60 (l'espace en tête est une quirk LendingClub)."""
    try:
        return int(str(term).strip().split()[0])
    except (ValueError, IndexError):
        return 36


def _emp_years(emp_length: str) -> float:
    s = str(emp_length).strip()
    if s.startswith("10"):
        return 10.0
    if s.startswith("< 1"):
        return 0.0
    try:
        return float(s.split()[0])
    except (ValueError, IndexError):
        return 0.0


def price_loan(annual_inc, emp_length, home_ownership, loan_amnt, term,
               dti, revol_util, delinq_2yrs, pub_rec):
    """Attribue grade / sous-grade / taux et calcule la mensualité à partir du profil.

    Retourne un dict : grade, sub_grade, int_rate, installment, risk_score, pricing_factors.
    Score = somme de points (positif = plus risqué), borné [0, 100].
    """
    factors = []

    def add(label, pts):
        if pts:
            factors.append({"label": label, "points": pts})

    # DTI
    if dti < 10:      add(f"DTI {dti:.0f} (faible)", 0)
    elif dti < 20:    add(f"DTI {dti:.0f}", 5)
    elif dti < 30:    add(f"DTI {dti:.0f}", 12)
    elif dti < 40:    add(f"DTI {dti:.0f} (élevé)", 22)
    else:             add(f"DTI {dti:.0f} (très élevé)", 32)

    # Taux d'utilisation du crédit renouvelable
    if revol_util < 30:    pass
    elif revol_util < 50:  add(f"Utilisation crédit {revol_util:.0f}%", 4)
    elif revol_util < 70:  add(f"Utilisation crédit {revol_util:.0f}%", 10)
    elif revol_util < 90:  add(f"Utilisation crédit {revol_util:.0f}% (élevée)", 18)
    else:                  add(f"Utilisation crédit {revol_util:.0f}% (saturée)", 26)

    # Montant emprunté rapporté au revenu
    loan_inc = loan_amnt / max(annual_inc, 1)
    if loan_inc < 0.15:    pass
    elif loan_inc < 0.30:  add(f"Prêt = {loan_inc:.1f}× le revenu", 5)
    elif loan_inc < 0.50:  add(f"Prêt = {loan_inc:.1f}× le revenu", 12)
    elif loan_inc < 1.0:   add(f"Prêt = {loan_inc:.1f}× le revenu (élevé)", 22)
    else:                  add(f"Prêt = {loan_inc:.1f}× le revenu (très élevé)", 34)

    # Revenu annuel
    if annual_inc >= 120000:   add(f"Revenu confortable ({annual_inc:,.0f} $)", -8)
    elif annual_inc >= 80000:  add(f"Revenu solide ({annual_inc:,.0f} $)", -4)
    elif annual_inc >= 50000:  pass
    elif annual_inc >= 30000:  add(f"Revenu modeste ({annual_inc:,.0f} $)", 6)
    else:                      add(f"Revenu faible ({annual_inc:,.0f} $)", 14)

    # Incidents
    if delinq_2yrs:
        add(f"{int(delinq_2yrs)} retard(s) de paiement (2 ans)", min(int(delinq_2yrs) * 9, 27))
    if pub_rec:
        add(f"{int(pub_rec)} incident(s) public(s)", min(int(pub_rec) * 12, 24))

    # Ancienneté professionnelle
    yrs = _emp_years(emp_length)
    if yrs >= 10:   add("Ancienneté 10+ ans", -5)
    elif yrs >= 5:  add(f"Ancienneté {yrs:.0f} ans", -2)
    elif yrs >= 2:  pass
    else:           add("Faible ancienneté professionnelle", 5)

    # Logement
    ho = str(home_ownership).upper()
    if ho in ("MORTGAGE", "OWN"):  add("Propriétaire / crédit immobilier", -3)
    elif ho == "RENT":             add("Locataire", 3)
    else:                          add("Logement non standard", 5)

    # Durée
    n = _term_months(term)
    if n >= 60:
        add("Durée 60 mois", 6)

    score = max(0, min(100, sum(f["points"] for f in factors)))

    # Score -> grade
    grade = "G"
    for hi, g in GRADE_BANDS:
        if score <= hi:
            grade = g
            break

    # Sous-grade : position dans la tranche du grade, en 5 paliers
    lo, hi = GRADE_RANGE[grade]
    span = max(hi - lo, 1)
    idx = int((score - lo) / span * 5)
    idx = max(0, min(4, idx))
    sub_grade = f"{grade}{idx + 1}"

    # Grade + sous-grade -> taux
    int_rate = round(GRADE_BASE_RATE[grade] + idx * 0.7, 2)

    # Mensualité
    installment = round(amortize(loan_amnt, int_rate, n), 2)

    factors.sort(key=lambda f: abs(f["points"]), reverse=True)
    return {
        "grade": grade, "sub_grade": sub_grade, "int_rate": int_rate,
        "installment": installment, "risk_score": score, "pricing_factors": factors,
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cat_values": CAT_VALUES,
        "metrics": METRICS,
        "default_rate": DEFAULT_RATE,
        "auc_lean": METRICS["rf_lean"]["auc"],
    })


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    # --- Demande de prêt ---
    loan_amnt: float = Form(...),
    term: str = Form(...),
    purpose: str = Form(...),
    # --- Identité & revenus ---
    annual_inc: float = Form(...),
    emp_length: str = Form(...),
    home_ownership: str = Form(...),
    verification_status: str = Form(...),
    # --- Rapport bureau de crédit ---
    dti: float = Form(...),
    revol_util: float = Form(...),
    revol_bal: float = Form(...),
    open_acc: float = Form(...),
    total_acc: float = Form(...),
    delinq_2yrs: float = Form(...),
    pub_rec: float = Form(...),
):
    # 0. Tarification du risque : la banque attribue grade / sous-grade / taux
    #    et calcule la mensualité (sorties, pas des saisies du conseiller).
    offer = price_loan(
        annual_inc=annual_inc, emp_length=emp_length, home_ownership=home_ownership,
        loan_amnt=loan_amnt, term=term, dti=dti, revol_util=revol_util,
        delinq_2yrs=delinq_2yrs, pub_rec=pub_rec,
    )

    raw = {
        "loan_amnt": loan_amnt,
        "int_rate": offer["int_rate"],          # dérivé
        "installment": offer["installment"],    # dérivé
        "annual_inc": annual_inc, "dti": dti, "delinq_2yrs": delinq_2yrs,
        "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal,
        "revol_util": revol_util, "total_acc": total_acc, "term": term,
        "grade": offer["grade"],                # dérivé
        "sub_grade": offer["sub_grade"],        # dérivé
        "emp_length": emp_length,
        "home_ownership": home_ownership, "verification_status": verification_status,
        "purpose": purpose,
    }

    # 1. Reconstruction + scaling
    df_enc = build_feature_vector(raw)
    x_scaled = SCALER.transform(df_enc)

    # 2. Prédictions
    p_rf = float(MODEL_RF.predict_proba(x_scaled)[0, 1]) * 100
    p_lr = float(MODEL_LR.predict_proba(x_scaled)[0, 1]) * 100

    # Modèle sans fuite (comparaison seulement) : même saisie, colonnes restreintes
    x_lean = SCALER_LEAN.transform(build_feature_vector(raw, FEATURE_NAMES_LEAN))
    p_lean = float(MODEL_RF_LEAN.predict_proba(x_lean)[0, 1]) * 100

    decision, style, message = decision_from_proba(p_rf)

    # 2b. Règles d'octroi (garde-fous) — peuvent forcer un refus
    policy_flags = apply_policy(raw)
    if policy_flags:
        decision, style, message = (
            "REFUSÉ", "rejected",
            "Refus automatique — une ou plusieurs règles d'octroi ne sont pas respectées.",
        )

    # 3. SHAP + contributions
    _, shap_def = compute_shap(x_scaled)
    contribs = aggregate_contributions(shap_def)
    top_contribs = contribs[:8]
    max_abs = max((abs(c["value"]) for c in top_contribs), default=1e-9)
    for c in top_contribs:
        c["width"] = int(abs(c["value"]) / max_abs * 100)  # largeur barre en %

    # 4. Synthèse langage clair
    strengths, warnings = make_narrative(raw, contribs)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "raw": raw,
        "p_default": round(p_rf, 1),
        "p_repay": round(100 - p_rf, 1),
        "p_lr": round(p_lr, 1),
        "p_lean": round(p_lean, 1),
        "auc_full": METRICS["rf"]["auc"],
        "auc_lean": METRICS["rf_lean"]["auc"],
        "decision": decision,
        "style": style,
        "message": message,
        "policy_flags": policy_flags,
        "offer": offer,
        "pricing_factors": offer["pricing_factors"],
        "top_contribs": top_contribs,
        "strengths": strengths,
        "warnings": warnings,
        "metrics": METRICS,
    })
