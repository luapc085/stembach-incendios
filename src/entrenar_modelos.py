#!/usr/bin/env python3
"""
Paso 12 — Entrenamiento y evaluacion de modelos de clasificacion.

Entrena modelos de forma incremental (baseline primero, luego mas complejos).
Cada modelo se evalua con F1, Recall, Precision, PR-AUC y umbral optimizado.

Uso: python3 src/entrenar_modelos.py
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(__file__))
from config import PROCESSED_DIR, METEO_VARIABLES

# --- PATHS ---
INPUT_CSV = os.path.join(PROCESSED_DIR, "dataset_modelo.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultados")

# --- TEMPORAL SPLIT ---
# Train: 2005-2018 (14 years), Test: 2019-2021 (3 years)
# Temporal split avoids data leakage from future to past
TEST_YEAR_START = 2019

# --- FEATURES ---
DERIVED_FEATURES = [
    "dias_sin_lluvia", "PP_acum_7d", "PP_acum_14d", "PP_acum_30d",
    "TA_AVG_3d", "TA_MAX_3d", "HR_MIN_3d", "amplitud_termica",
    "mes", "dia_del_ano",
]
ALL_FEATURES = METEO_VARIABLES + DERIVED_FEATURES


def load_and_split(path):
    """Load dataset and split temporally into train/test."""
    df = pd.read_csv(path, parse_dates=["fecha"])

    train = df[df["fecha"].dt.year < TEST_YEAR_START]
    test = df[df["fecha"].dt.year >= TEST_YEAR_START]

    X_train = train[ALL_FEATURES]
    y_train = train["incendio"]
    X_test = test[ALL_FEATURES]
    y_test = test["incendio"]

    return X_train, y_train, X_test, y_test


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TunedThresholdClassifierCV
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        average_precision_score, confusion_matrix,
        precision_recall_curve,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("PASO 12.1 — LOGISTIC REGRESSION (BASELINE)")
    print("=" * 70)

    # --- LOAD & SPLIT ---
    X_train, y_train, X_test, y_test = load_and_split(INPUT_CSV)

    print(f"\n--- SPLIT TEMPORAL (train < {TEST_YEAR_START}, test >= {TEST_YEAR_START}) ---")
    for name, y in [("Train", y_train), ("Test", y_test)]:
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        print(f"  {name}: {len(y):,} filas | "
              f"{n_pos:,} positivos ({n_pos/len(y)*100:.1f}%) | "
              f"{n_neg:,} negativos | "
              f"ratio {n_neg/n_pos:.1f}:1")

    train_nans = X_train.isna().sum().sum()
    test_nans = X_test.isna().sum().sum()
    print(f"\n  NaN en features: train={train_nans:,}, test={test_nans:,}")
    print(f"  Estrategia: SimpleImputer(strategy='median') en Pipeline")

    # --- MODEL ---
    print(f"\n--- MODELO: LOGISTIC REGRESSION ---")
    print(f"  class_weight='balanced' (compensa desbalanceo automaticamente)")
    print(f"  solver='lbfgs', max_iter=1000")

    from sklearn.preprocessing import StandardScaler

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)
    print(f"  Entrenamiento completado.")

    # --- EVALUATE BASE (threshold=0.5) ---
    y_pred_base = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1_base = f1_score(y_test, y_pred_base)
    recall_base = recall_score(y_test, y_pred_base)
    precision_base = precision_score(y_test, y_pred_base, zero_division=0)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\n--- RESULTADOS CON UMBRAL POR DEFECTO (0.5) ---")
    print(f"  F1-score:  {f1_base:.4f}")
    print(f"  Recall:    {recall_base:.4f}")
    print(f"  Precision: {precision_base:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred_base)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Matriz de confusion:")
    print(f"    Verdaderos negativos (dias sin incendio bien clasificados):  {tn:,}")
    print(f"    Falsos positivos (falsas alarmas):                          {fp:,}")
    print(f"    Falsos negativos (incendios NO detectados):                 {fn:,}")
    print(f"    Verdaderos positivos (incendios detectados):                {tp:,}")

    # --- THRESHOLD TUNING ---
    print(f"\n--- OPTIMIZACION DEL UMBRAL ---")
    print(f"  TunedThresholdClassifierCV(scoring='f1', cv=5)")
    tuned = TunedThresholdClassifierCV(model, scoring="f1", cv=5)
    tuned.fit(X_train, y_train)
    best_threshold = tuned.best_threshold_

    y_pred_tuned = (y_proba >= best_threshold).astype(int)
    f1_tuned = f1_score(y_test, y_pred_tuned)
    recall_tuned = recall_score(y_test, y_pred_tuned)
    precision_tuned = precision_score(y_test, y_pred_tuned, zero_division=0)

    print(f"\n--- RESULTADOS CON UMBRAL OPTIMIZADO ({best_threshold:.3f}) ---")
    print(f"  F1-score:  {f1_tuned:.4f}")
    print(f"  Recall:    {recall_tuned:.4f}")
    print(f"  Precision: {precision_tuned:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f} (no cambia, depende de probabilidades)")

    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    tn2, fp2, fn2, tp2 = cm_tuned.ravel()
    print(f"\n  Matriz de confusion:")
    print(f"    Verdaderos negativos:  {tn2:,}")
    print(f"    Falsos positivos:      {fp2:,}")
    print(f"    Falsos negativos:      {fn2:,}")
    print(f"    Verdaderos positivos:  {tp2:,}")

    # --- COEFFICIENTS (what LR learned) ---
    print(f"\n--- COEFICIENTES DEL MODELO (peso de cada variable) ---")
    coefs = model.named_steps["clf"].coef_[0]
    indices = np.argsort(np.abs(coefs))[::-1]
    for rank, idx in enumerate(indices, 1):
        sign = "+" if coefs[idx] > 0 else "-"
        print(f"  {rank:>2}. {ALL_FEATURES[idx]:<20} {sign} {abs(coefs[idx]):.4f}")

    # --- PLOT: PR curve ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_curve, prec_curve, color="#4e79a7",
            label=f"Logistic Regression (PR-AUC={pr_auc:.3f})")
    ax.axhline(y=(y_test == 1).sum() / len(y_test), color="gray",
               linestyle="--", label="Clasificador aleatorio")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall — Logistic Regression (baseline)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pr_curve_logistic_regression.png"), dpi=150)
    plt.close(fig)
    print(f"\n  Guardado: resultados/pr_curve_logistic_regression.png")

    # --- SAVE METRICS for downstream comparison (entrenar_random_forest.py) ---
    metrics = {
        "model": "logistic_regression",
        "test_year_start": TEST_YEAR_START,
        "test_positive_rate": float((y_test == 1).sum() / len(y_test)),
        "pr_auc": float(pr_auc),
        "f1_base": float(f1_base),
        "recall_base": float(recall_base),
        "precision_base": float(precision_base),
        "f1_tuned": float(f1_tuned),
        "recall_tuned": float(recall_tuned),
        "precision_tuned": float(precision_tuned),
        "best_threshold": float(best_threshold),
        "pr_curve": {
            "recall": rec_curve.tolist(),
            "precision": prec_curve.tolist(),
        },
    }
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_logistic_regression.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Guardado: resultados/metrics_logistic_regression.json (para comparativa con RF)")

    # --- SUMMARY ---
    print(f"\n{'='*70}")
    print("RESUMEN — LOGISTIC REGRESSION (BASELINE)")
    print(f"{'='*70}")
    print(f"  PR-AUC:              {pr_auc:.4f}")
    print(f"  F1 (umbral 0.5):     {f1_base:.4f}")
    print(f"  F1 (umbral {best_threshold:.3f}):   {f1_tuned:.4f}")
    print(f"  Recall:              {recall_tuned:.4f}")
    print(f"  Precision:           {precision_tuned:.4f}")
    print(f"  Umbral optimo:       {best_threshold:.3f}")
    print(f"\n  Siguiente paso: entrenar Random Forest y comparar.")


if __name__ == "__main__":
    main()
