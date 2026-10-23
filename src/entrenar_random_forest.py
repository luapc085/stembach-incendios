#!/usr/bin/env python3
"""
Paso 14 — Entrenamiento de Random Forest.

Entrena Random Forest sobre dataset_modelo.csv, compara con el baseline
de Logistic Regression, y genera graficos de feature importance.

Uso: python3 src/entrenar_random_forest.py
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
LR_METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics_logistic_regression.json")


def load_lr_metrics():
    """Load LR baseline metrics. Fail clearly if missing."""
    if not os.path.exists(LR_METRICS_JSON):
        raise FileNotFoundError(
            f"No se encontro {LR_METRICS_JSON}. "
            f"Ejecuta primero: python3 src/entrenar_modelos.py"
        )
    with open(LR_METRICS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

# --- TEMPORAL SPLIT ---
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
    return (
        train[ALL_FEATURES], train["incendio"],
        test[ALL_FEATURES], test["incendio"],
    )


def main():
    from sklearn.ensemble import RandomForestClassifier
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
    print("PASO 14 — RANDOM FOREST")
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

    # --- MODEL ---
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\n--- MODELO: RANDOM FOREST ---")
    print(f"  n_estimators=300 (300 arboles)")
    print(f"  max_depth=15 (profundidad maxima por arbol)")
    print(f"  min_samples_leaf=10 (minimo 10 muestras por hoja)")
    print(f"  class_weight='balanced' (compensa desbalanceo {ratio:.0f}:1)")
    print(f"  Nota: no necesita StandardScaler (arboles son invariantes a escala)")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=15,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print(f"\n  Entrenando...")
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
    print(f"    Verdaderos negativos:  {tn:,}")
    print(f"    Falsos positivos:      {fp:,}")
    print(f"    Falsos negativos:      {fn:,}")
    print(f"    Verdaderos positivos:  {tp:,}")

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
    print(f"  PR-AUC:    {pr_auc:.4f}")

    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    tn2, fp2, fn2, tp2 = cm_tuned.ravel()
    print(f"\n  Matriz de confusion:")
    print(f"    Verdaderos negativos:  {tn2:,}")
    print(f"    Falsos positivos:      {fp2:,}")
    print(f"    Falsos negativos:      {fn2:,}")
    print(f"    Verdaderos positivos:  {tp2:,}")

    # --- FEATURE IMPORTANCE ---
    print(f"\n--- FEATURE IMPORTANCE (top 22) ---")
    importances = model.named_steps["clf"].feature_importances_
    indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices, 1):
        bar = "#" * int(importances[idx] * 100)
        print(f"  {rank:>2}. {ALL_FEATURES[idx]:<20} {importances[idx]:.4f}  {bar}")

    # --- COMPARISON WITH BASELINE ---
    print(f"\n--- COMPARACION CON BASELINE (Logistic Regression) ---")
    print(f"  {'Metrica':<25} {'LogReg':>10} {'RandomForest':>14} {'Mejora':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*14} {'-'*10}")

    # Carga las metricas reales de LR escritas por entrenar_modelos.py
    lr_metrics = load_lr_metrics()
    lr_pr_auc = lr_metrics["pr_auc"]
    lr_f1 = lr_metrics["f1_tuned"]
    lr_recall = lr_metrics["recall_tuned"]
    lr_precision = lr_metrics["precision_tuned"]

    for metric, lr_val, rf_val in [
        ("PR-AUC", lr_pr_auc, pr_auc),
        ("F1 (tuned)", lr_f1, f1_tuned),
        ("Recall (tuned)", lr_recall, recall_tuned),
        ("Precision (tuned)", lr_precision, precision_tuned),
    ]:
        if lr_val > 0:
            mejora = (rf_val - lr_val) / lr_val * 100
            print(f"  {metric:<25} {lr_val:>10.4f} {rf_val:>14.4f} {mejora:>+9.1f}%")
        else:
            print(f"  {metric:<25} {lr_val:>10.4f} {rf_val:>14.4f}")

    # --- PLOTS ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Plot 1: Feature importance
    fig, ax = plt.subplots(figsize=(10, 7))
    sorted_idx = np.argsort(importances)
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="#59a14f")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([ALL_FEATURES[i] for i in sorted_idx])
    ax.set_xlabel("Importancia")
    ax.set_title("Feature Importance — Random Forest")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance_random_forest.png"), dpi=150)
    plt.close(fig)
    print(f"\n  Guardado: resultados/feature_importance_random_forest.png")

    # Plot 2: PR curve REAL de RF vs PR curve REAL de LR (cargada del JSON)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    lr_rec = lr_metrics["pr_curve"]["recall"]
    lr_prec = lr_metrics["pr_curve"]["precision"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_curve, prec_curve, color="#59a14f", linewidth=2,
            label=f"Random Forest (PR-AUC={pr_auc:.3f})")
    ax.plot(lr_rec, lr_prec, color="#4e79a7", linewidth=1.5, linestyle="--",
            label=f"Logistic Regression (PR-AUC={lr_pr_auc:.3f})")
    ax.axhline(y=(y_test == 1).sum() / len(y_test), color="gray",
               linestyle=":", alpha=0.7, label="Clasificador aleatorio")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall — Random Forest vs Logistic Regression")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pr_curve_random_forest.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: resultados/pr_curve_random_forest.png")

    # --- SUMMARY ---
    print(f"\n{'='*70}")
    print("RESUMEN — RANDOM FOREST")
    print(f"{'='*70}")
    print(f"  PR-AUC:              {pr_auc:.4f}")
    print(f"  F1 (umbral 0.5):     {f1_base:.4f}")
    print(f"  F1 (umbral {best_threshold:.3f}):   {f1_tuned:.4f}")
    print(f"  Recall:              {recall_tuned:.4f}")
    print(f"  Precision:           {precision_tuned:.4f}")
    print(f"  Umbral optimo:       {best_threshold:.3f}")
    print(f"\n  Siguiente paso: entrenar XGBoost y comparar los tres modelos.")


if __name__ == "__main__":
    main()
