#!/usr/bin/env python3
"""
Exporta el Random Forest entrenado a un Pipeline serializado.

Uso: python3 src/exportar_modelo.py

Genera:
  interfaz/modelo/rf_pipeline.pkl       Pipeline (imputer + RF) serializado
  interfaz/modelo/feature_importance.json   Top 22 features con su peso
  interfaz/modelo/metrics.json          PR-AUC, threshold optimo, F1, etc.
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(__file__))
from config import METEO_VARIABLES, PROCESSED_DIR

INPUT_CSV = os.path.join(PROCESSED_DIR, "dataset_modelo.csv")
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "interfaz", "modelo"
)
TEST_YEAR_START = 2019

DERIVED_FEATURES = [
    "dias_sin_lluvia", "PP_acum_7d", "PP_acum_14d", "PP_acum_30d",
    "TA_AVG_3d", "TA_MAX_3d", "HR_MIN_3d", "amplitud_termica",
    "mes", "dia_del_ano",
]
ALL_FEATURES = METEO_VARIABLES + DERIVED_FEATURES


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("EXPORTAR MODELO — Random Forest a pickle")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV, parse_dates=["fecha"])
    train = df[df["fecha"].dt.year < TEST_YEAR_START]
    test = df[df["fecha"].dt.year >= TEST_YEAR_START]
    X_train, y_train = train[ALL_FEATURES], train["incendio"]
    X_test, y_test = test[ALL_FEATURES], test["incendio"]
    print(f"\nTrain: {len(X_train):,} filas | Test: {len(X_test):,} filas")

    print("\nEntrenando Random Forest (300 árboles, max_depth=15)...")
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
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_proba)

    print("Optimizando umbral con TunedThresholdClassifierCV...")
    tuned = TunedThresholdClassifierCV(model, scoring="f1", cv=5)
    tuned.fit(X_train, y_train)
    threshold = tuned.best_threshold_

    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)

    print(f"\nMétricas finales:")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")

    # Guardar pickle del Pipeline + threshold + features
    bundle = {
        "pipeline": model,
        "threshold": float(threshold),
        "features": ALL_FEATURES,
        "pr_auc": float(pr_auc),
    }
    pkl_path = os.path.join(OUTPUT_DIR, "rf_pipeline.pkl")
    joblib.dump(bundle, pkl_path)
    size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"\nGuardado: {pkl_path}  ({size_mb:.1f} MB)")

    # Feature importance
    importances = model.named_steps["clf"].feature_importances_
    fi = sorted(
        [{"feature": ALL_FEATURES[i], "importance": float(importances[i])}
         for i in range(len(ALL_FEATURES))],
        key=lambda x: -x["importance"],
    )
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.json")
    with open(fi_path, "w", encoding="utf-8") as f:
        json.dump(fi, f, ensure_ascii=False, indent=2)
    print(f"Guardado: {fi_path}")

    # Metrics summary
    metrics = {
        "model": "random_forest",
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_leaf": 10,
        "test_year_start": TEST_YEAR_START,
        "pr_auc": float(pr_auc),
        "threshold": float(threshold),
        "f1_tuned": float(f1),
        "recall_tuned": float(recall),
        "precision_tuned": float(precision),
    }
    m_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Guardado: {m_path}")

    # Smoke test: cargar y predecir
    print("\nSmoke test: cargar pickle y predecir 1 muestra...")
    loaded = joblib.load(pkl_path)
    sample = X_test.iloc[[0]]
    p = loaded["pipeline"].predict_proba(sample)[0, 1]
    print(f"  Probabilidad muestra 0: {p:.4f}  (label real: {int(y_test.iloc[0])})")
    print(f"  Decision con threshold {loaded['threshold']:.3f}: "
          f"{'INCENDIO' if p >= loaded['threshold'] else 'no'}")
    print("\nOK.")


if __name__ == "__main__":
    main()
