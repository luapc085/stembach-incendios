"""Evaluar FWI canadiense vs Random Forest sobre el mismo test set 2019-2021.

Métricas comparables: PR-AUC, F1, recall, precision al umbral óptimo (maximizando F1).
Salidas: JSON con métricas y comparación, figura curva PR superpuesta.
"""
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

PROJ = Path(__file__).parent.parent
DATASET = PROJ / "data/processed/dataset_modelo.csv"
FWI_CSV = Path(__file__).parent / "data/fwi_diario.csv"
RF_PKL = PROJ / "interfaz/modelo/rf_pipeline.pkl"
RF_METRICS = PROJ / "interfaz/modelo/metrics.json"
OUT_DIR = Path(__file__).parent / "resultados"
TEST_YEAR_START = 2019


def metricas_score(y_true, score, etiqueta):
    """PR-AUC + F1/recall/precision al umbral óptimo (max F1)."""
    pr_auc = average_precision_score(y_true, score)
    p, r, th = precision_recall_curve(y_true, score)
    f1s = 2 * p * r / (p + r + 1e-12)
    i = int(np.argmax(f1s[:-1]))  # último punto no tiene threshold
    return {
        "modelo": etiqueta,
        "pr_auc": float(pr_auc),
        "threshold_opt": float(th[i]),
        "f1": float(f1s[i]),
        "recall": float(r[i]),
        "precision": float(p[i]),
        "n_test": int(len(y_true)),
        "n_positivos": int(y_true.sum()),
        "tasa_base": float(y_true.mean()),
    }, p, r


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Cargando dataset modelo y FWI…")
    df = pd.read_csv(DATASET, parse_dates=["fecha"])
    fwi = pd.read_csv(FWI_CSV, parse_dates=["fecha"])
    print(f"  dataset: {len(df):,} filas · FWI: {len(fwi):,} filas")

    df = df.merge(fwi[["estacion_id", "fecha", "fwi"]],
                   on=["estacion_id", "fecha"], how="left")
    print(f"  FWI nan post-merge: {df.fwi.isna().sum():,} ({100*df.fwi.isna().mean():.1f}%)")

    # Test set 2019-2021
    test = df[df.fecha.dt.year >= TEST_YEAR_START].copy()
    test = test.dropna(subset=["fwi"])  # solo filas con FWI calculado
    y_true = test["incendio"].values
    print(f"\nTest set 2019-2021 (con FWI no-nan):")
    print(f"  filas: {len(test):,} · positivos: {y_true.sum():,} ({100*y_true.mean():.2f}%)")

    # FWI como score directo
    print("\nEvaluando FWI…")
    fwi_metrics, p_fwi, r_fwi = metricas_score(y_true, test["fwi"].values, "FWI canadiense")
    print(f"  PR-AUC: {fwi_metrics['pr_auc']:.4f}")
    print(f"  F1: {fwi_metrics['f1']:.4f} @ FWI≥{fwi_metrics['threshold_opt']:.2f}")
    print(f"  Recall: {fwi_metrics['recall']:.4f} · Precision: {fwi_metrics['precision']:.4f}")

    # RF score sobre el mismo test set
    print("\nEvaluando RF sobre el mismo subset…")
    bundle = joblib.load(RF_PKL)
    rf_pipeline = bundle["pipeline"]
    feature_cols = bundle["features"]
    rf_score = rf_pipeline.predict_proba(test[feature_cols])[:, 1]
    rf_metrics, p_rf, r_rf = metricas_score(y_true, rf_score, "Random Forest")
    print(f"  PR-AUC: {rf_metrics['pr_auc']:.4f}")
    print(f"  F1: {rf_metrics['f1']:.4f} @ score≥{rf_metrics['threshold_opt']:.4f}")
    print(f"  Recall: {rf_metrics['recall']:.4f} · Precision: {rf_metrics['precision']:.4f}")

    # Tasa base = clasificador aleatorio
    base = float(y_true.mean())

    # Distribución FWI por nivel canónico
    cortes = [(0, 5.2, "Baixo"), (5.2, 11.2, "Moderado"),
              (11.2, 21.3, "Alto"), (21.3, 38.0, "Moi alto"),
              (38.0, 999, "Extremo")]
    niveles = []
    for lo, hi, lab in cortes:
        mask = (test.fwi >= lo) & (test.fwi < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        pos = int(test.loc[mask, "incendio"].sum())
        niveles.append({
            "nivel": lab,
            "rango_fwi": f"{lo:.1f}–{hi:.1f}" if hi < 999 else f"≥{lo:.1f}",
            "n_dias": n,
            "n_incendios": pos,
            "tasa_incendio": pos / n if n else 0.0,
            "lift_vs_base": (pos / n) / base if n else 0.0,
        })
    print("\nDistribución por nivel FWI canónico:")
    for d in niveles:
        print(f"  {d['nivel']:>10s} ({d['rango_fwi']:>10s}): "
              f"n={d['n_dias']:>6d} · incendios={d['n_incendios']:>4d} "
              f"· tasa={100*d['tasa_incendio']:.2f}% · lift={d['lift_vs_base']:.1f}×")

    # Comparación
    comp = {
        "test_year_start": TEST_YEAR_START,
        "tasa_base": base,
        "fwi": fwi_metrics,
        "random_forest": rf_metrics,
        "delta_rf_vs_fwi": {
            "pr_auc_pct": 100 * (rf_metrics["pr_auc"] - fwi_metrics["pr_auc"]) / fwi_metrics["pr_auc"],
            "f1_pct": 100 * (rf_metrics["f1"] - fwi_metrics["f1"]) / fwi_metrics["f1"],
            "precision_pct": 100 * (rf_metrics["precision"] - fwi_metrics["precision"]) / fwi_metrics["precision"],
            "recall_pct": 100 * (rf_metrics["recall"] - fwi_metrics["recall"]) / fwi_metrics["recall"],
        },
        "niveles_fwi": niveles,
    }
    out = OUT_DIR / "comparativa.json"
    out.write_text(json.dumps(comp, indent=2, ensure_ascii=False))
    print(f"\nGuardado: {out}")

    # Figura curva PR
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r_rf, p_rf, color="#1f9d55", linewidth=2,
            label=f"Random Forest (PR-AUC = {rf_metrics['pr_auc']:.3f})")
    ax.plot(r_fwi, p_fwi, color="#d97706", linewidth=2,
            label=f"FWI canadiense (PR-AUC = {fwi_metrics['pr_auc']:.3f})")
    ax.axhline(base, linestyle="--", color="gray", linewidth=1,
               label=f"Aleatorio (precision = {base:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall sobre test 2019-2021\n"
                 "Random Forest vs FWI canadiense")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(p_rf.max(), p_fwi.max()) * 1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pr_curve_rf_vs_fwi.png", dpi=140)
    print(f"Guardado: {OUT_DIR / 'pr_curve_rf_vs_fwi.png'}")

    # Conclusión
    print("\n" + "=" * 60)
    print("CONCLUSIÓN")
    print("=" * 60)
    delta = comp["delta_rf_vs_fwi"]["pr_auc_pct"]
    if delta > 5:
        print(f"Random Forest mejora al FWI: +{delta:.1f}% en PR-AUC.")
    elif delta < -5:
        print(f"FWI gana al Random Forest: {-delta:.1f}% mejor PR-AUC.")
    else:
        print(f"Empate técnico (Δ PR-AUC = {delta:+.1f}%).")
    print(f"  RF:  PR-AUC {rf_metrics['pr_auc']:.4f} · F1 {rf_metrics['f1']:.4f}")
    print(f"  FWI: PR-AUC {fwi_metrics['pr_auc']:.4f} · F1 {fwi_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
