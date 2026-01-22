from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

from ..config import Defaults


REGEX_FEATURE_COLS = [
    "has_cpf",
    "has_email",
    "has_phone",
    "has_rg",
    "has_zip",
    "injected_count",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Treina modelo híbrido: TF-IDF + features regex numéricas.")
    ap.add_argument("--synth_csv", required=True, help="CSV rotulado gerado pelo make_synth_dataset")
    ap.add_argument("--model_out", default="artifacts/models/hybrid_tfidf_logreg.joblib")
    ap.add_argument("--seed", type=int, default=Defaults.seed)
    ap.add_argument("--C", type=float, default=2.0, help="Regularização do LogisticRegression (maior = menos regularização)")
    args = ap.parse_args()

    df = pd.read_csv(args.synth_csv)
    if df.empty:
        raise SystemExit("Dataset sintético vazio.")

    # garante colunas
    for c in REGEX_FEATURE_COLS + ["text", "label"]:
        if c not in df.columns:
            raise SystemExit(f"Coluna ausente no synth_csv: {c}")

    X_text = df["text"].astype(str).tolist()
    y = df["label"].astype(int).values

    # features numéricas (regex)
    X_num = df[REGEX_FEATURE_COLS].fillna(0).astype(float).values
    X_num = csr_matrix(X_num)

    Xtr_text, Xva_text, Xtr_num, Xva_num, ytr, yva = train_test_split(
        X_text, X_num, y, test_size=0.25, random_state=args.seed, stratify=y
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    Xtr_tfidf = vec.fit_transform(Xtr_text)
    Xva_tfidf = vec.transform(Xva_text)

    # concatena TF-IDF + numéricas
    Xtr = hstack([Xtr_tfidf, Xtr_num])
    Xva = hstack([Xva_tfidf, Xva_num])

    clf = LogisticRegression(max_iter=400, C=args.C)
    clf.fit(Xtr, ytr)

    preds = clf.predict(Xva)
    probs = clf.predict_proba(Xva)[:, 1]

    print("\n=== CLASSIFICATION REPORT (val) ===")
    print(classification_report(yva, preds, digits=4))

    # salva artefato completo
    joblib.dump(
        {
            "vectorizer": vec,
            "model": clf,
            "regex_feature_cols": REGEX_FEATURE_COLS,
        },
        args.model_out,
    )
    print(f"OK: modelo híbrido salvo em {args.model_out}")
    print("prob stats:", float(np.min(probs)), float(np.mean(probs)), float(np.max(probs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
