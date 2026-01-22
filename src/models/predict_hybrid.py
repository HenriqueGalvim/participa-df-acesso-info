from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from ..io.load_data import load_records
from ..features.regex_features import regex_signals, regex_score

# Se o regex indicar fortemente presença de PII, não deixamos o ML derrubar o caso.
# Isso reduz falsos negativos mantendo explicabilidade e controle de FP via threshold.
def main() -> int:
    ap = argparse.ArgumentParser(description="Predição híbrida: TF-IDF+LogReg + regex score (mix por alpha).")
    ap.add_argument("--input", required=True, help="Arquivo de entrada (.xlsx/.csv/.jsonl)")
    ap.add_argument("--model", required=True, help="Modelo híbrido .joblib")
    ap.add_argument("--output", required=True, help="Saída .csv (padronizada)")
    ap.add_argument("--alpha", type=float, default=0.70, help="Peso do ML no score final (0..1)")
    ap.add_argument("--threshold", type=float, default=0.30, help="Threshold para pred_label (0/1)")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    vec = bundle["vectorizer"]
    clf = bundle["model"]
    regex_cols = bundle["regex_feature_cols"]

    records = load_records(Path(args.input))
    if not records:
        raise SystemExit("Entrada vazia.")

    texts = [r.text for r in records]
    ids = [r.id for r in records]

    # TF-IDF
    X_tfidf = vec.transform(texts)

    # regex features numéricas (mesma ordem do treino)
    feats = []
    regex_scores = []
    main_signals = []
    for t in texts:
        sig = regex_signals(t)
        rs = regex_score(sig)
        regex_scores.append(rs)

        row = []
        for c in regex_cols:
            if c == "injected_count":
                row.append(0.0)  # no real input não existe; mantemos 0
            else:
                row.append(float(bool(sig.get(c.replace("has_", "has_"), False)) if c.startswith("has_") else sig.get(c, 0)))
        feats.append(row)

        main_signals.append({
            "has_cpf": sig["has_cpf"],
            "has_email": sig["has_email"],
            "has_phone": sig["has_phone"],
        })

    X_num = csr_matrix(np.array(feats, dtype=float))
    X = hstack([X_tfidf, X_num])

    ml_score = clf.predict_proba(X)[:, 1].astype(float)
    regex_score_arr = np.array(regex_scores, dtype=float)

    alpha = float(args.alpha)
    score_final = alpha * ml_score + (1.0 - alpha) * regex_score_arr
    base_thr = float(args.threshold)
    regex_force_thr = 0.35  # casos óbvios pelo baseline

    pred_label = (score_final >= base_thr).astype(int)

    # fallback: não perder óbvios do regex
    pred_label = np.where(regex_score_arr >= regex_force_thr, 1, pred_label)

    out_rows = []
    for i in range(len(ids)):
        out_rows.append({
            "id": ids[i],
            "pred_label": int(pred_label[i]),
            "pred_score": float(score_final[i]),
            # debug explicável
            "ml_score": float(ml_score[i]),
            "regex_score": float(regex_score_arr[i]),
            "has_cpf": bool(main_signals[i]["has_cpf"]),
            "has_email": bool(main_signals[i]["has_email"]),
            "has_phone": bool(main_signals[i]["has_phone"]),
            "forced_by_regex": bool(regex_score_arr[i] >= regex_force_thr)
        })

    df = pd.DataFrame(out_rows).sort_values("pred_score", ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"OK: gerado {args.output} com {len(df)} linhas | alpha={alpha} thr={args.threshold}")
    print("pred_label counts:\n", df["pred_label"].value_counts(dropna=False))
    print("score stats:", df["pred_score"].min(), df["pred_score"].mean(), df["pred_score"].max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
