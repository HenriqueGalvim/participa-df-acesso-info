from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..io.load_data import load_records
from ..features.regex_features import regex_signals, regex_score


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline: detector de dados pessoais (regex) + score.")
    ap.add_argument("--input", required=True, help="Arquivo de entrada (.xlsx/.csv/.jsonl)")
    ap.add_argument("--output", required=True, help="Arquivo de saída (.csv)")
    ap.add_argument("--threshold", type=float, default=0.35, help="Threshold para pred_label (0/1)")
    args = ap.parse_args()

    records = load_records(Path(args.input))
    rows = []
    for r in records:
        sig = regex_signals(r.text)
        score = regex_score(sig)
        label = 1 if score >= args.threshold else 0
        rows.append({
            "id": r.id,
            "pred_label": label,
            "pred_score": float(score),
            "has_cpf": sig["has_cpf"],
            "has_email": sig["has_email"],
            "has_phone": sig["has_phone"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"OK: gerado {args.output} com {len(out)} linhas")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
