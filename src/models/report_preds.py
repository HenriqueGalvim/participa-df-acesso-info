from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Relatório rápido das predições (sem ground-truth).")
    ap.add_argument("--preds", required=True, help="CSV gerado pelo predict.py")
    ap.add_argument("--top", type=int, default=15, help="Quantidade de exemplos para mostrar")
    args = ap.parse_args()

    df = pd.read_csv(args.preds)

    print("\n=== Visão geral ===")
    print("linhas:", len(df))
    print("pred_label counts:\n", df["pred_label"].value_counts(dropna=False))

    print("\n=== Estatísticas de score ===")
    print(df["pred_score"].describe())

    # Distribuição simples por faixas
    bins = [-0.001, 0.05, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0]
    labels = ["<=0.05", "0.05-0.15", "0.15-0.25", "0.25-0.35", "0.35-0.50", "0.50-0.75", "0.75-1.0"]
    df["score_bucket"] = pd.cut(df["pred_score"], bins=bins, labels=labels)
    print("\n=== Buckets de score ===")
    print(df["score_bucket"].value_counts().sort_index())

    # Quais sinais estão gerando positivos
    if {"has_cpf", "has_email", "has_phone"}.issubset(df.columns):
        print("\n=== Positivos por sinal (apenas pred_label=1) ===")
        pos = df[df["pred_label"] == 1]
        if len(pos) == 0:
            print("Nenhum positivo com o threshold atual.")
        else:
            print("pos total:", len(pos))
            print("has_cpf:", int(pos["has_cpf"].sum()))
            print("has_email:", int(pos["has_email"].sum()))
            print("has_phone:", int(pos["has_phone"].sum()))

    # Top scores (prováveis PII)
    print("\n=== Top scores ===")
    print(df.sort_values("pred_score", ascending=False).head(args.top)[["id", "pred_score", "pred_label"]])

    # Borderline (perto do threshold 0.35 por padrão)
    thr = 0.35
    border = df[(df["pred_score"] >= thr - 0.05) & (df["pred_score"] < thr + 0.05)]
    print(f"\n=== Borderline (score em [{thr-0.05:.2f}, {thr+0.05:.2f})) ===")
    print(border.sort_values("pred_score", ascending=False).head(args.top)[["id", "pred_score", "pred_label"]])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
