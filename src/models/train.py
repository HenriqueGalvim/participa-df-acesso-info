from __future__ import annotations

import argparse
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ..config import Defaults
from ..io.load_data import load_records

SYNTH_EMAILS = ["maria.silva@email.com", "joao.souza@exemplo.com"]
SYNTH_CPFS = ["123.456.789-10", "98765432100"]
SYNTH_PHONES = ["(61) 91234-5678", "+55 61 3456-7890"]

def inject_pii(text: str, rng: random.Random) -> str:
    choice = rng.choice(["email", "cpf", "phone"])
    if choice == "email":
        return text + f"\nE-mail: {rng.choice(SYNTH_EMAILS)}"
    if choice == "cpf":
        return text + f"\nCPF: {rng.choice(SYNTH_CPFS)}"
    return text + f"\nTelefone: {rng.choice(SYNTH_PHONES)}"

def main() -> int:
    ap = argparse.ArgumentParser(description="Treino supervisionado com rótulos sintéticos (scaffold).")
    ap.add_argument("--input", required=True, help="Arquivo de entrada (.xlsx/.csv/.jsonl)")
    ap.add_argument("--model_out", default="artifacts/models/tfidf_logreg.joblib", help="Saída do modelo")
    ap.add_argument("--seed", type=int, default=Defaults.seed)
    ap.add_argument("--pos_ratio", type=float, default=0.5, help="Proporção de exemplos positivos sintéticos")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    records = load_records(Path(args.input))
    texts = [r.text for r in records]
    if not texts:
        raise SystemExit("Dataset vazio.")

    X, y = [], []
    for t in texts:
        if rng.random() < args.pos_ratio:
            X.append(inject_pii(t, rng)); y.append(1)
        else:
            X.append(t); y.append(0)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
    )

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)
    Xtr = vec.fit_transform(X_train)
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, y_train)

    joblib.dump({"vectorizer": vec, "model": clf}, args.model_out)
    print(f"OK: modelo salvo em {args.model_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
