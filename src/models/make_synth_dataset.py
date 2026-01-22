from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from ..io.load_data import load_records
from ..features.regex_features import regex_signals


SYNTH_EMAILS = ["maria.silva@email.com", "joao.souza@exemplo.com", "ana.pereira@dominio.org"]
SYNTH_CPFS = ["123.456.789-10", "98765432100", "111.222.333-44"]
SYNTH_PHONES = ["(61) 91234-5678", "+55 61 3456-7890", "61987654321"]
SYNTH_RGS = ["12.345.678-9", "1.234.567-X"]
SYNTH_ADDRS = ["Rua das Flores, 123", "Av. Central, 1000", "Quadra 10 Conjunto B"]
SYNTH_NAMES = ["Maria Silva", "João Souza", "Ana Pereira", "Carlos Oliveira"]


def inject_pii(text: str, rng: random.Random) -> tuple[str, dict]:
    """Retorna (texto_modificado, sinais_injetados)."""
    injections = []

    # escolhe 1..3 tipos para injetar
    k = rng.choice([1, 1, 2, 2, 3])
    types = rng.sample(["email", "cpf", "phone", "rg", "addr", "name"], k=k)

    for t in types:
        if t == "email":
            injections.append(f"E-mail: {rng.choice(SYNTH_EMAILS)}")
        elif t == "cpf":
            injections.append(f"CPF: {rng.choice(SYNTH_CPFS)}")
        elif t == "phone":
            injections.append(f"Telefone: {rng.choice(SYNTH_PHONES)}")
        elif t == "rg":
            injections.append(f"RG: {rng.choice(SYNTH_RGS)}")
        elif t == "addr":
            injections.append(f"Endereço: {rng.choice(SYNTH_ADDRS)}")
        elif t == "name":
            injections.append(f"Nome: {rng.choice(SYNTH_NAMES)}")

    new_text = text + "\n" + "\n".join(injections)
    return new_text, {"injected_types": types, "injected_count": len(types)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera dataset sintético rotulado (PII vs não-PII).")
    ap.add_argument("--input", required=True, help="Arquivo de entrada (.xlsx/.csv/.jsonl)")
    ap.add_argument("--out_csv", default="artifacts/reports/synth_dataset.csv", help="Saída CSV rotulada")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos_ratio", type=float, default=0.5, help="Proporção de positivos")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = load_records(Path(args.input))
    if not records:
        raise SystemExit("Dataset vazio.")

    rows = []
    for r in records:
        if rng.random() < args.pos_ratio:
            text2, meta = inject_pii(r.text, rng)
            y = 1
        else:
            text2 = r.text
            meta = {"injected_types": [], "injected_count": 0}
            y = 0

        sig = regex_signals(text2)
        rows.append(
            {
                "id": r.id,
                "text": text2,
                "label": y,
                "injected_count": meta["injected_count"],
                "injected_types": ",".join(meta["injected_types"]),
                # sinais reais detectáveis (para debug)
                "has_cpf": sig["has_cpf"],
                "has_email": sig["has_email"],
                "has_phone": sig["has_phone"],
                "has_rg": sig["has_rg"],
                "has_zip": sig["has_zip"],
            }
        )

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"OK: dataset sintético salvo em {args.out_csv} ({len(df)} linhas)")
    print("label counts:\n", df["label"].value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
