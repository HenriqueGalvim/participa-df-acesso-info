from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Runner único (baseline ou híbrido).")
    ap.add_argument("--input", required=True, help="Entrada (.xlsx/.csv/.jsonl)")
    ap.add_argument("--output", required=True, help="Saída (.csv)")
    ap.add_argument("--model", default="artifacts/models/hybrid_tfidf_logreg.joblib", help="Modelo híbrido (opcional)")
    ap.add_argument("--mode", choices=["auto", "regex", "hybrid"], default="auto")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--threshold", type=float, default=0.25)
    args = ap.parse_args()

    model_path = Path(args.model)

    if args.mode == "regex":
        cmd = [sys.executable, "-m", "src.models.predict", "--input", args.input, "--output", args.output, "--threshold", str(args.threshold)]
        return subprocess.call(cmd)

    if args.mode == "hybrid":
        cmd = [
            sys.executable, "-m", "src.models.predict_hybrid",
            "--input", args.input, "--model", str(model_path),
            "--output", args.output,
            "--alpha", str(args.alpha),
            "--threshold", str(args.threshold),
        ]
        return subprocess.call(cmd)

    # auto
    if model_path.exists():
        cmd = [
            sys.executable, "-m", "src.models.predict_hybrid",
            "--input", args.input, "--model", str(model_path),
            "--output", args.output,
            "--alpha", str(args.alpha),
            "--threshold", str(args.threshold),
        ]
        return subprocess.call(cmd)

    cmd = [sys.executable, "-m", "src.models.predict", "--input", args.input, "--output", args.output, "--threshold", str(args.threshold)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
