from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from ..config import Defaults
from .schemas import Record


def _normalize_col(col: str) -> str:
    return str(col).strip().lower()


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    norm = {_normalize_col(c): c for c in columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return None


def load_records(path: Path) -> List[Record]:
    """Loads records from .xlsx, .csv or .jsonl with auto-detect."""
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Formato não suportado: {suffix}. Use .xlsx, .csv ou .jsonl")

    if df.empty:
        return []

    cols = list(df.columns)
    id_col = _pick_column(cols, Defaults.id_column_candidates)
    text_col = _pick_column(cols, Defaults.text_column_candidates)

    if text_col is None:
        raise ValueError(
            "Não foi possível detectar a coluna de texto automaticamente. "
            f"Colunas encontradas: {cols}. "
            "Dica: renomeie para 'texto'/'mensagem' ou 'Texto Mascarado'."
        )

    if id_col is None:
        df = df.reset_index(drop=True)
        df["__id__"] = df.index.astype(str)
        id_col = "__id__"

    out: List[Record] = []
    for _, row in df.iterrows():
        rid = str(row[id_col])
        text = "" if pd.isna(row[text_col]) else str(row[text_col])
        out.append(Record(id=rid, text=text))
    return out
