from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Paths:
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    models_dir: Path = artifacts_dir / "models"
    reports_dir: Path = artifacts_dir / "reports"

@dataclass(frozen=True)
class Defaults:
    seed: int = 42
    text_column_candidates: tuple[str, ...] = (
        "texto mascarado",
        "texto",
        "mensagem",
        "pedido",
        "descricao",
        "descrição",
        "conteudo",
        "conteúdo",
        "manifestacao",
        "manifestação",
    )
    id_column_candidates: tuple[str, ...] = ("id", "protocolo", "numero", "número")
