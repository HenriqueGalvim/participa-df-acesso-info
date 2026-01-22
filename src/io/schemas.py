from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Record:
    id: str
    text: str

@dataclass
class Prediction:
    id: str
    pred_label: int
    pred_score: float
    signals: Optional[Dict[str, Any]] = None
