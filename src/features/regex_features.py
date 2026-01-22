from __future__ import annotations

import re
from typing import Dict, Any

RE_EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
RE_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
RE_PHONE = re.compile(r"\b(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?(?:9\d{4}|\d{4})-?\d{4}\b")
RE_RG = re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-?[0-9Xx]\b")
RE_ZIP = re.compile(r"\b\d{5}-?\d{3}\b")
RE_NAME = re.compile(r"\b[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][a-záàâãéèêíìîóòôõúùûç]+\s+[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][a-záàâãéèêíìîóòôõúùûç]+\b")

def regex_signals(text: str) -> Dict[str, Any]:
    return {
        "has_email": bool(RE_EMAIL.search(text)),
        "has_cpf": bool(RE_CPF.search(text)),
        "has_phone": bool(RE_PHONE.search(text)),
        "has_rg": bool(RE_RG.search(text)),
        "has_zip": bool(RE_ZIP.search(text)),
        "has_name_like": bool(RE_NAME.search(text)),
        "email_count": len(RE_EMAIL.findall(text)),
        "cpf_count": len(RE_CPF.findall(text)),
        "phone_count": len(RE_PHONE.findall(text)),
    }

def regex_score(signals: Dict[str, Any]) -> float:
    score = 0.0
    if signals.get("has_cpf"): score += 0.45
    if signals.get("has_email"): score += 0.35
    if signals.get("has_phone"): score += 0.25
    if signals.get("has_rg"): score += 0.20
    if signals.get("has_zip"): score += 0.10
    if signals.get("has_name_like"): score += 0.05
    return min(score, 1.0)
