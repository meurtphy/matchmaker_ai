#!/usr/bin/env python3
"""demo_request.py
=================
Petit script CLI pour tester l'API MatchMaker AI en local.

Exemple :
    python scripts/demo_request.py \
        --texte "on cherche des ballons de basket" \
        --cp 34090 \
        --k 5

Par défaut, l'API est supposée tourner sur http://127.0.0.1:8000
(p. ex. `uvicorn backend.app.main:app --reload`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import requests

DEFAULT_URL = "http://127.0.0.1:8000/match"


def send_request(url: str, payload: Dict[str, Any]) -> None:
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Requête échouée : {e}")
        sys.exit(1)

    data = r.json()
    print("\n📬 Réponse API :\n" + json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test rapide de l'endpoint /match")
    parser.add_argument("--texte", required=True, help="Besoin exprimé par l'association")
    parser.add_argument("--cp", required=True, help="Code postal (5 chiffres)")
    parser.add_argument("--k", type=int, default=5, help="Nombre de résultats à afficher")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"URL de l'API (défaut: {DEFAULT_URL})",
    )
    args = parser.parse_args()

    payload = {"texte": args.texte, "cp": args.cp, "k": args.k}
    print(f"➡️  Envoi : {payload} → {args.url}\n")
    send_request(args.url, payload)
