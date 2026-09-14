"""
tools/set_password.py — Définit le mot de passe d'accès à l'interface.

Le mot de passe est saisi sans écho, jamais affiché ni passé en argument (il
finirait dans l'historique du shell). Seul son hash est écrit dans `.env`,
sous ADMIN_PASSWORD_HASH : la ligne est remplacée si elle existe, ajoutée
sinon ; les autres lignes du fichier ne sont pas touchées.

Lancer : uv run python tools/set_password.py
         uv run python tools/set_password.py --remove   (accès ouvert)
Redémarrer l'application ensuite.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys

from werkzeug.security import generate_password_hash

KEY = "ADMIN_PASSWORD_HASH"
MIN_LENGTH = 8
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def set_env_value(path: str, key: str, value: str | None) -> None:
    """Remplace, ajoute (ou retire si `value` est None) `key` dans un fichier .env.

    La valeur est écrite entre apostrophes : python-dotenv n'y interprète pas
    les `$` que contient un hash werkzeug.
    """
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    pattern = re.compile(rf"^\s*(export\s+)?{re.escape(key)}\s*=")
    kept = [line for line in lines if not pattern.match(line)]
    if value is not None:
        kept.append(f"{key}='{value}'")

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--remove", action="store_true", help="retire le mot de passe (accès ouvert)")
    parser.add_argument("--env", default=ENV_PATH, help="fichier .env à modifier")
    args = parser.parse_args()

    if args.remove:
        set_env_value(args.env, KEY, None)
        print(f"{KEY} retiré de {args.env}. Redémarrer l'application.")
        return 0

    password = getpass.getpass("Nouveau mot de passe : ")
    if len(password) < MIN_LENGTH:
        print(f"Trop court : {MIN_LENGTH} caractères minimum.", file=sys.stderr)
        return 1
    if getpass.getpass("Confirmer : ") != password:
        print("Les deux saisies diffèrent.", file=sys.stderr)
        return 1

    set_env_value(args.env, KEY, generate_password_hash(password))
    print(f"{KEY} écrit dans {args.env}. Redémarrer l'application.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
