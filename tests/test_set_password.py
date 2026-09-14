"""
tests/test_set_password.py — Écriture du hash de mot de passe dans .env.

Le script ne doit modifier que la ligne ADMIN_PASSWORD_HASH : le .env de la
machine porte la source caméra et les moteurs TensorRT.

Lancer : pytest tests/test_set_password.py -v
"""

import os

from dotenv import dotenv_values
from werkzeug.security import check_password_hash, generate_password_hash

from tools.set_password import KEY, set_env_value

ORIGINAL = "USE_LOCAL_CAM=false\n# commentaire\nREMOTE_SOURCE=http://192.0.2.1:8080/video\n"


def write_env(tmp_path, content=ORIGINAL):
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_adds_the_hash_and_keeps_every_other_line(tmp_path):
    path = write_env(tmp_path)
    hashed = generate_password_hash("correct horse")

    set_env_value(path, KEY, hashed)

    content = open(path, encoding="utf-8").read()
    assert content.startswith(ORIGINAL)
    values = dotenv_values(path)
    # Les `$` du hash ne sont pas interprétés par python-dotenv.
    assert values[KEY] == hashed
    assert check_password_hash(values[KEY], "correct horse")
    assert values["REMOTE_SOURCE"] == "http://192.0.2.1:8080/video"


def test_replaces_an_existing_hash_instead_of_duplicating_it(tmp_path):
    path = write_env(tmp_path, ORIGINAL + f"{KEY}='old'\n")

    set_env_value(path, KEY, "new")

    lines = open(path, encoding="utf-8").read().splitlines()
    assert [line for line in lines if line.startswith(KEY)] == [f"{KEY}='new'"]
    assert lines[:3] == ORIGINAL.splitlines()


def test_remove_restores_open_access(tmp_path):
    path = write_env(tmp_path, ORIGINAL + f"{KEY}='old'\n")

    set_env_value(path, KEY, None)

    assert open(path, encoding="utf-8").read() == ORIGINAL


def test_env_file_is_private(tmp_path):
    path = str(tmp_path / ".env")

    set_env_value(path, KEY, "hash")

    assert dotenv_values(path)[KEY] == "hash"
    assert os.stat(path).st_mode & 0o777 == 0o600
