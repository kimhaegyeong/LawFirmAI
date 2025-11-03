# -*- coding: utf-8 -*-
"""
Version switching helpers (blue/green) for corpus/model.
Updates .env to flip ACTIVE_CORPUS_VERSION and ACTIVE_MODEL_VERSION atomically (with backup).
"""

from pathlib import Path
from typing import Dict


def _parse_env(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    content = path.read_text(encoding="utf-8").splitlines()
    for line in content:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def _dump_env(path: Path, data: Dict[str, str]) -> None:
    lines = [f"{k}={v}\n" for k, v in data.items()]
    path.write_text("".join(lines), encoding="utf-8")


def switch_active_versions(env_file: str, new_corpus_version: str, new_model_version: str) -> None:
    """Update .env ACTIVE_CORPUS_VERSION and ACTIVE_MODEL_VERSION with backup."""
    env_path = Path(env_file)
    backup = env_path.with_suffix(env_path.suffix + ".bak")
    env = _parse_env(env_path)
    env.setdefault("ACTIVE_CORPUS_VERSION", "v1")
    env.setdefault("ACTIVE_MODEL_VERSION", "default@1.0")
    env["ACTIVE_CORPUS_VERSION"] = new_corpus_version
    env["ACTIVE_MODEL_VERSION"] = new_model_version
    # backup
    if env_path.exists():
        backup.write_text(env_path.read_text(encoding="utf-8"), encoding="utf-8")
    _dump_env(env_path, env)
