# -*- coding: utf-8 -*-
"""
CLI: Switch active corpus/model versions by updating .env

Usage:
  python scripts/database/switch_active_version.py v2 bge-m3@1.0
"""

import sys
from pathlib import Path

from core.utils.versioning import switch_active_versions


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python scripts/database/switch_active_version.py <corpus_version> <model_version>")
        return 1
    corpus_version = sys.argv[1]
    model_version = sys.argv[2]
    env_file = Path(".env").as_posix()
    switch_active_versions(env_file, corpus_version, model_version)
    print(f"Switched ACTIVE_CORPUS_VERSION={corpus_version}, ACTIVE_MODEL_VERSION={model_version} in {env_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
