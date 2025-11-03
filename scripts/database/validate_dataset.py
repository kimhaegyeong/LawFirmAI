# -*- coding: utf-8 -*-
"""
Validate a versioned dataset: print table counts and simple sanity checks.

Usage:
  python scripts/database/validate_dataset.py v1
"""

import sqlite3
import sys

from core.data.versioned_schema import ensure_versioned_db
from core.utils.config import Config


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/database/validate_dataset.py <corpus_version>")
        return 1
    corpus = sys.argv[1]
    cfg = Config()
    db_path = ensure_versioned_db(cfg.versioned_database_dir, corpus)
    print(f"DB: {db_path}")
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        for table in [
            "laws",
            "provisions_meta",
            "cases",
            "interpretations",
            "text_store",
            "embeddings",
            "ministry_codes",
            "court_codes",
            "case_type_codes",
        ]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"{table}:", cur.fetchone()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
