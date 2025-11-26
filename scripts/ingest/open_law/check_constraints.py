#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check UNIQUE constraints on precedent_chunks table"""

import sys
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.env_loader import ensure_env_loaded
from scripts.ingest.open_law.utils import build_database_url
from sqlalchemy import create_engine, text

ensure_env_loaded(_PROJECT_ROOT)

db_url = build_database_url()
if not db_url:
    print("ERROR: DATABASE_URL not found")
    sys.exit(1)

engine = create_engine(db_url, pool_pre_ping=True)
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT conname, contype, pg_get_constraintdef(oid) as definition
        FROM pg_constraint 
        WHERE conrelid = 'precedent_chunks'::regclass 
        AND contype = 'u'
    """))
    print("UNIQUE constraints on precedent_chunks:")
    for row in result:
        print(f"  {row[0]}: {row[2]}")

