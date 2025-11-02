# -*- coding: utf-8 -*-
"""
Hash utilities for content change detection
"""

import hashlib
import json
from typing import Any, Dict


def canonicalize_record(record: Dict[str, Any]) -> str:
    """Serialize dict deterministically (keys sorted, UTF-8)."""
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_hash_from_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_hash_from_record(record: Dict[str, Any]) -> str:
    return hashlib.sha256(canonicalize_record(record).encode("utf-8")).hexdigest()
