# -*- coding: utf-8 -*-
"""
Date normalization utilities (YYYYMMDD -> YYYY-MM-DD)
"""

from typing import Optional


def yyyymmdd_to_iso(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    s = str(date_str).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s
