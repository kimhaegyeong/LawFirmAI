#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Law Open API 수집기 모듈

법령용어 수집을 위한 수집기 모듈들을 제공합니다.
"""

from .incremental_legal_term_collector import (
    IncrementalLegalTermCollector,
    create_collector,
    collect_incremental_updates,
    collect_full_data
)

__all__ = [
    'IncrementalLegalTermCollector',
    'create_collector',
    'collect_incremental_updates',
    'collect_full_data'
]




