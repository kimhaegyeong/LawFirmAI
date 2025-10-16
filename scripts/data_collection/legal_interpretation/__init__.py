#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법령해석례 수집 패키지

국가법령정보센터 LAW OPEN API를 사용하여 법령해석례를 수집하는 패키지입니다.
"""

from .legal_interpretation_collector import LegalInterpretationCollector
from .legal_interpretation_models import (
    CollectionStats, LegalInterpretationData, CollectionStatus, 
    InterpretationCategory, MinistryType
)
from .legal_interpretation_logger import setup_logging

__version__ = "1.0.0"
__author__ = "LawFirmAI Team"
__description__ = "법령해석례 수집 패키지"

__all__ = [
    "LegalInterpretationCollector",
    "CollectionStats", 
    "LegalInterpretationData", 
    "CollectionStatus",
    "InterpretationCategory", 
    "MinistryType",
    "setup_logging"
]
