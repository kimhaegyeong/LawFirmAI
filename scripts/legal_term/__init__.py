# -*- coding: utf-8 -*-
"""
법률 용어 수집 관련 모듈

국가법령정보센터 OpenAPI를 활용한 법률 용어 수집 및 관리
"""

from .term_collector import LegalTermCollector
from .synonym_manager import SynonymManager
from .term_validator import TermValidator

__all__ = ['LegalTermCollector', 'SynonymManager', 'TermValidator']
