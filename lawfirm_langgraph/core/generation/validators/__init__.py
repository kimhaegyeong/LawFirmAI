# -*- coding: utf-8 -*-
"""
Answer Validators Module
답변 검증기 모듈
"""

from .quality_validators import AnswerValidator
from .legal_basis_validator import LegalBasisValidator

# legal_validators.py는 legal_basis_validator.py의 별칭
from .legal_basis_validator import LegalBasisValidator as LegalValidator

__all__ = [
    "AnswerValidator",
    "LegalBasisValidator",
    "LegalValidator",
]

