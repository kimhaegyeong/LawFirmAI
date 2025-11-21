# -*- coding: utf-8 -*-
"""
통합 프롬프트 관리 시스템 (리팩토링 버전)
법률 도메인 특화 프롬프트의 통합 관리 및 최적화

참고: 이 파일은 unified_prompt_manager.py의 리팩토링 버전입니다.
기존 파일과의 호환성을 위해 동일한 인터페이스를 제공합니다.
"""

import json
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# QuestionType import
try:
    from core.classification.classifiers.question_classifier import QuestionType
except ImportError:
    try:
        from core.services.question_classifier import QuestionType
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        try:
            from question_classifier import QuestionType
        except ImportError:
            from enum import Enum
            class QuestionType(Enum):
                GENERAL_QUESTION = "general_question"
                LAW_INQUIRY = "law_inquiry"
                PRECEDENT_SEARCH = "precedent_search"
                DOCUMENT_ANALYSIS = "document_analysis"
                LEGAL_ADVICE = "legal_advice"

logger = get_logger(__name__)


class LegalDomain(Enum):
    """법률 도메인 분류"""
    CIVIL_LAW = "민사법"
    CRIMINAL_LAW = "형사법"
    FAMILY_LAW = "가족법"
    COMMERCIAL_LAW = "상사법"
    ADMINISTRATIVE_LAW = "행정법"
    LABOR_LAW = "노동법"
    PROPERTY_LAW = "부동산법"
    INTELLECTUAL_PROPERTY = "지적재산권법"
    TAX_LAW = "세법"
    CIVIL_PROCEDURE = "민사소송법"
    CRIMINAL_PROCEDURE = "형사소송법"
    GENERAL = "기타/일반"


class ModelType(Enum):
    """지원 모델 타입"""
    GEMINI = "gemini"
    OPENAI = "openai"


# 기존 unified_prompt_manager.py에서 UnifiedPromptManager를 import
# TODO: 점진적으로 리팩토링하여 이 파일로 이동
try:
    from core.services.unified_prompt_manager import UnifiedPromptManager as _UnifiedPromptManager
    UnifiedPromptManager = _UnifiedPromptManager
except ImportError:
    logger.warning("기존 unified_prompt_manager를 import할 수 없습니다. 새로 구현이 필요합니다.")
    # 여기에 새로운 구현을 추가할 예정
    UnifiedPromptManager = None

