# -*- coding: utf-8 -*-
"""
Query Extractor
쿼리 관련 추출 유틸리티
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class QueryExtractor:
    """쿼리 관련 추출 유틸리티"""

    @staticmethod
    def extract_legal_field(query_type: str, query: str) -> str:
        """법률 분야 추출"""
        # 키워드 매핑
        field_keywords = {
            "civil": ["민사", "계약", "손해배상", "재산", "계약서"],
            "criminal": ["형사", "범죄", "처벌", "형량", "범죄자"],
            "intellectual_property": ["특허", "상표", "저작권", "지적재산"],
            "administrative": ["행정", "행정처분", "행정소송", "행정심판"]
        }

        query_lower = query.lower()
        for field, keywords in field_keywords.items():
            if any(k in query_lower for k in keywords):
                return field

        # 질문 유형 기반 폴백
        type_to_field = {
            "precedent_search": "civil",
            "law_inquiry": "civil",
            "procedure_guide": "civil",
            "term_explanation": "civil",
            "legal_advice": "civil"
        }
        return type_to_field.get(query_type, "general")

