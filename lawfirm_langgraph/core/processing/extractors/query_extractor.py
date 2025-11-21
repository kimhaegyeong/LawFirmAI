# -*- coding: utf-8 -*-
"""
Query Extractor
쿼리 관련 추출 유틸리티
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Dict

logger = get_logger(__name__)


class QueryExtractor:
    """쿼리 관련 추출 유틸리티"""

    @staticmethod
    def extract_legal_field(query_type: str, query: str) -> str:
        """
        법률 분야 추출 (키워드 기반, LLM 호출 없음)
        
        키워드 매칭을 통해 법률 분야를 빠르게 추출합니다.
        우선순위: 가족법 > 형사법 > 지적재산권 > 행정법 > 노동법 > 민법 > 일반
        """
        # 키워드 매핑 (우선순위 순서대로)
        field_keywords = {
            "family_law": [
                "이혼", "양육권", "상속", "부양", "가족", "친권", "양육", "위자료",
                "재혼", "혼인", "결혼", "별거", "재산분할", "상속분", "유류분"
            ],
            "criminal": [
                "형사", "범죄", "처벌", "형량", "범죄자", "수사", "기소", "공소",
                "구속", "보석", "징역", "벌금", "사형", "무기징역", "유기징역"
            ],
            "intellectual_property": [
                "특허", "상표", "저작권", "지적재산", "디자인권", "영업비밀",
                "특허권", "상표권", "저작권법", "특허법", "상표법"
            ],
            "administrative": [
                "행정", "행정처분", "행정소송", "행정심판", "행정법", "행정구제",
                "행정쟁송", "행정상 손해배상", "행정상 불법행위"
            ],
            "labor_law": [
                "노동", "근로", "임금", "근로기준법", "근로계약", "해고", "퇴직금",
                "퇴직", "근로자", "사용자", "단체협약", "노동조합", "파업", "쟁의"
            ],
            "civil": [
                "민사", "계약", "손해배상", "재산", "계약서", "임대차", "임대인",
                "임차인", "부동산", "물권", "채권", "채무", "해지", "해제", "위약금",
                "계약금", "중도금", "잔금", "소유권", "점유", "등기", "전세", "보증금"
            ]
        }

        query_lower = query.lower()
        
        # 우선순위에 따라 매칭 (가족법이 가장 높은 우선순위)
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

