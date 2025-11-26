# -*- coding: utf-8 -*-
"""
법률 용어 정규화 모듈
법률 문서에서 사용되는 용어들을 표준화하고 정규화하는 기능 제공
"""

import re
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

logger = get_logger(__name__)

@dataclass
class LegalTerm:
    """법률 용어 데이터 클래스"""
    original: str
    normalized: str
    category: str
    synonyms: List[str]

class LegalTermNormalizer:
    """법률 용어 정규화 클래스"""
    
    def __init__(self):
        """초기화"""
        self.term_mappings = self._load_term_mappings()
        self.patterns = self._load_patterns()
        logger.info("LegalTermNormalizer initialized")
    
    def _load_term_mappings(self) -> Dict[str, str]:
        """법률 용어 매핑 로드"""
        return {
            # 법령 관련
            "민법": "민법",
            "형법": "형법",
            "상법": "상법",
            "노동법": "노동법",
            "행정법": "행정법",
            "헌법": "헌법",
            
            # 법원 관련
            "대법원": "대법원",
            "고등법원": "고등법원",
            "지방법원": "지방법원",
            "가정법원": "가정법원",
            "행정법원": "행정법원",
            
            # 절차 관련
            "소송": "소송",
            "재판": "재판",
            "판결": "판결",
            "선고": "선고",
            "기각": "기각",
            "인용": "인용",
            
            # 계약 관련
            "계약": "계약",
            "매매": "매매",
            "임대차": "임대차",
            "도급": "도급",
            "위임": "위임",
            "위탁": "위탁",
            
            # 손해 관련
            "손해배상": "손해배상",
            "불법행위": "불법행위",
            "과실": "과실",
            "고의": "고의",
            
            # 형사 관련
            "범죄": "범죄",
            "처벌": "처벌",
            "형벌": "형벌",
            "벌금": "벌금",
            "징역": "징역",
            
            # 행정 관련
            "허가": "허가",
            "인허가": "인허가",
            "신고": "신고",
            "신청": "신청",
            "처분": "처분",
            "제재": "제재"
        }
    
    def _load_patterns(self) -> Dict[str, str]:
        """정규화 패턴 로드"""
        return {
            # 조문 번호 패턴
            r'제(\d+)조': r'제\1조',
            r'제(\d+)조제(\d+)항': r'제\1조제\2항',
            r'제(\d+)조제(\d+)항제(\d+)호': r'제\1조제\2항제\3호',
            
            # 법원 판결 번호 패턴
            r'(\d{4})다(\d+)': r'\1다\2',
            r'(\d{4})나(\d+)': r'\1나\2',
            r'(\d{4})가(\d+)': r'\1가\2',
            
            # 날짜 패턴
            r'(\d{4})년(\d{1,2})월(\d{1,2})일': r'\1년\2월\3일',
            
            # 금액 패턴
            r'(\d+)원': r'\1원',
            r'(\d+)만원': r'\1만원',
            r'(\d+)억원': r'\1억원'
        }
    
    def normalize_term(self, term: str) -> str:
        """용어 정규화"""
        if not term:
            return term
        
        # 기본 정리
        normalized = term.strip()
        
        # 용어 매핑 적용
        if normalized in self.term_mappings:
            normalized = self.term_mappings[normalized]
        
        # 패턴 정규화 적용
        for pattern, replacement in self.patterns.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
    
    def normalize_text(self, text: str) -> str:
        """텍스트 전체 정규화"""
        if not text:
            return text
        
        # 문장 단위로 분리
        sentences = re.split(r'[.!?]\s*', text)
        normalized_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                # 단어 단위로 분리하여 정규화
                words = sentence.split()
                normalized_words = [self.normalize_term(word) for word in words]
                normalized_sentences.append(' '.join(normalized_words))
        
        return '. '.join(normalized_sentences)
    
    def extract_legal_terms(self, text: str) -> List[LegalTerm]:
        """법률 용어 추출"""
        terms = []
        
        for original, normalized in self.term_mappings.items():
            if original in text:
                # 동의어 찾기
                synonyms = [k for k, v in self.term_mappings.items() if v == normalized and k != original]
                
                term = LegalTerm(
                    original=original,
                    normalized=normalized,
                    category=self._get_category(normalized),
                    synonyms=synonyms
                )
                terms.append(term)
        
        return terms
    
    def _get_category(self, term: str) -> str:
        """용어 카테고리 분류"""
        categories = {
            "법령": ["민법", "형법", "상법", "노동법", "행정법", "헌법"],
            "법원": ["대법원", "고등법원", "지방법원", "가정법원", "행정법원"],
            "절차": ["소송", "재판", "판결", "선고", "기각", "인용"],
            "계약": ["계약", "매매", "임대차", "도급", "위임", "위탁"],
            "손해": ["손해배상", "불법행위", "과실", "고의"],
            "형사": ["범죄", "처벌", "형벌", "벌금", "징역"],
            "행정": ["허가", "인허가", "신고", "신청", "처분", "제재"]
        }
        
        for category, terms_list in categories.items():
            if term in terms_list:
                return category
        
        return "기타"
    
    def get_similar_terms(self, term: str) -> List[str]:
        """유사 용어 검색"""
        normalized = self.normalize_term(term)
        similar = []
        
        for original, norm in self.term_mappings.items():
            if norm == normalized and original != term:
                similar.append(original)
        
        return similar
    
    def is_legal_term(self, term: str) -> bool:
        """법률 용어 여부 확인"""
        normalized = self.normalize_term(term)
        return normalized in self.term_mappings.values()
    
    def get_term_frequency(self, text: str) -> Dict[str, int]:
        """용어 빈도 계산"""
        frequency = {}
        
        for term in self.term_mappings.keys():
            count = text.count(term)
            if count > 0:
                frequency[term] = count
        
        return frequency
    
    def validate_legal_document(self, text: str) -> Dict[str, any]:
        """법률 문서 유효성 검사"""
        result = {
            "is_valid": True,
            "issues": [],
            "suggestions": [],
            "term_count": len(self.extract_legal_terms(text)),
            "normalized_text": self.normalize_text(text)
        }
        
        # 기본 검사
        if not text or len(text.strip()) < 10:
            result["is_valid"] = False
            result["issues"].append("문서가 너무 짧습니다")
        
        # 법률 용어 검사
        legal_terms = self.extract_legal_terms(text)
        if len(legal_terms) == 0:
            result["issues"].append("법률 용어가 발견되지 않았습니다")
            result["suggestions"].append("법률 관련 용어를 포함해보세요")
        
        return result

# 편의 함수들
def normalize_legal_text(text: str) -> str:
    """법률 텍스트 정규화 편의 함수"""
    normalizer = LegalTermNormalizer()
    return normalizer.normalize_text(text)

def extract_legal_terms(text: str) -> List[LegalTerm]:
    """법률 용어 추출 편의 함수"""
    normalizer = LegalTermNormalizer()
    return normalizer.extract_legal_terms(text)

def is_legal_document(text: str) -> bool:
    """법률 문서 여부 확인 편의 함수"""
    normalizer = LegalTermNormalizer()
    result = normalizer.validate_legal_document(text)
    return result["is_valid"] and result["term_count"] > 0
