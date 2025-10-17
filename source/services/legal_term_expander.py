# -*- coding: utf-8 -*-
"""
법률 용어 확장 검색 시스템
법률 전문 용어와 일반 용어 간 매핑으로 검색 품질 향상
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExpandedQuery:
    """확장된 쿼리 정보"""
    original: str
    expanded_terms: List[str]
    related_laws: List[str]
    precedent_keywords: List[str]
    synonyms: List[str]
    confidence: float


class LegalTermExpander:
    """법률 용어 확장기"""
    
    def __init__(self, dictionary_path: str = "data/legal_term_dictionary.json"):
        """법률 용어 확장기 초기화"""
        self.logger = logging.getLogger(__name__)
        self.dictionary_path = dictionary_path
        self.term_dict = {}
        self.reverse_dict = {}  # 동의어 -> 원래 용어 매핑
        
        # 법률 용어 사전 로드
        self._load_legal_terms()
        
        # 역방향 사전 구축
        self._build_reverse_dictionary()
        
        # 법률 조문 패턴
        self.law_patterns = [
            r'제\d+조',  # 제750조
            r'제\d+항',  # 제1항
            r'제\d+호',  # 제1호
            r'법률\s*제\d+호',  # 법률 제20883호
            r'시행령\s*제\d+조',  # 시행령 제1조
            r'시행규칙\s*제\d+조'  # 시행규칙 제1조
        ]
        
        # 판례 패턴
        self.precedent_patterns = [
            r'\d{4}[가나다라마바사아자차카타파하]\d+',  # 2023다12345
            r'대법원\s*\d{4}\.\d+\.\d+',  # 대법원 2023.1.1
            r'고등법원\s*\d{4}\.\d+\.\d+',  # 고등법원 2023.1.1
            r'지방법원\s*\d{4}\.\d+\.\d+'  # 지방법원 2023.1.1
        ]
    
    def _load_legal_terms(self):
        """법률 용어 사전 로드"""
        try:
            if Path(self.dictionary_path).exists():
                with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                    self.term_dict = json.load(f)
                self.logger.info(f"Loaded {len(self.term_dict)} legal terms from dictionary")
            else:
                self.logger.warning(f"Legal term dictionary not found at {self.dictionary_path}")
                self.term_dict = {}
                
        except Exception as e:
            self.logger.error(f"Error loading legal term dictionary: {e}")
            self.term_dict = {}
    
    def _build_reverse_dictionary(self):
        """역방향 사전 구축 (동의어 -> 원래 용어)"""
        try:
            for main_term, term_info in self.term_dict.items():
                # 동의어들에 대해 역방향 매핑
                synonyms = term_info.get("synonyms", [])
                for synonym in synonyms:
                    self.reverse_dict[synonym] = main_term
                
                # 관련 용어들에 대해서도 역방향 매핑
                related_terms = term_info.get("related_terms", [])
                for related_term in related_terms:
                    if related_term not in self.reverse_dict:
                        self.reverse_dict[related_term] = main_term
                        
        except Exception as e:
            self.logger.error(f"Error building reverse dictionary: {e}")
    
    def expand_query(self, query: str) -> ExpandedQuery:
        """
        쿼리 확장 실행
        
        Args:
            query: 원본 쿼리
            
        Returns:
            ExpandedQuery: 확장된 쿼리 정보
        """
        try:
            self.logger.info(f"Expanding query: {query[:100]}...")
            
            # 원본 쿼리에서 법률 용어 추출
            detected_terms = self._extract_legal_terms(query)
            
            # 확장된 용어들 수집
            expanded_terms = []
            related_laws = []
            precedent_keywords = []
            synonyms = []
            
            for term in detected_terms:
                if term in self.term_dict:
                    term_info = self.term_dict[term]
                    
                    # 동의어 추가
                    term_synonyms = term_info.get("synonyms", [])
                    synonyms.extend(term_synonyms)
                    
                    # 관련 용어 추가
                    related_terms = term_info.get("related_terms", [])
                    expanded_terms.extend(related_terms)
                    
                    # 관련 법률 추가
                    laws = term_info.get("related_laws", [])
                    related_laws.extend(laws)
                    
                    # 판례 키워드 추가
                    precedent_kw = term_info.get("precedent_keywords", [])
                    precedent_keywords.extend(precedent_kw)
            
            # 쿼리에서 직접 추출된 법률 조문과 판례 패턴
            extracted_laws = self._extract_law_patterns(query)
            extracted_precedents = self._extract_precedent_patterns(query)
            
            related_laws.extend(extracted_laws)
            precedent_keywords.extend(extracted_precedents)
            
            # 중복 제거
            expanded_terms = list(set(expanded_terms))
            related_laws = list(set(related_laws))
            precedent_keywords = list(set(precedent_keywords))
            synonyms = list(set(synonyms))
            
            # 신뢰도 계산
            confidence = self._calculate_expansion_confidence(
                detected_terms, expanded_terms, related_laws
            )
            
            result = ExpandedQuery(
                original=query,
                expanded_terms=expanded_terms,
                related_laws=related_laws,
                precedent_keywords=precedent_keywords,
                synonyms=synonyms,
                confidence=confidence
            )
            
            self.logger.info(f"Query expanded: {len(expanded_terms)} terms, "
                           f"{len(related_laws)} laws, {len(precedent_keywords)} precedents")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error expanding query: {e}")
            return ExpandedQuery(
                original=query,
                expanded_terms=[],
                related_laws=[],
                precedent_keywords=[],
                synonyms=[],
                confidence=0.0
            )
    
    def _extract_legal_terms(self, query: str) -> List[str]:
        """쿼리에서 법률 용어 추출"""
        try:
            detected_terms = []
            query_lower = query.lower()
            
            # 직접 매칭
            for term in self.term_dict.keys():
                if term in query_lower:
                    detected_terms.append(term)
            
            # 동의어 매칭
            for synonym, main_term in self.reverse_dict.items():
                if synonym in query_lower and main_term not in detected_terms:
                    detected_terms.append(main_term)
            
            # 부분 매칭 (긴 용어 우선)
            detected_terms.sort(key=len, reverse=True)
            
            return detected_terms
            
        except Exception as e:
            self.logger.error(f"Error extracting legal terms: {e}")
            return []
    
    def _extract_law_patterns(self, query: str) -> List[str]:
        """쿼리에서 법률 조문 패턴 추출"""
        try:
            extracted_laws = []
            
            for pattern in self.law_patterns:
                matches = re.findall(pattern, query)
                extracted_laws.extend(matches)
            
            return extracted_laws
            
        except Exception as e:
            self.logger.error(f"Error extracting law patterns: {e}")
            return []
    
    def _extract_precedent_patterns(self, query: str) -> List[str]:
        """쿼리에서 판례 패턴 추출"""
        try:
            extracted_precedents = []
            
            for pattern in self.precedent_patterns:
                matches = re.findall(pattern, query)
                extracted_precedents.extend(matches)
            
            return extracted_precedents
            
        except Exception as e:
            self.logger.error(f"Error extracting precedent patterns: {e}")
            return []
    
    def _calculate_expansion_confidence(self, 
                                      detected_terms: List[str], 
                                      expanded_terms: List[str], 
                                      related_laws: List[str]) -> float:
        """확장 신뢰도 계산"""
        try:
            if not detected_terms:
                return 0.0
            
            # 기본 점수
            base_score = len(detected_terms) / 10.0  # 최대 1.0
            
            # 확장된 용어가 많을수록 높은 점수
            expansion_bonus = min(len(expanded_terms) / 20.0, 0.3)
            
            # 관련 법률이 있을수록 높은 점수
            law_bonus = min(len(related_laws) / 10.0, 0.2)
            
            confidence = min(base_score + expansion_bonus + law_bonus, 1.0)
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating expansion confidence: {e}")
            return 0.0
    
    def get_expanded_search_terms(self, query: str) -> List[str]:
        """확장된 검색 용어 리스트 반환"""
        try:
            expanded_query = self.expand_query(query)
            
            # 모든 용어 통합
            all_terms = [expanded_query.original]
            all_terms.extend(expanded_query.expanded_terms)
            all_terms.extend(expanded_query.synonyms)
            all_terms.extend(expanded_query.related_laws)
            all_terms.extend(expanded_query.precedent_keywords)
            
            # 중복 제거 및 정리
            unique_terms = list(set(all_terms))
            unique_terms = [term.strip() for term in unique_terms if term.strip()]
            
            return unique_terms
            
        except Exception as e:
            self.logger.error(f"Error getting expanded search terms: {e}")
            return [query]
    
    def find_related_laws(self, terms: List[str]) -> List[str]:
        """용어들로부터 관련 법률 찾기"""
        try:
            related_laws = []
            
            for term in terms:
                if term in self.term_dict:
                    laws = self.term_dict[term].get("related_laws", [])
                    related_laws.extend(laws)
                elif term in self.reverse_dict:
                    main_term = self.reverse_dict[term]
                    if main_term in self.term_dict:
                        laws = self.term_dict[main_term].get("related_laws", [])
                        related_laws.extend(laws)
            
            return list(set(related_laws))
            
        except Exception as e:
            self.logger.error(f"Error finding related laws: {e}")
            return []
    
    def get_term_info(self, term: str) -> Optional[Dict[str, Any]]:
        """특정 용어의 정보 반환"""
        try:
            if term in self.term_dict:
                return self.term_dict[term]
            elif term in self.reverse_dict:
                main_term = self.reverse_dict[term]
                return self.term_dict.get(main_term)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting term info: {e}")
            return None
    
    def add_custom_term(self, term: str, term_info: Dict[str, Any]):
        """사용자 정의 용어 추가"""
        try:
            self.term_dict[term] = term_info
            
            # 동의어에 대한 역방향 매핑 추가
            synonyms = term_info.get("synonyms", [])
            for synonym in synonyms:
                self.reverse_dict[synonym] = term
            
            self.logger.info(f"Added custom term: {term}")
            
        except Exception as e:
            self.logger.error(f"Error adding custom term: {e}")
    
    def save_dictionary(self):
        """사전을 파일에 저장"""
        try:
            with open(self.dictionary_path, 'w', encoding='utf-8') as f:
                json.dump(self.term_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved legal term dictionary to {self.dictionary_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving dictionary: {e}")


# 테스트 함수
def test_legal_term_expander():
    """법률 용어 확장기 테스트"""
    expander = LegalTermExpander()
    
    test_queries = [
        "손해배상 청구 방법",
        "계약 해지 절차",
        "임대차 보증금 반환",
        "민법 제750조 불법행위",
        "2023다12345 판례",
        "교통사고 과실비율"
    ]
    
    print("=== 법률 용어 확장기 테스트 ===")
    for query in test_queries:
        print(f"\n원본 쿼리: {query}")
        
        expanded_query = expander.expand_query(query)
        
        print(f"확장된 용어: {expanded_query.expanded_terms}")
        print(f"동의어: {expanded_query.synonyms}")
        print(f"관련 법률: {expanded_query.related_laws}")
        print(f"판례 키워드: {expanded_query.precedent_keywords}")
        print(f"신뢰도: {expanded_query.confidence:.3f}")
        
        # 확장된 검색 용어
        search_terms = expander.get_expanded_search_terms(query)
        print(f"검색 용어 ({len(search_terms)}개): {search_terms[:5]}...")


if __name__ == "__main__":
    test_legal_term_expander()
