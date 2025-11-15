# -*- coding: utf-8 -*-
"""
검색 쿼리 다변화 유틸리티
문서 타입별 최적화된 검색 쿼리 생성
"""

import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class QueryDiversifier:
    """검색 쿼리 다변화 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def diversify_search_queries(self, query: str) -> Dict[str, List[str]]:
        """
        문서 타입별 검색 쿼리 생성
        
        Args:
            query: 원본 쿼리
            
        Returns:
            Dict[str, List[str]]: 타입별 쿼리 리스트
        """
        queries = {
            "statute": [],
            "case": [],
            "decision": [],
            "interpretation": [],
            "general": [query]  # 원본 쿼리
        }
        
        # 법령 조문 검색 쿼리 (더 다양하게)
        queries["statute"] = [
            f"{query} 법령",
            f"{query} 조문",
            f"{query} 법률",
            f"법령 {query}",
            f"조문 {query}",
            f"법률 {query}",
            self._extract_legal_terms(query)
        ]
        
        # 판례 검색 쿼리 (더 다양하게)
        queries["case"] = [
            f"{query} 판례",
            f"{query} 사례",
            f"{query} 판결",
            f"판례 {query}",
            f"대법원 {query}",
            f"법원 {query}"
        ]
        
        # 결정례 검색 쿼리 (더 다양하게)
        queries["decision"] = [
            f"{query} 결정례",
            f"{query} 헌재 결정",
            f"{query} 결정",
            f"결정례 {query}",
            f"헌법재판소 {query}",
            f"헌재 {query}"
        ]
        
        # 해석례 검색 쿼리 (더 다양하게)
        queries["interpretation"] = [
            f"{query} 해석례",
            f"{query} 법령 해석",
            f"{query} 해석",
            f"해석례 {query}",
            f"법령해석 {query}",
            f"법령 해석 {query}"
        ]
        
        # 중복 제거
        for key in queries:
            queries[key] = list(dict.fromkeys(queries[key]))  # 순서 유지하면서 중복 제거
        
        return queries
    
    def _extract_legal_terms(self, query: str) -> str:
        """법률 용어 추출 및 확장"""
        # 간단한 법률 용어 매핑
        legal_term_mapping = {
            "전세금": ["전세금", "전세", "임대차"],
            "반환": ["반환", "반납", "회수"],
            "보증": ["보증", "담보", "보장"],
            "계약": ["계약", "계약서", "약정"],
            "해지": ["해지", "해제", "취소"],
            "손해배상": ["손해배상", "배상", "손해"],
            "소유권": ["소유권", "소유", "권리"],
            "임대차": ["임대차", "임대", "전세"]
        }
        
        # 쿼리에서 법률 용어 찾기
        found_terms = []
        for term, synonyms in legal_term_mapping.items():
            if term in query:
                found_terms.extend(synonyms[:2])  # 최대 2개만
        
        if found_terms:
            return " ".join(found_terms[:3])  # 최대 3개 용어 조합
        
        return query
    
    def generate_type_specific_queries(self, query: str, doc_type: str) -> List[str]:
        """
        특정 문서 타입에 최적화된 쿼리 생성
        
        Args:
            query: 원본 쿼리
            doc_type: 문서 타입 (statute_article, case_paragraph 등)
            
        Returns:
            List[str]: 최적화된 쿼리 리스트
        """
        type_mapping = {
            "statute_article": ["법령", "조문", "법률"],
            "case_paragraph": ["판례", "사례", "판결"],
            "decision_paragraph": ["결정례", "헌재 결정"],
            "interpretation_paragraph": ["해석례", "법령 해석"]
        }
        
        keywords = type_mapping.get(doc_type, [])
        queries = [query]  # 원본 쿼리 포함
        
        for keyword in keywords:
            queries.append(f"{query} {keyword}")
        
        return list(dict.fromkeys(queries))  # 중복 제거

