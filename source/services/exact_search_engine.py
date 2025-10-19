# -*- coding: utf-8 -*-
"""
Exact Search Engine
정확한 매칭을 위한 검색 엔진
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExactSearchResult:
    """정확한 검색 결과"""
    text: str
    score: float
    match_type: str
    metadata: Dict[str, Any]


class ExactSearchEngine:
    """정확한 검색 엔진"""
    
    def __init__(self):
        """검색 엔진 초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("ExactSearchEngine initialized")
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        쿼리를 파싱하여 검색에 필요한 정보를 추출합니다.
        Args:
            query: 검색 쿼리
        Returns:
            Dict[str, Any]: 파싱된 쿼리 정보
        """
        return {
            "original_query": query,
            "keywords": query.split(),
            "search_type": "exact"
        }
    
    def search(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[ExactSearchResult]:
        """
        정확한 검색 수행
        
        Args:
            query: 검색 쿼리
            documents: 검색할 문서 목록
            top_k: 반환할 결과 수
            
        Returns:
            List[ExactSearchResult]: 검색 결과
        """
        results = []
        
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # 정확한 매칭 점수 계산
            score = self._calculate_exact_score(query, text)
            
            if score > 0:
                match_type = self._determine_match_type(query, text)
                result = ExactSearchResult(
                    text=text,
                    score=score,
                    match_type=match_type,
                    metadata=metadata
                )
                results.append(result)
        
        # 점수순 정렬 및 상위 k개 반환
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def search_laws(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """법령 검색"""
        # 임시 구현 - 실제로는 데이터베이스에서 검색
        self.logger.info(f"Searching laws for query: {query}")
        return []
    
    def search_precedents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """판례 검색"""
        # 임시 구현 - 실제로는 데이터베이스에서 검색
        self.logger.info(f"Searching precedents for query: {query}")
        return [][:top_k]
    
    def _calculate_exact_score(self, query: str, text: str) -> float:
        """
        정확한 매칭 점수 계산
        
        Args:
            query: 검색 쿼리
            text: 검색할 텍스트
            
        Returns:
            float: 매칭 점수 (0.0-1.0)
        """
        query_lower = query.lower()
        text_lower = text.lower()
        
        # 완전 일치
        if query_lower == text_lower:
            return 1.0
        
        # 부분 일치
        if query_lower in text_lower:
            # 일치 비율 계산
            match_ratio = len(query_lower) / len(text_lower)
            return min(0.9, match_ratio)
        
        # 단어별 일치
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        
        if query_words.issubset(text_words):
            word_match_ratio = len(query_words) / len(text_words)
            return min(0.8, word_match_ratio)
        
        return 0.0
    
    def _determine_match_type(self, query: str, text: str) -> str:
        """
        매칭 타입 결정
        
        Args:
            query: 검색 쿼리
            text: 검색할 텍스트
            
        Returns:
            str: 매칭 타입
        """
        query_lower = query.lower()
        text_lower = text.lower()
        
        if query_lower == text_lower:
            return "exact"
        elif query_lower in text_lower:
            return "partial"
        else:
            return "word_match"


# 기본 인스턴스 생성
def create_exact_search_engine() -> ExactSearchEngine:
    """기본 정확한 검색 엔진 생성"""
    return ExactSearchEngine()


if __name__ == "__main__":
    # 테스트 코드
    engine = create_exact_search_engine()
    
    # 샘플 문서
    documents = [
        {"text": "민법 제543조 계약의 해지", "metadata": {"category": "civil"}},
        {"text": "형법 제250조 살인", "metadata": {"category": "criminal"}},
        {"text": "가족법 이혼 절차", "metadata": {"category": "family"}}
    ]
    
    # 검색 테스트
    results = engine.search("계약 해지", documents)
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"  Score: {result.score:.3f}, Type: {result.match_type}")
        print(f"  Text: {result.text}")