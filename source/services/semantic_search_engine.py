# -*- coding: utf-8 -*-
"""
Semantic Search Engine
의미적 검색을 위한 검색 엔진
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SemanticSearchResult:
    """의미적 검색 결과"""
    text: str
    score: float
    similarity_type: str
    metadata: Dict[str, Any]


class SemanticSearchEngine:
    """의미적 검색 엔진"""
    
    def __init__(self):
        """검색 엔진 초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("SemanticSearchEngine initialized")
    
    def search(self, query: str, documents: List[Dict[str, Any]], k: int = 10) -> List[SemanticSearchResult]:
        """
        의미적 검색 수행
        
        Args:
            query: 검색 쿼리
            documents: 검색할 문서 목록
            k: 반환할 결과 수
            
        Returns:
            List[SemanticSearchResult]: 검색 결과
        """
        results = []
        
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # 의미적 유사도 계산
            score = self._calculate_semantic_score(query, text)
            
            if score > 0.3:  # 임계값 설정
                similarity_type = self._determine_similarity_type(query, text)
                result = SemanticSearchResult(
                    text=text,
                    score=score,
                    similarity_type=similarity_type,
                    metadata=metadata
                )
                results.append(result)
        
        # 점수순 정렬 및 상위 k개 반환
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    def _calculate_semantic_score(self, query: str, text: str) -> float:
        """
        의미적 유사도 계산 (간단한 구현)
        
        Args:
            query: 검색 쿼리
            text: 검색할 텍스트
            
        Returns:
            float: 유사도 점수 (0.0-1.0)
        """
        # 실제 구현에서는 벡터 임베딩을 사용해야 함
        # 여기서는 간단한 키워드 기반 유사도 계산
        
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words or not text_words:
            return 0.0
        
        # Jaccard 유사도
        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))
        
        jaccard_score = intersection / union if union > 0 else 0.0
        
        # 추가 가중치 (법률 용어 매칭)
        legal_terms = {'계약', '해지', '손해배상', '이혼', '형사', '처벌'}
        legal_matches = len(query_words.intersection(legal_terms))
        legal_bonus = min(0.2, legal_matches * 0.1)
        
        return min(1.0, jaccard_score + legal_bonus)
    
    def _determine_similarity_type(self, query: str, text: str) -> str:
        """
        유사도 타입 결정
        
        Args:
            query: 검색 쿼리
            text: 검색할 텍스트
            
        Returns:
            str: 유사도 타입
        """
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        intersection = len(query_words.intersection(text_words))
        
        if intersection == len(query_words):
            return "high_similarity"
        elif intersection >= len(query_words) * 0.7:
            return "medium_similarity"
        else:
            return "low_similarity"


# 기본 인스턴스 생성
def create_semantic_search_engine() -> SemanticSearchEngine:
    """기본 의미적 검색 엔진 생성"""
    return SemanticSearchEngine()


if __name__ == "__main__":
    # 테스트 코드
    engine = create_semantic_search_engine()
    
    # 샘플 문서
    documents = [
        {"text": "민법 제543조 계약의 해지에 관한 규정", "metadata": {"category": "civil"}},
        {"text": "형법 제250조 살인죄의 구성요건", "metadata": {"category": "criminal"}},
        {"text": "가족법상 이혼 절차 및 요건", "metadata": {"category": "family"}}
    ]
    
    # 검색 테스트
    results = engine.search("계약 해지", documents)
    print(f"Semantic search results: {len(results)}")
    for result in results:
        print(f"  Score: {result.score:.3f}, Type: {result.similarity_type}")
        print(f"  Text: {result.text}")