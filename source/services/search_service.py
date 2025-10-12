# -*- coding: utf-8 -*-
"""
Search Service
검색 서비스
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore as VectorStore
from source.models.model_manager import LegalModelManager
from source.utils.config import Config

logger = logging.getLogger(__name__)


class SearchService:
    """검색 서비스 클래스"""
    
    def __init__(self, config: Config, database: DatabaseManager, 
                 vector_store: VectorStore, model_manager: LegalModelManager):
        """검색 서비스 초기화"""
        self.config = config
        self.database = database
        self.vector_store = vector_store
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("SearchService initialized")
    
    def search_documents(self, query: str, search_type: str = "semantic", 
                        limit: int = 10) -> List[Dict[str, Any]]:
        """문서 검색"""
        try:
            if search_type == "semantic":
                return self._semantic_search(query, limit)
            elif search_type == "keyword":
                return self._keyword_search(query, limit)
            elif search_type == "hybrid":
                return self._hybrid_search(query, limit)
            else:
                raise ValueError(f"Unknown search type: {search_type}")
                
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def _semantic_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """의미적 검색"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.model_manager.encode_text(query)
            if not query_embedding:
                return []
            
            # 벡터 저장소에서 검색
            import numpy as np
            query_array = np.array(query_embedding)
            similar_docs = self.vector_store.search_similar(query_array, limit)
            
            # 결과 포맷팅
            results = []
            for idx, similarity, metadata in similar_docs:
                result = {
                    "document_id": metadata.get("document_id"),
                    "title": metadata.get("title", ""),
                    "content": metadata.get("content", ""),
                    "similarity": similarity,
                    "search_type": "semantic",
                    "chunk_index": metadata.get("chunk_index", 0)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []
    
    def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """키워드 검색"""
        try:
            # 키워드 추출
            keywords = self._extract_keywords(query)
            if not keywords:
                return []
            
            # SQL 쿼리 구성
            keyword_conditions = []
            params = []
            
            for keyword in keywords:
                keyword_conditions.append("(title LIKE ? OR content LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%"])
            
            where_clause = " OR ".join(keyword_conditions)
            sql = f"""
                SELECT id, title, content, document_type, created_at
                FROM legal_documents
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
            """
            params.append(limit)
            
            # 데이터베이스에서 검색
            rows = self.database.execute_query(sql, tuple(params))
            
            # 결과 포맷팅
            results = []
            for row in rows:
                result = {
                    "document_id": row["id"],
                    "title": row["title"],
                    "content": row["content"],
                    "document_type": row["document_type"],
                    "created_at": row["created_at"],
                    "search_type": "keyword",
                    "matched_keywords": self._find_matched_keywords(row["content"], keywords)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in keyword search: {e}")
            return []
    
    def _hybrid_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """하이브리드 검색 (의미적 + 키워드)"""
        try:
            # 의미적 검색
            semantic_results = self._semantic_search(query, limit // 2)
            
            # 키워드 검색
            keyword_results = self._keyword_search(query, limit // 2)
            
            # 결과 병합 및 정렬
            all_results = semantic_results + keyword_results
            
            # 중복 제거 (document_id 기준)
            seen_ids = set()
            unique_results = []
            
            for result in all_results:
                doc_id = result.get("document_id")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_results.append(result)
            
            # 유사도 기준 정렬
            unique_results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
            
            return unique_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 키워드 추출"""
        try:
            # 불용어 제거
            stopwords = {'의', '을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로', '와', '과', '도', '만', '부터', '까지', '에', '대해', '관련', '질문'}
            
            # 단어 분리
            words = re.findall(r'[가-힣]+', query)
            
            # 불용어 제거 및 길이 필터링
            keywords = [word for word in words if len(word) > 1 and word not in stopwords]
            
            return keywords[:10]  # 상위 10개만
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _find_matched_keywords(self, content: str, keywords: List[str]) -> List[str]:
        """콘텐츠에서 매칭된 키워드 찾기"""
        try:
            matched = []
            for keyword in keywords:
                if keyword in content:
                    matched.append(keyword)
            return matched
            
        except Exception as e:
            self.logger.error(f"Error finding matched keywords: {e}")
            return []
    
    def search_legal_entities(self, query: str) -> Dict[str, List[str]]:
        """법률 엔티티 검색"""
        try:
            # 법률명 패턴
            law_pattern = r'([가-힣]+법|([가-힣]+법률))'
            laws = re.findall(law_pattern, query)
            
            # 조문 패턴
            article_pattern = r'제(\d+)조'
            articles = re.findall(article_pattern, query)
            
            # 판례 패턴
            case_pattern = r'([가-힣]+[0-9]+[가-힣]*[0-9]*[가-힣]*)'
            cases = re.findall(case_pattern, query)
            
            return {
                "laws": [law[0] for law in laws],
                "articles": [f"제{article}조" for article in articles],
                "cases": cases[:5]  # 상위 5개만
            }
            
        except Exception as e:
            self.logger.error(f"Error searching legal entities: {e}")
            return {"laws": [], "articles": [], "cases": []}
    
    def get_search_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """검색 제안"""
        try:
            # 간단한 제안 로직 (실제로는 더 복잡한 알고리즘 사용)
            suggestions = []
            
            # 키워드 기반 제안
            keywords = self._extract_keywords(query)
            for keyword in keywords[:3]:
                suggestions.append(f"{keyword} 관련 법률")
                suggestions.append(f"{keyword} 판례")
                suggestions.append(f"{keyword} 계약서")
            
            return suggestions[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting search suggestions: {e}")
            return []
