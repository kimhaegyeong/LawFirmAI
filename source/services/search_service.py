# -*- coding: utf-8 -*-
"""
Search Service (ML Enhanced)
검색 서비스 - ML 강화 버전
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from data.database import DatabaseManager
from data.vector_store import LegalVectorStore as VectorStore
from models.model_manager import LegalModelManager
from utils.config import Config

logger = logging.getLogger(__name__)


class MLEnhancedSearchService:
    """ML 강화 검색 서비스 클래스"""
    
    def __init__(self, config: Config, database: DatabaseManager, 
                 vector_store: VectorStore, model_manager: LegalModelManager):
        """ML 강화 검색 서비스 초기화"""
        self.config = config
        self.database = database
        self.vector_store = vector_store
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        # ML 강화 검색 설정
        self.use_ml_enhanced_search = True
        self.quality_threshold = 0.7
        self.supplementary_weight = 0.8
        self.hybrid_weights = {
            'semantic': 0.6,
            'keyword': 0.3,
            'ml_quality': 0.1
        }
        
        self.logger.info("MLEnhancedSearchService initialized")
    
    def search_documents(self, query: str, search_type: str = "hybrid", 
                        limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """ML 강화 문서 검색"""
        try:
            if search_type == "semantic":
                return self._ml_enhanced_semantic_search(query, limit, filters)
            elif search_type == "keyword":
                return self._ml_enhanced_keyword_search(query, limit, filters)
            elif search_type == "hybrid":
                return self._ml_enhanced_hybrid_search(query, limit, filters)
            elif search_type == "supplementary":
                return self._search_supplementary_provisions(query, limit)
            elif search_type == "high_quality":
                return self._search_high_quality_documents(query, limit)
            else:
                raise ValueError(f"Unknown search type: {search_type}")
                
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def _ml_enhanced_semantic_search(self, query: str, limit: int, 
                                   filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """ML 강화 의미적 검색"""
        try:
            # 기본 필터 설정
            search_filters = filters or {}
            if self.use_ml_enhanced_search:
                search_filters['ml_enhanced'] = True
            
            # 벡터 저장소에서 검색
            similar_docs = self.vector_store.search(query, limit * 2, search_filters)
            
            # ML 강화 결과 필터링 및 스코어링
            filtered_docs = self._filter_and_score_documents(similar_docs, query)
            
            # 결과 포맷팅
            results = []
            for doc in filtered_docs[:limit]:
                metadata = doc.get("metadata", {})
                result = {
                    "document_id": metadata.get("document_id"),
                    "title": metadata.get("law_name", ""),
                    "content": doc.get("text", ""),
                    "similarity": doc.get("score", 0.0),
                    "search_type": "semantic",
                    "article_number": metadata.get("article_number", ""),
                    "article_title": metadata.get("article_title", ""),
                    "article_type": metadata.get("article_type", "main"),
                    "is_supplementary": metadata.get("is_supplementary", False),
                    "ml_confidence_score": metadata.get("ml_confidence_score"),
                    "parsing_method": metadata.get("parsing_method", "ml_enhanced"),
                    "quality_score": metadata.get("parsing_quality_score", 0.0),
                    "chunk_index": metadata.get("chunk_id", 0)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in ML-enhanced semantic search: {e}")
            return []
    
    def _ml_enhanced_keyword_search(self, query: str, limit: int, 
                                  filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """ML 강화 키워드 검색"""
        try:
            # 키워드 추출
            keywords = self._extract_keywords(query)
            if not keywords:
                return []
            
            # ML 강화 필터 추가
            ml_filters = []
            if self.use_ml_enhanced_search:
                ml_filters.append("ml_enhanced = 1")
            
            # 추가 필터 적용
            if filters:
                for key, value in filters.items():
                    if key == "article_type":
                        ml_filters.append(f"article_type = '{value}'")
                    elif key == "is_supplementary":
                        ml_filters.append(f"is_supplementary = {1 if value else 0}")
            
            # SQL 쿼리 구성
            keyword_conditions = []
            params = []
            
            for keyword in keywords:
                keyword_conditions.append("(law_name LIKE ? OR article_content LIKE ? OR article_title LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
            
            where_clause = " OR ".join(keyword_conditions)
            if ml_filters:
                where_clause += " AND " + " AND ".join(ml_filters)
            
            sql = f"""
                SELECT law_id, law_name, article_number, article_title, article_content,
                       article_type, is_supplementary, ml_confidence_score, parsing_method,
                       parsing_quality_score, word_count, char_count
                FROM assembly_articles
                WHERE {where_clause}
                ORDER BY parsing_quality_score DESC, word_count DESC
                LIMIT ?
            """
            params.append(limit)
            
            # 데이터베이스에서 검색
            rows = self.database.execute_query(sql, tuple(params))
            
            # 결과 포맷팅
            results = []
            for row in rows:
                matched_keywords = self._find_matched_keywords(
                    f"{row['law_name']} {row['article_content']}", keywords
                )
                
                result = {
                    "document_id": f"{row['law_id']}_article_{row['article_number']}",
                    "title": row["law_name"],
                    "content": row["article_content"],
                    "article_number": row["article_number"],
                    "article_title": row["article_title"],
                    "article_type": row["article_type"],
                    "is_supplementary": bool(row["is_supplementary"]),
                    "ml_confidence_score": row["ml_confidence_score"],
                    "parsing_method": row["parsing_method"],
                    "quality_score": row["parsing_quality_score"],
                    "word_count": row["word_count"],
                    "char_count": row["char_count"],
                    "search_type": "keyword",
                    "matched_keywords": matched_keywords
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in ML-enhanced keyword search: {e}")
            return []
    
    def _ml_enhanced_hybrid_search(self, query: str, limit: int, 
                                 filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """ML 강화 하이브리드 검색"""
        try:
            # 의미적 검색
            semantic_results = self._ml_enhanced_semantic_search(query, limit // 2, filters)
            
            # 키워드 검색
            keyword_results = self._ml_enhanced_keyword_search(query, limit // 2, filters)
            
            # 결과 병합 및 하이브리드 스코어링
            all_results = self._merge_and_score_results(semantic_results, keyword_results)
            
            # 중복 제거 (document_id 기준)
            seen_ids = set()
            unique_results = []
            
            for result in all_results:
                doc_id = result.get("document_id")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_results.append(result)
            
            # 하이브리드 스코어 기준 정렬
            unique_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
            
            return unique_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in ML-enhanced hybrid search: {e}")
            return []
    
    def _merge_and_score_results(self, semantic_results: List[Dict[str, Any]], 
                               keyword_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과 병합 및 하이브리드 스코어링"""
        try:
            all_results = []
            
            # 의미적 검색 결과에 하이브리드 스코어 추가
            for result in semantic_results:
                semantic_score = result.get("similarity", 0.0)
                quality_score = result.get("quality_score", 0.0)
                ml_confidence = result.get("ml_confidence_score", 0.5)
                
                # 하이브리드 스코어 계산
                hybrid_score = (
                    self.hybrid_weights['semantic'] * semantic_score +
                    self.hybrid_weights['ml_quality'] * quality_score +
                    self.hybrid_weights['keyword'] * ml_confidence
                )
                
                result["hybrid_score"] = hybrid_score
                result["search_method"] = "semantic"
                all_results.append(result)
            
            # 키워드 검색 결과에 하이브리드 스코어 추가
            for result in keyword_results:
                keyword_score = len(result.get("matched_keywords", [])) / 10.0  # 키워드 매칭 비율
                quality_score = result.get("quality_score", 0.0)
                ml_confidence = result.get("ml_confidence_score", 0.5)
                
                # 하이브리드 스코어 계산
                hybrid_score = (
                    self.hybrid_weights['keyword'] * keyword_score +
                    self.hybrid_weights['ml_quality'] * quality_score +
                    self.hybrid_weights['semantic'] * ml_confidence
                )
                
                result["hybrid_score"] = hybrid_score
                result["search_method"] = "keyword"
                all_results.append(result)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error merging and scoring results: {e}")
            return semantic_results + keyword_results
    
    def _filter_and_score_documents(self, documents: List[Dict[str, Any]], 
                                   query: str) -> List[Dict[str, Any]]:
        """ML 강화 문서 필터링 및 스코어링"""
        try:
            filtered_docs = []
            
            for doc in documents:
                metadata = doc.get("metadata", {})
                
                # 품질 필터링
                quality_score = metadata.get("parsing_quality_score", 0.0)
                if quality_score < self.quality_threshold:
                    continue
                
                # 부칙 가중치 적용
                is_supplementary = metadata.get("is_supplementary", False)
                if is_supplementary:
                    doc["score"] *= self.supplementary_weight
                
                # ML 신뢰도 점수 고려
                ml_confidence = metadata.get("ml_confidence_score")
                if ml_confidence is not None:
                    doc["score"] *= (0.5 + 0.5 * ml_confidence)
                
                filtered_docs.append(doc)
            
            # 스코어 기준으로 정렬
            filtered_docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            
            return filtered_docs
            
        except Exception as e:
            self.logger.error(f"Error filtering documents: {e}")
            return documents
    
    def _search_supplementary_provisions(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """부칙 조문 전용 검색"""
        try:
            filters = {"is_supplementary": True}
            return self._ml_enhanced_hybrid_search(query, limit, filters)
            
        except Exception as e:
            self.logger.error(f"Error searching supplementary provisions: {e}")
            return []
    
    def _search_high_quality_documents(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """고품질 문서 전용 검색"""
        try:
            # 임시로 품질 임계값을 높게 설정
            original_threshold = self.quality_threshold
            self.quality_threshold = 0.9
            
            results = self._ml_enhanced_hybrid_search(query, limit)
            
            # 원래 임계값 복원
            self.quality_threshold = original_threshold
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching high quality documents: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 키워드 추출 (ML 강화 버전)"""
        try:
            # 불용어 제거 (확장된 목록)
            stopwords = {
                '의', '을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로', 
                '와', '과', '도', '만', '부터', '까지', '에', '대해', '관련', '질문',
                '어떻게', '무엇', '언제', '어디', '왜', '어떤', '누구', '몇', '얼마',
                '법률', '법', '조문', '항', '호', '목', '단', '절', '장', '편'
            }
            
            # 단어 분리 (한글, 숫자, 영문 포함)
            words = re.findall(r'[가-힣0-9a-zA-Z]+', query)
            
            # 불용어 제거 및 길이 필터링
            keywords = [word for word in words if len(word) > 1 and word not in stopwords]
            
            return keywords[:15]  # 상위 15개로 확장
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _find_matched_keywords(self, content: str, keywords: List[str]) -> List[str]:
        """콘텐츠에서 매칭된 키워드 찾기 (ML 강화 버전)"""
        try:
            matched = []
            content_lower = content.lower()
            
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    matched.append(keyword)
            
            return matched
            
        except Exception as e:
            self.logger.error(f"Error finding matched keywords: {e}")
            return []
    
    def search_legal_entities(self, query: str) -> Dict[str, List[str]]:
        """ML 강화 법률 엔티티 검색"""
        try:
            # 법률명 패턴 (확장된 패턴)
            law_pattern = r'([가-힣]+법|([가-힣]+법률)|([가-힣]+규칙)|([가-힣]+령)|([가-힣]+고시))'
            laws = re.findall(law_pattern, query)
            
            # 조문 패턴 (항, 호, 목 포함)
            article_pattern = r'제(\d+)조(?:제(\d+)항)?(?:제(\d+)호)?(?:제(\d+)목)?'
            articles = re.findall(article_pattern, query)
            
            # 판례 패턴 (확장된 패턴)
            case_pattern = r'([가-힣]+[0-9]+[가-힣]*[0-9]*[가-힣]*[0-9]*[가-힣]*)'
            cases = re.findall(case_pattern, query)
            
            # 부칙 패턴
            supplementary_pattern = r'부칙(?:제(\d+)조)?'
            supplementary = re.findall(supplementary_pattern, query)
            
            return {
                "laws": [law[0] for law in laws if law[0]],
                "articles": [f"제{article[0]}조" + (f"제{article[1]}항" if article[1] else "") + 
                           (f"제{article[2]}호" if article[2] else "") + 
                           (f"제{article[3]}목" if article[3] else "") for article in articles],
                "cases": cases[:5],
                "supplementary": [f"부칙제{sup}조" if sup else "부칙" for sup in supplementary]
            }
            
        except Exception as e:
            self.logger.error(f"Error searching legal entities: {e}")
            return {"laws": [], "articles": [], "cases": [], "supplementary": []}
    
    def get_search_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """ML 강화 검색 제안"""
        try:
            suggestions = []
            
            # 키워드 기반 제안
            keywords = self._extract_keywords(query)
            for keyword in keywords[:3]:
                suggestions.append(f"{keyword} 관련 법률")
                suggestions.append(f"{keyword} 판례")
                suggestions.append(f"{keyword} 계약서")
                suggestions.append(f"{keyword} 부칙")
            
            # 법률 엔티티 기반 제안
            entities = self.search_legal_entities(query)
            if entities["laws"]:
                for law in entities["laws"][:2]:
                    suggestions.append(f"{law} 조문 검색")
                    suggestions.append(f"{law} 부칙 검색")
            
            if entities["articles"]:
                for article in entities["articles"][:2]:
                    suggestions.append(f"{article} 내용")
                    suggestions.append(f"{article} 해석")
            
            return suggestions[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting search suggestions: {e}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """ML 강화 검색 통계"""
        try:
            return {
                "ml_enhanced": True,
                "quality_threshold": self.quality_threshold,
                "supplementary_weight": self.supplementary_weight,
                "hybrid_weights": self.hybrid_weights,
                "use_ml_enhanced_search": self.use_ml_enhanced_search
            }
            
        except Exception as e:
            self.logger.error(f"Error getting search stats: {e}")
            return {"error": str(e)}


# 레거시 호환성을 위한 기존 SearchService 클래스
class SearchService(MLEnhancedSearchService):
    """레거시 호환성을 위한 검색 서비스 클래스"""
    
    def __init__(self, config: Config, database: DatabaseManager, 
                 vector_store: VectorStore, model_manager: LegalModelManager):
        """레거시 검색 서비스 초기화"""
        super().__init__(config, database, vector_store, model_manager)
        self.logger.info("Legacy SearchService initialized (using MLEnhancedSearchService)")
