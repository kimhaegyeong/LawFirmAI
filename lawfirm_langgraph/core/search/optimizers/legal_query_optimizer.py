# -*- coding: utf-8 -*-
"""
법률 쿼리 최적화기 (HuggingFace 모델 기반)
프롬프트 3 대체: 쿼리 최적화 (optimized_query, semantic_query, keyword_query)
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import numpy as np
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = get_logger(__name__)


class LegalQueryOptimizer:
    """법률 쿼리 최적화기 (HuggingFace 모델 사용)"""
    
    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        LegalQueryOptimizer 초기화
        
        Args:
            embedding_model_name: 임베딩 모델명
            logger: 로거
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # 임베딩 모델 초기화
        self.embedding_model = None
        self.embedding_model_name = embedding_model_name or "woong0322/ko-legal-sbert-finetuned"
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.logger.trace(f"✅ [HF MODEL] Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                self.logger.warning(f"⚠️ [HF MODEL] Failed to load {self.embedding_model_name}: {e}")
                try:
                    self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                    self.logger.info("✅ [HF MODEL] Using fallback embedding model")
                except Exception as e2:
                    self.logger.error(f"❌ [HF MODEL] Failed to load fallback model: {e2}")
    
    def optimize_query(
        self,
        query: str,
        core_keywords: List[str],
        expanded_keywords: List[str],
        query_type: str
    ) -> Dict[str, Any]:
        """
        쿼리 최적화 (프롬프트 3 대체)
        
        Args:
            query: 원본 쿼리
            core_keywords: 핵심 키워드
            expanded_keywords: 확장된 키워드
            query_type: 질문 유형
            
        Returns:
            {
                "optimized_query": "최적화 쿼리 (50자 이내)",
                "semantic_query": "벡터 검색용",
                "keyword_query": "키워드 검색용"
            }
        """
        # 1. 벡터 검색용 쿼리 생성
        semantic_query = self._build_semantic_query(
            query, core_keywords, expanded_keywords
        )
        
        # 2. 키워드 검색용 쿼리 생성
        keyword_query = self._build_keyword_query(core_keywords, expanded_keywords)
        
        # 3. 최적화된 쿼리 (50자 이내)
        optimized_query = self._optimize_query_length(semantic_query, max_length=50)
        
        # 4. 품질 점수 계산
        quality_score = self._calculate_query_quality_score(
            optimized_query, semantic_query, keyword_query, query
        )
        
        return {
            "optimized_query": optimized_query,
            "semantic_query": semantic_query,
            "keyword_query": keyword_query,
            "keyword_queries": [keyword_query] + core_keywords[:4],
            "quality_score": quality_score,
            "llm_enhanced": False
        }
    
    def _build_semantic_query(
        self,
        query: str,
        core_keywords: List[str],
        expanded_keywords: List[str]
    ) -> str:
        """의미적 쿼리 생성 (임베딩 기반)"""
        if not self.embedding_model:
            # 폴백: 키워드 조합
            all_keywords = (core_keywords + expanded_keywords[:5])[:5]
            return ' '.join(all_keywords) if all_keywords else query
        
        try:
            # 핵심 키워드와 확장 키워드의 임베딩
            all_keywords = core_keywords + expanded_keywords[:10]
            if not all_keywords:
                return query
            
            keyword_embeddings = self.embedding_model.encode(all_keywords)
            
            # 원본 쿼리 임베딩
            query_embedding = self.embedding_model.encode([query])[0]
            
            # 유사도 기반 키워드 선택
            similarities = []
            for i, kw_emb in enumerate(keyword_embeddings):
                similarity = np.dot(query_embedding, kw_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(kw_emb)
                )
                similarities.append((similarity, all_keywords[i]))
            
            # 상위 키워드 선택 (최대 5개)
            similarities.sort(reverse=True)
            selected_keywords = [kw for _, kw in similarities[:5]]
            
            # 의미 보존을 위해 원본 쿼리의 핵심 단어 포함
            if selected_keywords:
                return ' '.join(selected_keywords)
            else:
                return query
                
        except Exception as e:
            self.logger.debug(f"Semantic query building failed: {e}")
            # 폴백
            all_keywords = (core_keywords + expanded_keywords[:5])[:5]
            return ' '.join(all_keywords) if all_keywords else query
    
    def _build_keyword_query(
        self,
        core_keywords: List[str],
        expanded_keywords: List[str]
    ) -> str:
        """키워드 검색용 쿼리 생성"""
        # 핵심 키워드 우선, 확장 키워드 보조
        all_keywords = core_keywords[:3] + expanded_keywords[:2]
        return ' '.join(all_keywords) if all_keywords else ""
    
    def _optimize_query_length(self, query: str, max_length: int = 50) -> str:
        """쿼리 길이 최적화"""
        if len(query) <= max_length:
            return query
        
        # 단어 단위로 자르기
        words = query.split()
        optimized = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length:
                optimized.append(word)
                current_length += len(word) + 1
            else:
                break
        
        return ' '.join(optimized) if optimized else query[:max_length]
    
    def _calculate_query_quality_score(
        self,
        optimized_query: str,
        semantic_query: str,
        keyword_query: str,
        original_query: str
    ) -> float:
        """쿼리 품질 점수 계산"""
        score = 0.0
        
        # 1. 길이 적절성 (30-50자)
        if 30 <= len(optimized_query) <= 50:
            score += 0.3
        elif 20 <= len(optimized_query) < 30 or 50 < len(optimized_query) <= 60:
            score += 0.2
        else:
            score += 0.1
        
        # 2. 키워드 포함 여부
        if semantic_query and keyword_query:
            score += 0.3
        elif semantic_query or keyword_query:
            score += 0.2
        
        # 3. 원본 쿼리와의 유사성 (간단한 단어 매칭)
        original_words = set(original_query.split())
        optimized_words = set(optimized_query.split())
        overlap = len(original_words & optimized_words) / max(len(original_words), 1)
        score += overlap * 0.4
        
        return min(score, 1.0)

