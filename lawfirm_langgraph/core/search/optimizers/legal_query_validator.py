# -*- coding: utf-8 -*-
"""
법률 쿼리 검증기 (HuggingFace 모델 기반)
프롬프트 4 대체: 쿼리 검증 (is_valid, quality_score, improvements)
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import numpy as np
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None

try:
    from lawfirm_langgraph.core.shared.utils.model_cache_manager import get_model_cache_manager
except ImportError:
    try:
        from core.shared.utils.model_cache_manager import get_model_cache_manager
    except ImportError:
        get_model_cache_manager = None

logger = get_logger(__name__)


class LegalQueryValidator:
    """법률 쿼리 검증기 (HuggingFace 모델 사용)"""
    
    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        LegalQueryValidator 초기화
        
        Args:
            embedding_model_name: 임베딩 모델명
            logger: 로거
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # 임베딩 모델 초기화 (캐시 매니저 사용)
        self.embedding_model = None
        self.embedding_model_name = embedding_model_name or "woong0322/ko-legal-sbert-finetuned"
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # ModelCacheManager 사용 (항상 시도)
            if get_model_cache_manager:
                try:
                    model_cache = get_model_cache_manager()
                    self.embedding_model = model_cache.get_model(
                        self.embedding_model_name,
                        fallback_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )
                    if self.embedding_model:
                        self.logger.trace(f"✅ [HF MODEL] Loaded embedding model (cached): {self.embedding_model_name}")
                    else:
                        self.logger.warning(f"⚠️ [HF MODEL] Failed to load {self.embedding_model_name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ [HF MODEL] Failed to load via cache manager: {e}")
                    # 캐시 매니저 실패 시 직접 로드하지 않음 (중복 로딩 방지)
                    self.embedding_model = None
            else:
                # get_model_cache_manager가 없으면 직접 로드 (최후의 수단)
                try:
                    self.embedding_model = SentenceTransformer(self.embedding_model_name)
                    self.logger.trace(f"✅ [HF MODEL] Loaded embedding model (direct): {self.embedding_model_name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ [HF MODEL] Failed to load {self.embedding_model_name}: {e}")
                    try:
                        self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                        self.logger.info("✅ [HF MODEL] Using fallback embedding model")
                    except Exception as e2:
                        self.logger.error(f"❌ [HF MODEL] Failed to load fallback model: {e2}")
                        self.embedding_model = None
        
        # Cross-Encoder 초기화 (검증용)
        self.validator_model = None
        if CrossEncoder:
            try:
                self.validator_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.logger.info("✅ [HF MODEL] Loaded CrossEncoder for validation")
            except Exception as e:
                self.logger.warning(f"⚠️ [HF MODEL] Failed to load CrossEncoder: {e}")
    
    def validate_query(
        self,
        optimized_queries: Dict[str, Any],
        original_query: str
    ) -> Dict[str, Any]:
        """
        쿼리 검증 (프롬프트 4 대체)
        
        Args:
            optimized_queries: 최적화된 쿼리 딕셔너리
            original_query: 원본 쿼리
            
        Returns:
            {
                "is_valid": True/False,
                "quality_score": 0.0-1.0,
                "improvements": ["개선 제안1", ...],
                "final_reasoning": "검증 결과"
            }
        """
        semantic_query = optimized_queries.get("semantic_query", "")
        keyword_query = optimized_queries.get("keyword_query", "")
        
        # 1. 의도 유지 확인
        intent_preserved = self._check_intent_preservation(original_query, semantic_query)
        
        # 2. 검색 정확도 향상 확인
        accuracy_score = self._check_accuracy_improvement(
            original_query, semantic_query, keyword_query
        )
        
        # 3. 검색 범위 적절성 확인
        range_score = self._check_search_range(original_query, keyword_query)
        
        # 4. 법률 전문성 확인
        professional_score = self._check_legal_professionalism(semantic_query)
        
        # 5. 종합 품질 점수
        quality_score = (
            intent_preserved * 0.3 +
            accuracy_score * 0.3 +
            range_score * 0.2 +
            professional_score * 0.2
        )
        
        # 6. 개선 제안 생성
        improvements = self._generate_improvements(
            intent_preserved, accuracy_score, range_score, professional_score
        )
        
        return {
            "is_valid": quality_score >= 0.6,
            "quality_score": quality_score,
            "improvements": improvements,
            "final_reasoning": f"HuggingFace 모델 기반 검증 완료 (점수: {quality_score:.2f})"
        }
    
    def _check_intent_preservation(
        self,
        original: str,
        optimized: str
    ) -> float:
        """의도 유지 확인 (임베딩 유사도)"""
        if not self.embedding_model or not optimized:
            return 0.5  # 중간 점수
        
        try:
            orig_emb = self.embedding_model.encode([original])[0]
            opt_emb = self.embedding_model.encode([optimized])[0]
            
            similarity = np.dot(orig_emb, opt_emb) / (
                np.linalg.norm(orig_emb) * np.linalg.norm(opt_emb)
            )
            
            return float(similarity)
        except Exception as e:
            self.logger.debug(f"Intent preservation check failed: {e}")
            return 0.5
    
    def _check_accuracy_improvement(
        self,
        original: str,
        semantic: str,
        keyword: str
    ) -> float:
        """검색 정확도 향상 확인"""
        if not semantic and not keyword:
            return 0.3
        
        # Cross-Encoder 사용 (가능한 경우)
        if self.validator_model:
            try:
                pairs = []
                if semantic:
                    pairs.append([original, semantic])
                if keyword:
                    pairs.append([original, keyword])
                
                if pairs:
                    scores = self.validator_model.predict(pairs)
                    return float(np.mean(scores))
            except Exception as e:
                self.logger.debug(f"CrossEncoder validation failed: {e}")
        
        # 폴백: 임베딩 기반 유사도
        if self.embedding_model:
            try:
                orig_emb = self.embedding_model.encode([original])[0]
                scores = []
                
                if semantic:
                    sem_emb = self.embedding_model.encode([semantic])[0]
                    similarity = np.dot(orig_emb, sem_emb) / (
                        np.linalg.norm(orig_emb) * np.linalg.norm(sem_emb)
                    )
                    scores.append(similarity)
                
                if keyword:
                    kw_emb = self.embedding_model.encode([keyword])[0]
                    similarity = np.dot(orig_emb, kw_emb) / (
                        np.linalg.norm(orig_emb) * np.linalg.norm(kw_emb)
                    )
                    scores.append(similarity)
                
                if scores:
                    return float(np.mean(scores))
            except Exception as e:
                self.logger.debug(f"Embedding-based accuracy check failed: {e}")
        
        return 0.5
    
    def _check_search_range(
        self,
        original: str,
        keyword_query: str
    ) -> float:
        """검색 범위 적절성 확인"""
        if not keyword_query:
            return 0.3
        
        # 원본 쿼리와 키워드 쿼리의 단어 수 비교
        original_words = len(original.split())
        keyword_words = len(keyword_query.split())
        
        # 적절한 확장 (1.2-2.0배)
        if 1.2 <= keyword_words / max(original_words, 1) <= 2.0:
            return 1.0
        elif 1.0 <= keyword_words / max(original_words, 1) < 1.2:
            return 0.7
        elif 2.0 < keyword_words / max(original_words, 1) <= 3.0:
            return 0.6
        else:
            return 0.4
    
    def _check_legal_professionalism(self, semantic_query: str) -> float:
        """법률 전문성 확인"""
        if not semantic_query:
            return 0.3
        
        # 법률 용어 패턴
        legal_terms = [
            "계약", "해지", "손해", "배상", "소송", "판례", "법령",
            "조문", "규정", "의무", "권리", "책임", "법률"
        ]
        
        query_lower = semantic_query.lower()
        found_terms = sum(1 for term in legal_terms if term in query_lower)
        
        # 법률 용어 비율
        if found_terms >= 2:
            return 1.0
        elif found_terms == 1:
            return 0.7
        else:
            return 0.4
    
    def _generate_improvements(
        self,
        intent_preserved: float,
        accuracy_score: float,
        range_score: float,
        professional_score: float
    ) -> List[str]:
        """개선 제안 생성"""
        improvements = []
        
        if intent_preserved < 0.6:
            improvements.append("원본 쿼리의 핵심 의도를 더 명확히 반영하세요")
        
        if accuracy_score < 0.6:
            improvements.append("검색 정확도를 높이기 위해 키워드를 구체화하세요")
        
        if range_score < 0.6:
            improvements.append("검색 범위를 적절히 조정하세요")
        
        if professional_score < 0.6:
            improvements.append("법률 전문 용어를 추가하세요")
        
        return improvements

