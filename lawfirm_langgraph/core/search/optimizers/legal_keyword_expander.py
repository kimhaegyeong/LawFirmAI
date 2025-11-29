# -*- coding: utf-8 -*-
"""
법률 키워드 확장기 (HuggingFace 모델 기반)
프롬프트 2 대체: 키워드 확장 (동의어, 유사어, 법률 용어)
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

try:
    from lawfirm_langgraph.core.shared.utils.model_cache_manager import get_model_cache_manager
except ImportError:
    try:
        from core.shared.utils.model_cache_manager import get_model_cache_manager
    except ImportError:
        get_model_cache_manager = None

logger = get_logger(__name__)


class LegalKeywordExpander:
    """법률 키워드 확장기 (HuggingFace 모델 사용)"""
    
    def __init__(
        self,
        term_integrator: Optional[Any] = None,
        embedding_model_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        LegalKeywordExpander 초기화
        
        Args:
            term_integrator: 법률 용어 통합기
            embedding_model_name: 임베딩 모델명
            logger: 로거
        """
        self.logger = logger or logging.getLogger(__name__)
        self.term_integrator = term_integrator
        
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
                        self.logger.debug("✅ [HF MODEL] Using fallback embedding model")
                    except Exception as e2:
                        self.logger.error(f"❌ [HF MODEL] Failed to load fallback model: {e2}")
                        self.embedding_model = None
        
        # 법률 도메인 동의어 사전
        self.legal_synonyms_dict = {
            "계약": ["계약서", "계약관계", "계약체결", "계약서작성"],
            "해지": ["해제", "취소", "해약", "계약종료"],
            "손해": ["손해배상", "손해전보", "배상책임", "손해보상", "손해배상청구"],
            "불법행위": ["불법", "위법행위", "불법적 행위", "불법행위책임"],
            "소송": ["소송절차", "소송요건", "소송제기", "소송비용"],
            "판례": ["판결", "사례", "재판례", "대법원판례"],
            "법령": ["법률", "조문", "규정", "법조문"]
        }
        
        # 법률 용어 임베딩 캐시
        self._legal_terms_cache: Dict[str, np.ndarray] = {}
    
    def expand_keywords(
        self,
        query: str,
        core_keywords: List[str],
        extracted_keywords: List[str],
        legal_field: str
    ) -> Dict[str, Any]:
        """
        키워드 확장 (프롬프트 2 대체)
        
        Args:
            query: 원본 쿼리
            core_keywords: 핵심 키워드
            extracted_keywords: 추출된 키워드
            legal_field: 법률 분야
            
        Returns:
            {
                "expanded_keywords": ["확장키워드1", ...],
                "synonyms": ["동의어1", ...],
                "keyword_variants": ["변형1", ...],
                "legal_references": ["법령1", ...]
            }
        """
        if not core_keywords and not extracted_keywords:
            return {
                "expanded_keywords": [],
                "synonyms": [],
                "keyword_variants": [],
                "legal_references": []
            }
        
        all_keywords = list(set(core_keywords + extracted_keywords))
        
        # 1. 동의어 확장
        synonyms = self._expand_synonyms(all_keywords, legal_field)
        
        # 2. 키워드 변형 생성
        keyword_variants = self._generate_keyword_variants(all_keywords)
        
        # 3. 법률 용어 매칭
        legal_references = self._find_legal_references(query, all_keywords, legal_field)
        
        # 4. 임베딩 기반 유사 키워드 찾기
        similar_keywords = []
        if self.embedding_model:
            similar_keywords = self._find_similar_keywords_with_embedding(all_keywords)
        
        # 5. 모든 키워드 통합
        expanded_keywords = list(set(
            all_keywords +
            synonyms +
            keyword_variants +
            similar_keywords
        ))
        
        return {
            "expanded_keywords": expanded_keywords[:20],  # 최대 20개
            "synonyms": list(set(synonyms))[:10],
            "keyword_variants": keyword_variants[:5],
            "legal_references": list(set(legal_references))[:5]
        }
    
    def _expand_synonyms(self, keywords: List[str], legal_field: str) -> List[str]:
        """동의어 확장"""
        synonyms = []
        
        # 사전 기반 동의어
        for keyword in keywords:
            if keyword in self.legal_synonyms_dict:
                synonyms.extend(self.legal_synonyms_dict[keyword])
        
        # TermIntegrator 활용
        if self.term_integrator:
            try:
                for keyword in keywords[:5]:  # 상위 5개만
                    term_synonyms = self.term_integrator.get_synonyms(keyword, legal_field)
                    if term_synonyms:
                        synonyms.extend(term_synonyms[:3])  # 최대 3개
            except Exception as e:
                self.logger.debug(f"TermIntegrator synonym expansion failed: {e}")
        
        return list(set(synonyms))
    
    def _generate_keyword_variants(self, keywords: List[str]) -> List[str]:
        """키워드 변형 생성"""
        variants = []
        
        for keyword in keywords:
            # 공백 제거/추가 변형
            if ' ' in keyword:
                variants.append(keyword.replace(' ', ''))
            else:
                # 복합어 분리 시도
                if len(keyword) >= 4:
                    mid = len(keyword) // 2
                    variants.append(f"{keyword[:mid]} {keyword[mid:]}")
        
        return variants[:5]
    
    def _find_legal_references(
        self,
        query: str,
        keywords: List[str],
        legal_field: str
    ) -> List[str]:
        """법률 용어 매칭"""
        references = []
        
        # TermIntegrator 활용
        if self.term_integrator:
            try:
                for keyword in keywords[:3]:  # 상위 3개만
                    related_laws = self.term_integrator.get_related_laws(keyword, legal_field)
                    if related_laws:
                        references.extend(related_laws[:2])  # 최대 2개
            except Exception as e:
                self.logger.debug(f"TermIntegrator legal reference finding failed: {e}")
        
        # 패턴 기반 법령 추출
        import re
        law_patterns = [
            r'민법\s*제?\s*\d+조',
            r'상법\s*제?\s*\d+조',
            r'형법\s*제?\s*\d+조',
            r'민사소송법\s*제?\s*\d+조'
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, query)
            references.extend(matches)
        
        return list(set(references))
    
    def _find_similar_keywords_with_embedding(
        self,
        keywords: List[str],
        threshold: float = 0.7
    ) -> List[str]:
        """임베딩 기반 유사 키워드 찾기"""
        if not self.embedding_model or not keywords:
            return []
        
        try:
            # 키워드 임베딩 생성
            keyword_embeddings = self.embedding_model.encode(keywords)
            
            # 법률 용어 DB와 비교 (캐시 활용)
            similar = []
            
            # 법률 동의어 사전의 모든 용어와 비교
            all_legal_terms = []
            for synonyms in self.legal_synonyms_dict.values():
                all_legal_terms.extend(synonyms)
            
            if all_legal_terms:
                # 법률 용어 임베딩 (캐시 활용)
                legal_term_embeddings = []
                for term in all_legal_terms:
                    if term not in self._legal_terms_cache:
                        emb = self.embedding_model.encode([term])[0]
                        self._legal_terms_cache[term] = emb
                    legal_term_embeddings.append((term, self._legal_terms_cache[term]))
                
                # 유사도 계산
                for kw_emb in keyword_embeddings:
                    for term, term_emb in legal_term_embeddings:
                        similarity = np.dot(kw_emb, term_emb) / (
                            np.linalg.norm(kw_emb) * np.linalg.norm(term_emb)
                        )
                        if similarity > threshold and term not in keywords:
                            similar.append(term)
            
            return list(set(similar))[:10]  # 최대 10개
            
        except Exception as e:
            self.logger.debug(f"Embedding-based similarity search failed: {e}")
            return []

