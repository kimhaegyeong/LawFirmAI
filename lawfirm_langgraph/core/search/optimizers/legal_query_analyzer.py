# -*- coding: utf-8 -*-
"""
법률 쿼리 분석기 (HuggingFace 모델 기반)
프롬프트 1 대체: 쿼리 분석 (core_keywords, query_intent, key_concepts)
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class LegalQueryAnalyzer:
    """법률 쿼리 분석기 (HuggingFace 모델 사용)"""
    
    def __init__(
        self,
        keyword_extractor: Optional[Any] = None,
        embedding_model_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        LegalQueryAnalyzer 초기화
        
        Args:
            keyword_extractor: 키워드 추출기 (KeywordExtractor)
            embedding_model_name: 임베딩 모델명 (기본값: 법률 특화 모델)
            logger: 로거
        """
        self.logger = logger or logging.getLogger(__name__)
        self.keyword_extractor = keyword_extractor
        
        # 임베딩 모델 초기화
        self.embedding_model = None
        self.embedding_model_name = embedding_model_name or "woong0322/ko-legal-sbert-finetuned"
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.logger.info(f"✅ [HF MODEL] Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                self.logger.warning(f"⚠️ [HF MODEL] Failed to load {self.embedding_model_name}: {e}")
                # 폴백: 일반 모델
                try:
                    self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                    self.logger.info("✅ [HF MODEL] Using fallback embedding model")
                except Exception as e2:
                    self.logger.error(f"❌ [HF MODEL] Failed to load fallback model: {e2}")
        
        # 질문 의도 패턴
        self.intent_patterns = {
            "계약": "계약 관련 법률 조항 및 판례 검색",
            "해지": "계약 해지 요건 및 절차 검색",
            "손해": "손해배상 요건 및 범위 검색",
            "소송": "소송 절차 및 요건 검색",
            "법령": "법령 조문 검색",
            "판례": "관련 판례 검색",
            "해석": "법률 해석 검색",
            "의견": "법률 의견 검색"
        }
    
    def analyze_query(
        self,
        query: str,
        query_type: str,
        legal_field: str
    ) -> Dict[str, Any]:
        """
        쿼리 분석 (프롬프트 1 대체)
        
        Args:
            query: 원본 쿼리
            query_type: 질문 유형
            legal_field: 법률 분야
            
        Returns:
            {
                "core_keywords": ["키워드1", "키워드2", ...],
                "query_intent": "의도",
                "key_concepts": ["개념1", "개념2", ...]
            }
        """
        if not query or not query.strip():
            return {
                "core_keywords": [],
                "query_intent": "법률 정보 검색",
                "key_concepts": []
            }
        
        # 1. 핵심 키워드 추출
        core_keywords = self._extract_core_keywords(query)
        
        # 2. 질문 의도 추출
        query_intent = self._extract_query_intent(query, query_type)
        
        # 3. 핵심 개념 추출
        key_concepts = self._extract_key_concepts(query, query_type, legal_field)
        
        return {
            "core_keywords": core_keywords,
            "query_intent": query_intent,
            "key_concepts": key_concepts
        }
    
    def _extract_core_keywords(self, query: str) -> List[str]:
        """핵심 키워드 추출"""
        if self.keyword_extractor:
            try:
                keywords = self.keyword_extractor.extract_keywords(
                    query, max_keywords=10, prefer_morphology=True
                )
                if keywords:
                    return keywords
            except Exception as e:
                self.logger.debug(f"KeywordExtractor failed: {e}")
        
        # 폴백: 간단한 키워드 추출
        import re
        words = re.findall(r'[가-힣]+', query)
        return [w for w in words if len(w) >= 2][:10]
    
    def _extract_query_intent(self, query: str, query_type: str) -> str:
        """질문 의도 추출 (규칙 기반)"""
        # 패턴 매칭
        for keyword, intent in self.intent_patterns.items():
            if keyword in query:
                return intent
        
        # query_type 기반 의도
        type_intent_map = {
            "statute": "법령 조문 검색",
            "case": "관련 판례 검색",
            "decision": "판결 검색",
            "interpretation": "법률 해석 검색",
            "legal_advice": "법률 상담 검색"
        }
        
        return type_intent_map.get(query_type, "법률 정보 검색")
    
    def _extract_key_concepts(
        self,
        query: str,
        query_type: str,
        legal_field: str
    ) -> List[str]:
        """핵심 개념 추출 (임베딩 기반)"""
        if not self.embedding_model:
            # 폴백: 키워드 기반
            return self._extract_core_keywords(query)[:5]
        
        try:
            # 법률 도메인 개념 패턴
            legal_concepts = {
                "계약": ["계약서", "계약관계", "계약체결", "계약이행"],
                "해지": ["해제", "취소", "해약", "계약종료"],
                "손해": ["손해배상", "손해전보", "배상책임", "손해보상"],
                "소송": ["소송절차", "소송요건", "소송제기", "소송비용"]
            }
            
            concepts = []
            for keyword, related_concepts in legal_concepts.items():
                if keyword in query:
                    concepts.extend(related_concepts[:2])
            
            # 임베딩 기반 유사 개념 찾기
            if concepts:
                query_embedding = self.embedding_model.encode([query])[0]
                concept_embeddings = self.embedding_model.encode(concepts)
                
                # 유사도 계산
                similarities = []
                for i, concept_emb in enumerate(concept_embeddings):
                    similarity = np.dot(query_embedding, concept_emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(concept_emb)
                    )
                    similarities.append((similarity, concepts[i]))
                
                # 상위 개념 선택
                similarities.sort(reverse=True)
                return [concept for _, concept in similarities[:5]]
            
            return concepts[:5]
            
        except Exception as e:
            self.logger.debug(f"Concept extraction failed: {e}")
            return self._extract_core_keywords(query)[:5]

