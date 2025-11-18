# -*- coding: utf-8 -*-
"""
Advanced Reranker
고급 Reranker 모델 지원 및 앙상블
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Reranker 설정"""
    model_name: str
    max_length: int = 512
    device: str = "cpu"
    batch_size: int = 32


class AdvancedReranker:
    """고급 Reranker (다중 모델 지원 및 앙상블)"""
    
    # 지원하는 Reranker 모델 목록
    SUPPORTED_MODELS = {
        "ko-reranker": "Dongjin-kr/ko-reranker",  # 기본 한국어 Reranker
        "bge-reranker-base": "BAAI/bge-reranker-base",  # BGE Reranker Base
        "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",  # BGE Reranker V2 M3
        "ko-sroberta-reranker": "Dongjin-kr/ko-sroberta-reranker",  # Ko-SRoBERTa Reranker
    }
    
    def __init__(
        self,
        primary_model: str = "ko-reranker",
        ensemble_models: Optional[List[str]] = None,
        use_ensemble: bool = False
    ):
        """
        초기화
        
        Args:
            primary_model: 주 Reranker 모델명
            ensemble_models: 앙상블에 사용할 추가 모델 목록
            use_ensemble: 앙상블 사용 여부
        """
        self.logger = logging.getLogger(__name__)
        self.primary_model = primary_model
        self.ensemble_models = ensemble_models or []
        self.use_ensemble = use_ensemble and len(self.ensemble_models) > 0
        
        # Reranker 인스턴스
        self.primary_reranker = None
        self.ensemble_rerankers = []
        
        # 주 Reranker 초기화
        self._initialize_primary_reranker()
        
        # 앙상블 Reranker 초기화
        if self.use_ensemble:
            self._initialize_ensemble_rerankers()
        
        self.logger.info(
            f"AdvancedReranker initialized: primary={primary_model}, "
            f"ensemble={self.use_ensemble} ({len(self.ensemble_rerankers)} models)"
        )
    
    def _initialize_primary_reranker(self):
        """주 Reranker 초기화"""
        try:
            from sentence_transformers import CrossEncoder
            
            model_name = self.SUPPORTED_MODELS.get(
                self.primary_model,
                self.primary_model
            )
            
            self.primary_reranker = CrossEncoder(
                model_name,
                max_length=512,
                device="cpu"
            )
            
            self.logger.info(f"Primary reranker initialized: {model_name}")
        
        except ImportError:
            self.logger.warning("sentence-transformers not available, reranker disabled")
        except Exception as e:
            self.logger.warning(f"Failed to initialize primary reranker: {e}")
    
    def _initialize_ensemble_rerankers(self):
        """앙상블 Reranker 초기화"""
        try:
            from sentence_transformers import CrossEncoder
            
            for model_key in self.ensemble_models:
                if model_key not in self.SUPPORTED_MODELS:
                    self.logger.warning(f"Unsupported ensemble model: {model_key}")
                    continue
                
                try:
                    model_name = self.SUPPORTED_MODELS[model_key]
                    reranker = CrossEncoder(
                        model_name,
                        max_length=512,
                        device="cpu"
                    )
                    self.ensemble_rerankers.append((model_key, reranker))
                    self.logger.info(f"Ensemble reranker initialized: {model_name}")
                
                except Exception as e:
                    self.logger.warning(f"Failed to initialize ensemble reranker {model_key}: {e}")
        
        except ImportError:
            self.logger.warning("sentence-transformers not available for ensemble")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        문서 재정렬
        
        Args:
            query: 검색 쿼리
            documents: 재정렬할 문서 리스트
            top_k: 반환할 최대 문서 수
            batch_size: 배치 크기
        
        Returns:
            List[Dict[str, Any]]: 재정렬된 문서 리스트
        """
        if not documents or not self.primary_reranker:
            return documents[:top_k] if top_k else documents
        
        try:
            # 앙상블 사용 여부에 따라 분기
            if self.use_ensemble and self.ensemble_rerankers:
                return self._ensemble_rerank(query, documents, top_k, batch_size)
            else:
                return self._single_rerank(query, documents, top_k, batch_size)
        
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return documents[:top_k] if top_k else documents
    
    def _single_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int],
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """단일 Reranker로 재정렬"""
        # 쿼리-문서 쌍 생성
        pairs = []
        for doc in documents:
            text = doc.get("text", doc.get("content", ""))
            if text:
                pairs.append([query, text])
        
        if not pairs:
            return documents[:top_k] if top_k else documents
        
        # 점수 계산
        scores = self.primary_reranker.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )
        
        # 점수 추가 및 정렬
        for i, doc in enumerate(documents):
            if i < len(scores):
                doc["rerank_score"] = float(scores[i])
                # 기존 점수와 결합 (가중 평균)
                original_score = doc.get("relevance_score", doc.get("similarity", 0.0))
                doc["final_score"] = 0.7 * doc["rerank_score"] + 0.3 * original_score
        
        # 최종 점수로 정렬
        reranked = sorted(
            documents,
            key=lambda x: x.get("final_score", x.get("rerank_score", 0.0)),
            reverse=True
        )
        
        return reranked[:top_k] if top_k else reranked
    
    def _ensemble_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int],
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """앙상블 Reranker로 재정렬"""
        # 모든 Reranker의 점수 수집
        all_scores = {}
        
        # 주 Reranker 점수
        primary_scores = self._get_rerank_scores(
            self.primary_reranker,
            query,
            documents,
            batch_size
        )
        all_scores["primary"] = primary_scores
        
        # 앙상블 Reranker 점수
        for model_key, reranker in self.ensemble_rerankers:
            try:
                scores = self._get_rerank_scores(
                    reranker,
                    query,
                    documents,
                    batch_size
                )
                all_scores[model_key] = scores
            except Exception as e:
                self.logger.warning(f"Ensemble reranker {model_key} failed: {e}")
        
        # 앙상블 점수 계산 (가중 평균)
        # 주 Reranker: 0.5, 앙상블 각각: 0.5 / len(ensemble)
        ensemble_weight = 0.5 / len(self.ensemble_rerankers) if self.ensemble_rerankers else 0.0
        
        for i, doc in enumerate(documents):
            # 주 Reranker 점수
            primary_score = all_scores["primary"][i] if i < len(all_scores["primary"]) else 0.0
            
            # 앙상블 점수 평균
            ensemble_scores = [
                all_scores[key][i]
                for key in all_scores.keys()
                if key != "primary" and i < len(all_scores[key])
            ]
            ensemble_avg = sum(ensemble_scores) / len(ensemble_scores) if ensemble_scores else 0.0
            
            # 최종 앙상블 점수
            doc["rerank_score"] = 0.5 * primary_score + ensemble_weight * sum(ensemble_scores)
            
            # 기존 점수와 결합
            original_score = doc.get("relevance_score", doc.get("similarity", 0.0))
            doc["final_score"] = 0.7 * doc["rerank_score"] + 0.3 * original_score
        
        # 최종 점수로 정렬
        reranked = sorted(
            documents,
            key=lambda x: x.get("final_score", x.get("rerank_score", 0.0)),
            reverse=True
        )
        
        return reranked[:top_k] if top_k else reranked
    
    def _get_rerank_scores(
        self,
        reranker: Any,
        query: str,
        documents: List[Dict[str, Any]],
        batch_size: int
    ) -> List[float]:
        """Reranker 점수 계산"""
        pairs = []
        for doc in documents:
            text = doc.get("text", doc.get("content", ""))
            if text:
                pairs.append([query, text])
        
        if not pairs:
            return [0.0] * len(documents)
        
        try:
            scores = reranker.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False
            )
            return [float(s) for s in scores]
        except Exception as e:
            self.logger.warning(f"Reranker score calculation failed: {e}")
            return [0.0] * len(documents)

