# -*- coding: utf-8 -*-
"""
Result Merger and Ranker
검색 결과 병합 및 순위 결정
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class MergedResult:
    """병합된 검색 결과"""
    text: str
    score: float
    source: str
    metadata: Dict[str, Any]


class ResultMerger:
    """검색 결과 병합기"""
    
    def __init__(self):
        """병합기 초기화"""
        self.logger = get_logger(__name__)
        self.logger.info("ResultMerger initialized")
    
    def merge_results(self, 
                     exact_results: Dict[str, List[Dict[str, Any]]], 
                     semantic_results: List[Dict[str, Any]],
                     weights: Dict[str, float] = None,
                     query: str = "") -> List[MergedResult]:
        """
        검색 결과 병합
        
        Args:
            exact_results: 정확한 검색 결과 (딕셔너리)
            semantic_results: 의미적 검색 결과 (리스트)
            weights: 가중치 딕셔너리
            query: 검색 쿼리 (metadata 저장용)
            
        Returns:
            List[MergedResult]: 병합된 결과
        """
        if weights is None:
            weights = {"exact": 0.7, "semantic": 0.3}
        
        merged_results = []
        
        # 정확한 검색 결과 처리 (딕셔너리 형태)
        for search_type, results in exact_results.items():
            for result in results:
                if isinstance(result, dict):
                    # content 필드도 text로 매핑 (검색 결과가 content 필드를 사용하는 경우)
                    text_content = (
                        result.get('text', '') or
                        result.get('content', '') or
                        str(result.get('metadata', {}).get('content', '')) or
                        str(result.get('metadata', {}).get('text', '')) or
                        ''
                    )
                    
                    # text가 비어있으면 건너뛰기 (필터링)
                    if not text_content or len(text_content.strip()) == 0:
                        self.logger.warning(f"Skipping MergedResult with empty text from exact_{search_type} (id: {result.get('id', 'unknown')})")
                        continue
                    
                    # metadata에 content 필드 명시적으로 저장 (향후 딕셔너리 변환 시 복원용)
                    metadata = result.get('metadata', {})
                    if not isinstance(metadata, dict):
                        metadata = result if isinstance(result, dict) else {}
                    metadata['content'] = text_content
                    metadata['text'] = text_content
                    
                    # metadata에 query 저장
                    if query:
                        metadata['query'] = query
                    
                    merged_result = MergedResult(
                        text=text_content,
                        score=result.get('similarity', result.get('relevance_score', result.get('score', 0.0))) * weights["exact"],
                        source=f"exact_{search_type}",
                        metadata=metadata
                    )
                    merged_results.append(merged_result)
        
        # 의미적 검색 결과 처리 (리스트 형태)
        for result in semantic_results:
            if isinstance(result, dict):
                # content 필드도 text로 매핑
                text_content = (
                    result.get('text', '') or
                    result.get('content', '') or
                    str(result.get('metadata', {}).get('content', '')) or
                    str(result.get('metadata', {}).get('text', '')) or
                    ''
                )
                
                # text가 비어있으면 건너뛰기 (필터링)
                if not text_content or len(text_content.strip()) == 0:
                    self.logger.warning(f"Skipping MergedResult with empty text from semantic (id: {result.get('id', 'unknown')})")
                    continue
                
                # metadata에 content 필드 명시적으로 저장 (향후 딕셔너리 변환 시 복원용)
                metadata = result.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = result if isinstance(result, dict) else {}
                metadata['content'] = text_content
                metadata['text'] = text_content
                
                # metadata에 query 저장
                if query:
                    metadata['query'] = query
                
                merged_result = MergedResult(
                    text=text_content,
                    score=result.get('similarity', result.get('relevance_score', result.get('score', 0.0))) * weights["semantic"],
                    source="semantic",
                    metadata=metadata
                )
                merged_results.append(merged_result)
        
        return merged_results


class ResultRanker:
    """검색 결과 순위 결정기"""
    
    # 클래스 변수: SentenceTransformer 모델 캐싱 (성능 최적화)
    _semantic_model = None
    _semantic_model_name = None
    
    def __init__(self, use_cross_encoder: bool = True, rerank_original_weight: float = None, rerank_cross_encoder_weight: float = None):
        """순위 결정기 초기화"""
        self.logger = get_logger(__name__)
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder = None
        
        # 재랭킹 가중치 설정 (환경 변수 또는 파라미터로 설정)
        if rerank_original_weight is None or rerank_cross_encoder_weight is None:
            import os
            rerank_original_weight = float(os.getenv("RERANK_ORIGINAL_WEIGHT", "0.6"))
            rerank_cross_encoder_weight = float(os.getenv("RERANK_CROSS_ENCODER_WEIGHT", "0.4"))
        
        # 가중치 합이 1.0이 되도록 정규화
        total_weight = rerank_original_weight + rerank_cross_encoder_weight
        if total_weight > 0:
            self.rerank_original_weight = rerank_original_weight / total_weight
            self.rerank_cross_encoder_weight = rerank_cross_encoder_weight / total_weight
        else:
            # 잘못된 값이면 기본값으로 재설정
            self.rerank_original_weight = 0.6
            self.rerank_cross_encoder_weight = 0.4
        
        if use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder('Dongjin-kr/ko-reranker', max_length=512)
                self.logger.info("Cross-Encoder reranker initialized (Dongjin-kr/ko-reranker)")
            except ImportError:
                self.logger.warning("sentence-transformers not available, Cross-Encoder disabled")
                self.use_cross_encoder = False
            except Exception as e:
                self.logger.warning(f"Failed to initialize Cross-Encoder: {e}, falling back to standard ranking")
                self.use_cross_encoder = False
        
        self.logger.info(
            f"ResultRanker initialized (cross_encoder={self.use_cross_encoder}, "
            f"weights: original={self.rerank_original_weight:.2f}, cross_encoder={self.rerank_cross_encoder_weight:.2f})"
        )
    
    @classmethod
    def _get_semantic_model(cls):
        """SentenceTransformer 모델 가져오기 (캐싱)"""
        if cls._semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import os
                
                if cls._semantic_model_name is None:
                    model_name = os.getenv("EMBEDDING_MODEL")
                    if model_name is None:
                        try:
                            from ...utils.config import Config
                        except ImportError:
                            from core.utils.config import Config
                        config = Config()
                        model_name = config.embedding_model
                    cls._semantic_model_name = model_name
                
                cls._semantic_model = SentenceTransformer(cls._semantic_model_name)
                logger = logging.getLogger(__name__)
                logger.debug(f"Semantic model loaded and cached: {cls._semantic_model_name}")
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.debug(f"Failed to load semantic model: {e}")
                return None
        return cls._semantic_model
    
    def rank_results(self, results: List[Any], top_k: int = 10, query: str = "") -> List[Any]:
        """
        검색 결과 순위 결정
        
        Args:
            results: 병합된 검색 결과 (MergedResult 또는 Dict)
            top_k: 반환할 결과 수
            query: 검색 쿼리 (Cross-Encoder reranking용)
            
        Returns:
            List[MergedResult] 또는 List[Dict]: 순위가 매겨진 결과
        """
        if not results:
            return []
        
        # Dict를 MergedResult로 변환
        converted_results = []
        for result in results:
            if isinstance(result, dict):
                # Dict를 MergedResult로 변환
                text = result.get("text") or result.get("content") or result.get("chunk_text") or ""
                score = result.get("score") or result.get("relevance_score") or result.get("similarity", 0.0)
                source = result.get("source") or result.get("title") or result.get("document_id") or ""
                metadata = result.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = result if isinstance(result, dict) else {}
                
                converted_result = MergedResult(
                    text=text,
                    score=score,
                    source=source,
                    metadata=metadata
                )
                converted_results.append(converted_result)
            elif isinstance(result, MergedResult):
                converted_results.append(result)
            else:
                # 기타 타입은 건너뛰기
                continue
        
        # 중복 제거 (텍스트 기준)
        unique_results = {}
        for result in converted_results:
            if result.text not in unique_results:
                unique_results[result.text] = result
            else:
                # 더 높은 점수로 업데이트
                if result.score > unique_results[result.text].score:
                    unique_results[result.text] = result
        
        # 점수순 정렬
        ranked_results = list(unique_results.values())
        ranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Cross-Encoder reranking 적용 (상위 후보만)
        if self.use_cross_encoder and self.cross_encoder and len(ranked_results) > 0:
            try:
                # extracted_keywords 추출 (metadata에서)
                extracted_keywords = None
                for result in ranked_results:
                    if isinstance(result.metadata, dict):
                        extracted_keywords = result.metadata.get("extracted_keywords", [])
                        if extracted_keywords:
                            break
                
                reranked_results = self.cross_encoder_rerank(
                    ranked_results[:top_k * 2],  # 상위 후보만 rerank
                    query=query,  # query 전달
                    top_k=top_k,
                    extracted_keywords=extracted_keywords
                )
                ranked_results = reranked_results + ranked_results[top_k * 2:]
            except Exception as e:
                self.logger.warning(f"Cross-Encoder reranking failed: {e}, using standard ranking")
        
        # Dict로 변환하여 반환 (호환성 유지)
        return [self._merged_result_to_dict(r) for r in ranked_results[:top_k]]
    
    def cross_encoder_rerank(
        self,
        results: List[MergedResult],
        query: str = "",
        top_k: int = 10,
        extracted_keywords: List[str] = None
    ) -> List[MergedResult]:
        """
        Cross-Encoder를 사용한 정확한 재정렬 (Phase 2: Keyword Coverage 고려)
        
        Args:
            results: 재정렬할 MergedResult 리스트
            query: 검색 쿼리 (우선순위: 파라미터 > metadata)
            top_k: 반환할 최대 결과 수
            extracted_keywords: 추출된 키워드 리스트 (Keyword Coverage 계산용)
            
        Returns:
            List[MergedResult]: 재정렬된 결과
        """
        if not results or not self.cross_encoder:
            return results[:top_k]
        
        # query 추출 우선순위: 파라미터 > metadata > 원본 쿼리
        extracted_query = query
        
        if not extracted_query:
            for result in results:
                if isinstance(result.metadata, dict):
                    extracted_query = result.metadata.get("query", "")
                    if extracted_query:
                        break
        
        # 추가: 원본 쿼리 필드 확인
        if not extracted_query:
            for result in results:
                if isinstance(result.metadata, dict):
                    extracted_query = result.metadata.get("original_query", "") or result.metadata.get("search_query", "")
                    if extracted_query:
                        break
        
        if not extracted_query:
            self.logger.warning("No query provided for Cross-Encoder reranking, using standard ranking")
            return results[:top_k]
        
        # extracted_keywords 추출 (metadata에서)
        if extracted_keywords is None:
            for result in results:
                if isinstance(result.metadata, dict):
                    extracted_keywords = result.metadata.get("extracted_keywords", [])
                    if extracted_keywords:
                        break
        
        try:
            # query-document 쌍 생성
            pairs = []
            for result in results:
                text = result.text[:500]  # 길이 제한
                pairs.append([extracted_query, text])
            
            # 개선: 배치 크기 최적화 (Cross-Encoder의 predict는 이미 배치 처리 지원)
            # batch_size를 명시적으로 설정하여 메모리 효율성 향상
            batch_size = min(16, len(pairs))  # 배치 크기 제한 (메모리 효율성)
            
            # 점수 계산 (배치 처리)
            if len(pairs) <= batch_size:
                scores = self.cross_encoder.predict(pairs)
            else:
                # 큰 경우 배치로 나누어 처리
                scores = []
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i:i + batch_size]
                    batch_scores = self.cross_encoder.predict(batch_pairs)
                    scores.extend(batch_scores)
            
            # Cross-Encoder 점수 정규화 (0-1 범위로 정규화하여 기존 점수와 스케일 일치)
            if SKLEARN_AVAILABLE and len(scores) > 1:
                scores_array = np.array([float(s) for s in scores])
                min_score = scores_array.min()
                max_score = scores_array.max()
                if max_score > min_score:
                    # Min-Max 정규화 (0-1 범위)
                    normalized_scores = (scores_array - min_score) / (max_score - min_score)
                else:
                    normalized_scores = scores_array
                scores = normalized_scores.tolist()
            
            # 점수 반영 및 정렬
            reranked_results = []
            for result, score in zip(results, scores):
                # 기존 점수와 Cross-Encoder 점수 결합
                original_score = result.score
                cross_encoder_score = float(score)  # 정규화된 점수 사용
                
                # Phase 2: Keyword Coverage 기반 보너스 계산
                keyword_bonus = 0.0
                if extracted_keywords:
                    # 문서에서 키워드 매칭 확인
                    text_lower = result.text.lower()
                    matched_keywords = [kw for kw in extracted_keywords if kw.lower() in text_lower]
                    
                    if matched_keywords:
                        # Keyword Coverage 계산
                        keyword_coverage = len(matched_keywords) / len(extracted_keywords)
                        
                        # 핵심 키워드(상위 3개) 매칭 확인
                        core_keywords = extracted_keywords[:3] if len(extracted_keywords) >= 3 else extracted_keywords
                        core_matched = sum(1 for kw in core_keywords if kw.lower() in text_lower)
                        core_ratio = core_matched / len(core_keywords) if core_keywords else 0.0
                        
                        # Keyword Coverage 보너스 (최대 15%)
                        coverage_bonus = keyword_coverage * 0.15
                        
                        # 핵심 키워드 매칭 보너스 (최대 10%)
                        core_bonus = core_ratio * 0.10
                        
                        keyword_bonus = coverage_bonus + core_bonus
                
                # 가중 평균 (가중치 튜닝 가능, 환경 변수로 조정)
                base_combined_score = (
                    self.rerank_original_weight * original_score + 
                    self.rerank_cross_encoder_weight * cross_encoder_score
                )
                
                # 디버그 로그 (가중치 확인용)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"Score combination: original={original_score:.3f} (weight={self.rerank_original_weight:.2f}), "
                        f"cross_encoder={cross_encoder_score:.3f} (weight={self.rerank_cross_encoder_weight:.2f}), "
                        f"combined={base_combined_score:.3f}"
                    )
                
                # Keyword Coverage 보너스 적용
                combined_score = base_combined_score * (1.0 + keyword_bonus)
                
                # MergedResult 업데이트 (새 객체 생성)
                updated_result = MergedResult(
                    text=result.text,
                    score=combined_score,
                    source=result.source,
                    metadata={
                        **result.metadata,
                        "cross_encoder_score": cross_encoder_score,
                        "original_score": original_score,
                        "keyword_bonus": keyword_bonus
                    }
                )
                reranked_results.append(updated_result)
            
            # 점수순 정렬
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            self.logger.info(
                f"Cross-Encoder reranking: {len(results)} documents reranked, "
                f"top score: {reranked_results[0].score:.3f}" if reranked_results else "no results"
            )
            
            return reranked_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Cross-Encoder reranking error: {e}")
            return results[:top_k]
    
    def _merged_result_to_dict(self, result: MergedResult) -> Dict[str, Any]:
        """MergedResult를 Dict로 변환"""
        return {
            "text": result.text,
            "content": result.text,
            "score": result.score,
            "relevance_score": result.score,
            "similarity": result.score,
            "source": result.source,
            "metadata": result.metadata
        }
    
    def _dict_to_merged_result(self, doc: Dict[str, Any]) -> MergedResult:
        """Dict를 MergedResult로 변환"""
        text = doc.get("text") or doc.get("content") or doc.get("chunk_text") or ""
        score = doc.get("final_weighted_score") or doc.get("score") or doc.get("relevance_score") or doc.get("similarity", 0.0)
        source = doc.get("source") or doc.get("title") or doc.get("document_id") or ""
        metadata = doc.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = doc if isinstance(doc, dict) else {}
        
        return MergedResult(
            text=text,
            score=score,
            source=source,
            metadata=metadata
        )
    
    def multi_stage_rerank(
        self,
        documents: List[Dict[str, Any]],
        query: str = "",
        query_type: str = "",
        extracted_keywords: List[str] = None,
        search_quality: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        다단계 재정렬 전략 (개선: 키워드 매칭, Citation 매칭, 질문 유형별 특화, MMR 다양성, 검색 품질 기반 조정, 정보 밀도, 최신성, Cross-Encoder)
        
        재랭킹 단계:
        - Stage 1: 관련성 점수로 초기 정렬
        - Stage 1.5: 키워드 매칭 점수 직접 반영
        - Stage 2: Citation 포함 문서 우선순위
        - Stage 3: 신뢰도 점수 적용
        - Stage 3.5: 질문 유형별 특화 재정렬
        - Stage 3.6: 정보 밀도 점수 적용
        - Stage 3.7: 최신성 점수 적용
        - Stage 4: MMR 기반 다양성 적용
        - Stage 5: Cross-Encoder 재랭킹 (동적 후보 수 조정)
        
        Args:
            documents: 재정렬할 문서 리스트
            query: 검색 쿼리 (Cross-Encoder 재랭킹에 필요)
            query_type: 질문 유형
            extracted_keywords: 추출된 키워드
            search_quality: 검색 품질 점수
            
        Returns:
            List[Dict]: 재정렬된 문서 리스트
        """
        import time
        start_time = time.time()
        
        if not documents:
            return []
        
        if extracted_keywords is None:
            extracted_keywords = []
        
        # Stage 1: 관련성 점수로 초기 정렬
        stage1 = sorted(
            documents,
            key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)),
            reverse=True
        )
        
        # Stage 1.5: 키워드 매칭 점수 직접 반영 (개선: 키워드 매칭 점수 직접 반영)
        for doc in stage1:
            keyword_score = doc.get("weighted_keyword_score", 0.0)
            keyword_coverage = doc.get("keyword_coverage", 0.0)
            matched_keywords = doc.get("matched_keywords", [])
            
            # 핵심 키워드 매칭 보너스 계산
            core_keyword_bonus = 0.0
            if extracted_keywords and matched_keywords:
                # 핵심 키워드 매칭 비율
                core_keywords_matched = sum(1 for kw in extracted_keywords[:3] if kw in matched_keywords)
                core_keyword_ratio = core_keywords_matched / min(3, len(extracted_keywords)) if extracted_keywords else 0.0
                core_keyword_bonus = core_keyword_ratio * 0.25  # 최대 25% 증가
            
            # 키워드 커버리지 보너스 (개선: Phase 1 - 가중치 증가)
            coverage_bonus = keyword_coverage * 0.20  # 최대 20% 증가 (0.15 → 0.20)
            
            # 키워드 매칭 점수 보너스 (개선: Phase 1 - 가중치 증가)
            keyword_bonus = keyword_score * 0.30  # 최대 30% 증가 (0.2 → 0.3)
            
            # 총 키워드 보너스 적용
            current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
            total_keyword_bonus = core_keyword_bonus + coverage_bonus + keyword_bonus
            doc["final_weighted_score"] = current_score * (1.0 + total_keyword_bonus)
            doc["keyword_bonus"] = total_keyword_bonus
        
        # Stage 2: Citation 포함 문서 우선순위 (개선: Citation 매칭 정확도 개선)
        citation_docs = []
        non_citation_docs = []
        for doc in stage1:
            if self._has_citation(doc):
                # Citation 매칭 점수 계산
                citation_match_score = self._calculate_citation_match_score(
                    document=doc,
                    query=query,
                    extracted_keywords=extracted_keywords
                )
                
                # Citation 매칭 점수에 따른 보너스 적용
                current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                if citation_match_score >= 0.5:
                    # 정확 일치: 30-50% 증가
                    citation_bonus = 0.3 + (citation_match_score - 0.5) * 0.4
                elif citation_match_score > 0.0:
                    # 부분 일치: 10-30% 증가
                    citation_bonus = 0.1 + citation_match_score * 0.4
                else:
                    # Citation만 있음: 10% 증가
                    citation_bonus = 0.1
                
                doc["final_weighted_score"] = current_score * (1.0 + citation_bonus)
                doc["citation_match_score"] = citation_match_score
                doc["citation_bonus"] = citation_bonus
                citation_docs.append(doc)
            else:
                non_citation_docs.append(doc)
        
        # Stage 3: 신뢰도 점수 적용
        for doc in citation_docs + non_citation_docs:
            trust_score = self._calculate_trust_score(doc)
            current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
            doc["trust_score"] = trust_score
            doc["final_weighted_score"] = current_score * (1.0 + trust_score * 0.1)
        
        # Stage 3.5: 질문 유형별 특화 재정렬 (개선: 질문 유형별 특화 재정렬)
        if query_type == "law_inquiry":
            # 법령 문의: 법령 문서 우선순위 강화
            for doc in citation_docs + non_citation_docs:
                doc_type = doc.get("type", "").lower() if doc.get("type") else ""
                source = doc.get("source", "").lower()
                
                if "법령" in doc_type or "statute" in doc_type or "법령" in source:
                    current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                    doc["final_weighted_score"] = current_score * 1.3  # 30% 증가
                    doc["query_type_boost"] = 0.3
                elif "판례" in doc_type or "precedent" in doc_type:
                    current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                    doc["final_weighted_score"] = current_score * 0.9  # 10% 감소
        
        elif query_type == "precedent_search":
            # 판례 검색: 판례 문서 우선순위 강화
            for doc in citation_docs + non_citation_docs:
                doc_type = doc.get("type", "").lower() if doc.get("type") else ""
                source = doc.get("source", "").lower()
                
                if "판례" in doc_type or "precedent" in doc_type or "대법원" in source:
                    current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                    doc["final_weighted_score"] = current_score * 1.3  # 30% 증가
                    doc["query_type_boost"] = 0.3
                elif "법령" in doc_type or "statute" in doc_type:
                    current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                    doc["final_weighted_score"] = current_score * 0.9  # 10% 감소
        
        # Stage 3.6: 정보 밀도 점수 적용 (개선: 품질이 낮을 때만 적용)
        if search_quality < 0.85:  # 품질이 낮을 때만 적용
            for doc in citation_docs + non_citation_docs:
                density_score = self._calculate_information_density(doc)
                current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                doc["final_weighted_score"] = current_score * (1.0 + density_score * 0.1)  # 최대 10% 증가
                doc["information_density"] = density_score
        
        # Stage 3.7: 최신성 점수 적용 (개선: 품질이 낮을 때만 적용)
        if search_quality < 0.85:  # 품질이 낮을 때만 적용
            for doc in citation_docs + non_citation_docs:
                doc_type = doc.get("type", "").lower() if doc.get("type") else ""
                if "판례" in doc_type or "precedent" in doc_type:
                    # 판례만 최신성 고려
                    recency_score = self._calculate_recency_score(doc)
                    current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                    doc["final_weighted_score"] = current_score * (1.0 + recency_score * 0.15)  # 최대 15% 증가
                    doc["recency_score"] = recency_score
        
        # Stage 4: MMR 기반 다양성 적용 (개선: 동적 가중치 사용, 품질이 높으면 가중치 감소)
        dynamic_lambda = self._get_dynamic_mmr_lambda(
            query_type=query_type,
            search_quality=search_quality,
            num_results=len(citation_docs + non_citation_docs)
        )
        # 품질이 높으면 다양성 가중치 감소 (더 빠른 처리)
        if search_quality >= 0.85:
            dynamic_lambda = min(dynamic_lambda, 0.3)  # 다양성 가중치 상한선 설정
        
        diverse_docs = self._apply_mmr_diversity(
            citation_docs + non_citation_docs,
            query,
            lambda_score=dynamic_lambda
        )
        
        # Stage 5: Cross-Encoder 재랭킹 (개선: 조기 종료 조건 강화)
        if self.use_cross_encoder and self.cross_encoder and len(diverse_docs) > 0 and query:
            # 개선: 조기 종료 조건 강화 (품질 >= 0.75 또는 문서 <= 10)
            should_skip_cross_encoder = (
                search_quality >= 0.75 or  # 품질이 중간 이상이면 스킵 (0.85 → 0.75로 완화)
                len(diverse_docs) <= 10  # 문서 수가 적으면 스킵 (8 → 10으로 완화)
            )
            
            if should_skip_cross_encoder:
                self.logger.info(
                    f"⏭️ [CROSS-ENCODER SKIP] Skipping Cross-Encoder reranking "
                    f"(quality: {search_quality:.2f}, docs: {len(diverse_docs)})"
                )
                return diverse_docs
            
            try:
                # 개선: Cross-Encoder 재랭킹 후보 수 감소 (10-20개 → 5-10개)
                # 문서 수에 따라 동적으로 조정: 최소 5개, 최대 10개
                if len(diverse_docs) <= 5:
                    rerank_candidates_count = len(diverse_docs)  # 문서 수가 매우 적으면 모두 재랭킹
                elif len(diverse_docs) <= 10:
                    rerank_candidates_count = min(8, len(diverse_docs))  # 중간 크기면 8개
                elif len(diverse_docs) <= 15:
                    rerank_candidates_count = min(10, len(diverse_docs))  # 큰 경우 10개
                else:
                    rerank_candidates_count = min(10, max(5, len(diverse_docs) // 4))  # 매우 큰 경우 동적 조정 (최대 10개)
                
                candidates_for_rerank = diverse_docs[:rerank_candidates_count]
                
                print(f"[CROSS-ENCODER RERANK] Selecting {rerank_candidates_count} candidates out of {len(diverse_docs)} documents for reranking", flush=True)
                self.logger.info(
                    f"🔍 [CROSS-ENCODER RERANK] Selecting {rerank_candidates_count} candidates "
                    f"out of {len(diverse_docs)} documents for reranking"
                )
                
                # 개선: Dict를 MergedResult로 변환 최적화 (불필요한 복사 최소화)
                merged_candidates = []
                for doc in candidates_for_rerank:
                    # 기존 _dict_to_merged_result 호출 대신 직접 생성 (성능 향상)
                    text = doc.get("text") or doc.get("content") or doc.get("chunk_text", "")
                    score = doc.get("final_weighted_score") or doc.get("relevance_score", 0.0)
                    source = doc.get("source") or doc.get("title") or doc.get("document_id", "")
                    metadata = doc.get("metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = doc if isinstance(doc, dict) else {}
                    
                    merged_candidates.append(MergedResult(
                        text=text,
                        score=score,
                        source=source,
                        metadata=metadata
                    ))
                
                # Cross-Encoder 재랭킹 수행
                reranked_merged = self.cross_encoder_rerank(
                    results=merged_candidates,
                    query=query,
                    top_k=rerank_candidates_count,
                    extracted_keywords=extracted_keywords
                )
                
                # MergedResult를 Dict로 변환하고 기존 점수 정보 유지
                reranked_docs = []
                # 원본 문서를 텍스트로 인덱싱 (빠른 매칭을 위해)
                original_docs_by_text = {doc.get("text") or doc.get("content") or doc.get("chunk_text", ""): doc for doc in candidates_for_rerank}
                
                for merged_result in reranked_merged:
                    doc_dict = self._merged_result_to_dict(merged_result)
                    # 텍스트로 원본 문서 찾기
                    doc_text = merged_result.text
                    original_doc = original_docs_by_text.get(doc_text)
                    
                    if original_doc:
                        # 기존 필드 유지하면서 점수만 업데이트
                        doc_dict.update({
                            **original_doc,
                            "final_weighted_score": merged_result.score,
                            "relevance_score": merged_result.score,
                            "score": merged_result.score,
                            "cross_encoder_applied": True
                        })
                    else:
                        # 원본 문서를 찾지 못한 경우에도 점수 업데이트
                        doc_dict.update({
                            "final_weighted_score": merged_result.score,
                            "relevance_score": merged_result.score,
                            "score": merged_result.score,
                            "cross_encoder_applied": True
                        })
                    reranked_docs.append(doc_dict)
                
                # 재랭킹된 문서와 나머지 문서 병합
                remaining_docs = diverse_docs[rerank_candidates_count:]
                final_docs = reranked_docs + remaining_docs
                
                # 개선: 재랭킹 결과 검증 - 전후 점수 분포 비교
                # Cross-Encoder 재랭킹 직전 점수와 재랭킹 후 점수 비교
                if reranked_docs:
                    # 재랭킹 직전 점수: merged_candidates의 score (MergedResult.score)
                    before_scores = [result.score for result in merged_candidates]
                    # 재랭킹 후 점수: reranked_merged의 score (Cross-Encoder 재랭킹 후)
                    after_scores = [result.score for result in reranked_merged]
                    
                    if before_scores and after_scores and len(before_scores) == len(after_scores):
                        before_avg = sum(before_scores) / len(before_scores)
                        after_avg = sum(after_scores) / len(after_scores)
                        before_max = max(before_scores)
                        after_max = max(after_scores)
                        before_min = min(before_scores)
                        after_min = min(after_scores)
                        
                        # 순위 변경 추적 (상위 3개 문서)
                        before_top3_indices = sorted(range(len(before_scores)), key=lambda i: before_scores[i], reverse=True)[:3]
                        after_top3_indices = sorted(range(len(after_scores)), key=lambda i: after_scores[i], reverse=True)[:3]
                        rank_change = len(set(before_top3_indices) & set(after_top3_indices))  # 상위 3개 중 유지된 문서 수
                        
                        improvement_pct = (after_avg - before_avg) * 100 / before_avg if before_avg > 0 else 0
                        max_improvement_pct = (after_max - before_max) * 100 / before_max if before_max > 0 else 0
                        
                        print(f"[CROSS-ENCODER RERANK] Applied Cross-Encoder reranking to {len(reranked_docs)} documents. Score: avg {before_avg:.3f}→{after_avg:.3f} ({improvement_pct:+.1f}%), max {before_max:.3f}→{after_max:.3f} ({max_improvement_pct:+.1f}%), min {before_min:.3f}→{after_min:.3f}. Top 3 rank stability: {rank_change}/3", flush=True)
                        self.logger.info(
                            f"🔄 [CROSS-ENCODER RERANK] Applied Cross-Encoder reranking to {len(reranked_docs)} documents "
                            f"(out of {len(diverse_docs)} total documents). "
                            f"Score: avg {before_avg:.3f}→{after_avg:.3f} ({improvement_pct:+.1f}%), "
                            f"max {before_max:.3f}→{after_max:.3f} ({max_improvement_pct:+.1f}%), "
                            f"min {before_min:.3f}→{after_min:.3f}. "
                            f"Top 3 rank stability: {rank_change}/3"
                        )
                    else:
                        self.logger.info(
                            f"🔄 [CROSS-ENCODER RERANK] Applied Cross-Encoder reranking to {len(reranked_docs)} documents "
                            f"(out of {len(diverse_docs)} total documents)"
                        )
                else:
                    self.logger.warning(
                        f"⚠️ [CROSS-ENCODER RERANK] No documents were reranked "
                        f"(expected {rerank_candidates_count}, got {len(reranked_docs)})"
                    )
                
                # 개선: 재랭킹 성능 모니터링
                rerank_time = time.time() - start_time
                self.logger.info(
                    f"⏱️ [RERANK PERFORMANCE] Total reranking time: {rerank_time:.3f}s "
                    f"({len(diverse_docs)} documents, {rerank_candidates_count} reranked)"
                )
                
                return final_docs
                
            except Exception as e:
                rerank_time = time.time() - start_time
                self.logger.warning(
                    f"Cross-Encoder reranking in multi_stage_rerank failed: {e}, using MMR results "
                    f"(time: {rerank_time:.3f}s)"
                )
                return diverse_docs
        else:
            # Cross-Encoder가 비활성화되었거나 쿼리가 없는 경우 MMR 결과 반환
            rerank_time = time.time() - start_time
            if not self.use_cross_encoder or not self.cross_encoder:
                self.logger.info(f"⏭️ [CROSS-ENCODER SKIP] Cross-Encoder not available, skipping Stage 5 (time: {rerank_time:.3f}s)")
            elif not query:
                self.logger.info(f"⏭️ [CROSS-ENCODER SKIP] Query not provided, skipping Cross-Encoder reranking (time: {rerank_time:.3f}s)")
            return diverse_docs
    
    def evaluate_search_quality(
        self,
        query: str,
        results: List[Dict[str, Any]],
        query_type: str = "",
        extracted_keywords: List[str] = None
    ) -> Dict[str, float]:
        """
        검색 품질 평가 메트릭 수집
        
        Args:
            query: 검색 쿼리
            results: 검색 결과 리스트
            query_type: 질문 유형
            extracted_keywords: 추출된 키워드
            
        Returns:
            Dict[str, float]: 검색 품질 메트릭
        """
        metrics = {
            "avg_relevance": 0.0,
            "min_relevance": 0.0,
            "max_relevance": 0.0,
            "diversity_score": 0.0,
            "keyword_coverage": 0.0
        }
        
        if not results:
            return metrics
        
        # 관련성 점수 계산
        scores = [doc.get("relevance_score", doc.get("final_weighted_score", 0.0)) for doc in results]
        if scores:
            metrics["avg_relevance"] = sum(scores) / len(scores)
            metrics["min_relevance"] = min(scores)
            metrics["max_relevance"] = max(scores)
        
        # 다양성 점수 계산 (다차원 다양성 통합)
        metrics["diversity_score"] = self._calculate_comprehensive_diversity_score(results)
        
        # 키워드 커버리지 점수 계산 (Phase 3: 의미적 유사도 고려)
        if extracted_keywords:
            covered_keywords = set()
            semantic_matches = {}  # 의미적 매칭 점수
            
            # Phase 3: 임베딩 모델을 사용한 의미적 유사도 기반 매칭 (성능 최적화: 모델 캐싱)
            use_semantic_matching = False
            model = None
            if SKLEARN_AVAILABLE:
                model = self._get_semantic_model()
                if model is not None:
                    use_semantic_matching = True
                    self.logger.debug("Using semantic similarity for keyword matching (cached model)")
                else:
                    self.logger.debug("Semantic model not available, using string matching only")
            else:
                self.logger.debug("sklearn not available, using string matching only")
            
            if use_semantic_matching:
                try:
                    # 성능 최적화: 배치 임베딩 생성 및 선택적 의미 기반 매칭
                    # 1. 직접 문자열 매칭 먼저 수행 (빠른 필터링)
                    for doc in results:
                        content = doc.get("content", doc.get("text", ""))
                        if not isinstance(content, str) or not content:
                            continue
                        
                        content_lower = content.lower()
                        
                        # 직접 문자열 매칭
                        for keyword in extracted_keywords:
                            if keyword.lower() in content_lower:
                                covered_keywords.add(keyword.lower())
                    
                    # 2. 의미 기반 매칭이 필요한 경우만 수행 (직접 매칭이 부족한 경우)
                    # Keyword Coverage가 이미 높으면 의미 기반 매칭 생략
                    current_coverage = len(covered_keywords) / len(extracted_keywords) if extracted_keywords else 0.0
                    use_semantic_if_needed = current_coverage < 0.7  # 70% 미만일 때만 의미 기반 매칭
                    
                    if use_semantic_if_needed:
                        # 키워드 임베딩 생성 (한 번만)
                        keyword_embeddings = model.encode(extracted_keywords, show_progress_bar=False)
                        
                        # 문서 임베딩 배치 생성 (성능 최적화)
                        doc_texts = []
                        doc_indices = []
                        for idx, doc in enumerate(results):
                            content = doc.get("content", doc.get("text", ""))
                            if isinstance(content, str) and content:
                                doc_text = content[:512] if len(content) > 512 else content
                                doc_texts.append(doc_text)
                                doc_indices.append(idx)
                        
                        if doc_texts:
                            # 배치 임베딩 생성 (개별 생성 대신 배치로 처리)
                            doc_embeddings = model.encode(doc_texts, show_progress_bar=False, batch_size=8)
                            
                            # 배치 유사도 계산
                            for doc_idx, doc_embedding in zip(doc_indices, doc_embeddings):
                                doc = results[doc_idx]
                                try:
                                    # 코사인 유사도 계산
                                    similarities = cosine_similarity(
                                        [doc_embedding],
                                        keyword_embeddings
                                    )[0]
                                    
                                    for idx, similarity in enumerate(similarities):
                                        if similarity >= 0.5:  # 의미적 유사도 임계값
                                            keyword = extracted_keywords[idx]
                                            # 직접 매칭이 없고 의미적 매칭만 있는 경우
                                            if keyword.lower() not in covered_keywords:
                                                if keyword not in semantic_matches:
                                                    semantic_matches[keyword] = similarity
                                                else:
                                                    semantic_matches[keyword] = max(semantic_matches[keyword], similarity)
                                except Exception as e:
                                    self.logger.debug(f"Semantic matching error for document: {e}")
                                    continue
                    else:
                        self.logger.debug(f"Skipping semantic matching (coverage already high: {current_coverage:.2f})")
                    
                    # 의미적 매칭도 포함하여 계산 (의미적 매칭은 80% 가중치로 증가)
                    total_coverage = len(covered_keywords) + len(semantic_matches) * 0.8
                    if extracted_keywords:
                        metrics["keyword_coverage"] = min(1.0, total_coverage / len(extracted_keywords))
                    
                except Exception as e:
                    self.logger.warning(f"Semantic keyword matching failed: {e}, falling back to string matching")
                    use_semantic_matching = False
            
            # 폴백: 기존 문자열 매칭 방식
            if not use_semantic_matching:
                for doc in results:
                    content = doc.get("content", doc.get("text", "")).lower()
                    if isinstance(content, str):
                        for keyword in extracted_keywords:
                            if keyword.lower() in content:
                                covered_keywords.add(keyword.lower())
                
                if extracted_keywords:
                    metrics["keyword_coverage"] = len(covered_keywords) / len(extracted_keywords)
        
        return metrics
    
    def _calculate_comprehensive_diversity_score(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        """
        다차원 다양성 점수 계산
        
        Args:
            results: 검색 결과 리스트
            
        Returns:
            float: 통합 다양성 점수 (0.0 ~ 1.0)
        """
        if not results or len(results) < 2:
            return 0.0
        
        # 1. 단어 기반 다양성 (30%)
        word_diversity = self._calculate_word_diversity(results)
        
        # 2. 의미적 다양성 (40%) - 문서 임베딩 간 코사인 유사도 기반
        semantic_diversity = self._calculate_semantic_diversity(results)
        
        # 3. 타입 다양성 (20%) - 엔트로피 기반
        type_diversity = self._calculate_type_diversity(results)
        
        # 4. 문서 간 유사도 다양성 (10%) - 평균 최소 거리
        inter_doc_diversity = self._calculate_inter_document_diversity(results)
        
        # 가중 평균
        comprehensive_score = (
            0.3 * word_diversity +
            0.4 * semantic_diversity +
            0.2 * type_diversity +
            0.1 * inter_doc_diversity
        )
        
        return max(0.0, min(1.0, comprehensive_score))
    
    def _calculate_word_diversity(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        """
        단어 기반 다양성 계산 (기존 로직)
        
        Returns:
            float: 단어 다양성 점수 (0.0 ~ 1.0)
        """
        contents = [doc.get("content", doc.get("text", "")) for doc in results]
        unique_terms = set()
        total_terms = 0
        for content in contents:
            if isinstance(content, str):
                terms = content.lower().split()
                unique_terms.update(terms)
                total_terms += len(terms)
        
        if total_terms > 0:
            return len(unique_terms) / total_terms
        return 0.0
    
    def _calculate_semantic_diversity(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        """
        의미적 다양성 계산 (임베딩 벡터 간 유사도 기반)
        
        Returns:
            float: 의미적 다양성 점수 (0.0 ~ 1.0, 높을수록 다양함)
        """
        if len(results) < 2:
            return 0.0
        
        # 문서 임베딩 벡터 추출
        embeddings = []
        for doc in results:
            embedding = doc.get("embedding") or doc.get("vector") or doc.get("metadata", {}).get("embedding")
            if embedding is not None:
                if isinstance(embedding, list):
                    if SKLEARN_AVAILABLE:
                        embedding = np.array(embedding)
                    else:
                        continue
                embeddings.append(embedding)
        
        if len(embeddings) < 2 or not SKLEARN_AVAILABLE:
            # 임베딩이 없거나 sklearn이 없으면 폴백: 문서 간 유사도 사용
            return self._calculate_inter_document_diversity(results)
        
        try:
            # 모든 문서 쌍 간 코사인 유사도 계산
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    try:
                        similarity = cosine_similarity(
                            embeddings[i].reshape(1, -1),
                            embeddings[j].reshape(1, -1)
                        )[0][0]
                        similarities.append(float(similarity))
                    except Exception:
                        continue
            
            if not similarities:
                return self._calculate_inter_document_diversity(results)
            
            # 평균 유사도가 낮을수록 다양성 높음
            avg_similarity = sum(similarities) / len(similarities)
            semantic_diversity = 1.0 - avg_similarity
            
            return max(0.0, min(1.0, semantic_diversity))
        except Exception:
            # 오류 발생 시 폴백
            return self._calculate_inter_document_diversity(results)
    
    def _calculate_inter_document_diversity(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        """
        문서 간 유사도 기반 다양성 계산
        
        Returns:
            float: 문서 간 다양성 점수 (0.0 ~ 1.0)
        """
        if len(results) < 2:
            return 0.0
        
        # 문서 쌍 간 유사도 계산
        similarities = []
        for i in range(len(results)):
            doc1_content = (results[i].get("content") or results[i].get("text") or "").lower()
            doc1_words = set(doc1_content.split())
            
            for j in range(i + 1, len(results)):
                doc2_content = (results[j].get("content") or results[j].get("text") or "").lower()
                doc2_words = set(doc2_content.split())
                
                # Jaccard 유사도
                if doc1_words or doc2_words:
                    intersection = len(doc1_words & doc2_words)
                    union = len(doc1_words | doc2_words)
                    similarity = intersection / union if union > 0 else 0.0
                    similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # 평균 유사도가 낮을수록 다양성 높음
        avg_similarity = sum(similarities) / len(similarities)
        diversity = 1.0 - avg_similarity
        
        return max(0.0, min(1.0, diversity))
    
    def _calculate_type_diversity(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        """
        타입 다양성 계산 (엔트로피 기반)
        
        Returns:
            float: 타입 다양성 점수 (0.0 ~ 1.0)
        """
        if not results:
            return 0.0
        
        # 타입별 카운트
        type_counts = {}
        for doc in results:
            doc_type = (
                doc.get("type") or
                doc.get("source_type") or
                doc.get("metadata", {}).get("source_type", "unknown")
            )
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        # 엔트로피 계산
        total = sum(type_counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in type_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # 정규화 (최대 엔트로피)
        max_entropy = math.log2(len(type_counts)) if len(type_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def _has_citation(self, document: Dict[str, Any]) -> bool:
        """문서에 Citation이 있는지 확인"""
        import re
        content = document.get("content", "") or document.get("text", "") or ""
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        law_pattern = r'[가-힣]+법\s*제?\s*\d+\s*조'
        precedent_pattern = r'대법원|법원.*\d{4}[다나마]\d+'
        
        has_law = bool(re.search(law_pattern, content))
        has_precedent = bool(re.search(precedent_pattern, content))
        
        return has_law or has_precedent
    
    def _calculate_citation_match_score(
        self,
        document: Dict[str, Any],
        query: str = "",
        extracted_keywords: List[str] = None
    ) -> float:
        """질문의 법령/판례와 문서 Citation 일치도 계산 (개선: Citation 매칭 정확도 개선)"""
        import re
        
        if extracted_keywords is None:
            extracted_keywords = []
        
        content = document.get("content", "") or document.get("text", "") or ""
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        if not content:
            return 0.0
        
        # 질문에서 법령/판례 추출
        query_laws = re.findall(r'([가-힣]+법)\s*제?\s*(\d+)\s*조', query)
        query_precedents = re.findall(r'대법원.*?(\d{4}[다나마]\d+)', query)
        
        # 문서에서 법령/판례 추출
        doc_laws = re.findall(r'([가-힣]+법)\s*제?\s*(\d+)\s*조', content)
        doc_precedents = re.findall(r'대법원.*?(\d{4}[다나마]\d+)', content)
        
        match_score = 0.0
        
        # 법령 일치도 계산
        if query_laws and doc_laws:
            for q_law, q_article in query_laws:
                for d_law, d_article in doc_laws:
                    if q_law == d_law and q_article == d_article:
                        match_score += 0.5  # 정확 일치
                    elif q_law == d_law:
                        match_score += 0.2  # 법령명만 일치
        
        # 판례 일치도 계산
        if query_precedents and doc_precedents:
            for q_precedent in query_precedents:
                if q_precedent in doc_precedents:
                    match_score += 0.5  # 정확 일치
        
        # 키워드에서 법령/판례 추출
        for keyword in extracted_keywords:
            keyword_laws = re.findall(r'([가-힣]+법)\s*제?\s*(\d+)\s*조', keyword)
            keyword_precedents = re.findall(r'대법원.*?(\d{4}[다나마]\d+)', keyword)
            
            if keyword_laws and doc_laws:
                for k_law, k_article in keyword_laws:
                    for d_law, d_article in doc_laws:
                        if k_law == d_law and k_article == d_article:
                            match_score += 0.3  # 키워드 일치
        
        return min(1.0, match_score)
    
    def _calculate_trust_score(self, document: Dict[str, Any]) -> float:
        """문서 신뢰도 점수 계산"""
        trust_score = 0.5
        
        source = document.get("source", "").lower()
        doc_type = document.get("type", "").lower() if document.get("type") else ""
        
        if "법령" in source or "statute" in source or "법령" in doc_type:
            trust_score += 0.3
        elif "판례" in source or "precedent" in source or "대법원" in source or "판례" in doc_type:
            trust_score += 0.2
        elif "해석" in source or "interpretation" in source:
            trust_score += 0.1
        
        metadata = document.get("metadata", {})
        if isinstance(metadata, dict):
            citation_count = metadata.get("citation_count", 0)
            if citation_count > 0:
                trust_score += min(0.2, citation_count / 100.0)
        
        return min(1.0, trust_score)
    
    def _apply_mmr_diversity(
        self,
        documents: List[Dict[str, Any]],
        query: str = "",
        lambda_score: float = 0.7,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        MMR (Maximal Marginal Relevance) 기반 다양성 적용
        
        Args:
            documents: 문서 리스트
            query: 검색 쿼리
            lambda_score: 관련성 가중치 (0.0-1.0)
            top_k: 반환할 결과 수
            
        Returns:
            List[Dict]: 다양성이 적용된 문서 리스트
        """
        if not documents:
            return []
        
        selected = []
        remaining = documents.copy()
        
        if not remaining:
            return []
        
        selected.append(remaining.pop(0))
        
        while remaining and len(selected) < top_k:
            best_doc = None
            best_score = -1
            
            for doc in remaining:
                relevance = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                
                diversity = 1.0
                if selected:
                    max_similarity = max([
                        self._calculate_doc_similarity(doc, sel_doc)
                        for sel_doc in selected
                    ])
                    diversity = 1.0 - max_similarity
                
                mmr_score = lambda_score * relevance + (1 - lambda_score) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = doc
            
            if best_doc:
                selected.append(best_doc)
                remaining.remove(best_doc)
            else:
                break
        
        return selected
    
    def _get_dynamic_mmr_lambda(
        self,
        query_type: str = "",
        search_quality: float = 0.7,
        num_results: int = 10
    ) -> float:
        """
        동적 MMR lambda 가중치 계산
        
        Args:
            query_type: 질문 유형
            search_quality: 검색 품질 점수 (0.0 ~ 1.0)
            num_results: 검색 결과 수
        
        Returns:
            float: lambda 가중치 (0.0 ~ 1.0, 높을수록 관련성 중시)
        """
        base_lambda = 0.7
        
        # 질문 유형별 조정
        if query_type == "law_inquiry":
            base_lambda = 0.8  # 법령 조회: 관련성 우선
        elif query_type == "precedent_search":
            base_lambda = 0.6  # 판례 검색: 다양성 중시
        elif query_type == "complex_question":
            base_lambda = 0.65  # 복합 질문: 균형
        
        # 검색 품질에 따른 조정
        if search_quality >= 0.8:
            # 품질이 높으면 다양성 강화 가능
            base_lambda -= 0.1
        elif search_quality < 0.6:
            # 품질이 낮으면 관련성 우선
            base_lambda += 0.1
        
        # 결과 수에 따른 조정
        if num_results > 20:
            # 결과가 많으면 다양성 강화
            base_lambda -= 0.05
        elif num_results < 10:
            # 결과가 적으면 관련성 우선
            base_lambda += 0.05
        
        return max(0.5, min(0.9, base_lambda))
    
    def _calculate_doc_similarity(
        self,
        doc1: Dict[str, Any],
        doc2: Dict[str, Any]
    ) -> float:
        """두 문서 간의 유사도 계산 (개선: MMR 다양성 계산 개선)"""
        content1 = doc1.get("content", "") or doc1.get("text", "") or ""
        content2 = doc2.get("content", "") or doc2.get("text", "") or ""
        
        if not isinstance(content1, str):
            content1 = str(content1) if content1 else ""
        if not isinstance(content2, str):
            content2 = str(content2) if content2 else ""
        
        if not content1 or not content2:
            return 0.0
        
        # 1. Jaccard 유사도 (단어 기반)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            jaccard_sim = 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard_sim = intersection / union if union > 0 else 0.0
        
        # 2. 문서 타입 유사도
        type1 = doc1.get("type", "").lower() if doc1.get("type") else ""
        type2 = doc2.get("type", "").lower() if doc2.get("type") else ""
        type_sim = 1.0 if type1 == type2 and type1 else 0.0
        
        # 3. Citation 유사도
        citations1 = self._extract_citations(doc1)
        citations2 = self._extract_citations(doc2)
        
        if citations1 and citations2:
            citation_intersection = len(set(citations1).intersection(set(citations2)))
            citation_union = len(set(citations1).union(set(citations2)))
            citation_sim = citation_intersection / citation_union if citation_union > 0 else 0.0
        else:
            citation_sim = 0.0
        
        # 가중 평균 유사도
        combined_sim = (
            0.5 * jaccard_sim +  # 단어 유사도
            0.2 * type_sim +     # 타입 유사도
            0.3 * citation_sim    # Citation 유사도
        )
        
        return min(1.0, combined_sim)
    
    def _extract_citations(self, document: Dict[str, Any]) -> List[str]:
        """문서에서 Citation 추출 (개선: MMR 다양성 계산 개선)"""
        import re
        content = document.get("content", "") or document.get("text", "") or ""
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        citations = []
        
        # 법령 추출
        laws = re.findall(r'([가-힣]+법)\s*제?\s*(\d+)\s*조', content)
        citations.extend([f"{law} 제{article}조" for law, article in laws])
        
        # 판례 추출
        precedents = re.findall(r'대법원.*?(\d{4}[다나마]\d+)', content)
        citations.extend(precedents)
        
        return citations
    
    def _calculate_information_density(self, document: Dict[str, Any]) -> float:
        """문서 정보 밀도 계산 (개선: 문서 길이/정보 밀도 고려)"""
        import re
        
        content = document.get("content", "") or document.get("text", "") or ""
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        if not content:
            return 0.0
        
        content_length = len(content)
        
        # 최적 길이 범위 (200-2000자)
        if 200 <= content_length <= 2000:
            length_score = 1.0
        elif content_length < 200:
            # 너무 짧음: 페널티
            length_score = max(0.3, content_length / 200.0)
        else:
            # 너무 김: 페널티
            length_score = max(0.5, 1.0 - (content_length - 2000) / 2000.0)
        
        # 법률 용어 밀도
        legal_terms = re.findall(r'[가-힣]+법|제\d+조|대법원|판례|법령', content)
        legal_term_density = len(legal_terms) / max(1, content_length / 100.0)
        legal_term_score = min(1.0, legal_term_density / 5.0)  # 최적 밀도: 5개/100자
        
        # Citation 밀도
        citations = self._extract_citations(document)
        citation_density = len(citations) / max(1, content_length / 500.0)
        citation_score = min(1.0, citation_density / 2.0)  # 최적 밀도: 2개/500자
        
        # 정보 밀도 점수
        density_score = (
            0.4 * length_score +
            0.3 * legal_term_score +
            0.3 * citation_score
        )
        
        return min(1.0, density_score)
    
    def _calculate_recency_score(self, document: Dict[str, Any]) -> float:
        """문서 최신성 점수 계산 (개선: 시간적 관련성 고려)"""
        from datetime import datetime
        
        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        
        # 판례 날짜 추출
        date_str = metadata.get("date") or metadata.get("decision_date") or ""
        if not date_str:
            return 0.5  # 날짜 정보 없음: 중간 점수
        
        try:
            # 날짜 파싱 (형식: "2020-01-01" 또는 "2020.01.01")
            if "." in date_str:
                date_obj = datetime.strptime(date_str.split()[0], "%Y.%m.%d")
            else:
                date_obj = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
            
            # 현재 날짜와의 차이 계산
            current_date = datetime.now()
            days_diff = (current_date - date_obj).days
            
            # 최신성 점수 (최근일수록 높은 점수)
            if days_diff <= 365:
                recency_score = 1.0  # 1년 이내
            elif days_diff <= 1825:
                recency_score = 0.8 - (days_diff - 365) / 1825.0 * 0.3  # 5년 이내
            else:
                recency_score = max(0.3, 0.5 - (days_diff - 1825) / 3650.0 * 0.2)  # 5년 이상
            
            return min(1.0, recency_score)
        except:
            return 0.5
    
    def apply_diversity_filter(self, results: List[Any], max_per_type: int = 5, diversity_weight: float = None) -> List[Any]:
        """
        다양성 필터 적용
        
        Args:
            results: 순위가 매겨진 결과 (MergedResult 또는 Dict)
            max_per_type: 타입별 최대 결과 수
            diversity_weight: 다양성 가중치 (사용하지 않지만 호환성을 위해 유지)
            
        Returns:
            List[Dict]: 다양성이 적용된 결과
        """
        if not results:
            return []
        
        # Dict를 MergedResult로 변환
        converted_results = []
        for result in results:
            if isinstance(result, dict):
                # Dict를 MergedResult로 변환
                text = result.get("text") or result.get("content") or result.get("chunk_text") or ""
                score = result.get("score") or result.get("relevance_score") or result.get("similarity", 0.0)
                source = result.get("source") or result.get("title") or result.get("document_id") or ""
                metadata = result.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = result if isinstance(result, dict) else {}
                
                converted_result = MergedResult(
                    text=text,
                    score=score,
                    source=source,
                    metadata=metadata
                )
                converted_results.append(converted_result)
            elif isinstance(result, MergedResult):
                converted_results.append(result)
            else:
                # 기타 타입은 건너뛰기
                continue
        
        # 타입별 카운터 (source를 타입으로 사용)
        type_counts = {}
        filtered_results = []
        
        for result in converted_results:
            result_type = result.source
            if result_type not in type_counts:
                type_counts[result_type] = 0
            
            if type_counts[result_type] < max_per_type:
                filtered_results.append(result)
                type_counts[result_type] += 1
        
        # Dict로 변환하여 반환 (호환성 유지)
        return [self._merged_result_to_dict(r) for r in filtered_results]


# 기본 인스턴스 생성
def create_result_merger() -> ResultMerger:
    """기본 결과 병합기 생성"""
    return ResultMerger()


def create_result_ranker() -> ResultRanker:
    """기본 결과 순위 결정기 생성"""
    return ResultRanker()


if __name__ == "__main__":
    # 테스트 코드
    merger = create_result_merger()
    ranker = create_result_ranker()
    
    # 샘플 결과
    exact_results = [
        type('obj', (object,), {'text': '민법 제543조', 'score': 0.9, 'metadata': {}})()
    ]
    semantic_results = [
        type('obj', (object,), {'text': '계약 해지 규정', 'score': 0.7, 'metadata': {}})()
    ]
    
    # 병합 및 순위 결정
    merged = merger.merge_results(exact_results, semantic_results)
    ranked = ranker.rank_results(merged)
    
    print(f"Ranked results: {len(ranked)}")
    for result in ranked:
        print(f"  Score: {result.score:.3f}, Source: {result.source}")
        print(f"  Text: {result.text}")