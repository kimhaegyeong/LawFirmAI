# -*- coding: utf-8 -*-
"""
검색 핸들러 모듈
의미적 검색, 키워드 검색 및 결과 병합 로직을 독립 모듈로 분리
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState
try:
    from lawfirm_langgraph.core.workflow.utils.workflow_constants import WorkflowConstants
except ImportError:
    from core.workflow.utils.workflow_constants import WorkflowConstants
try:
    from lawfirm_langgraph.core.workflow.utils.workflow_utils import WorkflowUtils
except ImportError:
    from core.workflow.utils.workflow_utils import WorkflowUtils

# 개선 기능 import (선택적)
try:
    from core.search.optimizers.enhanced_query_expander import EnhancedQueryExpander
    from core.search.optimizers.adaptive_hybrid_weights import AdaptiveHybridWeights
    from core.search.optimizers.adaptive_threshold import AdaptiveThreshold
    from core.search.optimizers.diversity_ranker import DiversityRanker
    from core.search.optimizers.metadata_enhancer import MetadataEnhancer
    from core.search.optimizers.multi_dimensional_quality import MultiDimensionalQualityScorer
    IMPROVEMENT_FEATURES_AVAILABLE = True
except ImportError:
    IMPROVEMENT_FEATURES_AVAILABLE = False

# MergedResult import
try:
    from lawfirm_langgraph.core.search.processors.result_merger import MergedResult
except ImportError:
    try:
        from core.search.processors.result_merger import MergedResult
    except ImportError:
        MergedResult = None

# Score normalization utilities
try:
    from lawfirm_langgraph.core.search.utils.score_utils import normalize_score
except ImportError:
    try:
        from core.search.utils.score_utils import normalize_score
    except ImportError:
        def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
            """Fallback normalization function"""
            if score < min_val:
                return float(min_val)
            elif score > max_val:
                excess = score - max_val
                normalized = max_val + (excess / (1.0 + excess * 10))
                return float(max(0.0, min(1.0, normalized)))
            return float(score)


class SearchHandler:
    """
    검색 실행 및 결과 병합 클래스

    의미적 검색, 키워드 검색, 결과 병합 및 재정렬을 처리합니다.
    """
    
    def _expand_synonyms_simple(self, query: str, keywords: List[str]) -> List[str]:
        """
        간단한 법률 용어 동의어 확장
        
        Args:
            query: 원본 쿼리
            keywords: 추출된 키워드
        
        Returns:
            동의어 리스트
        """
        synonyms = []
        
        # 법률 용어 동의어 사전 (간단한 버전)
        legal_synonyms = {
            "보증금": ["임대보증금", "전세금", "보증금액"],
            "반환": ["반납", "지급", "환불", "반환금"],
            "임대차": ["임대", "임차", "전세", "월세"],
            "계약": ["약정", "계약서", "계약관계"],
            "해지": ["해제", "종료", "만료", "해약"],
            "손해": ["손실", "피해", "손해액"],
            "배상": ["보상", "배상금", "손해배상"],
            "이혼": ["이혼소송", "이혼절차"],
            "재산": ["재산분할", "재산권", "재산분할청구"],
            "상속": ["상속분", "상속재산", "상속인"],
        }
        
        # 키워드와 쿼리에서 동의어 찾기
        all_terms = keywords + query.split()
        for term in all_terms:
            if term in legal_synonyms:
                synonyms.extend(legal_synonyms[term])
        
        # 중복 제거 및 최대 5개로 제한
        return list(set(synonyms))[:5]

    def __init__(
        self,
        semantic_search: Any,
        keyword_mapper: Any,
        data_connector: Any,
        result_merger: Any,
        result_ranker: Any,
        performance_optimizer: Any,
        config: Any,
        logger: Optional[logging.Logger] = None
    ):
        """
        SearchHandler 초기화

        Args:
            semantic_search: 의미적 검색 엔진
            keyword_mapper: 키워드 매퍼
            data_connector: 데이터 커넥터
            result_merger: 결과 병합기
            result_ranker: 결과 재정렬기
            performance_optimizer: 성능 최적화기
            config: 설정 객체
            logger: 로거 (없으면 자동 생성)
        """
        self.semantic_search_engine = semantic_search
        self.keyword_mapper = keyword_mapper
        self.data_connector = data_connector
        self.result_merger = result_merger
        self.result_ranker = result_ranker
        self.performance_optimizer = performance_optimizer
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # 개선 기능 초기화 (선택적)
        self._init_improvement_features()
    
    def _init_improvement_features(self):
        """개선 기능 초기화"""
        self.use_improvements = getattr(self.config, 'enable_search_improvements', True)
        
        if not IMPROVEMENT_FEATURES_AVAILABLE:
            self.use_improvements = False
            self.logger.debug("Search improvement features not available")
            return
        
        if self.use_improvements:
            try:
                self.query_expander = EnhancedQueryExpander()
                self.adaptive_weights = AdaptiveHybridWeights()
                self.adaptive_threshold = AdaptiveThreshold(
                    base_threshold=getattr(self.config, 'similarity_threshold', 0.5)
                )
                self.diversity_ranker = DiversityRanker()
                self.metadata_enhancer = MetadataEnhancer()
                self.quality_scorer = MultiDimensionalQualityScorer()
                self.logger.info("Search improvement features initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize improvement features: {e}")
                self.use_improvements = False
        else:
            self.logger.debug("Search improvements disabled by config")

    def check_cache(
        self,
        state: LegalWorkflowState,
        query: str,
        query_type_str: str,
        start_time: float
    ) -> bool:
        """캐시에서 문서 확인"""
        cached_documents = self.performance_optimizer.cache.get_cached_documents(query, query_type_str)
        if cached_documents:
            WorkflowUtils.set_state_value(state, "retrieved_docs", cached_documents)
            WorkflowUtils.add_step(state, "문서 검색 완료", f"문서 검색 완료: {len(cached_documents)}개 (캐시)")
            self.logger.info(f"Using cached documents for query: {query[:50]}...")
            WorkflowUtils.update_processing_time(state, start_time)
            return True
        return False

    def semantic_search(self, query: str, k: Optional[int] = None, extracted_keywords: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], int]:
        """의미적 벡터 검색"""
        if not self.semantic_search_engine:
            self.logger.info("Semantic search not available")
            return [], 0

        if hasattr(self.semantic_search_engine, 'is_available'):
            if not self.semantic_search_engine.is_available():
                self.logger.warning("Semantic search engine is not available")
                if hasattr(self.semantic_search_engine, 'diagnose'):
                    diagnosis = self.semantic_search_engine.diagnose()
                    self.logger.warning(f"Semantic search diagnosis: {diagnosis}")
                return [], 0

        try:
            search_k = k if k is not None else WorkflowConstants.SEMANTIC_SEARCH_K
            
            # 검색 결과 수 증가: 다양성 보장을 위해 더 많은 후보 확보
            # 원래 k의 1.5배로 검색하여 판례/결정례 포함 확률 증가 (개선: 2배 → 1.5배로 최적화)
            expanded_k = int(search_k * 1.5)
            
            # 검색 품질 강화: similarity_threshold 조정
            # IndexIVFPQ 인덱스 사용 시 더 낮은 threshold 사용 (압축 인덱스이므로)
            config_threshold = getattr(self.config, 'similarity_threshold', 0.3)
            
            # IndexIVFPQ 인덱스 사용 여부 확인
            is_indexivfpq = False
            if self.semantic_search_engine and hasattr(self.semantic_search_engine, 'index') and self.semantic_search_engine.index:
                index_type = type(self.semantic_search_engine.index).__name__
                if 'IndexIVFPQ' in index_type:
                    is_indexivfpq = True
                    self.logger.info(f"🔍 [SEMANTIC SEARCH] IndexIVFPQ detected, using lower similarity_threshold")
            
            # IndexIVFPQ 인덱스 사용 시 더 낮은 threshold (0.15로 낮춤, 0.85 목표 달성)
            if is_indexivfpq:
                similarity_threshold = max(0.15, config_threshold)  # IndexIVFPQ는 0.15부터 시작 (더 많은 결과 확보)
            else:
                # Ko-Legal-SBERT 모델 사용 시 config_threshold 사용 (높은 관련성 결과만 반환)
                similarity_threshold = config_threshold  # 일반 인덱스는 config 값 사용 (기본값 0.75)
            
            # Query Expansion (개선 기능)
            enhanced_query = query
            if self.use_improvements and hasattr(self, 'query_expander') and extracted_keywords:
                try:
                    expanded_query = self.query_expander.expand_query(
                        query=query,
                        query_type="general_question",  # query_type은 별도로 전달받을 수 있음
                        extracted_keywords=extracted_keywords
                    )
                    if expanded_query.expanded_keywords:
                        # 확장된 키워드를 쿼리에 추가
                        expanded_keywords_str = ' '.join(expanded_query.expanded_keywords[:5])
                        enhanced_query = f"{query} {expanded_keywords_str}"
                        self.logger.info(f"🔍 [QUERY EXPANSION] Expanded query: {len(expanded_query.expanded_keywords)} keywords")
                except Exception as e:
                    self.logger.debug(f"Query expansion failed: {e}, using original query")
            
            # 기존 방식 (폴백)
            if enhanced_query == query and extracted_keywords and len(extracted_keywords) > 0:
                # 핵심 키워드 추출 (법령명, 조문번호, 핵심 용어 우선)
                core_keywords = []
                for kw in extracted_keywords[:5]:
                    if isinstance(kw, str):
                        # 법령명이나 조문번호가 포함된 키워드 우선
                        if any(term in kw for term in ["법", "조", "제", "민법", "형법", "상법"]):
                            core_keywords.insert(0, kw)
                        else:
                            core_keywords.append(kw)
                
                if core_keywords:
                    # 쿼리에 핵심 키워드 추가 (중복 제거)
                    query_keywords = set(query.split())
                    new_keywords = [kw for kw in core_keywords if kw not in query_keywords]
                    if new_keywords:
                        enhanced_query = f"{query} {' '.join(new_keywords[:5])}"
                        
                        # 동의어 확장 추가 (간단한 법률 용어 동의어)
                        synonyms = self._expand_synonyms_simple(query, new_keywords)
                        if synonyms:
                            enhanced_query = f"{enhanced_query} {' '.join(synonyms[:3])}"
                        
                        self.logger.info(f"🔍 [SEMANTIC SEARCH] Enhanced query with keywords: '{enhanced_query[:100]}...'")
            
            # 기본 검색 수행 (향상된 쿼리 사용)
            # 벡터 검색 실행 (쿼리 전처리 포함)
            self.logger.info(f"Calling semantic_search_engine.search with query: '{enhanced_query[:50]}...'")
            results = self.semantic_search_engine.search(enhanced_query, k=expanded_k, similarity_threshold=similarity_threshold)
            self.logger.info(f"Semantic search returned {len(results)} results")
            
            # 검색 결과의 관련성 검증 강화
            if extracted_keywords and len(extracted_keywords) > 0:
                # 질문의 핵심 키워드가 검색 결과에 포함되어 있는지 확인
                core_keywords_lower = [str(kw).lower() for kw in extracted_keywords[:5] if isinstance(kw, str)]
                filtered_results = []
                for r in results:
                    text_content = (
                        r.get('text', '') or
                        r.get('content', '') or
                        str(r.get('metadata', {}).get('content', '')) or
                        str(r.get('metadata', {}).get('text', '')) or
                        ''
                    ).lower()
                    
                    # 핵심 키워드 중 하나라도 포함되어 있으면 관련성 있음
                    has_relevant_keyword = any(kw in text_content for kw in core_keywords_lower if len(kw) > 2)
                    
                    if has_relevant_keyword or len(core_keywords_lower) == 0:
                        filtered_results.append(r)
                    else:
                        self.logger.debug(f"🔍 [SEMANTIC SEARCH] Filtered out result (no relevant keywords): {r.get('id', 'unknown')[:50]}")
                
                if len(filtered_results) < len(results):
                    self.logger.info(f"🔍 [SEMANTIC SEARCH] Filtered {len(results) - len(filtered_results)} irrelevant results")
                    results = filtered_results
            
            # 확장된 키워드가 있으면 추가 검색 수행
            if extracted_keywords and len(extracted_keywords) > 0:
                # 판례/결정례/해석례/법령 검색 강화를 위한 키워드 필터링
                precedent_keywords = [kw for kw in extracted_keywords if any(term in str(kw).lower() for term in ["판례", "대법원", "판결", "선고", "사건", "참고", "유사"])]
                decision_keywords = [kw for kw in extracted_keywords if any(term in str(kw).lower() for term in ["결정", "심판", "의견", "통보", "결정례"])]
                interpretation_keywords = [kw for kw in extracted_keywords if any(term in str(kw).lower() for term in ["해석", "해석례", "유권해석", "법리 해석"])]
                statute_keywords = [kw for kw in extracted_keywords if any(term in str(kw).lower() for term in ["법령", "법률", "조문", "조", "항", "호", "민법", "형법", "상법", "행정법", "헌법", "노동법", "가족법"])]
                
                # 검색 결과 타입 분포 확인
                result_types = {}
                for r in results:
                    r_type = r.get("type") or r.get("source_type") or (r.get("metadata", {}).get("source_type") if isinstance(r.get("metadata"), dict) else "")
                    result_types[r_type] = result_types.get(r_type, 0) + 1
                
                has_precedent = result_types.get("case_paragraph", 0) > 0
                has_decision = result_types.get("decision_paragraph", 0) > 0
                has_interpretation = result_types.get("interpretation_paragraph", 0) > 0
                has_statute = result_types.get("statute_article", 0) > 0
                
                self.logger.info(f"🔍 [SEMANTIC SEARCH] Initial results type distribution: {result_types}")
                
                # 판례 검색 강화: 판례 관련 키워드가 있지만 결과에 판례가 없으면 별도 검색
                if precedent_keywords and not has_precedent:
                    self.logger.info(f"🔍 [PRECEDENT SEARCH] Performing dedicated precedent search with {len(precedent_keywords)} keywords")
                    precedent_query = f"{query} {' '.join(precedent_keywords[:3])}"
                    precedent_results = self.semantic_search_engine.search(
                        query=precedent_query,
                        k=search_k // 2,
                        source_types=["case_paragraph"],
                        similarity_threshold=max(0.35, similarity_threshold - 0.05)  # 판례는 더 낮은 임계값
                    )
                    results.extend(precedent_results)
                    self.logger.info(f"🔍 [PRECEDENT SEARCH] Found {len(precedent_results)} additional precedent results")
                
                # 결정례 검색 강화: 결정례 관련 키워드가 있지만 결과에 결정례가 없으면 별도 검색
                if decision_keywords and not has_decision:
                    self.logger.info(f"🔍 [DECISION SEARCH] Performing dedicated decision search with {len(decision_keywords)} keywords")
                    decision_query = f"{query} {' '.join(decision_keywords[:3])}"
                    decision_results = self.semantic_search_engine.search(
                        query=decision_query,
                        k=search_k // 2,
                        source_types=["decision_paragraph"],
                        similarity_threshold=max(0.35, similarity_threshold - 0.05)  # 결정례는 더 낮은 임계값
                    )
                    results.extend(decision_results)
                    self.logger.info(f"🔍 [DECISION SEARCH] Found {len(decision_results)} additional decision results")
                
                # 해석례 검색 강화: 해석례 관련 키워드가 있지만 결과에 해석례가 없으면 별도 검색
                if interpretation_keywords and not has_interpretation:
                    self.logger.info(f"🔍 [INTERPRETATION SEARCH] Performing dedicated interpretation search with {len(interpretation_keywords)} keywords")
                    interpretation_query = f"{query} {' '.join(interpretation_keywords[:3])}"
                    interpretation_results = self.semantic_search_engine.search(
                        query=interpretation_query,
                        k=search_k // 2,
                        source_types=["interpretation_paragraph"],
                        similarity_threshold=max(0.35, similarity_threshold - 0.05)  # 해석례는 더 낮은 임계값
                    )
                    results.extend(interpretation_results)
                    self.logger.info(f"🔍 [INTERPRETATION SEARCH] Found {len(interpretation_results)} additional interpretation results")
                
                # 법령 검색 강화: 법령 관련 키워드가 있지만 결과에 법령이 없으면 별도 검색
                if statute_keywords and not has_statute:
                    self.logger.info(f"🔍 [STATUTE SEARCH] Performing dedicated statute search with {len(statute_keywords)} keywords")
                    statute_query = f"{query} {' '.join(statute_keywords[:3])}"
                    statute_results = self.semantic_search_engine.search(
                        query=statute_query,
                        k=search_k // 2,
                        source_types=["statute_article"],
                        similarity_threshold=max(0.35, similarity_threshold - 0.05)  # 법령은 더 낮은 임계값
                    )
                    results.extend(statute_results)
                    self.logger.info(f"🔍 [STATUTE SEARCH] Found {len(statute_results)} additional statute results")
                
                # 쿼리 확장 검색도 수행 (추가 결과 확보)
                if precedent_keywords or decision_keywords or interpretation_keywords or statute_keywords:
                    self.logger.info(f"🔍 [SEMANTIC SEARCH] Using query expansion with {len(extracted_keywords)} expanded keywords")
                    expansion_results = self.semantic_search_engine.search_with_query_expansion(
                        query=query,
                        k=search_k // 2,
                        similarity_threshold=similarity_threshold,
                        expanded_keywords=extracted_keywords,
                        use_query_variations=True
                    )
                    # 중복 제거하면서 추가
                    seen_ids = {r.get("metadata", {}).get("chunk_id") for r in results if r.get("metadata", {}).get("chunk_id")}
                    for r in expansion_results:
                        r_id = r.get("metadata", {}).get("chunk_id")
                        if r_id and r_id not in seen_ids:
                            results.append(r)
                            seen_ids.add(r_id)
            
            self.logger.info(f"Semantic search found {len(results)} results (expanded k={expanded_k}, threshold={similarity_threshold})")

            # 검색 결과 타입별 다양성 보장
            results = self._ensure_diverse_source_types(results, search_k)
            
            formatted_results = []
            for result in results:
                # content 필드도 text로 매핑 (검색 결과 형식 통일)
                text_content = (
                    result.get('text', '') or
                    result.get('content', '') or
                    str(result.get('metadata', {}).get('content', '')) or
                    str(result.get('metadata', {}).get('text', '')) or
                    ''
                )
                
                # content 필드가 비어있으면 경고 및 로깅
                if not text_content or len(text_content.strip()) == 0:
                    self.logger.warning(f"⚠️ [SEMANTIC SEARCH] Empty content for result: {result.get('id', 'unknown')}, metadata: {result.get('metadata', {})}")
                    # metadata에서 추가 시도
                    metadata = result.get('metadata', {})
                    if isinstance(metadata, dict):
                        text_content = metadata.get('content') or metadata.get('text') or ''
                    if not text_content:
                        continue  # content가 없으면 건너뛰기
                
                # metadata에 content 필드 명시적으로 저장
                metadata = result.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = result if isinstance(result, dict) else {}
                metadata['content'] = text_content
                metadata['text'] = text_content
                
                formatted_results.append({
                    'id': f"semantic_{result.get('metadata', {}).get('id', hash(text_content))}",
                    'content': text_content,  # content 필드 보장
                    'text': text_content,  # text 필드도 추가 (호환성)
                    'source': result.get('source', 'Vector Search'),
                    'relevance_score': result.get('score', 0.8),
                    'type': result.get('type', 'unknown'),
                    'metadata': metadata,  # content 필드가 포함된 metadata 저장
                    'search_type': 'semantic'
                })

            return formatted_results, len(results)
        except Exception as e:
            self.logger.warning(f"Semantic search failed: {e}")
            return [], 0
    
    def semantic_search_batch(self, queries: List[str], k: Optional[int] = None, extracted_keywords: Optional[List[str]] = None) -> List[Tuple[List[Dict[str, Any]], int]]:
        """여러 쿼리를 배치로 의미적 벡터 검색 (배치 검색 최적화)"""
        if not self.semantic_search_engine:
            self.logger.info("Semantic search not available")
            return [([], 0) for _ in queries]
        
        if hasattr(self.semantic_search_engine, 'is_available'):
            if not self.semantic_search_engine.is_available():
                self.logger.warning("Semantic search engine is not available")
                return [([], 0) for _ in queries]
        
        try:
            search_k = k if k is not None else WorkflowConstants.SEMANTIC_SEARCH_K
            expanded_k = int(search_k * 1.5)
            
            config_threshold = getattr(self.config, 'similarity_threshold', 0.3)
            
            # IndexIVFPQ 인덱스 사용 여부 확인
            is_indexivfpq = False
            if self.semantic_search_engine and hasattr(self.semantic_search_engine, 'index') and self.semantic_search_engine.index:
                index_type = type(self.semantic_search_engine.index).__name__
                if 'IndexIVFPQ' in index_type:
                    is_indexivfpq = True
            
            similarity_threshold = max(0.15, config_threshold) if is_indexivfpq else config_threshold
            
            # 배치 검색 수행
            if hasattr(self.semantic_search_engine, '_search_batch_with_threshold'):
                batch_results = self.semantic_search_engine._search_batch_with_threshold(
                    queries=queries,
                    k=expanded_k,
                    source_types=None,
                    similarity_threshold=similarity_threshold,
                    embedding_version_id=None
                )
                
                # 결과 포맷팅 (semantic_search와 동일한 형식)
                formatted_batch_results = []
                for results in batch_results:
                    formatted_results = []
                    for result in results:
                        text_content = (
                            result.get('text', '') or
                            result.get('content', '') or
                            str(result.get('metadata', {}).get('content', '')) or
                            str(result.get('metadata', {}).get('text', '')) or
                            ''
                        )
                        if not text_content:
                            continue
                        
                        metadata = result.get('metadata', {})
                        if not isinstance(metadata, dict):
                            metadata = result if isinstance(result, dict) else {}
                        metadata['content'] = text_content
                        metadata['text'] = text_content
                        
                        formatted_results.append({
                            'id': f"semantic_{result.get('id', hash(text_content))}",
                            'content': text_content,
                            'text': text_content,
                            'source': result.get('source', 'Vector Search'),
                            'relevance_score': result.get('relevance_score', 0.8),
                            'type': result.get('type', 'unknown'),
                            'metadata': metadata,
                            'search_type': 'semantic'
                        })
                    
                    formatted_batch_results.append((formatted_results, len(formatted_results)))
                
                return formatted_batch_results
            else:
                # 폴백: 개별 검색
                self.logger.warning("Batch search not available, falling back to individual searches")
                return [self.semantic_search(q, k, extracted_keywords) for q in queries]
        
        except Exception as e:
            self.logger.warning(f"Batch semantic search failed: {e}")
            return [([], 0) for _ in queries]

    def keyword_search(
        self,
        query: str,
        query_type_str: str,
        limit: Optional[int] = None,
        legal_field: str = "",
        extracted_keywords: List[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """향상된 키워드 기반 검색"""
        try:
            category_mapping = WorkflowUtils.get_category_mapping()
            categories_to_search = category_mapping.get(query_type_str, ["civil_law"])

            # 지원되는 법률 분야만 매핑
            field_category_map = {
                "civil": "civil_law",
                "criminal": "criminal_law",
                "intellectual_property": "intellectual_property",
                "administrative": "administrative_law"
            }

            preferred_category = None
            if legal_field and legal_field in field_category_map:
                preferred_category = field_category_map[legal_field]
                if preferred_category in categories_to_search:
                    categories_to_search.remove(preferred_category)
                    categories_to_search.insert(0, preferred_category)

            keyword_results = []
            search_limit = limit if limit is not None else WorkflowConstants.CATEGORY_SEARCH_LIMIT

            # 확장된 키워드를 쿼리에 추가
            enhanced_query = query
            if extracted_keywords and len(extracted_keywords) > 0:
                safe_keywords = [kw for kw in extracted_keywords[:3] if isinstance(kw, str)]
                if safe_keywords:
                    enhanced_query = f"{query} {' '.join(safe_keywords)}"

            for category in categories_to_search:
                # 키워드 검색은 항상 FTS5 검색 수행 (force_fts=True)
                category_docs = self.data_connector.search_documents(
                    enhanced_query, category, limit=search_limit, force_fts=True
                )

                # 결과가 없으면 원본 쿼리로도 검색 시도 (폴백)
                if len(category_docs) == 0 and enhanced_query != query:
                    self.logger.debug(f"No results with enhanced query, trying original query for {category}")
                    category_docs = self.data_connector.search_documents(
                        query, category, limit=search_limit, force_fts=True
                    )

                # 여전히 결과가 없으면 키워드만 추출하여 검색 시도
                if len(category_docs) == 0 and extracted_keywords:
                    safe_keywords = [kw for kw in extracted_keywords[:5] if isinstance(kw, str) and len(kw) > 1]
                    if safe_keywords:
                        keyword_only_query = " ".join(safe_keywords)
                        self.logger.debug(f"No results with full query, trying keywords only: {keyword_only_query}")
                        category_docs = self.data_connector.search_documents(
                            keyword_only_query, category, limit=search_limit, force_fts=True
                        )

                for doc in category_docs:
                    doc['search_type'] = 'keyword'
                    doc['category'] = category
                    # 카테고리 일치도 점수 추가
                    if preferred_category and category == preferred_category:
                        doc['category_boost'] = 1.2
                    else:
                        doc['category_boost'] = 1.0

                keyword_results.extend(category_docs)
                self.logger.info(f"Found {len(category_docs)} documents in category: {category}")

            return keyword_results, len(keyword_results)
        except Exception as e:
            self.logger.warning(f"Keyword search failed: {e}")
            return [], 0

    def merge_and_rerank_search_results(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        query: str,
        optimized_queries: Dict[str, Any],
        rerank_params: Dict[str, Any]
    ) -> List[Dict]:
        """결과 통합 및 재정렬 (Rerank)"""
        try:
            # 1. 중복 제거 강화 (우선순위: source_id > doc_id > content_hash)
            seen_source_ids = {}  # 동일 소스 문서의 중복 청크 제거용 (최우선)
            seen_doc_ids = {}  # 동일 문서의 중복 청크 제거용 (차순위)
            seen_texts = set()  # 텍스트 해시 기반 중복 제거 (최후)
            unique_results = []

            for doc in semantic_results + keyword_results:
                doc_content = doc.get("content", "") or doc.get("text", "")
                if not doc_content:
                    continue

                source_type = doc.get("source_type") or doc.get("type", "")
                source_id = doc.get("metadata", {}).get("source_id") if isinstance(doc.get("metadata"), dict) else None
                doc_id = doc.get("doc_id")
                
                is_duplicate = False
                
                # 우선순위 1: source_id 기반 중복 체크 (최우선)
                if source_id and source_type:
                    source_key = f"{source_type}_{source_id}"
                    if source_key in seen_source_ids:
                        # 동일 소스 문서에서 이미 결과가 있으면 무조건 건너뛰기 (점수와 무관)
                        is_duplicate = True
                    else:
                        seen_source_ids[source_key] = doc
                
                # 우선순위 2: doc_id 기반 중복 체크 (차순위)
                if not is_duplicate and doc_id and source_type:
                    doc_key = f"{source_type}_doc_{doc_id}"
                    if doc_key in seen_doc_ids:
                        # 동일 문서에서 이미 결과가 있으면 무조건 건너뛰기 (점수와 무관)
                        is_duplicate = True
                    else:
                        seen_doc_ids[doc_key] = doc
                
                # 우선순위 3: 텍스트 해시 기반 중복 체크 (최후)
                if not is_duplicate:
                    content_preview = doc_content[:200]
                    content_hash = hash(content_preview)
                    if content_hash in seen_texts:
                        is_duplicate = True
                    else:
                        seen_texts.add(content_hash)
                
                if not is_duplicate:
                    unique_results.append(doc)

            # 2. 유사도 점수 정규화 및 통합 (개선: 동적 가중치 조정)
            query_type = optimized_queries.get("query_type", "")
            
            for doc in unique_results:
                semantic_score = doc.get("relevance_score", 0.0)
                keyword_score = doc.get("score", doc.get("relevance_score", 0.0))
                
                # 질문 타입별 동적 가중치 조정
                if query_type == "law_inquiry":
                    # 법령 조회: 키워드 검색 강조 (정확한 법령명/조문번호 매칭)
                    if doc.get("search_type") == "semantic":
                        combined_score = 0.5 * semantic_score + 0.5 * keyword_score
                    else:
                        combined_score = 0.3 * semantic_score + 0.7 * keyword_score
                elif query_type == "precedent_search":
                    # 판례 검색: 의미적 검색 강조 (의미적 유사도 중요)
                    if doc.get("search_type") == "semantic":
                        combined_score = 0.8 * semantic_score + 0.2 * keyword_score
                    else:
                        combined_score = 0.6 * semantic_score + 0.4 * keyword_score
                else:
                    # 기본: 검색 타입별 가중치
                    if doc.get("search_type") == "semantic":
                        combined_score = 0.7 * semantic_score + 0.3 * keyword_score
                    else:
                        combined_score = 0.5 * semantic_score + 0.5 * keyword_score

                # 카테고리 부스트 적용
                category_boost = doc.get("category_boost", 1.0)
                combined_score *= category_boost
                
                # 문서 타입별 추가 보너스
                doc_type = doc.get("doc_type", "").lower() if doc.get("doc_type") else ""
                if doc_type == "statute_article" and query_type == "law_inquiry":
                    combined_score *= 1.2  # 법령 조회 시 법령 조문 20% 보너스
                elif "case" in doc_type and query_type == "precedent_search":
                    combined_score *= 1.15  # 판례 검색 시 판례 15% 보너스

                doc["combined_score"] = combined_score

            # 2.5. 점수 정규화 체크포인트 (모든 점수 필드 정규화)
            for doc in unique_results:
                # 모든 점수 필드 정규화
                if "relevance_score" in doc:
                    original_score = doc["relevance_score"]
                    doc["relevance_score"] = normalize_score(float(original_score))
                    if original_score > 1.0:
                        self.logger.warning(
                            f"⚠️ [SCORE NORMALIZATION] relevance_score 초과 감지: "
                            f"original={original_score:.3f}, normalized={doc['relevance_score']:.3f}"
                        )
                if "similarity" in doc:
                    original_score = doc["similarity"]
                    doc["similarity"] = normalize_score(float(original_score))
                    if original_score > 1.0:
                        self.logger.warning(
                            f"⚠️ [SCORE NORMALIZATION] similarity 초과 감지: "
                            f"original={original_score:.3f}, normalized={doc['similarity']:.3f}"
                        )
                if "score" in doc:
                    original_score = doc["score"]
                    doc["score"] = normalize_score(float(original_score))
                    if original_score > 1.0:
                        self.logger.warning(
                            f"⚠️ [SCORE NORMALIZATION] score 초과 감지: "
                            f"original={original_score:.3f}, normalized={doc['score']:.3f}"
                        )
                if "final_weighted_score" in doc:
                    original_score = doc["final_weighted_score"]
                    doc["final_weighted_score"] = normalize_score(float(original_score))
                    if original_score > 1.0:
                        self.logger.warning(
                            f"⚠️ [SCORE NORMALIZATION] final_weighted_score 초과 감지: "
                            f"original={original_score:.3f}, normalized={doc['final_weighted_score']:.3f}"
                        )
                if "combined_score" in doc:
                    original_score = doc["combined_score"]
                    doc["combined_score"] = normalize_score(float(original_score))
                    if original_score > 1.0:
                        self.logger.warning(
                            f"⚠️ [SCORE NORMALIZATION] combined_score 초과 감지: "
                            f"original={original_score:.3f}, normalized={doc['combined_score']:.3f}"
                        )

            # 3. Reranker를 사용한 재정렬
            if self.result_ranker and len(unique_results) > 0:
                try:
                    # 문서 구조 검증: text 필드가 있는지 확인 (복사본 생성)
                    validated_results = []
                    for doc in unique_results[:rerank_params["top_k"]]:
                        if isinstance(doc, dict):
                            # 문서 복사본 생성
                            doc_copy = dict(doc)
                            # text 필드가 없으면 content나 chunk_text에서 가져오기
                            if "text" not in doc_copy or not doc_copy.get("text"):
                                doc_copy["text"] = doc_copy.get("content") or doc_copy.get("chunk_text") or doc_copy.get("text_content", "")
                            # source 필드도 확인
                            if "source" not in doc_copy or not doc_copy.get("source"):
                                doc_copy["source"] = doc_copy.get("title") or doc_copy.get("document_id") or doc_copy.get("source_name", "")
                            validated_results.append(doc_copy)
                    
                    if validated_results:
                        rerank_results = self.result_ranker.rank_results(
                            validated_results,
                            top_k=rerank_params["top_k"],
                            query=query
                        )
                    else:
                        raise ValueError("No valid documents for reranking")
                except Exception as e:
                    self.logger.warning(f"Reranker failed, using combined score: {e}")
                    rerank_results = sorted(
                        unique_results,
                        key=lambda x: x.get("combined_score", 0.0),
                        reverse=True
                    )[:rerank_params["top_k"]]
            else:
                # 폴백: 결합 점수로 정렬
                rerank_results = sorted(
                    unique_results,
                    key=lambda x: x.get("combined_score", 0.0),
                    reverse=True
                )[:rerank_params["top_k"]]

            # 4. 다양성 필터 적용
            try:
                if self.result_ranker and hasattr(self.result_ranker, 'apply_diversity_filter'):
                    # 문서 구조 검증: source 필드가 있는지 확인 (복사본 생성)
                    validated_for_diversity = []
                    for doc in rerank_results:
                        if isinstance(doc, dict):
                            # 문서 복사본 생성
                            doc_copy = dict(doc)
                            # source 필드가 없으면 다른 필드에서 가져오기
                            if "source" not in doc_copy or not doc_copy.get("source"):
                                doc_copy["source"] = doc_copy.get("title") or doc_copy.get("document_id") or doc_copy.get("source_name", "")
                            validated_for_diversity.append(doc_copy)
                    
                    if validated_for_diversity:
                        # apply_diversity_filter 메서드의 시그니처 확인
                        import inspect
                        sig = inspect.signature(self.result_ranker.apply_diversity_filter)
                        params = list(sig.parameters.keys())
                        
                        # diversity_weight 파라미터가 있는지 확인
                        if "diversity_weight" in params:
                            diverse_results = self.result_ranker.apply_diversity_filter(
                                validated_for_diversity,
                                max_per_type=5,
                                diversity_weight=rerank_params.get("diversity_weight", 0.3)
                            )
                        else:
                            # diversity_weight 파라미터가 없으면 제외하고 호출
                            diverse_results = self.result_ranker.apply_diversity_filter(
                                validated_for_diversity,
                                max_per_type=5
                            )
                    else:
                        diverse_results = rerank_results
                else:
                    diverse_results = rerank_results
            except Exception as e:
                self.logger.warning(f"Diversity filter failed: {e}")
                diverse_results = rerank_results

            # 5. 타입별 다양성 보장 (최종 결과에 판례/결정례 포함)
            final_diverse_results = self._ensure_diverse_source_types(
                diverse_results,
                rerank_params.get("top_k", 10)
            )

            return final_diverse_results

        except Exception as e:
            self.logger.warning(f"Merge and rerank failed: {e}, using simple merge")
            # 폴백: 간단한 병합
            all_results = semantic_results + keyword_results
            return sorted(
                all_results,
                key=lambda x: x.get("relevance_score", 0.0),
                reverse=True
            )[:rerank_params.get("top_k", 10)]
    
    def _ensure_diverse_source_types(self, results: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
        """
        검색 결과의 타입별 다양성 보장
        
        검색 결과가 특정 타입(예: 법령)에 치우쳐 있을 때,
        다른 타입(판례, 해석례, 결정례)도 포함되도록 재분배
        
        Args:
            results: 검색 결과 리스트
            max_results: 최대 결과 수
            
        Returns:
            타입별로 균형있게 재분배된 결과 리스트
        """
        if not results or len(results) <= 1:
            return results
        
        # source_type 매핑
        source_type_mapping = {
            "statute_article": "law",
            "case_paragraph": "precedent",
            "decision_paragraph": "decision",
            "interpretation_paragraph": "interpretation"
        }
        
        # 결과를 타입별로 그룹화
        results_by_type = {
            "law": [],
            "precedent": [],
            "decision": [],
            "interpretation": [],
            "unknown": []
        }
        
        for result in results:
            # 타입 추출
            doc_type = (
                result.get("type") or
                result.get("source_type") or
                (result.get("metadata", {}).get("source_type") if isinstance(result.get("metadata"), dict) else "") or
                "unknown"
            )
            
            # source_type 매핑 적용
            if doc_type in source_type_mapping:
                doc_type = source_type_mapping[doc_type]
            elif doc_type not in results_by_type:
                doc_type = "unknown"
            
            results_by_type[doc_type].append(result)
        
        # 타입별 개수 확인
        type_counts = {k: len(v) for k, v in results_by_type.items()}
        total_count = sum(type_counts.values())
        
        # 모든 결과가 같은 타입이면 그대로 반환
        if len([c for c in type_counts.values() if c > 0]) <= 1:
            self.logger.debug(f"All results are of the same type, returning as-is")
            return results[:max_results]
        
        # 개선 #6: 타입별로 균형있게 재분배 (타입별 최소 할당량 설정)
        # 타입별 최소 할당량: max_results에 따라 동적 조정 (소스 타입 다양성 개선)
        # max_results가 5개 이상일 때만 최소 할당량 적용
        if max_results >= 5:
            min_allocations = {
                "law": min(2, max_results // 3),  # statute_article: 최소 2개 (max_results의 1/3)
                "precedent": min(2, max_results // 3),  # case_paragraph: 최소 2개 (max_results의 1/3)
                "decision": min(1, max_results // 5),  # decision_paragraph: 최소 1개 (max_results의 1/5)
                "interpretation": min(1, max(1, max_results // 4))  # interpretation_paragraph: 최소 1개 (max_results의 1/4로 증가)
            }
        else:
            # max_results가 5개 미만일 때는 최소 할당량 없이 점수 순서대로
            min_allocations = {
                "law": 0,
                "precedent": 0,
                "decision": 0,
                "interpretation": 0
            }
        
        diverse_results = []
        seen_ids = set()  # 개선: set 사용으로 중복 체크 최적화
        
        # 1단계: 각 타입에서 최소 할당량만큼 선택 (다양성 보장)
        for doc_type in ["law", "precedent", "decision", "interpretation"]:
            min_count = min_allocations.get(doc_type, 0)
            if results_by_type[doc_type] and min_count > 0:
                # 해당 타입에서 최소 할당량만큼 선택
                for i, result in enumerate(results_by_type[doc_type][:min_count]):
                    result_id = result.get("id") or result.get("doc_id") or result.get("document_id") or str(hash(str(result)))
                    if result_id not in seen_ids:
                        diverse_results.append(result)
                        seen_ids.add(result_id)
                        # 개선: 조기 종료 조건 추가
                        if len(diverse_results) >= max_results:
                            break
                    if len(diverse_results) >= max_results:
                        break
                if len(diverse_results) >= max_results:
                    break
        
        # 2단계: 나머지 결과를 점수 순서대로 추가
        remaining_results = []
        for result in results:
            result_id = result.get("id") or str(hash(str(result)))
            if result_id not in seen_ids:
                remaining_results.append(result)
        
        # 점수 기준 정렬
        remaining_results.sort(
            key=lambda x: (
                x.get("score", 0.0) or
                x.get("similarity", 0.0) or
                x.get("relevance_score", 0.0) or
                0.0
            ),
            reverse=True
        )
        
        # 나머지 결과 추가 (최대 개수까지)
        for result in remaining_results:
            if len(diverse_results) >= max_results:
                break
            diverse_results.append(result)
        
        # 타입 분포 로깅
        final_type_counts = {}
        for result in diverse_results:
            doc_type = (
                result.get("type") or
                result.get("source_type") or
                (result.get("metadata", {}).get("source_type") if isinstance(result.get("metadata"), dict) else "") or
                "unknown"
            )
            if doc_type in source_type_mapping:
                doc_type = source_type_mapping[doc_type]
            final_type_counts[doc_type] = final_type_counts.get(doc_type, 0) + 1
        
        self.logger.info(
            f"🔀 [DIVERSITY] Rebalanced results: "
            f"law={final_type_counts.get('law', 0)}, "
            f"precedent={final_type_counts.get('precedent', 0)}, "
            f"decision={final_type_counts.get('decision', 0)}, "
            f"interpretation={final_type_counts.get('interpretation', 0)}"
        )
        
        return diverse_results[:max_results]

    def filter_low_quality_results(
        self,
        documents: List[Dict],
        min_relevance: float,
        max_diversity: int
    ) -> List[Dict]:
        """낮은 품질 결과 필터링"""
        filtered = []

        for doc in documents:
            score = doc.get("combined_score") or doc.get("relevance_score", 0.0)

            # 유사도 임계값 체크
            if score < min_relevance:
                continue

            # 내용 길이 체크 (너무 짧은 결과 제외)
            content = doc.get("content", "")
            if len(content) < 20:
                continue

            # 중복 내용 체크 (이미 처리됨)
            filtered.append(doc)

            if len(filtered) >= max_diversity:
                break

        return filtered

    def apply_metadata_filters(
        self,
        documents: List[Dict],
        query_type: str,
        legal_field: str
    ) -> List[Dict]:
        """메타데이터 기반 필터링"""
        filtered = []

        for doc in documents:
            metadata = doc.get("metadata", {})

            # 카테고리 매칭도 계산
            doc_category = metadata.get("category", "")
            if legal_field and doc_category:
                field_match = self.calculate_field_match(legal_field, doc_category)
                doc["field_match_score"] = field_match
            else:
                doc["field_match_score"] = 0.5

            # 날짜 기반 점수 (최신 문서 우선)
            doc_date = metadata.get("date", None)
            if doc_date:
                doc["recency_score"] = self.calculate_recency_score(doc_date)
            else:
                doc["recency_score"] = 0.5

            # 신뢰도 점수 (출처 기반)
            source = doc.get("source", "")
            doc["source_credibility"] = self.calculate_source_credibility(source)

            filtered.append(doc)

        return filtered

    def calculate_field_match(self, legal_field: str, doc_category: str) -> float:
        """법률 분야 매칭도 계산"""
        field_category_map = {
            "family": ["family_law", "가족법", "이혼", "상속"],
            "civil": ["civil_law", "민사법", "계약", "손해배상"],
            "criminal": ["criminal_law", "형사법", "범죄"],
            "labor": ["labor_law", "노동법", "근로"],
            "corporate": ["corporate_law", "회사법", "기업"],
            "tax": ["tax_law", "세법", "세금"],
            "intellectual_property": ["ip_law", "특허", "지적재산"]
        }

        related_terms = field_category_map.get(legal_field, [])
        doc_category_lower = str(doc_category).lower()

        for term in related_terms:
            if term.lower() in doc_category_lower:
                return 1.0

        return 0.5  # 부분 매칭

    def calculate_recency_score(self, doc_date: Any) -> float:
        """문서 날짜 기반 최신도 점수 계산"""
        try:
            if isinstance(doc_date, str):
                # 문자열 날짜 파싱 시도
                try:
                    date_obj = datetime.fromisoformat(doc_date.replace('Z', '+00:00'))
                except:
                    return 0.5
            elif hasattr(doc_date, 'year'):
                # datetime 객체
                date_obj = doc_date
            else:
                return 0.5

            # 현재 날짜와의 차이 계산 (일 단위)
            current_date = datetime.now()
            days_diff = (current_date - date_obj.replace(tzinfo=None)).days

            # 최신도 점수: 1년 이내 = 1.0, 5년 이내 = 0.7, 그 외 = 0.5
            if days_diff <= 365:
                return 1.0
            elif days_diff <= 1825:  # 5년
                return 0.7
            else:
                return 0.5

        except Exception:
            return 0.5

    def calculate_source_credibility(self, source: str) -> float:
        """출처 신뢰도 점수 계산"""
        if not source:
            return 0.5

        source_lower = str(source).lower()

        # 신뢰도 출처 매핑
        high_credibility = ["법원", "대법원", "법제처", "법무부", "판례"]
        medium_credibility = ["법률", "조항", "법령", "규정"]

        for cred_source in high_credibility:
            if cred_source in source_lower:
                return 1.0

        for cred_source in medium_credibility:
            if cred_source in source_lower:
                return 0.8

        return 0.6  # 기본 신뢰도
    
    def _calculate_keyword_coverage(
        self,
        results: List[Dict[str, Any]],
        keywords: List[str]
    ) -> float:
        """Keyword Coverage 계산"""
        if not keywords or not results:
            return 0.5
        
        matched_count = 0
        for result in results[:10]:  # 상위 10개만 확인
            text = result.get("text", result.get("content", ""))
            if text:
                text_lower = text.lower()
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        matched_count += 1
                        break
        
        coverage = matched_count / min(len(keywords), 10)
        return min(1.0, coverage)

    def merge_search_results(
        self, 
        semantic_results: List[Dict], 
        keyword_results: List[Dict],
        query: str = "",
        query_type: str = "general_question",
        extracted_keywords: Optional[List[str]] = None
    ) -> List[Dict]:
        """검색 결과 통합 및 중복 제거 (Rerank 로직 적용 + 유사도 필터링)"""
        try:
            # Step 0: 적응형 유사도 임계값 (개선 기능)
            if self.use_improvements and hasattr(self, 'adaptive_threshold'):
                similarity_threshold = self.adaptive_threshold.calculate_threshold(
                    query=query,
                    query_type=query_type,
                    result_count=len(semantic_results) + len(keyword_results)
                )
            else:
                similarity_threshold = self.config.similarity_threshold
            filtered_semantic = [
                doc for doc in semantic_results
                if doc.get('relevance_score', doc.get('similarity', 0.0)) >= similarity_threshold
            ]
            filtered_keyword = [
                doc for doc in keyword_results
                if doc.get('relevance_score', doc.get('similarity', 0.0)) >= similarity_threshold
            ]

            if len(filtered_semantic) < len(semantic_results) or len(filtered_keyword) < len(keyword_results):
                self.logger.info(
                    f"Similarity filtering: {len(semantic_results)} → {len(filtered_semantic)}, "
                    f"{len(keyword_results)} → {len(filtered_keyword)}"
                )

            # Step 1: 결과를 ResultMerger가 처리할 수 있는 형태로 변환
            exact_results = {"semantic": filtered_semantic}

            # Step 1.5: 동적 가중치 계산 (개선 기능)
            if self.use_improvements and hasattr(self, 'adaptive_weights'):
                # Keyword Coverage 계산
                keyword_coverage = self._calculate_keyword_coverage(
                    filtered_semantic + filtered_keyword,
                    extracted_keywords or []
                )
                weights = self.adaptive_weights.calculate_weights(
                    query=query,
                    query_type=query_type,
                    keyword_coverage=keyword_coverage
                )
                # 기존 가중치 형식으로 변환
                merge_weights = {
                    "exact": weights["semantic"],
                    "semantic": weights["keyword"]
                }
            else:
                merge_weights = {"exact": 0.6, "semantic": 0.4}

            # Step 2: 결과 병합 (가중치 적용)
            merged = self.result_merger.merge_results(
                exact_results=exact_results,
                semantic_results=filtered_keyword,
                weights=merge_weights,
                query=query
            )

            # Step 3: 순위 결정
            ranked = self.result_ranker.rank_results(merged, top_k=20, query=query)
            
            # 🔥 개선: rank_results 후 메타데이터 복원 (원본 문서에서)
            # 원본 문서를 ID로 매핑 (text/content 기반 해시도 포함)
            original_docs_by_id = {}
            original_docs_by_content = {}
            for doc in semantic_results + keyword_results:
                if isinstance(doc, dict):
                    doc_id = doc.get("id") or doc.get("chunk_id") or doc.get("document_id")
                    if doc_id:
                        original_docs_by_id[doc_id] = doc
                    # content 기반 매칭도 추가
                    content = doc.get("text") or doc.get("content", "")
                    if content:
                        content_hash = str(hash(content[:200]))  # 처음 200자로 해시
                        original_docs_by_content[content_hash] = doc
            
            # ranked 문서의 메타데이터를 원본 문서에서 복원
            from lawfirm_langgraph.core.utils.document_type_normalizer import normalize_document_type
            
            for doc in ranked:
                if not isinstance(doc, dict):
                    continue
                
                # ID 기반 매칭 시도
                merged_id = doc.get("id") or doc.get("chunk_id") or doc.get("document_id")
                original_doc = None
                
                if merged_id and merged_id in original_docs_by_id:
                    original_doc = original_docs_by_id[merged_id]
                else:
                    # content 기반 매칭 시도
                    content = doc.get("text") or doc.get("content", "")
                    if content:
                        content_hash = str(hash(content[:200]))
                        if content_hash in original_docs_by_content:
                            original_doc = original_docs_by_content[content_hash]
                
                if original_doc:
                    # 원본 문서의 메타데이터를 doc에 복원
                    if not doc.get("type") and original_doc.get("type"):
                        doc["type"] = original_doc.get("type")
                    # 🔥 source_type 제거: source_type이 있으면 type으로 변환
                    if not doc.get("type") and original_doc.get("source_type"):
                        doc["type"] = original_doc.get("source_type")
                    
                    # 법령/판례 관련 필드 복원
                    for key in ["statute_name", "law_name", "article_no", "article_number", 
                               "case_id", "court", "ccourt", "doc_id", "casenames", "precedent_id"]:
                        if not doc.get(key) and original_doc.get(key):
                            doc[key] = original_doc.get(key)
                    
                    # metadata에도 복원
                    if "metadata" not in doc:
                        doc["metadata"] = {}
                    if not isinstance(doc["metadata"], dict):
                        doc["metadata"] = {}
                    
                    original_metadata = original_doc.get("metadata", {})
                    if isinstance(original_metadata, dict):
                        # 🔥 source_type을 type으로 변환
                        original_type = original_metadata.get("type") or original_metadata.get("source_type")
                        if original_type and not doc["metadata"].get("type"):
                            doc["metadata"]["type"] = original_type
                        
                        for key in ["statute_name", "law_name", "article_no", 
                                   "article_number", "case_id", "court", "ccourt", "doc_id", 
                                   "casenames", "precedent_id", "chunk_id", "source_id"]:
                            if original_metadata.get(key) and not doc["metadata"].get(key):
                                doc["metadata"][key] = original_metadata.get(key)
                
                # 🔥 정규화 함수로 type 통합 (단일 소스 원칙)
                normalize_document_type(doc)
                
                # metadata에서 최상위 필드로 복원 (백업)
                metadata = doc.get("metadata", {})
                if isinstance(metadata, dict):
                    for key in ["statute_name", "law_name", "article_no", 
                               "article_number", "case_id", "court", "ccourt", "doc_id", 
                               "casenames", "precedent_id", "id", "chunk_id", "document_id", "source_id"]:
                        if metadata.get(key) and key not in doc:
                            doc[key] = metadata[key]
            
            # Step 3.3: 키워드 매칭 점수 기반 재정렬 (검색 결과 관련성 검증 개선)
            if extracted_keywords and len(extracted_keywords) > 0:
                # MergedResult를 Dict로 변환하여 키워드 매칭 점수 계산 (메타데이터 보존)
                ranked_with_keywords = []
                for result in ranked:
                    if isinstance(result, dict):
                        doc = result.copy()
                    elif MergedResult and isinstance(result, MergedResult):
                        # result_merger의 _merged_result_to_dict 사용 (메타데이터 보존)
                        doc = self.result_merger._merged_result_to_dict(result)
                    else:
                        # 폴백: 직접 변환
                        doc = {
                            "text": result.text if hasattr(result, 'text') else str(result),
                            "content": result.text if hasattr(result, 'text') else str(result),
                            "relevance_score": result.score if hasattr(result, 'score') else 0.0,
                            "source": result.source if hasattr(result, 'source') else "",
                            "metadata": result.metadata if hasattr(result, 'metadata') else {}
                        }
                    
                    # 키워드 매칭 점수 계산
                    text_content = doc.get("text") or doc.get("content", "")
                    keyword_match_count = 0
                    for keyword in extracted_keywords[:10]:
                        if keyword.lower() in text_content.lower():
                            keyword_match_count += 1
                    keyword_match_score = keyword_match_count / min(len(extracted_keywords), 10)
                    
                    # 유사도와 키워드 매칭 점수 결합
                    similarity_score = doc.get("relevance_score", doc.get("similarity", 0.0))
                    combined_relevance = 0.7 * similarity_score + 0.3 * keyword_match_score
                    
                    # 검색 결과 순위 개선: 소스 타입별 가중치 적용 (법령 > 해석례 > 판례 순)
                    source_type = doc.get("source_type") or doc.get("type", "")
                    source_type_weights = {
                        "statute_article": 1.3,  # 법령: 가장 높은 가중치
                        "interpretation_paragraph": 1.2,  # 해석례: 두 번째
                        "decision_paragraph": 1.1,  # 결정례: 세 번째
                        "case_paragraph": 1.0  # 판례: 기본 가중치
                    }
                    source_weight = source_type_weights.get(source_type, 1.0)
                    combined_relevance *= source_weight
                    
                    doc["keyword_match_score"] = keyword_match_score
                    doc["combined_relevance_score"] = combined_relevance
                    doc["source_type_weight"] = source_weight
                    
                    ranked_with_keywords.append(doc)
                
                # combined_relevance_score 기준으로 재정렬
                ranked_with_keywords.sort(key=lambda x: x.get("combined_relevance_score", x.get("relevance_score", 0.0)), reverse=True)
                ranked = ranked_with_keywords

            # Step 3.5: Citation 포함 문서 우선순위 부여
            import re
            law_pattern = r'[가-힣]+법\s*제?\s*\d+\s*조'
            precedent_pattern = r'대법원|법원.*\d{4}[다나마]\d+'
            
            citation_boosted = []
            non_citation = []
            
            for result in ranked:
                content = result.text if hasattr(result, 'text') else str(result)
                has_law = bool(re.search(law_pattern, content))
                has_precedent = bool(re.search(precedent_pattern, content))
                
                if has_law or has_precedent:
                    # Citation이 있는 문서는 점수 부스트
                    if hasattr(result, 'score'):
                        result.score *= 1.2  # 20% 부스트
                    citation_boosted.append(result)
                else:
                    non_citation.append(result)
            
            # Citation이 있는 문서를 먼저 배치
            ranked = citation_boosted + non_citation
            
            if citation_boosted:
                self.logger.info(
                    f"🔍 [SEARCH FILTERING] Citation boost applied: "
                    f"{len(citation_boosted)} documents with citations prioritized"
                )

            # Step 4: 다양성 필터 적용 (개선 기능: MMR 사용)
            if self.use_improvements and hasattr(self, 'diversity_ranker'):
                # MergedResult를 Dict로 변환
                ranked_dicts = []
                for result in ranked:
                    doc = {
                        "text": result.text if hasattr(result, 'text') else str(result),
                        "content": result.text if hasattr(result, 'text') else str(result),
                        "relevance_score": result.score if hasattr(result, 'score') else 0.0,
                        "source": result.source if hasattr(result, 'source') else "",
                        "metadata": result.metadata if hasattr(result, 'metadata') else {}
                    }
                    ranked_dicts.append(doc)
                
                # MMR 기반 다양성 보장
                diverse_results = self.diversity_ranker.rank_with_diversity(
                    results=ranked_dicts,
                    query=query,
                    lambda_param=0.5,
                    top_k=20
                )
                filtered = diverse_results
            else:
                filtered = self.result_ranker.apply_diversity_filter(ranked, max_per_type=5)
                # MergedResult를 Dict로 변환 (메타데이터 보존)
                ranked_dicts = []
                for result in filtered:
                    if MergedResult and isinstance(result, MergedResult):
                        # result_merger의 _merged_result_to_dict 사용 (메타데이터 보존)
                        doc = self.result_merger._merged_result_to_dict(result)
                    elif isinstance(result, dict):
                        doc = result.copy()
                    else:
                        # 폴백: 직접 변환
                        doc = {
                            "text": result.text if hasattr(result, 'text') else str(result),
                            "content": result.text if hasattr(result, 'text') else str(result),
                            "relevance_score": result.score if hasattr(result, 'score') else 0.0,
                            "source": result.source if hasattr(result, 'source') else "",
                            "metadata": result.metadata if hasattr(result, 'metadata') else {}
                        }
                    ranked_dicts.append(doc)
                filtered = ranked_dicts

            # Step 5: 메타데이터 강화 및 품질 점수 계산 (개선 기능)
            documents = []
            for result in filtered:
                if isinstance(result, dict):
                    doc = result.copy()
                else:
                    doc = {
                        "content": result.text if hasattr(result, 'text') else str(result),
                        "relevance_score": result.score if hasattr(result, 'score') else 0.0,
                        "source": result.source if hasattr(result, 'source') else "",
                        "id": f"{result.source}_{hash(result.text)}" if hasattr(result, 'source') else "",
                        "type": "merged"
                    }
                    if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                        doc.update(result.metadata)
                
                # 메타데이터 강화 및 부스팅
                if self.use_improvements and hasattr(self, 'metadata_enhancer'):
                    try:
                        metadata_boost = self.metadata_enhancer.boost_by_metadata(
                            doc, query, query_type
                        )
                        doc["metadata_boost"] = metadata_boost
                        original_score = doc.get("relevance_score", 0.0)
                        doc["relevance_score"] = original_score * (1.0 + metadata_boost * 0.2)
                    except Exception as e:
                        self.logger.debug(f"Metadata boost failed: {e}")
                
                # 키워드 매칭 점수 계산 (검색 결과 관련성 검증 개선)
                if extracted_keywords and len(extracted_keywords) > 0:
                    text_content = doc.get("text") or doc.get("content", "")
                    keyword_match_count = 0
                    for keyword in extracted_keywords[:10]:  # 상위 10개 키워드만 확인
                        if keyword.lower() in text_content.lower():
                            keyword_match_count += 1
                    keyword_match_score = keyword_match_count / min(len(extracted_keywords), 10)
                    doc["keyword_match_score"] = keyword_match_score
                    doc["matched_keywords_count"] = keyword_match_count
                else:
                    doc["keyword_match_score"] = 0.0
                    doc["matched_keywords_count"] = 0
                
                # 유사도와 키워드 매칭 점수 결합 (검색 결과 관련성 검증 개선)
                similarity_score = doc.get("relevance_score", doc.get("similarity", 0.0))
                keyword_score = doc.get("keyword_match_score", 0.0)
                # 가중치: 유사도 70%, 키워드 매칭 30%
                combined_relevance = 0.7 * similarity_score + 0.3 * keyword_score
                
                # 검색 결과 순위 개선: 소스 타입별 가중치 적용 (법령 > 해석례 > 판례 순)
                source_type = doc.get("source_type") or doc.get("type", "")
                source_type_weights = {
                    "statute_article": 1.3,  # 법령: 가장 높은 가중치
                    "interpretation_paragraph": 1.2,  # 해석례: 두 번째
                    "decision_paragraph": 1.1,  # 결정례: 세 번째
                    "case_paragraph": 1.0  # 판례: 기본 가중치
                }
                source_weight = source_type_weights.get(source_type, 1.0)
                combined_relevance *= source_weight
                
                doc["combined_relevance_score"] = combined_relevance
                doc["source_type_weight"] = source_weight
                
                # 다차원 품질 점수 계산
                if self.use_improvements and hasattr(self, 'quality_scorer'):
                    try:
                        quality_scores = self.quality_scorer.calculate_quality(
                            doc, query, query_type, extracted_keywords
                        )
                        doc["quality_scores"] = {
                            "relevance": quality_scores.relevance,
                            "accuracy": quality_scores.accuracy,
                            "completeness": quality_scores.completeness,
                            "recency": quality_scores.recency,
                            "source_credibility": quality_scores.source_credibility,
                            "overall": quality_scores.overall
                        }
                        # 품질 점수를 최종 점수에 반영
                        final_score = doc.get("relevance_score", 0.0)
                        doc["final_score"] = 0.7 * final_score + 0.3 * quality_scores.overall
                        doc["relevance_score"] = doc["final_score"]
                    except Exception as e:
                        self.logger.debug(f"Quality scoring failed: {e}")
                
                # id가 없으면 생성
                if "id" not in doc:
                    doc["id"] = f"{doc.get('source', 'unknown')}_{hash(doc.get('content', ''))}"
                
                documents.append(doc)

            self.logger.info(
                f"Rerank applied: {len(semantic_results)} semantic + {len(keyword_results)} keyword → {len(documents)} final"
            )
            return documents

        except Exception as e:
            self.logger.warning(f"Rerank failed, using simple merge: {e}")
            # 폴백: 간단한 병합 및 정렬
            seen_ids = set()
            documents = []

            for doc in semantic_results:
                doc_id = doc.get('id')
                if doc_id is not None:
                    try:
                        if doc_id not in seen_ids:
                            documents.append(doc)
                            seen_ids.add(doc_id)
                    except TypeError:
                        doc_id_str = str(doc_id)
                        if doc_id_str not in seen_ids:
                            documents.append(doc)
                            seen_ids.add(doc_id_str)

            for doc in keyword_results:
                doc_id = doc.get('id')
                if doc_id is not None:
                    try:
                        if doc_id not in seen_ids:
                            documents.append(doc)
                            seen_ids.add(doc_id)
                    except TypeError:
                        doc_id_str = str(doc_id)
                        if doc_id_str not in seen_ids:
                            documents.append(doc)
                            seen_ids.add(doc_id_str)

            documents.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

            # 폴백에도 유사도 필터링 적용
            similarity_threshold = self.config.similarity_threshold
            documents = [
                doc for doc in documents
                if doc.get('relevance_score', 0.0) >= similarity_threshold
            ]
            return documents

    def update_search_metadata(
        self,
        state: LegalWorkflowState,
        semantic_count: int,
        keyword_count: int,
        documents: List[Dict],
        query_type_str: str,
        start_time: float,
        optimized_queries: Optional[Dict[str, Any]] = None
    ) -> None:
        """검색 메타데이터 업데이트 (consolidated metadata)"""
        # 중요: 기존 metadata의 query_complexity 보존
        metadata = WorkflowUtils.get_state_value(state, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        # query_complexity와 needs_search 보존
        preserved_complexity = metadata.get("query_complexity")
        preserved_needs_search = metadata.get("needs_search")

        retrieved_docs = WorkflowUtils.get_state_value(state, "retrieved_docs", [])
        search_meta = {
            "semantic_results_count": semantic_count,
            "keyword_results_count": keyword_count,
            "total_candidates": len(documents),
            "final_count": len(retrieved_docs),
            "search_time": time.time() - start_time,
            "query_type": query_type_str,
            "search_mode": "hybrid_improved"
        }

        # 최적화된 쿼리 정보 추가
        if optimized_queries:
            search_meta["optimized_queries"] = {
                "semantic_query": optimized_queries.get("semantic_query", ""),
                "keyword_query_count": len(optimized_queries.get("keyword_queries", [])),
                "expanded_keywords_count": len(optimized_queries.get("expanded_keywords", []))
            }

        metadata["search"] = search_meta
        # 중요: query_complexity와 needs_search 보존
        if preserved_complexity:
            metadata["query_complexity"] = preserved_complexity
        if preserved_needs_search is not None:
            metadata["needs_search"] = preserved_needs_search

        WorkflowUtils.set_state_value(state, "metadata", metadata)

        retrieved_docs_count = len(retrieved_docs)
        WorkflowUtils.add_step(
            state,
            "하이브리드 검색 완료",
            f"하이브리드 검색 완료: 의미적 {semantic_count}개, 키워드 {keyword_count}개, 최종 {retrieved_docs_count}개"
        )

    def fallback_search(self, state: LegalWorkflowState) -> None:
        """폴백 검색"""
        try:
            query_type_str = WorkflowUtils.get_query_type_str(
                WorkflowUtils.get_state_value(state, "query_type", "")
            )
            category_mapping = WorkflowUtils.get_category_mapping()
            fallback_categories = category_mapping.get(query_type_str, ["civil_law"])

            fallback_docs = []
            for category in fallback_categories:
                category_docs = self.data_connector.get_document_by_category(category, limit=2)
                fallback_docs.extend(category_docs)
                if len(fallback_docs) >= 3:
                    break

            if fallback_docs:
                WorkflowUtils.set_state_value(state, "retrieved_docs", fallback_docs)
                WorkflowUtils.add_step(state, "폴백", f"폴백: {len(fallback_docs)}개 문서 사용")
                self.logger.info(f"Using fallback documents: {len(fallback_docs)} docs")
            else:
                query = WorkflowUtils.get_state_value(state, "query", "")
                WorkflowUtils.set_state_value(state, "retrieved_docs", [
                    {"content": f"'{query}'에 대한 기본 법률 정보입니다.", "source": "Default DB"}
                ])
                self.logger.warning("No fallback documents available")
        except Exception as fallback_error:
            self.logger.error(f"Fallback also failed: {fallback_error}")
            query = WorkflowUtils.get_state_value(state, "query", "")
            WorkflowUtils.set_state_value(state, "retrieved_docs", [
                {"content": f"'{query}'에 대한 기본 법률 정보입니다.", "source": "Default DB"}
            ])
