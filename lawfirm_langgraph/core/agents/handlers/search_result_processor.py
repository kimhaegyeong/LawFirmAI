# -*- coding: utf-8 -*-
"""
검색 결과 처리 핸들러 모듈
검색 결과 품질 평가, 병합, 재순위, 필터링 로직을 독립 모듈로 분리
LangGraph의 Node, Task, Subgraph 패턴 활용
"""

import asyncio
import logging
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.agents.state_definitions import LegalWorkflowState
from core.agents.workflow_constants import WorkflowConstants
from core.agents.workflow_utils import WorkflowUtils
from core.agents.tasks.search_result_tasks import SearchResultTasks


class SearchResultProcessor:
    """
    검색 결과 처리 클래스
    
    검색 결과의 품질 평가, 병합, 재순위, 필터링을 처리합니다.
    """

    def __init__(
        self,
        result_merger: Any,
        config: Any,
        evaluate_semantic_quality_func: Optional[Callable] = None,
        evaluate_keyword_quality_func: Optional[Callable] = None,
        calculate_keyword_weights_func: Optional[Callable] = None,
        calculate_keyword_match_score_func: Optional[Callable] = None,
        calculate_weighted_final_score_func: Optional[Callable] = None,
        execute_semantic_search_func: Optional[Callable] = None,
        execute_keyword_search_func: Optional[Callable] = None,
        get_query_type_str_func: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        SearchResultProcessor 초기화 (의존성 주입 패턴)
        
        Args:
            result_merger: 결과 병합기
            config: 설정 객체
            evaluate_semantic_quality_func: 의미 검색 품질 평가 함수
            evaluate_keyword_quality_func: 키워드 검색 품질 평가 함수
            calculate_keyword_weights_func: 키워드 가중치 계산 함수
            calculate_keyword_match_score_func: 키워드 매칭 점수 계산 함수
            calculate_weighted_final_score_func: 가중치 최종 점수 계산 함수
            execute_semantic_search_func: 의미 검색 실행 함수
            execute_keyword_search_func: 키워드 검색 실행 함수
            get_query_type_str_func: 질의 유형 문자열 변환 함수
            logger: 로거 (없으면 자동 생성)
        """
        self.result_merger = result_merger
        self.config = config
        self.evaluate_semantic_quality = evaluate_semantic_quality_func
        self.evaluate_keyword_quality = evaluate_keyword_quality_func
        self.calculate_keyword_weights = calculate_keyword_weights_func
        self.calculate_keyword_match_score = calculate_keyword_match_score_func
        self.calculate_weighted_final_score = calculate_weighted_final_score_func
        self.execute_semantic_search = execute_semantic_search_func
        self.execute_keyword_search = execute_keyword_search_func
        self.get_query_type_str = get_query_type_str_func or (lambda x: str(x) if x else "")
        self.logger = logger or logging.getLogger(__name__)

    def process_search_results_combined(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """검색 결과 처리 통합 노드 (6개 노드를 1개로 병합)"""
        print("🔄 [SEARCH RESULTS] process_search_results_combined 실행 시작", flush=True, file=sys.stdout)
        self.logger.info("🔄 [SEARCH RESULTS] process_search_results_combined 실행 시작")

        try:
            start_time = time.time()

            # 배치로 State 값 가져오기 (성능 최적화)
            state_values = WorkflowUtils.get_state_values_batch(
                state,
                keys=["semantic_results", "keyword_results", "semantic_count", "keyword_count", 
                      "query", "query_type", "search_params", "extracted_keywords", "legal_field"],
                defaults={
                    "semantic_results": [],
                    "keyword_results": [],
                    "semantic_count": 0,
                    "keyword_count": 0,
                    "query": "",
                    "query_type": "",
                    "search_params": {},
                    "extracted_keywords": [],
                    "legal_field": ""
                }
            )
            
            semantic_results = state_values["semantic_results"]
            keyword_results = state_values["keyword_results"]
            semantic_count = state_values["semantic_count"]
            keyword_count = state_values["keyword_count"]
            query = state_values["query"]
            query_type_str = self.get_query_type_str(state_values["query_type"])
            search_params = state_values["search_params"]
            extracted_keywords = state_values["extracted_keywords"]
            legal_field = state_values["legal_field"]

            input_msg = f"📥 [SEARCH RESULTS] 입력 데이터 - semantic: {len(semantic_results)}, keyword: {len(keyword_results)}, semantic_count: {semantic_count}, keyword_count: {keyword_count}"
            print(input_msg, flush=True, file=sys.stdout)
            self.logger.info(input_msg)

            # 1. 품질 평가 (병렬 Task 사용)
            quality_evaluation = asyncio.run(SearchResultTasks.evaluate_quality_parallel(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                query=query,
                query_type=query_type_str,
                search_params=search_params,
                evaluate_semantic_func=self.evaluate_semantic_quality,
                evaluate_keyword_func=self.evaluate_keyword_quality
            ))
            WorkflowUtils.set_state_value(state, "search_quality_evaluation", quality_evaluation)

            overall_quality = quality_evaluation["overall_quality"]
            needs_retry = quality_evaluation["needs_retry"]

            # 2. 조건부 재검색
            if needs_retry and overall_quality < 0.6 and semantic_count + keyword_count < 10:
                semantic_results, keyword_results, semantic_count, keyword_count = self._perform_conditional_retry(
                    state, semantic_results, keyword_results, semantic_count, keyword_count,
                    quality_evaluation, query, query_type_str, search_params, extracted_keywords, legal_field
                )

            # 3. 병합 및 재순위
            merged_docs = self._merge_and_rerank_results(
                semantic_results, keyword_results, query, query_type_str, 
                extracted_keywords, legal_field, search_params, state
            )

            # 4. 키워드 가중치 적용 (병렬 Task 사용)
            if merged_docs and self.calculate_keyword_weights:
                weighted_docs = asyncio.run(SearchResultTasks.apply_keyword_weights_parallel(
                    documents=merged_docs,
                    extracted_keywords=extracted_keywords,
                    query=query,
                    query_type=query_type_str,
                    legal_field=legal_field,
                    calculate_keyword_weights_func=self.calculate_keyword_weights,
                    calculate_keyword_match_score_func=self.calculate_keyword_match_score,
                    calculate_weighted_final_score_func=self.calculate_weighted_final_score,
                    search_params=search_params
                ))
                merged_docs = weighted_docs

            # 5. 필터링 및 검증 (병렬 Task 사용)
            final_docs, filter_stats = asyncio.run(SearchResultTasks.filter_documents_parallel(
                documents=merged_docs,
                min_relevance=0.80,
                min_content_length=5,
                min_final_score=0.55
            ))
            
            self.logger.info(
                f"📊 [FILTER] Total: {filter_stats['total']}, "
                f"Filtered: {filter_stats['filtered']}, "
                f"Skipped: {filter_stats['skipped']}"
            )

            # 6. 메타데이터 업데이트
            self._update_search_metadata(
                state, merged_docs, filtered_docs=final_docs, 
                overall_quality=overall_quality, needs_retry=needs_retry,
                semantic_count=semantic_count, keyword_count=keyword_count
            )

            # 6. State 저장
            self._save_final_results_to_state(state, final_docs)

            processing_time = WorkflowUtils.update_processing_time(state, start_time)
            WorkflowUtils.add_step(
                state,
                "검색 결과 처리",
                f"검색 결과 처리 완료: {len(final_docs)}개 문서 (품질 점수: {overall_quality:.2f}, 시간: {processing_time:.3f}s)"
            )

            if len(final_docs) > 0:
                processed_msg = f"✅ [SEARCH RESULTS] Processed {len(final_docs)} documents (quality: {overall_quality:.2f}, retry: {needs_retry}, time: {processing_time:.3f}s)"
                print(processed_msg, flush=True, file=sys.stdout)
                self.logger.info(processed_msg)

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self.logger.error(
                f"❌ [SEARCH RESULTS ERROR] process_search_results_combined 실행 중 예외 발생: {str(e)}\n"
                f"   스택 트레이스:\n{error_traceback}"
            )
            WorkflowUtils.handle_error(state, str(e), "검색 결과 처리 중 오류 발생")

            # 폴백: 기존 검색 결과가 있으면 사용 시도
            existing_semantic = WorkflowUtils.get_state_value(state, "semantic_results", [])
            existing_keyword = WorkflowUtils.get_state_value(state, "keyword_results", [])
            
            fallback_docs = []
            for doc in (existing_semantic + existing_keyword)[:10]:
                if isinstance(doc, dict) and (doc.get("content") or doc.get("text")):
                    fallback_docs.append(doc)

            if fallback_docs:
                self.logger.info(f"🔄 [FALLBACK] Using {len(fallback_docs)} documents from original search results")
                WorkflowUtils.set_state_value(state, "retrieved_docs", fallback_docs)
                WorkflowUtils.set_state_value(state, "merged_documents", fallback_docs)
            else:
                self.logger.warning("⚠️ [FALLBACK] No fallback documents available")
                WorkflowUtils.set_state_value(state, "retrieved_docs", [])
                WorkflowUtils.set_state_value(state, "merged_documents", [])

        return state

    def _evaluate_search_quality(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        query: str,
        query_type_str: str,
        search_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """검색 품질 평가"""
        # 이 메서드는 legal_workflow_enhanced.py의 _evaluate_semantic_search_quality와 
        # _evaluate_keyword_search_quality를 호출해야 합니다.
        # 현재는 간단한 버전으로 구현
        semantic_quality = {
            "score": 0.8 if len(semantic_results) >= 5 else 0.5,
            "needs_retry": len(semantic_results) < 5
        }
        keyword_quality = {
            "score": 0.8 if len(keyword_results) >= 3 else 0.5,
            "needs_retry": len(keyword_results) < 3
        }
        
        overall_quality = (semantic_quality["score"] + keyword_quality["score"]) / 2.0
        needs_retry = semantic_quality["needs_retry"] or keyword_quality["needs_retry"]
        
        return {
            "semantic_quality": semantic_quality,
            "keyword_quality": keyword_quality,
            "overall_quality": overall_quality,
            "needs_retry": needs_retry
        }

    def _perform_conditional_retry(
        self,
        state: LegalWorkflowState,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        semantic_count: int,
        keyword_count: int,
        quality_evaluation: Dict[str, Any],
        query: str,
        query_type_str: str,
        search_params: Dict[str, Any],
        extracted_keywords: List[str],
        legal_field: str
    ) -> Tuple[List[Dict], List[Dict], int, int]:
        """조건부 재검색 수행"""
        # 이 메서드는 legal_workflow_enhanced.py의 재검색 로직을 호출해야 합니다.
        # 현재는 빈 구현으로 남겨둡니다.
        return semantic_results, keyword_results, semantic_count, keyword_count

    def _merge_and_rerank_results(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        query: str,
        query_type_str: str,
        extracted_keywords: List[str],
        legal_field: str,
        search_params: Dict[str, Any],
        state: LegalWorkflowState
    ) -> List[Dict]:
        """결과 병합 및 재순위"""
        # result_merger.merge_results 사용
        exact_results_dict = {
            "keyword": keyword_results if isinstance(keyword_results, list) else []
        } if keyword_results else {}

        merged_results = self.result_merger.merge_results(
            exact_results=exact_results_dict,
            semantic_results=semantic_results if isinstance(semantic_results, list) else [],
            weights={"exact": 0.7, "semantic": 0.3}
        )

        # MergedResult 객체를 dict로 변환
        merged_docs = []
        for merged_result in merged_results:
            if hasattr(merged_result, 'text'):
                text_value = merged_result.text
                if not text_value or len(str(text_value).strip()) == 0:
                    if hasattr(merged_result, 'content'):
                        text_value = merged_result.content
                    elif hasattr(merged_result, 'metadata') and isinstance(merged_result.metadata, dict):
                        text_value = (
                            merged_result.metadata.get('content') or
                            merged_result.metadata.get('text') or
                            merged_result.metadata.get('document') or
                            ''
                        )

                merged_docs.append({
                    "content": str(text_value) if text_value else "",
                    "text": str(text_value) if text_value else "",
                    "relevance_score": getattr(merged_result, 'score', 0.0),
                    "source": getattr(merged_result, 'source', 'Unknown'),
                    "metadata": getattr(merged_result, 'metadata', {}) if hasattr(merged_result, 'metadata') else {}
                })
            elif isinstance(merged_result, dict):
                doc = merged_result.copy()
                if "content" not in doc and "text" in doc:
                    doc["content"] = doc["text"]
                elif "text" not in doc and "content" in doc:
                    doc["text"] = doc["content"]
                elif "content" not in doc and "text" not in doc:
                    doc["content"] = ""
                    doc["text"] = ""
                merged_docs.append(doc)

        return merged_docs

    def _filter_and_validate_documents(
        self,
        merged_docs: List[Dict],
        query: str,
        query_type_str: str,
        search_params: Dict[str, Any]
    ) -> List[Dict]:
        """문서 필터링 및 검증"""
        min_relevance_score = 0.80
        filtered_docs = []
        
        for doc in merged_docs:
            relevance_score = (
                doc.get("relevance_score") or
                doc.get("score") or
                doc.get("final_weighted_score") or
                doc.get("similarity") or
                0.0
            )
            
            if relevance_score >= min_relevance_score:
                content = doc.get("content", "") or doc.get("text", "")
                if content and len(content.strip()) >= 5:
                    filtered_docs.append(doc)

        max_docs = self.config.max_retrieved_docs or 20
        return filtered_docs[:max_docs]

    def _update_search_metadata(
        self,
        state: LegalWorkflowState,
        merged_docs: List[Dict],
        filtered_docs: List[Dict],
        overall_quality: float,
        needs_retry: bool,
        semantic_count: int,
        keyword_count: int
    ) -> None:
        """검색 메타데이터 업데이트"""
        search_metadata = {
            "total_results": len(merged_docs),
            "filtered_results": len(filtered_docs),
            "final_results": len(filtered_docs),
            "quality_score": overall_quality,
            "semantic_count": semantic_count,
            "keyword_count": keyword_count,
            "retry_performed": needs_retry,
            "has_results": len(filtered_docs) > 0,
            "used_fallback": len(filtered_docs) > 0 and len(merged_docs) == 0,
            "timestamp": time.time()
        }
        WorkflowUtils.set_state_value(state, "search_metadata", search_metadata)

    def _save_final_results_to_state(
        self,
        state: LegalWorkflowState,
        final_docs: List[Dict]
    ) -> None:
        """최종 결과를 State에 저장"""
        WorkflowUtils.set_state_value(state, "retrieved_docs", final_docs)
        WorkflowUtils.set_state_value(state, "merged_documents", final_docs)

        # search 그룹에도 저장
        if "search" not in state:
            state["search"] = {}
        state["search"]["retrieved_docs"] = final_docs
        state["search"]["merged_documents"] = final_docs

        # common 그룹에도 저장
        if "common" not in state:
            state["common"] = {}
        if "search" not in state["common"]:
            state["common"]["search"] = {}
        state["common"]["search"]["retrieved_docs"] = final_docs
        state["common"]["search"]["merged_documents"] = final_docs

        # 전역 캐시에도 저장
        try:
            from core.agents.node_wrappers import _global_search_results_cache
            if not _global_search_results_cache:
                _global_search_results_cache = {}
            _global_search_results_cache["retrieved_docs"] = final_docs
            _global_search_results_cache["merged_documents"] = final_docs
        except Exception:
            pass

