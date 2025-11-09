# -*- coding: utf-8 -*-
"""
워크플로우 유틸리티 함수 집합
상태 관리, 파싱, 정규화 등의 공통 기능 제공
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from core.processing.extractors.extractors import ResponseExtractor
from core.workflow.state.state_definitions import LegalWorkflowState
from core.workflow.state.state_helpers import ensure_state_group, get_field, set_field
from core.workflow.state.state_utils import (
    MAX_PROCESSING_STEPS,
    prune_processing_steps,
)
from core.classification.classifiers.question_classifier import QuestionType
from core.services.unified_prompt_manager import LegalDomain


class WorkflowUtils:
    """
    워크플로우 유틸리티 함수 집합

    정적 메서드로 구현하여 어디서든 사용 가능하도록 설계
    """

    @staticmethod
    def get_state_value(state: LegalWorkflowState, key: str, default: Any = None) -> Any:
        """
        State에서 값을 안전하게 가져오기 (flat/nested 모두 지원)

        state_helpers의 get_field를 사용하여 일관된 접근 제공
        개선: query_type 전용 검색 로직 추가

        Args:
            state: State 객체 (flat 또는 nested)
            key: 접근할 키
            default: 기본값

        Returns:
            State에서 가져온 값 또는 기본값
        """
        # query_type 전용 검색 로직 (개선: 여러 위치에서 검색)
        if key == "query_type":
            return WorkflowUtils._get_query_type_enhanced(state, default)
        
        result = get_field(state, key)
        return result if result is not None else default
    
    @staticmethod
    def _get_query_type_enhanced(state: LegalWorkflowState, default: Any = None) -> Any:
        """
        query_type을 여러 위치에서 검색 (개선된 로직)
        
        검색 순서:
        1. 최상위 레벨: state.get("query_type")
        2. classification 그룹: state["classification"]["query_type"]
        3. common.classification 그룹: state["common"]["classification"]["query_type"]
        4. metadata 그룹: state["metadata"]["query_type"]
        5. common.metadata 그룹: state["common"]["metadata"]["query_type"]
        6. Global cache: _global_classification_cache
        
        Args:
            state: LegalWorkflowState
            default: 기본값
            
        Returns:
            query_type 값 또는 기본값
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 1. 최상위 레벨에서 검색
        if "query_type" in state:
            value = state.get("query_type")
            if value:
                logger.debug(f"[QUERY_TYPE] Found in top-level: {value}")
                return value
        
        # 2. classification 그룹에서 검색
        if "classification" in state and isinstance(state["classification"], dict):
            value = state["classification"].get("query_type")
            if value:
                logger.debug(f"[QUERY_TYPE] Found in classification group: {value}")
                return value
        
        # 3. common.classification 그룹에서 검색
        if "common" in state and isinstance(state.get("common"), dict):
            common = state["common"]
            if "classification" in common and isinstance(common["classification"], dict):
                value = common["classification"].get("query_type")
                if value:
                    logger.debug(f"[QUERY_TYPE] Found in common.classification group: {value}")
                    return value
        
        # 4. metadata 그룹에서 검색
        if "metadata" in state and isinstance(state.get("metadata"), dict):
            value = state["metadata"].get("query_type")
            if value:
                logger.debug(f"[QUERY_TYPE] Found in metadata group: {value}")
                return value
        
        # 5. common.metadata 그룹에서 검색
        if "common" in state and isinstance(state.get("common"), dict):
            common = state["common"]
            if "metadata" in common and isinstance(common.get("metadata"), dict):
                value = common["metadata"].get("query_type")
                if value:
                    logger.debug(f"[QUERY_TYPE] Found in common.metadata group: {value}")
                    return value
        
        # 6. Global cache에서 검색
        try:
            from core.shared.wrappers.node_wrappers import _global_search_results_cache
            if _global_search_results_cache:
                # 여러 위치에서 검색
                cached_value = (
                    _global_search_results_cache.get("common", {}).get("classification", {}).get("query_type") or
                    _global_search_results_cache.get("metadata", {}).get("query_type") or
                    _global_search_results_cache.get("classification", {}).get("query_type") or
                    _global_search_results_cache.get("query_type")
                )
                if cached_value:
                    logger.debug(f"[QUERY_TYPE] Found in global cache: {cached_value}")
                    # state에도 복원하여 다음 노드에서 사용 가능하도록
                    WorkflowUtils.set_state_value(state, "query_type", cached_value)
                    return cached_value
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"[QUERY_TYPE] Could not access global cache: {e}")
        
        # 모든 위치에서 찾지 못한 경우
        logger.debug(f"[QUERY_TYPE] Not found in any location, using default: {default}")
        return default

    @staticmethod
    def set_state_value(state: LegalWorkflowState, key: str, value: Any, logger: Optional[logging.Logger] = None) -> None:
        """
        State에 값을 안전하게 설정하기 (flat/nested 모두 지원)

        state_helpers의 set_field를 사용하여 일관된 설정 제공
        필요한 State 그룹이 없으면 자동으로 초기화합니다.

        Args:
            state: State 객체 (flat 또는 nested)
            key: 설정할 키
            value: 설정할 값
            logger: 로거 (선택사항)
        """
        # 중요: metadata 전체 딕셔너리를 설정할 때 query_complexity 보존
        if key == "metadata" and isinstance(value, dict):
            # 기존 metadata에서 query_complexity와 needs_search 보존
            existing_metadata = WorkflowUtils.get_state_value(state, "metadata", {})
            if isinstance(existing_metadata, dict):
                preserved_complexity = existing_metadata.get("query_complexity")
                preserved_needs_search = existing_metadata.get("needs_search")
                # 보존된 값 복원
                if preserved_complexity:
                    value["query_complexity"] = preserved_complexity
                if preserved_needs_search is not None:
                    value["needs_search"] = preserved_needs_search

        # Classification 필드인 경우 그룹 초기화
        if key in ["query_type", "confidence", "legal_field", "legal_domain",
                   "urgency_level", "urgency_reasoning", "emergency_type",
                   "complexity_level", "requires_expert", "expert_subgraph"]:
            ensure_state_group(state, "classification")
        # Search 필드인 경우 그룹 초기화
        elif key in ["search_query", "extracted_keywords", "ai_keyword_expansion", "retrieved_docs",
                     "optimized_queries", "search_params", "semantic_results", "keyword_results",
                     "semantic_count", "keyword_count", "merged_documents", "keyword_weights",
                     "prompt_optimized_context"]:
            ensure_state_group(state, "search")
        # Analysis 필드인 경우 그룹 초기화
        elif key in ["analysis", "legal_references", "legal_citations"]:
            ensure_state_group(state, "analysis")
        # Answer 필드인 경우 그룹 초기화
        elif key in ["answer", "sources", "structure_confidence"]:
            ensure_state_group(state, "answer")
        # Document 필드인 경우 그룹 초기화
        elif key in ["document_type", "document_analysis", "key_clauses", "potential_issues"]:
            ensure_state_group(state, "document")
        # MultiTurn 필드인 경우 그룹 초기화
        elif key in ["is_multi_turn", "multi_turn_confidence", "conversation_history", "conversation_context"]:
            ensure_state_group(state, "multi_turn")
        # Validation 필드인 경우 그룹 초기화
        elif key in ["legal_validity_check", "legal_basis_validation", "outdated_laws"]:
            ensure_state_group(state, "validation")
        # Control 필드인 경우 그룹 초기화
        elif key in ["retry_count", "quality_check_passed", "needs_enhancement"]:
            ensure_state_group(state, "control")
        # Common 필드는 항상 존재
        elif key in ["processing_steps", "errors", "metadata", "processing_time", "tokens_used"]:
            ensure_state_group(state, "common")

        set_field(state, key, value)

    @staticmethod
    def update_processing_time(state: LegalWorkflowState, start_time: float) -> float:
        """처리 시간 업데이트"""
        processing_time = time.time() - start_time
        current_time = WorkflowUtils.get_state_value(state, "processing_time", 0.0)
        WorkflowUtils.set_state_value(state, "processing_time", current_time + processing_time)
        return processing_time

    @staticmethod
    def add_step(state: LegalWorkflowState, step_prefix: str, step_message: str) -> None:
        """처리 단계 추가 (중복 방지 및 pruning)"""
        processing_steps = WorkflowUtils.get_state_value(state, "processing_steps", [])
        if not processing_steps:
            processing_steps = []
            WorkflowUtils.set_state_value(state, "processing_steps", processing_steps)

        if not any(step_prefix in step for step in processing_steps):
            processing_steps.append(step_message)
            WorkflowUtils.set_state_value(state, "processing_steps", processing_steps)

        # Always prune if too many steps (check on every add)
        if len(processing_steps) > MAX_PROCESSING_STEPS:
            pruned_steps = prune_processing_steps(
                processing_steps,
                max_items=MAX_PROCESSING_STEPS
            )
            WorkflowUtils.set_state_value(state, "processing_steps", pruned_steps)

    @staticmethod
    def handle_error(state: LegalWorkflowState, error_msg: str, context: str = "",
                     logger: Optional[logging.Logger] = None) -> None:
        """에러 처리 헬퍼"""
        full_error = f"{context}: {error_msg}" if context else error_msg

        # errors 리스트 가져오기 및 초기화
        errors = WorkflowUtils.get_state_value(state, "errors", [])
        if not errors:
            errors = []
            WorkflowUtils.set_state_value(state, "errors", errors)
        errors.append(full_error)
        WorkflowUtils.set_state_value(state, "errors", errors)

        # processing_steps에 추가
        WorkflowUtils.add_step(state, "ERROR", full_error)

        if logger:
            logger.error(full_error)

    @staticmethod
    def normalize_answer(answer_raw: Any) -> str:
        """
        답변을 안전하게 문자열로 변환하는 통합 메서드

        Args:
            answer_raw: 답변 (str, dict, 또는 다른 타입)

        Returns:
            정규화된 답변 문자열
        """
        if answer_raw is None:
            return ""
        if isinstance(answer_raw, str):
            return answer_raw
        if isinstance(answer_raw, dict):
            # dict에서 content나 answer 키를 찾거나, 전체 dict를 문자열로 변환
            content = answer_raw.get("content") or answer_raw.get("answer")
            if content:
                # content가 여전히 dict일 수 있으므로 재귀적으로 처리
                if isinstance(content, str):
                    return content
                elif isinstance(content, dict):
                    return content.get("content", content.get("answer", str(content)))
                else:
                    return str(content)
            return str(answer_raw)
        if isinstance(answer_raw, list):
            # list인 경우 첫 번째 항목 사용
            if answer_raw:
                return WorkflowUtils.normalize_answer(answer_raw[0])
            return ""
        return str(answer_raw) if answer_raw else ""

    @staticmethod
    def save_metadata_safely(state: LegalWorkflowState, key: str, value: Any,
                             save_to_top_level: bool = False) -> None:
        """
        메타데이터를 모든 경로에 안전하게 저장

        Args:
            state: LegalWorkflowState
            key: 메타데이터 키
            value: 메타데이터 값
            save_to_top_level: 최상위 레벨에도 저장할지 여부 (조건부 엣지 접근용)
        """
        # metadata 설정
        if "metadata" not in state or not isinstance(state.get("metadata"), dict):
            state["metadata"] = {}
        # 중요: query_complexity와 needs_search 보존
        preserved_complexity = state.get("metadata", {}).get("query_complexity")
        preserved_needs_search = state.get("metadata", {}).get("needs_search")
        state["metadata"] = dict(state["metadata"])  # 복사본 생성
        # 보존된 값 복원
        if preserved_complexity:
            state["metadata"]["query_complexity"] = preserved_complexity
        if preserved_needs_search is not None:
            state["metadata"]["needs_search"] = preserved_needs_search
        state["metadata"][key] = value

        # common.metadata 설정 (상태 최적화에서 항상 포함됨)
        if "common" not in state or not isinstance(state.get("common"), dict):
            state["common"] = {}
        if "metadata" not in state["common"]:
            state["common"]["metadata"] = {}
        state["common"]["metadata"] = dict(state["common"]["metadata"])  # 복사본 생성
        state["common"]["metadata"][key] = value

        # 특정 키는 top-level에도 저장 (조건부 엣지 접근용)
        if save_to_top_level and isinstance(state, dict):
            state[f"_{key}"] = value

    @staticmethod
    def get_quality_metadata(state: LegalWorkflowState) -> Dict[str, Any]:
        """
        품질 검증 메타데이터를 모든 경로에서 안전하게 읽기

        Args:
            state: LegalWorkflowState

        Returns:
            품질 메타데이터 딕셔너리 (quality_check_passed, quality_score)
        """
        quality_check_passed = False
        quality_score = None

        # 1순위: 최상위 레벨 (조건부 엣지에서 가장 확실하게 접근 가능)
        if isinstance(state, dict):
            quality_check_passed = state.get("_quality_check_passed", False)
            if "_quality_score" in state:
                quality_score = state.get("_quality_score")

        # 2순위: common.metadata (상태 최적화에서 항상 포함됨)
        if not quality_check_passed or quality_score is None:
            if "common" in state and isinstance(state.get("common"), dict):
                common_meta = state["common"].get("metadata", {})
                if isinstance(common_meta, dict):
                    if not quality_check_passed:
                        quality_check_passed = common_meta.get("quality_check_passed", False)
                    if quality_score is None:
                        quality_score = common_meta.get("quality_score")

        # 3순위: 일반 경로 (get_field를 통한 접근)
        if not quality_check_passed:
            quality_check_passed = WorkflowUtils.get_state_value(state, "quality_check_passed", False)
        if quality_score is None:
            quality_score = WorkflowUtils.get_state_value(state, "quality_score", 0.0)

        return {
            "quality_check_passed": quality_check_passed,
            "quality_score": float(quality_score) if quality_score is not None else 0.0
        }

    @staticmethod
    def extract_response_content(response: Any) -> str:
        """응답에서 내용 추출"""
        return ResponseExtractor.extract_response_content(response)

    @staticmethod
    def get_query_type_str(query_type: Any) -> str:
        """QueryType을 문자열로 변환"""
        return query_type.value if hasattr(query_type, 'value') else str(query_type)

    @staticmethod
    def normalize_query_type_for_prompt(query_type: Any, logger: Optional[logging.Logger] = None) -> str:
        """질문 유형을 프롬프트용 표준 문자열로 변환"""
        if not query_type:
            return "general_question"

        # 문자열로 변환
        if hasattr(query_type, 'value'):
            query_type_str = query_type.value
        elif hasattr(query_type, 'name'):
            query_type_str = query_type.name.lower()
        else:
            query_type_str = str(query_type).lower()

        # 표준 형태로 변환 (snake_case)
        query_type_mapping = {
            "precedent_search": "precedent_search",
            "law_inquiry": "law_inquiry",
            "legal_advice": "legal_advice",
            "document_analysis": "document_analysis",
            "procedure_guide": "procedure_guide",
            "term_explanation": "term_explanation",
            "general_question": "general_question",
            # 변형 형태 매핑
            "precedent": "precedent_search",
            "law": "law_inquiry",
            "advice": "legal_advice",
            "analysis": "document_analysis",
            "procedure": "procedure_guide",
            "term": "term_explanation",
            "general": "general_question",
        }

        # 매핑이 있으면 사용, 없으면 원본 반환 (하지만 소문자로)
        normalized = query_type_mapping.get(query_type_str, query_type_str)

        # 유효한 query_type 목록에 없으면 general_question으로
        valid_types = ["precedent_search", "law_inquiry", "legal_advice",
                      "document_analysis", "procedure_guide", "term_explanation", "general_question"]
        if normalized not in valid_types:
            if logger:
                logger.debug(f"Unknown query_type '{query_type_str}', defaulting to 'general_question'")
            normalized = "general_question"

        return normalized

    @staticmethod
    def get_domain_from_query_type(query_type: str) -> str:
        """
        질문 유형에서 도메인 추출

        현재 지원 도메인만 반환:
        - 민사법 (CIVIL_LAW)
        - 지식재산권법 (INTELLECTUAL_PROPERTY)
        - 행정법 (ADMINISTRATIVE_LAW)
        - 형사법 (CRIMINAL_LAW)

        이외는 기타/일반으로 처리
        """
        domain_mapping = {
            "precedent_search": "민사법",
            "law_inquiry": "민사법",
            "legal_advice": "민사법",
            "procedure_guide": "기타/일반",  # 절차 가이드는 기타로 처리
            "term_explanation": "기타/일반",
            "general_question": "기타/일반"
        }
        return domain_mapping.get(query_type, "기타/일반")

    @staticmethod
    def get_supported_domains() -> List[LegalDomain]:
        """현재 지원되는 도메인 목록 반환"""
        return [
            LegalDomain.CIVIL_LAW,
            LegalDomain.INTELLECTUAL_PROPERTY,
            LegalDomain.ADMINISTRATIVE_LAW,
            LegalDomain.CRIMINAL_LAW
        ]

    @staticmethod
    def is_supported_domain(domain: Optional[LegalDomain]) -> bool:
        """도메인이 지원되는지 확인"""
        if domain is None:
            return False
        return domain in WorkflowUtils.get_supported_domains()

    @staticmethod
    def get_question_type_and_domain(query_type: Any, query: str = "",
                                     logger: Optional[logging.Logger] = None) -> Tuple[QuestionType, Optional[LegalDomain]]:
        """
        질문 유형과 도메인 매핑 - LegalDomain enum 반환

        Args:
            query_type: 질문 유형 (문자열 또는 QuestionType enum)
            query: 사용자 질문 내용 (도메인 추출용, 선택사항)
            logger: 로거 (선택사항)

        현재 지원 도메인:
        - 민사법 (CIVIL_LAW)
        - 지식재산권법 (INTELLECTUAL_PROPERTY)
        - 행정법 (ADMINISTRATIVE_LAW)
        - 형사법 (CRIMINAL_LAW)

        이외의 모든 도메인은 기타(GENERAL)로 처리됩니다.
        """
        # 1. QuestionType 추출 (query_type 문자열을 enum으로 변환)
        question_type = WorkflowUtils.normalize_question_type(query_type, logger)

        # 2. 도메인 추출 (질문 내용에서 지원 도메인만 필터링)
        domain = WorkflowUtils.extract_supported_domain_from_query(query)

        # 로깅: QuestionType과 Domain 매핑 결과
        if logger:
            logger.info(
                f"📋 [QUESTION TYPE & DOMAIN] "
                f"query_type='{query_type}', "
                f"normalized_question_type={question_type.name if hasattr(question_type, 'name') else question_type}, "
                f"extracted_domain={domain.value if domain else 'None'}"
            )

        return (question_type, domain)

    @staticmethod
    def normalize_question_type(query_type: Any, logger: Optional[logging.Logger] = None) -> QuestionType:
        """query_type을 QuestionType enum으로 정규화"""
        # 이미 QuestionType enum인 경우
        if isinstance(query_type, QuestionType):
            return query_type

        # 문자열인 경우 매핑
        if isinstance(query_type, str):
            query_type_lower = query_type.lower().strip()

            # 직접 매핑
            type_mapping = {
                "precedent_search": QuestionType.PRECEDENT_SEARCH,
                "law_inquiry": QuestionType.LAW_INQUIRY,
                "legal_advice": QuestionType.LEGAL_ADVICE,
                "procedure_guide": QuestionType.PROCEDURE_GUIDE,
                "term_explanation": QuestionType.TERM_EXPLANATION,
                "general_question": QuestionType.GENERAL_QUESTION,
                "general": QuestionType.GENERAL_QUESTION,
            }

            # 직접 매핑 시도
            if query_type_lower in type_mapping:
                return type_mapping[query_type_lower]

            # QuestionType enum의 value로 찾기
            for qt in QuestionType:
                if qt.value.lower() == query_type_lower:
                    return qt

            # QuestionType enum의 name으로 찾기
            try:
                return QuestionType[query_type.upper()]
            except (KeyError, AttributeError):
                pass

        # 기본값
        if logger:
            logger.warning(f"⚠️ [QUESTION TYPE] Unknown query_type: '{query_type}', defaulting to GENERAL_QUESTION")
        return QuestionType.GENERAL_QUESTION

    @staticmethod
    def extract_supported_domain_from_query(query: str) -> Optional[LegalDomain]:
        """질문 내용에서 지원되는 도메인만 추출"""
        if not query or not isinstance(query, str):
            return LegalDomain.GENERAL

        query_lower = query.lower()

        # 지원 도메인별 키워드 매핑
        domain_keywords = {
            LegalDomain.CIVIL_LAW: [
                "민사", "계약", "손해배상", "채권", "채무", "임대차",
                "상속", "부동산", "계약서", "민법"
            ],
            LegalDomain.CRIMINAL_LAW: [
                "형사", "범죄", "처벌", "형량", "형법", "벌금",
                "징역", "교통사고", "절도", "사기", "폭행"
            ],
            LegalDomain.ADMINISTRATIVE_LAW: [
                "행정", "행정처분", "행정소송", "행정심판", "허가",
                "신고", "공무원", "행정법"
            ],
            LegalDomain.INTELLECTUAL_PROPERTY: [
                "특허", "상표", "저작권", "지적재산", "지식재산",
                "디자인", "영업비밀", "지적재산권"
            ]
        }

        # 도메인별 점수 계산
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score

        # 가장 높은 점수의 도메인 반환
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]

        # 기본값
        return LegalDomain.GENERAL

    @staticmethod
    def parse_validation_response(response: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """검증 응답 파싱"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # JSON이 없으면 간단한 파싱
            result = {
                "is_valid": "invalid" not in response.lower() and "문제" not in response,
                "quality_score": 0.7 if "good" in response.lower() or "좋" in response else 0.5,
                "issues": [],
                "strengths": [],
                "recommendations": []
            }

            # 문제점 추출 시도
            if "문제" in response:
                issues_match = re.findall(r'문제[점]?\s*[:\-]\s*([^\n]+)', response)
                result["issues"] = issues_match[:5]

            return result
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse validation response: {e}")
            return {
                "is_valid": True,
                "quality_score": 0.7,
                "issues": [],
                "strengths": [],
                "recommendations": []
            }

    @staticmethod
    def parse_improvement_instructions(response: str, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
        """개선 지시 파싱"""
        try:
            # None이면 건너뛰기
            if not response or "needs_improvement" not in response.lower():
                return None

            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                if result.get("needs_improvement", False):
                    return result

            return None
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse improvement instructions: {e}")
            return None

    @staticmethod
    def parse_final_validation_response(response: str, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
        """최종 검증 응답 파싱"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            return None
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse final validation response: {e}")
            return None

    @staticmethod
    def parse_query_type_analysis_response(response: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """질문 유형 분석 응답 파싱"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # 기본값
            return {
                "query_type": "simple_question",
                "confidence": 0.7,
                "reasoning": "JSON 파싱 실패"
            }
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse query type analysis response: {e}")
            return {
                "query_type": "simple_question",
                "confidence": 0.7,
                "reasoning": f"파싱 에러: {e}"
            }

    @staticmethod
    def parse_quality_validation_response(response: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """답변 품질 검증 응답 파싱"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # 기본값
            return {
                "is_valid": True,
                "quality_score": 0.8,
                "issues": [],
                "needs_improvement": False
            }
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse quality validation response: {e}")
            return {
                "is_valid": True,
                "quality_score": 0.8,
                "issues": [],
                "needs_improvement": False
            }

    @staticmethod
    def get_category_mapping() -> Dict[str, List[str]]:
        """카테고리 매핑 반환"""
        return {
            "precedent_search": ["family_law", "civil_law", "criminal_law"],
            "law_inquiry": ["family_law", "civil_law", "contract_review"],
            "legal_advice": ["family_law", "civil_law", "labor_law"],
            "procedure_guide": ["civil_procedure", "family_law", "labor_law"],
            "term_explanation": ["civil_law", "family_law", "contract_review"],
            "general_question": ["civil_law", "family_law", "contract_review"]
        }
