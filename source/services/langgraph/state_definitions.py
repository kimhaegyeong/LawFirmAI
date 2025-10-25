# -*- coding: utf-8 -*-
"""
향상된 상태 정의
LangGraph 기반 법률 워크플로우를 위한 확장된 상태 정의
"""

from typing import TypedDict, List, Optional, Dict, Any, Union
from datetime import datetime
from langchain_core.messages import BaseMessage


class LegalWorkflowState(TypedDict):
    """법률 워크플로우 상태 정의"""
    
    # === 기본 입력 정보 ===
    user_query: str
    context: Optional[str]
    session_id: str
    user_id: str
    
    # === 처리 단계별 결과 ===
    input_validation: Dict[str, Any]
    question_classification: Dict[str, Any]
    domain_analysis: Dict[str, Any]
    retrieved_documents: List[Dict[str, Any]]
    legal_analysis: Dict[str, Any]
    generated_response: str
    quality_metrics: Dict[str, Any]
    
    # === 메타데이터 ===
    workflow_steps: List[str]
    processing_time: float
    confidence_score: float
    error_messages: List[str]
    
    # === 대화 히스토리 ===
    conversation_history: List[BaseMessage]
    user_preferences: Dict[str, Any]
    
    # === 중간 결과들 ===
    intermediate_results: Dict[str, Any]
    validation_results: Dict[str, Any]
    
    # === 확장된 필드들 ===
    enriched_context: Dict[str, Any]
    agent_coordination: Dict[str, Any]
    synthesis_result: Dict[str, Any]
    quality_assurance_result: Dict[str, Any]
    
    # === 멀티 에이전트 결과 ===
    research_agent_result: Dict[str, Any]
    analysis_agent_result: Dict[str, Any]
    review_agent_result: Dict[str, Any]
    
    # === 성능 메트릭 ===
    performance_metrics: Dict[str, Any]
    memory_usage: Dict[str, Any]
    
    # === 사용자 컨텍스트 ===
    user_expertise_level: str
    preferred_response_style: str
    device_info: Dict[str, Any]
    
    # === 법률 특화 정보 ===
    legal_domain: str
    statute_references: List[Dict[str, Any]]
    precedent_references: List[Dict[str, Any]]
    legal_confidence: float
    
    # === 워크플로우 제어 ===
    current_step: str
    next_step: Optional[str]
    workflow_completed: bool
    requires_human_review: bool
    
    # === 캐싱 및 최적화 ===
    cache_hits: List[str]
    cache_misses: List[str]
    optimization_applied: List[str]
    
    # === 모니터링 및 로깅 ===
    trace_id: Optional[str]
    span_id: Optional[str]
    log_entries: List[Dict[str, Any]]
    
    # === 확장성 필드 ===
    custom_fields: Dict[str, Any]
    plugin_results: Dict[str, Any]
    external_api_responses: Dict[str, Any]


class WorkflowStep(TypedDict):
    """워크플로우 단계 정의"""
    step_name: str
    step_type: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    status: str  # pending, running, completed, failed
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    error_message: Optional[str]
    retry_count: int
    dependencies: List[str]


class AgentResult(TypedDict):
    """에이전트 결과 정의"""
    agent_name: str
    agent_type: str
    task_assigned: str
    task_completed: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    status: str  # pending, running, completed, failed
    confidence: float
    result_data: Dict[str, Any]
    error_message: Optional[str]
    retry_count: int


class QualityMetrics(TypedDict):
    """품질 메트릭 정의"""
    response_length: int
    confidence_score: float
    relevance_score: float
    completeness_score: float
    accuracy_score: float
    clarity_score: float
    overall_quality: float
    quality_threshold_met: bool
    improvement_suggestions: List[str]


class PerformanceMetrics(TypedDict):
    """성능 메트릭 정의"""
    total_processing_time: float
    step_processing_times: Dict[str, float]
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    api_call_count: int
    api_response_times: List[float]
    error_rate: float
    throughput_per_minute: float


class UserContext(TypedDict):
    """사용자 컨텍스트 정의"""
    user_id: str
    session_id: str
    expertise_level: str  # beginner, intermediate, expert, professional
    preferred_detail_level: str  # low, medium, high
    response_style: str  # formal, casual, technical, simple
    language_preference: str
    device_type: str  # mobile, desktop, tablet
    timezone: str
    interaction_history: List[Dict[str, Any]]
    preferences: Dict[str, Any]


class LegalContext(TypedDict):
    """법률 컨텍스트 정의"""
    legal_domain: str
    jurisdiction: str
    legal_system: str
    applicable_laws: List[str]
    relevant_statutes: List[Dict[str, Any]]
    relevant_precedents: List[Dict[str, Any]]
    legal_complexity: str  # simple, moderate, complex
    requires_expert_review: bool
    confidentiality_level: str  # public, internal, confidential, restricted


class WorkflowConfiguration(TypedDict):
    """워크플로우 설정 정의"""
    workflow_id: str
    workflow_version: str
    enabled_steps: List[str]
    disabled_steps: List[str]
    timeout_settings: Dict[str, float]
    retry_settings: Dict[str, int]
    quality_thresholds: Dict[str, float]
    performance_targets: Dict[str, float]
    custom_settings: Dict[str, Any]


class StateTransition(TypedDict):
    """상태 전환 정의"""
    from_state: str
    to_state: str
    transition_condition: str
    transition_time: datetime
    transition_reason: str
    transition_data: Dict[str, Any]


class WorkflowCheckpoint(TypedDict):
    """워크플로우 체크포인트 정의"""
    checkpoint_id: str
    checkpoint_time: datetime
    state_snapshot: LegalWorkflowState
    workflow_position: str
    can_resume: bool
    resume_data: Dict[str, Any]


# 상태 유틸리티 함수들
def create_empty_state() -> LegalWorkflowState:
    """빈 상태 생성"""
    return LegalWorkflowState(
        user_query="",
        context=None,
        session_id="",
        user_id="",
        input_validation={},
        question_classification={},
        domain_analysis={},
        retrieved_documents=[],
        legal_analysis={},
        generated_response="",
        quality_metrics={},
        workflow_steps=[],
        processing_time=0.0,
        confidence_score=0.0,
        error_messages=[],
        conversation_history=[],
        user_preferences={},
        intermediate_results={},
        validation_results={},
        enriched_context={},
        agent_coordination={},
        synthesis_result={},
        quality_assurance_result={},
        research_agent_result={},
        analysis_agent_result={},
        review_agent_result={},
        performance_metrics={},
        memory_usage={},
        user_expertise_level="beginner",
        preferred_response_style="formal",
        device_info={},
        legal_domain="",
        statute_references=[],
        precedent_references=[],
        legal_confidence=0.0,
        current_step="",
        next_step=None,
        workflow_completed=False,
        requires_human_review=False,
        cache_hits=[],
        cache_misses=[],
        optimization_applied=[],
        trace_id=None,
        span_id=None,
        log_entries=[],
        custom_fields={},
        plugin_results={},
        external_api_responses={}
    )


def create_initial_state(query: str, session_id: str, user_id: str, 
                        context: Optional[str] = None) -> LegalWorkflowState:
    """초기 상태 생성"""
    state = create_empty_state()
    state.update({
        "user_query": query,
        "context": context,
        "session_id": session_id,
        "user_id": user_id,
        "current_step": "input_validation",
        "workflow_completed": False,
        "requires_human_review": False
    })
    return state


def update_workflow_step(state: LegalWorkflowState, step_name: str, 
                        step_data: Dict[str, Any]) -> LegalWorkflowState:
    """워크플로우 단계 업데이트"""
    workflow_steps = state.get("workflow_steps", [])
    if step_name not in workflow_steps:
        workflow_steps.append(step_name)
    
    return {
        **state,
        "workflow_steps": workflow_steps,
        "current_step": step_name,
        step_name: step_data
    }


def add_error_message(state: LegalWorkflowState, error_message: str) -> LegalWorkflowState:
    """오류 메시지 추가"""
    error_messages = state.get("error_messages", [])
    error_messages.append(error_message)
    
    return {
        **state,
        "error_messages": error_messages
    }


def update_quality_metrics(state: LegalWorkflowState, metrics: QualityMetrics) -> LegalWorkflowState:
    """품질 메트릭 업데이트"""
    return {
        **state,
        "quality_metrics": metrics,
        "confidence_score": metrics["overall_quality"]
    }


def update_performance_metrics(state: LegalWorkflowState, metrics: PerformanceMetrics) -> LegalWorkflowState:
    """성능 메트릭 업데이트"""
    return {
        **state,
        "performance_metrics": metrics,
        "processing_time": metrics["total_processing_time"]
    }


def mark_workflow_completed(state: LegalWorkflowState) -> LegalWorkflowState:
    """워크플로우 완료 표시"""
    return {
        **state,
        "workflow_completed": True,
        "current_step": "completed"
    }


def requires_human_review(state: LegalWorkflowState, reason: str) -> LegalWorkflowState:
    """인간 검토 필요 표시"""
    return {
        **state,
        "requires_human_review": True,
        "next_step": "human_review",
        "intermediate_results": {
            **state.get("intermediate_results", {}),
            "human_review_reason": reason
        }
    }