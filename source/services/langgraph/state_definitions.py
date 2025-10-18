# -*- coding: utf-8 -*-
"""
LangGraph State Definitions
LangGraph 워크플로우 상태 정의 모듈
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add


class LegalWorkflowState(TypedDict):
    """법률 워크플로우 상태 정의"""
    
    # 입력 데이터
    query: str
    session_id: str
    
    # 질문 분류 결과
    query_type: str  # "simple", "complex", "contract_review", "precedent_search"
    confidence: float
    
    # 문서 검색 결과
    retrieved_docs: Annotated[List[Dict[str, Any]], add]
    search_metadata: Dict[str, Any]
    
    # 컨텍스트 분석 결과
    analysis: Optional[str]
    legal_references: List[str]
    
    # 최종 답변
    answer: str
    sources: List[str]
    
    # 처리 과정 메타데이터
    processing_steps: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    metadata: Dict[str, Any]
    
    # 성능 메트릭
    processing_time: float
    tokens_used: int


class AgentWorkflowState(TypedDict):
    """에이전트 워크플로우 상태 정의 (향후 확장용)"""
    
    # 기본 상태
    query: str
    session_id: str
    
    # 에이전트 상태
    current_agent: str
    agent_history: Annotated[List[Dict[str, Any]], add]
    
    # 작업 결과
    task_results: Annotated[List[Dict[str, Any]], add]
    final_result: Optional[Dict[str, Any]]
    
    # 메타데이터
    processing_steps: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    metadata: Dict[str, Any]


class StreamingWorkflowState(TypedDict):
    """스트리밍 워크플로우 상태 정의 (향후 확장용)"""
    
    # 기본 상태
    query: str
    session_id: str
    
    # 스트리밍 데이터
    stream_chunks: Annotated[List[str], add]
    is_streaming: bool
    
    # 최종 결과
    final_answer: str
    sources: List[str]
    
    # 메타데이터
    processing_steps: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    metadata: Dict[str, Any]


# 상태 초기화 헬퍼 함수들
def create_initial_legal_state(query: str, session_id: str) -> LegalWorkflowState:
    """법률 워크플로우 초기 상태 생성"""
    return LegalWorkflowState(
        query=query,
        session_id=session_id,
        query_type="",
        confidence=0.0,
        retrieved_docs=[],
        search_metadata={},
        analysis=None,
        legal_references=[],
        answer="",
        sources=[],
        processing_steps=[],
        errors=[],
        metadata={},
        processing_time=0.0,
        tokens_used=0
    )


def create_initial_agent_state(query: str, session_id: str) -> AgentWorkflowState:
    """에이전트 워크플로우 초기 상태 생성"""
    return AgentWorkflowState(
        query=query,
        session_id=session_id,
        current_agent="",
        agent_history=[],
        task_results=[],
        final_result=None,
        processing_steps=[],
        errors=[],
        metadata={}
    )


def create_initial_streaming_state(query: str, session_id: str) -> StreamingWorkflowState:
    """스트리밍 워크플로우 초기 상태 생성"""
    return StreamingWorkflowState(
        query=query,
        session_id=session_id,
        stream_chunks=[],
        is_streaming=False,
        final_answer="",
        sources=[],
        processing_steps=[],
        errors=[],
        metadata={}
    )
