# -*- coding: utf-8 -*-
"""
LangGraph Studio Graph Export
Studio에서 사용할 그래프를 export하는 파일
"""

import sys
from pathlib import Path

# 상위 프로젝트 경로 추가 (필수 모듈 참조용)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from source.services.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from source.services.workflow_service import LangGraphWorkflowService
from config.langgraph_config import LangGraphConfig
from langgraph.graph import StateGraph

# 설정 로드
_config = None

def get_config():
    """설정 인스턴스 반환 (지연 로딩)"""
    global _config
    if _config is None:
        _config = LangGraphConfig.from_env()
    return _config

# 그래프 생성 함수
def create_graph() -> StateGraph:
    """Studio에서 사용할 그래프 생성"""
    config = get_config()
    workflow = EnhancedLegalQuestionWorkflow(config)
    return workflow.graph

# 컴파일된 앱 생성 함수
def create_app():
    """Studio에서 사용할 컴파일된 앱 생성"""
    config = get_config()
    service = LangGraphWorkflowService(config)
    return service.app

# 그래프 인스턴스 (Studio에서 직접 사용, 지연 로딩)
_graph_instance = None
def _get_graph() -> StateGraph:
    """그래프 인스턴스 반환 (지연 로딩)"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = create_graph()
    return _graph_instance

# 컴파일된 앱 인스턴스 (Studio에서 직접 사용, 지연 로딩)
_app_instance = None
def _get_app():
    """컴파일된 앱 인스턴스 반환 (지연 로딩)"""
    global _app_instance
    if _app_instance is None:
        _app_instance = create_app()
    return _app_instance

# Studio에서 참조할 수 있는 변수 (lazy property pattern)
class _LazyGraph:
    """지연 로딩 그래프 wrapper"""
    def __call__(self) -> StateGraph:
        return _get_graph()

    def __getattr__(self, name):
        return getattr(_get_graph(), name)

class _LazyApp:
    """지연 로딩 앱 wrapper"""
    def __call__(self):
        return _get_app()

    def __getattr__(self, name):
        return getattr(_get_app(), name)

# Studio에서 참조할 변수
graph = _LazyGraph()
app = _LazyApp()
