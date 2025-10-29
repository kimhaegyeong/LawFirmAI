# -*- coding: utf-8 -*-
"""
Pytest Configuration
Pytest 공통 설정 및 픽스처
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 설정
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"
os.environ["LANGFUSE_ENABLED"] = "false"  # 테스트 시에는 기본적으로 비활성화

# LangGraph 모니터링 전환 테스트 픽스처
import pytest

from tests.langgraph.fixtures.workflow_factory import WorkflowFactory
from tests.langgraph.monitoring_switch import MonitoringMode, MonitoringSwitch


@pytest.fixture
def monitoring_switch():
    """모니터링 전환 유틸리티 픽스처"""
    return MonitoringSwitch


@pytest.fixture
def workflow_factory():
    """워크플로우 팩토리 픽스처"""
    # 테스트 전 캐시 정리
    WorkflowFactory.clear_cache()
    yield WorkflowFactory
    # 테스트 후 캐시 정리
    WorkflowFactory.clear_cache()


@pytest.fixture(params=[
    MonitoringMode.LANGSMITH,
    MonitoringMode.LANGFUSE,
    MonitoringMode.BOTH,
    MonitoringMode.NONE
])
def monitoring_mode(request):
    """각 모니터링 모드별 테스트 파라미터"""
    return request.param


@pytest.fixture
def langsmith_mode(monitoring_switch):
    """LangSmith 모드 컨텍스트 픽스처"""
    with monitoring_switch.set_mode(MonitoringMode.LANGSMITH):
        yield MonitoringMode.LANGSMITH


@pytest.fixture
def langfuse_mode(monitoring_switch):
    """Langfuse 모드 컨텍스트 픽스처"""
    with monitoring_switch.set_mode(MonitoringMode.LANGFUSE):
        yield MonitoringMode.LANGFUSE
