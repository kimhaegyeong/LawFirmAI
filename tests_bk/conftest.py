# -*- coding: utf-8 -*-
"""
Pytest Configuration
Pytest κ³µν†µ ?¤μ • λ°??½μ¤μ²?
"""

import os
import sys
from pathlib import Path

# ?„λ΅?νΈ λ£¨νΈ κ²½λ΅λ¥?sys.path??μ¶”κ?
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ?κ²½ λ³€???¤μ •
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"
os.environ["LANGFUSE_ENABLED"] = "false"  # ?μ¤???μ—??κΈ°λ³Έ?μΌλ΅?λΉ„ν™?±ν™”

# LangGraph λª¨λ‹?°λ§ ?„ν™ ?μ¤???½μ¤μ²?
import pytest

from tests.langgraph.fixtures.workflow_factory import WorkflowFactory
from tests.langgraph.monitoring_switch import MonitoringMode, MonitoringSwitch


@pytest.fixture
def monitoring_switch():
    """λª¨λ‹?°λ§ ?„ν™ ? ν‹Έλ¦¬ν‹° ?½μ¤μ²?""
    return MonitoringSwitch


@pytest.fixture
def workflow_factory():
    """?ν¬?λ΅???©ν† λ¦??½μ¤μ²?""
    # ?μ¤????μΊμ‹ ?•λ¦¬
    WorkflowFactory.clear_cache()
    yield WorkflowFactory
    # ?μ¤????μΊμ‹ ?•λ¦¬
    WorkflowFactory.clear_cache()


@pytest.fixture(params=[
    MonitoringMode.LANGSMITH,
    MonitoringMode.LANGFUSE,
    MonitoringMode.BOTH,
    MonitoringMode.NONE
])
def monitoring_mode(request):
    """κ°?λª¨λ‹?°λ§ λª¨λ“λ³??μ¤???λΌλ―Έν„°"""
    return request.param


@pytest.fixture
def langsmith_mode(monitoring_switch):
    """LangSmith λª¨λ“ μ»¨ν…?¤νΈ ?½μ¤μ²?""
    with monitoring_switch.set_mode(MonitoringMode.LANGSMITH):
        yield MonitoringMode.LANGSMITH


@pytest.fixture
def langfuse_mode(monitoring_switch):
    """Langfuse λª¨λ“ μ»¨ν…?¤νΈ ?½μ¤μ²?""
    with monitoring_switch.set_mode(MonitoringMode.LANGFUSE):
        yield MonitoringMode.LANGFUSE
