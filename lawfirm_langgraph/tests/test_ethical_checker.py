# -*- coding: utf-8 -*-
"""
윤리적 검사 기능 테스트
"""

import pytest
from core.workflow.utils.ethical_checker import EthicalChecker


def test_ethical_checker_illegal_activity():
    """불법 행위 조장 키워드 테스트"""
    checker = EthicalChecker()
    
    # 불법 행위 방법을 묻는 질문
    is_problematic, reason, severity = checker.check_query("어떻게 해킹하는지 알려주세요")
    assert is_problematic is True
    assert reason is not None
    assert severity == "high"
    
    # 불법 행위 도움 요청
    is_problematic, reason, severity = checker.check_query("탈세 방법을 도와주세요")
    assert is_problematic is True
    
    # 법적 맥락에서 묻는 질문은 허용
    is_problematic, reason, severity = checker.check_query("해킹에 대한 법적 처벌은 무엇인가요?")
    assert is_problematic is False


def test_ethical_checker_unethical_patterns():
    """윤리적으로 문제되는 패턴 테스트"""
    checker = EthicalChecker()
    
    # 증거 인멸
    is_problematic, reason, severity = checker.check_query("증거를 인멸하는 방법은?")
    assert is_problematic is True
    
    # 재판 조작
    is_problematic, reason, severity = checker.check_query("재판을 조작하는 방법 알려주세요")
    assert is_problematic is True


def test_ethical_checker_normal_queries():
    """정상적인 질문 테스트"""
    checker = EthicalChecker()
    
    # 정상적인 법률 질문
    is_problematic, reason, severity = checker.check_query("계약 해지 사유에 대해 알려주세요")
    assert is_problematic is False
    
    # 정상적인 법률 상담
    is_problematic, reason, severity = checker.check_query("임대차 계약서 작성 방법을 알려주세요")
    assert is_problematic is False


def test_ethical_rejection_node():
    """윤리적 거부 노드 테스트"""
    from core.workflow.nodes.ethical_rejection_node import EthicalRejectionNode
    from core.workflow.state.state_definitions import LegalWorkflowState
    
    node = EthicalRejectionNode()
    state: LegalWorkflowState = {
        "query": "해킹 방법 알려주세요",
        "is_ethically_problematic": True,
        "ethical_rejection_reason": "불법 행위 조장"
    }
    
    result_state = node.generate_rejection_response(state, "불법 행위 조장")
    
    assert result_state.get("is_ethically_problematic") is True
    assert result_state.get("answer") is not None
    assert "불법 행위" in result_state.get("answer", "")

