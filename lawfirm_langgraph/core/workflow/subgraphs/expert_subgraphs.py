# -*- coding: utf-8 -*-
"""
전문가 서브그래프 모듈
특정 법률 분야에 특화된 워크플로우
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

from langgraph.graph import END, StateGraph

from .state_definitions import LegalWorkflowState

logger = get_logger(__name__)


class FamilyLawExpertGraph:
    """가족법 전문 서브그래프"""

    def __init__(self, parent_workflow):
        self.parent = parent_workflow
        self.logger = get_logger(__name__)

    def build_graph(self) -> StateGraph:
        """가족법 전문 그래프 구축"""
        graph = StateGraph(LegalWorkflowState)

        # 가족법 특화 노드
        graph.add_node("analyze_family_case", self.analyze_family_case)
        graph.add_node("search_family_precedents", self.search_family_precedents)
        graph.add_node("generate_family_advice", self.generate_family_advice)

        # 플로우
        graph.set_entry_point("analyze_family_case")
        graph.add_edge("analyze_family_case", "search_family_precedents")
        graph.add_edge("search_family_precedents", "generate_family_advice")
        graph.add_edge("generate_family_advice", END)

        return graph

    def analyze_family_case(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """가족법 사건 분석"""
        # 이혼, 양육권, 상속 등 가족법 특화 분석
        self.logger.info("Analyzing family law case")

        query = state.get("query", "")

        # 가족법 키워드 추출
        family_keywords = {
            "divorce": ["이혼", "별거"],
            "custody": ["양육권", "친권"],
            "inheritance": ["상속", "유산"],
            "adoption": ["입양", "친양자"],
            "alimony": ["위자료", "부양비"]
        }

        detected_issues = []
        query_lower = query.lower()

        for issue_type, keywords in family_keywords.items():
            if any(k in query_lower for k in keywords):
                detected_issues.append(issue_type)

        state["metadata"]["family_law_issues"] = detected_issues

        return state

    def search_family_precedents(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """가족법 판례 검색"""
        # 가족법 관련 판례 집중 검색
        self.logger.info("Searching family law precedents")

        # 기존 검색 결과를 가족법 중심으로 필터링
        retrieved_docs = state.get("retrieved_docs", [])

        # 가족법 관련 문서 우선순위 상승
        for doc in retrieved_docs:
            content = doc.get("content", "").lower()
            if any(keyword in content for keyword in ["이혼", "상속", "양육권", "친권"]):
                doc["relevance_score"] = doc.get("relevance_score", 0.5) * 1.3

        return state

    def generate_family_advice(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """가족법 조언 생성"""
        # 가족법 특화 답변 생성
        self.logger.info("Generating family law advice")

        # 기존 답변 생성 로직 사용
        state["metadata"]["expert_advice_type"] = "family_law"

        return state


class CorporateLawExpertGraph:
    """기업법 전문 서브그래프"""

    def __init__(self, parent_workflow):
        self.parent = parent_workflow
        self.logger = get_logger(__name__)

    def build_graph(self) -> StateGraph:
        """기업법 전문 그래프 구축"""
        graph = StateGraph(LegalWorkflowState)

        # 기업법 특화 노드
        graph.add_node("analyze_corporate_issue", self.analyze_corporate_issue)
        graph.add_node("search_corporate_law", self.search_corporate_law)
        graph.add_node("generate_corporate_advice", self.generate_corporate_advice)

        # 플로우
        graph.set_entry_point("analyze_corporate_issue")
        graph.add_edge("analyze_corporate_issue", "search_corporate_law")
        graph.add_edge("search_corporate_law", "generate_corporate_advice")
        graph.add_edge("generate_corporate_advice", END)

        return graph

    def analyze_corporate_issue(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """기업법 이슈 분석"""
        self.logger.info("Analyzing corporate law issue")

        query = state.get("query", "")

        # 기업법 키워드 추출
        corporate_keywords = {
            "shareholders": ["주주", "지분"],
            "board": ["이사", "임원"],
            "merger": ["합병", "인수"],
            "ipo": ["상장", "공모"],
            "compliance": ["준법", "내부통제"]
        }

        detected_issues = []
        query_lower = query.lower()

        for issue_type, keywords in corporate_keywords.items():
            if any(k in query_lower for k in keywords):
                detected_issues.append(issue_type)

        state["metadata"]["corporate_law_issues"] = detected_issues

        return state

    def search_corporate_law(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """기업법 검색"""
        self.logger.info("Searching corporate law")

        # 기존 검색 결과를 기업법 중심으로 필터링
        retrieved_docs = state.get("retrieved_docs", [])

        # 기업법 관련 문서 우선순위 상승
        for doc in retrieved_docs:
            content = doc.get("content", "").lower()
            if any(keyword in content for keyword in ["주주", "이사회", "임원", "자본금"]):
                doc["relevance_score"] = doc.get("relevance_score", 0.5) * 1.3

        return state

    def generate_corporate_advice(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """기업법 조언 생성"""
        self.logger.info("Generating corporate law advice")

        # 기존 답변 생성 로직 사용
        state["metadata"]["expert_advice_type"] = "corporate_law"

        return state


class IPLawExpertGraph:
    """지적재산권법 전문 서브그래프"""

    def __init__(self, parent_workflow):
        self.parent = parent_workflow
        self.logger = get_logger(__name__)

    def build_graph(self) -> StateGraph:
        """지적재산권법 전문 그래프 구축"""
        graph = StateGraph(LegalWorkflowState)

        # 지적재산권법 특화 노드
        graph.add_node("analyze_ip_issue", self.analyze_ip_issue)
        graph.add_node("search_ip_law", self.search_ip_law)
        graph.add_node("generate_ip_advice", self.generate_ip_advice)

        # 플로우
        graph.set_entry_point("analyze_ip_issue")
        graph.add_edge("analyze_ip_issue", "search_ip_law")
        graph.add_edge("search_ip_law", "generate_ip_advice")
        graph.add_edge("generate_ip_advice", END)

        return graph

    def analyze_ip_issue(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """지적재산권 이슈 분석"""
        self.logger.info("Analyzing intellectual property law issue")

        query = state.get("query", "")

        # 지적재산권 키워드 추출
        ip_keywords = {
            "patent": ["특허", "발명"],
            "trademark": ["상표", "브랜드"],
            "copyright": ["저작권", "작품"],
            "design": ["디자인", "의장"]
        }

        detected_issues = []
        query_lower = query.lower()

        for issue_type, keywords in ip_keywords.items():
            if any(k in query_lower for k in keywords):
                detected_issues.append(issue_type)

        state["metadata"]["ip_law_issues"] = detected_issues

        return state

    def search_ip_law(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """지적재산권법 검색"""
        self.logger.info("Searching intellectual property law")

        # 기존 검색 결과를 지적재산권법 중심으로 필터링
        retrieved_docs = state.get("retrieved_docs", [])

        # 지적재산권법 관련 문서 우선순위 상승
        for doc in retrieved_docs:
            content = doc.get("content", "").lower()
            if any(keyword in content for keyword in ["특허", "상표", "저작권", "디자인"]):
                doc["relevance_score"] = doc.get("relevance_score", 0.5) * 1.3

        return state

    def generate_ip_advice(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """지적재산권법 조언 생성"""
        self.logger.info("Generating intellectual property law advice")

        # 기존 답변 생성 로직 사용
        state["metadata"]["expert_advice_type"] = "intellectual_property"

        return state
