# -*- coding: utf-8 -*-
"""
대화 맥락 통합 테스트
- ConversationContext를 LangChain Message로 변환
- 관련성 기반 맥락 선택
- 토큰 기반 크기 관리
- 체크포인트 복원 시 대화 맥락 동기화
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_path))

try:
    from lawfirm_langgraph.core.conversation.conversation_manager import ConversationManager, ConversationContext, ConversationTurn
    from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
except ImportError:
    from core.conversation.conversation_manager import ConversationManager, ConversationContext, ConversationTurn
    from core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
    from config.langgraph_config import LangGraphConfig


class TestConversationContextConversion:
    """ConversationContext를 LangChain Message로 변환하는 기능 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.config = LangGraphConfig.from_env()
        self.workflow = EnhancedLegalQuestionWorkflow(self.config)
        self.session_id = "test_session_001"
        
        # 테스트용 ConversationContext 생성
        self.conversation_manager = ConversationManager()
        self.workflow.conversation_manager = self.conversation_manager
        
        # 테스트 턴 추가
        self.conversation_manager.add_turn(
            session_id=self.session_id,
            user_query="손해배상 청구 방법을 알려주세요",
            bot_response="민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...",
            question_type="legal_advice"
        )
        self.conversation_manager.add_turn(
            session_id=self.session_id,
            user_query="계약 해지 절차는 어떻게 되나요?",
            bot_response="계약 해지 절차는 다음과 같습니다...",
            question_type="procedure_guide"
        )
    
    def test_convert_conversation_context_to_messages_basic(self):
        """기본 변환 테스트"""
        context = self.conversation_manager.sessions.get(self.session_id)
        assert context is not None
        assert len(context.turns) == 2
        
        messages = self.workflow._convert_conversation_context_to_messages(
            context,
            max_turns=5
        )
        
        assert len(messages) == 4  # 2턴 * 2메시지 (user + assistant)
        assert messages[0].content == "손해배상 청구 방법을 알려주세요"
        assert messages[1].content.startswith("민법 제750조")
        assert messages[2].content == "계약 해지 절차는 어떻게 되나요?"
    
    def test_convert_conversation_context_to_messages_with_relevance(self):
        """관련성 기반 선택 테스트"""
        context = self.conversation_manager.sessions.get(self.session_id)
        
        # 관련 질문 추가
        self.conversation_manager.add_turn(
            session_id=self.session_id,
            user_query="위의 손해배상 사건에서 과실비율은 어떻게 정해지나요?",
            bot_response="과실비율은 교통사고의 경우...",
            question_type="legal_advice"
        )
        
        current_query = "손해배상 관련 판례를 더 찾아주세요"
        messages = self.workflow._convert_conversation_context_to_messages(
            context,
            max_turns=5,
            current_query=current_query,
            use_relevance=True
        )
        
        # 관련성 기반으로 선택된 메시지가 있어야 함
        assert len(messages) > 0
        # 손해배상 관련 메시지가 포함되어야 함
        message_contents = [msg.content for msg in messages]
        assert any("손해배상" in content for content in message_contents)
    
    def test_convert_conversation_context_to_messages_with_tokens(self):
        """토큰 기반 선택 테스트"""
        context = self.conversation_manager.sessions.get(self.session_id)
        
        # 긴 답변 추가
        long_response = "긴 답변입니다. " * 100  # 약 1500자
        self.conversation_manager.add_turn(
            session_id=self.session_id,
            user_query="긴 질문입니다",
            bot_response=long_response,
            question_type="legal_advice"
        )
        
        messages = self.workflow._convert_conversation_context_to_messages(
            context,
            max_turns=10,
            max_tokens=500  # 작은 토큰 제한
        )
        
        # 토큰 제한 내에서 선택된 메시지만 있어야 함
        assert len(messages) > 0
        # 토큰 제한으로 인해 일부만 선택되었을 수 있음
        assert len(messages) <= 10  # 최대 턴 수 제한


class TestTokenBasedPruning:
    """토큰 기반 크기 관리 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.config = LangGraphConfig.from_env()
        self.workflow = EnhancedLegalQuestionWorkflow(self.config)
        self.session_id = "test_session_002"
        
        self.conversation_manager = ConversationManager()
        self.workflow.conversation_manager = self.conversation_manager
    
    def test_prune_conversation_history_by_tokens(self):
        """토큰 기반 정리 테스트"""
        # 여러 턴 추가
        for i in range(10):
            self.conversation_manager.add_turn(
                session_id=self.session_id,
                user_query=f"질문 {i+1}",
                bot_response=f"답변 {i+1}입니다. " * 20,  # 각 답변 약 200자
                question_type="legal_advice"
            )
        
        context = self.conversation_manager.sessions.get(self.session_id)
        assert len(context.turns) == 10
        
        # 토큰 제한으로 정리
        selected_turns = self.workflow._prune_conversation_history_by_tokens(
            context,
            max_tokens=1000  # 약 3-4개 턴 정도
        )
        
        assert len(selected_turns) <= 10
        assert len(selected_turns) > 0
        # 최신 턴이 포함되어야 함
        assert selected_turns[-1].user_query == "질문 10"
    
    def test_prune_conversation_history_empty_context(self):
        """빈 컨텍스트 테스트"""
        context = ConversationContext(
            session_id="empty_session",
            turns=[],
            entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
            topic_stack=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        selected_turns = self.workflow._prune_conversation_history_by_tokens(
            context,
            max_tokens=1000
        )
        
        assert len(selected_turns) == 0


class TestRelevanceBasedSelection:
    """관련성 기반 선택 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.conversation_manager = ConversationManager()
        self.session_id = "test_session_003"
        
        # 다양한 주제의 대화 추가
        self.conversation_manager.add_turn(
            session_id=self.session_id,
            user_query="손해배상 청구 방법을 알려주세요",
            bot_response="민법 제750조에 따른 손해배상 청구 방법...",
            question_type="legal_advice"
        )
        self.conversation_manager.add_turn(
            session_id=self.session_id,
            user_query="계약 해지 절차는 어떻게 되나요?",
            bot_response="계약 해지 절차는 다음과 같습니다...",
            question_type="procedure_guide"
        )
        self.conversation_manager.add_turn(
            session_id=self.session_id,
            user_query="위의 손해배상 사건에서 과실비율은 어떻게 정해지나요?",
            bot_response="과실비율은 교통사고의 경우...",
            question_type="legal_advice"
        )
    
    def test_get_relevant_context(self):
        """관련 맥락 조회 테스트"""
        current_query = "손해배상 관련 판례를 더 찾아주세요"
        relevant_context = self.conversation_manager.get_relevant_context(
            self.session_id,
            current_query,
            max_turns=2
        )
        
        assert relevant_context is not None
        assert "relevant_turns" in relevant_context
        assert len(relevant_context["relevant_turns"]) > 0
        
        # 손해배상 관련 턴이 포함되어야 함
        relevant_turns = relevant_context["relevant_turns"]
        assert any("손해배상" in turn.get("user_query", "") for turn in relevant_turns)
    
    def test_get_relevant_context_no_matches(self):
        """관련 맥락이 없는 경우 테스트"""
        current_query = "완전히 다른 주제의 질문입니다"
        relevant_context = self.conversation_manager.get_relevant_context(
            self.session_id,
            current_query,
            max_turns=2
        )
        
        # 관련 맥락이 없을 수도 있음
        if relevant_context:
            assert "relevant_turns" in relevant_context


class TestAgenticDecisionWithChatHistory:
    """Agentic Decision 노드에서 chat_history 전달 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.config = LangGraphConfig.from_env()
        # Agentic 모드 비활성화 (테스트 환경)
        self.config.use_agentic_mode = False
        self.workflow = EnhancedLegalQuestionWorkflow(self.config)
        self.session_id = "test_session_004"
        
        self.conversation_manager = ConversationManager()
        self.workflow.conversation_manager = self.conversation_manager
        
        # 테스트 턴 추가
        self.conversation_manager.add_turn(
            session_id=self.session_id,
            user_query="손해배상 청구 방법",
            bot_response="민법 제750조에 따른 손해배상...",
            question_type="legal_advice"
        )
    
    def test_agentic_decision_chat_history_loading(self):
        """Agentic Decision 노드에서 chat_history 로딩 테스트"""
        try:
            from lawfirm_langgraph.core.workflow.state.state_definitions import create_initial_legal_state
        except ImportError:
            from core.workflow.state.state_definitions import create_initial_legal_state
        
        state = create_initial_legal_state(
            query="손해배상 관련 판례를 찾아주세요",
            session_id=self.session_id
        )
        
        # conversation_context 가져오기
        context = self.workflow._get_or_create_conversation_context(self.session_id)
        assert context is not None
        
        # 메시지 변환
        messages = self.workflow._convert_conversation_context_to_messages(
            context,
            max_turns=5,
            current_query=state.get("query", ""),
            use_relevance=True
        )
        
        assert len(messages) > 0
        assert messages[0].content == "손해배상 청구 방법"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

