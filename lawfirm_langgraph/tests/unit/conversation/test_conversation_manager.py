# -*- coding: utf-8 -*-
"""
Conversation Manager 테스트
대화 관리 conversation_manager 모듈 단위 테스트
"""

import pytest
from datetime import datetime, timedelta

from lawfirm_langgraph.core.conversation.conversation_manager import (
    ConversationManager,
    ConversationTurn,
    ConversationContext
)


class TestConversationManager:
    """Conversation Manager 테스트"""
    
    def test_init_default(self):
        """기본 초기화 테스트"""
        manager = ConversationManager()
        
        assert manager.max_context_turns == 10
        assert manager.max_session_age_hours == 24
        assert isinstance(manager.sessions, dict)
        assert len(manager.sessions) == 0
    
    def test_init_custom(self):
        """커스텀 초기화 테스트"""
        manager = ConversationManager(
            max_context_turns=5,
            max_session_age_hours=12
        )
        
        assert manager.max_context_turns == 5
        assert manager.max_session_age_hours == 12
    
    def test_add_turn_new_session(self):
        """새 세션에 턴 추가 테스트"""
        manager = ConversationManager()
        session_id = "test_session_1"
        
        context = manager.add_turn(
            session_id,
            "민법 제750조에 대해 알려주세요",
            "민법 제750조는 불법행위로 인한 손해배상에 관한 조문입니다.",
            "legal_advice"
        )
        
        assert context.session_id == session_id
        assert len(context.turns) == 1
        assert context.turns[0].user_query == "민법 제750조에 대해 알려주세요"
        assert context.turns[0].question_type == "legal_advice"
    
    def test_add_turn_existing_session(self):
        """기존 세션에 턴 추가 테스트"""
        manager = ConversationManager()
        session_id = "test_session_2"
        
        manager.add_turn(session_id, "질문1", "답변1")
        context = manager.add_turn(session_id, "질문2", "답변2")
        
        assert len(context.turns) == 2
        assert context.turns[0].user_query == "질문1"
        assert context.turns[1].user_query == "질문2"
    
    def test_add_turn_entity_extraction(self):
        """엔티티 추출 테스트"""
        manager = ConversationManager()
        session_id = "test_session_3"
        
        context = manager.add_turn(
            session_id,
            "민법 제750조와 형법 제250조에 대해 알려주세요",
            "민법 제750조는 손해배상, 형법 제250조는 살인에 관한 조문입니다."
        )
        
        assert len(context.entities["laws"]) > 0
        assert len(context.entities["articles"]) > 0
    
    def test_add_turn_topic_identification(self):
        """주제 식별 테스트"""
        manager = ConversationManager()
        session_id = "test_session_4"
        
        context = manager.add_turn(
            session_id,
            "계약서 작성 시 주의할 사항은?",
            "계약서 작성 시 주의할 사항은 다음과 같습니다..."
        )
        
        assert len(context.topic_stack) > 0
    
    def test_add_turn_context_limit(self):
        """컨텍스트 크기 제한 테스트"""
        manager = ConversationManager(max_context_turns=3)
        session_id = "test_session_5"
        
        for i in range(5):
            manager.add_turn(session_id, f"질문{i}", f"답변{i}")
        
        context = manager.sessions[session_id]
        assert len(context.turns) == 3
        assert context.turns[0].user_query == "질문2"
    
    def test_get_relevant_context_no_session(self):
        """세션이 없는 경우 관련 맥락 조회 테스트"""
        manager = ConversationManager()
        
        result = manager.get_relevant_context("nonexistent", "질문")
        
        assert result is None
    
    def test_get_relevant_context_empty_turns(self):
        """턴이 없는 경우 관련 맥락 조회 테스트"""
        manager = ConversationManager()
        session_id = "test_session_6"
        
        manager.sessions[session_id] = ConversationContext(
            session_id=session_id,
            turns=[],
            entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
            topic_stack=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        result = manager.get_relevant_context(session_id, "질문")
        
        assert result is None
    
    def test_get_relevant_context_with_turns(self):
        """턴이 있는 경우 관련 맥락 조회 테스트"""
        manager = ConversationManager()
        session_id = "test_session_7"
        
        manager.add_turn(session_id, "계약서 작성", "계약서 작성 방법은...")
        manager.add_turn(session_id, "손해배상 청구", "손해배상 청구 방법은...")
        
        result = manager.get_relevant_context(session_id, "계약서", max_turns=1)
        
        assert result is not None
        assert result["session_id"] == session_id
        assert "relevant_turns" in result
        assert "entities" in result
        assert "current_topics" in result
    
    def test_cleanup_old_sessions(self):
        """오래된 세션 정리 테스트"""
        manager = ConversationManager(max_session_age_hours=1)
        session_id = "test_session_8"
        
        manager.add_turn(session_id, "질문", "답변")
        
        old_context = manager.sessions[session_id]
        old_context.last_updated = datetime.now() - timedelta(hours=2)
        
        manager.cleanup_old_sessions()
        
        assert session_id not in manager.sessions
    
    def test_cleanup_old_sessions_recent(self):
        """최근 세션은 정리되지 않음 테스트"""
        manager = ConversationManager(max_session_age_hours=24)
        session_id = "test_session_9"
        
        manager.add_turn(session_id, "질문", "답변")
        
        manager.cleanup_old_sessions()
        
        assert session_id in manager.sessions
    
    def test_get_session_stats_empty(self):
        """빈 세션 통계 테스트"""
        manager = ConversationManager()
        
        stats = manager.get_session_stats()
        
        assert stats["total_sessions"] == 0
        assert stats["total_turns"] == 0
        assert stats["avg_turns_per_session"] == 0
    
    def test_get_session_stats_with_sessions(self):
        """세션이 있는 경우 통계 테스트"""
        manager = ConversationManager()
        
        manager.add_turn("session1", "질문1", "답변1")
        manager.add_turn("session1", "질문2", "답변2")
        manager.add_turn("session2", "질문3", "답변3")
        
        stats = manager.get_session_stats()
        
        assert stats["total_sessions"] == 2
        assert stats["total_turns"] == 3
        assert stats["avg_turns_per_session"] == 1.5
    
    def test_export_session_nonexistent(self):
        """존재하지 않는 세션 내보내기 테스트"""
        manager = ConversationManager()
        
        result = manager.export_session("nonexistent")
        
        assert result is None
    
    def test_export_session_existing(self):
        """존재하는 세션 내보내기 테스트"""
        manager = ConversationManager()
        session_id = "test_session_10"
        
        manager.add_turn(session_id, "질문", "답변", "legal_advice")
        
        result = manager.export_session(session_id)
        
        assert result is not None
        assert result["session_id"] == session_id
        assert len(result["turns"]) == 1
        assert result["turns"][0]["user_query"] == "질문"
        assert "entities" in result
        assert "topic_stack" in result
    
    def test_entity_extraction_laws(self):
        """법령 엔티티 추출 테스트"""
        manager = ConversationManager()
        session_id = "test_session_11"
        
        context = manager.add_turn(
            session_id,
            "민법과 형법에 대해 알려주세요",
            "민법과 형법은 중요한 법률입니다."
        )
        
        assert len(context.entities["laws"]) >= 2
    
    def test_entity_extraction_articles(self):
        """조문 엔티티 추출 테스트"""
        manager = ConversationManager()
        session_id = "test_session_12"
        
        context = manager.add_turn(
            session_id,
            "민법 제750조와 제751조에 대해 알려주세요",
            "민법 제750조와 제751조는 손해배상에 관한 조문입니다."
        )
        
        assert len(context.entities["articles"]) >= 2
    
    def test_entity_extraction_precedents(self):
        """판례 엔티티 추출 테스트"""
        manager = ConversationManager()
        session_id = "test_session_13"
        
        context = manager.add_turn(
            session_id,
            "2023다12345 판례에 대해 알려주세요",
            "2023다12345 판례는 손해배상에 관한 판례입니다."
        )
        
        assert len(context.entities["precedents"]) > 0
    
    def test_topic_identification_multiple(self):
        """다중 주제 식별 테스트"""
        manager = ConversationManager()
        session_id = "test_session_14"
        
        context = manager.add_turn(
            session_id,
            "계약서 작성과 손해배상 청구에 대해 알려주세요",
            "계약서 작성과 손해배상 청구에 대한 설명입니다."
        )
        
        assert len(context.topic_stack) >= 2

