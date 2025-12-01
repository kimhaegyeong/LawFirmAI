# -*- coding: utf-8 -*-
"""
ConversationStore 테스트
대화 저장소 모듈 단위 테스트
"""

import pytest
import sqlite3
import tempfile
import os
import json
from datetime import datetime
from typing import Dict, Any

from lawfirm_langgraph.core.data.conversation_store import ConversationStore


class TestConversationStore:
    """ConversationStore 테스트"""
    
    @pytest.fixture
    def temp_db(self):
        """임시 데이터베이스 픽스처"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
    
    @pytest.fixture
    def conversation_store(self, temp_db):
        """ConversationStore 인스턴스"""
        return ConversationStore(temp_db)
    
    def test_conversation_store_initialization(self, temp_db):
        """ConversationStore 초기화 테스트"""
        store = ConversationStore(temp_db)
        
        assert store.db_path == temp_db
        assert os.path.exists(temp_db)
    
    def test_get_connection(self, conversation_store):
        """데이터베이스 연결 테스트"""
        with conversation_store.get_connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
            assert conn.row_factory == sqlite3.Row
    
    def test_create_tables(self, conversation_store):
        """테이블 생성 테스트"""
        with conversation_store.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'conversation_sessions' in tables
            assert 'conversation_turns' in tables
            assert 'legal_entities' in tables
    
    def test_save_session(self, conversation_store):
        """세션 저장 테스트"""
        session_data = {
            "session_id": "test_session",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": [],
            "metadata": {"user_id": "test_user"},
            "turns": [
                {
                    "user_query": "테스트 질문",
                    "bot_response": "테스트 응답",
                    "timestamp": datetime.now().isoformat(),
                    "question_type": "general",
                    "entities": {}
                }
            ],
            "entities": {
                "laws": ["법률1"],
                "articles": ["조항1"]
            }
        }
        
        result = conversation_store.save_session(session_data)
        
        assert result is True
    
    def test_load_session(self, conversation_store):
        """세션 로드 테스트"""
        session_data = {
            "session_id": "test_session",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": [],
            "metadata": {"user_id": "test_user"},
            "turns": [
                {
                    "user_query": "테스트 질문",
                    "bot_response": "테스트 응답",
                    "timestamp": datetime.now().isoformat(),
                    "question_type": "general",
                    "entities": {}
                }
            ],
            "entities": {}
        }
        
        conversation_store.save_session(session_data)
        loaded_session = conversation_store.load_session("test_session")
        
        assert loaded_session is not None
        assert loaded_session["session_id"] == "test_session"
        assert len(loaded_session["turns"]) == 1
    
    def test_load_session_not_found(self, conversation_store):
        """존재하지 않는 세션 로드 테스트"""
        loaded_session = conversation_store.load_session("non_existent_session")
        
        assert loaded_session is None
    
    def test_add_turn(self, conversation_store):
        """턴 추가 테스트"""
        session_data = {
            "session_id": "test_session",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": [],
            "metadata": {},
            "turns": [],
            "entities": {}
        }
        
        conversation_store.save_session(session_data)
        
        turn_data = {
            "user_query": "새 질문",
            "bot_response": "새 응답",
            "timestamp": datetime.now().isoformat(),
            "question_type": "general",
            "entities": {}
        }
        
        result = conversation_store.add_turn("test_session", turn_data)
        
        assert result is True
        
        loaded_session = conversation_store.load_session("test_session")
        assert len(loaded_session["turns"]) == 1
    
    def test_get_session_list(self, conversation_store):
        """세션 목록 조회 테스트"""
        session_data = {
            "session_id": "test_session",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": [],
            "metadata": {},
            "turns": [],
            "entities": {}
        }
        
        conversation_store.save_session(session_data)
        
        sessions = conversation_store.get_session_list(limit=10)
        
        assert isinstance(sessions, list)
        assert len(sessions) > 0
    
    def test_delete_session(self, conversation_store):
        """세션 삭제 테스트"""
        session_data = {
            "session_id": "test_session",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": [],
            "metadata": {},
            "turns": [],
            "entities": {}
        }
        
        conversation_store.save_session(session_data)
        result = conversation_store.delete_session("test_session")
        
        assert result is True
        
        loaded_session = conversation_store.load_session("test_session")
        assert loaded_session is None
    
    def test_get_user_sessions(self, conversation_store):
        """사용자 세션 조회 테스트"""
        session_data = {
            "session_id": "test_session",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": [],
            "metadata": {"user_id": "test_user"},
            "turns": [],
            "entities": {}
        }
        
        conversation_store.save_session(session_data)
        
        sessions = conversation_store.get_user_sessions("test_user", limit=10)
        
        assert isinstance(sessions, list)
    
    def test_get_statistics(self, conversation_store):
        """통계 조회 테스트"""
        session_data = {
            "session_id": "test_session",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": [],
            "metadata": {},
            "turns": [],
            "entities": {}
        }
        
        conversation_store.save_session(session_data)
        
        stats = conversation_store.get_statistics()
        
        assert isinstance(stats, dict)
        assert "session_count" in stats
        assert "turn_count" in stats
        assert "total_entities" in stats or "entity_count" in stats

