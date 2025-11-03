# -*- coding: utf-8 -*-
"""
통합 세션 관리자
ConversationManager와 ConversationStore를 통합하여 메모리와 DB를 동시에 관리
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..data.conversation_store import ConversationStore
from .conversation_manager import ConversationManager, ConversationContext, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class SessionSyncInfo:
    """세션 동기화 정보"""
    session_id: str
    last_sync_time: datetime
    sync_count: int
    needs_sync: bool = False


class IntegratedSessionManager:
    """통합 세션 관리자"""
    
    def __init__(self, db_path: str = "data/conversations.db", sync_interval: int = 5):
        """
        통합 세션 관리자 초기화
        
        Args:
            db_path: 데이터베이스 경로
            sync_interval: DB 동기화 간격 (턴 수)
        """
        self.logger = logging.getLogger(__name__)
        self.sync_interval = sync_interval
        
        # 컴포넌트 초기화
        self.conversation_manager = ConversationManager()
        self.conversation_store = ConversationStore(db_path)
        
        # 동기화 정보 추적
        self.sync_info: Dict[str, SessionSyncInfo] = {}
        
        # 사용자 ID 추적
        self.session_users: Dict[str, str] = {}
        
        # 캐시 설정
        self.cache_enabled = True
        self.cache_ttl = 3600  # 1시간
        
        self.logger.info(f"IntegratedSessionManager initialized with sync_interval={sync_interval}")
    
    def add_turn(self, 
                 session_id: str, 
                 user_query: str, 
                 bot_response: str,
                 question_type: Optional[str] = None,
                 user_id: Optional[str] = None,
                 auto_sync: bool = True) -> ConversationContext:
        """
        대화 턴 추가 (메모리 + DB 통합)
        
        Args:
            session_id: 세션 ID
            user_query: 사용자 질문
            bot_response: 봇 응답
            question_type: 질문 유형
            user_id: 사용자 ID
            auto_sync: 자동 동기화 여부
            
        Returns:
            ConversationContext: 업데이트된 대화 맥락
        """
        try:
            self.logger.info(f"Adding turn to session {session_id}")
            
            # 1. 메모리에서 세션 가져오기 또는 생성
            context = self.get_or_create_session(session_id, user_id)
            
            # 사용자 ID 추적
            if user_id:
                self.session_users[session_id] = user_id
            
            # 2. 턴 추가 (메모리)
            context = self.conversation_manager.add_turn(
                session_id, user_query, bot_response, question_type
            )
            
            # 3. 동기화 정보 업데이트
            self._update_sync_info(session_id)
            
            # 4. 자동 동기화 (필요시)
            if auto_sync and self._should_sync(session_id):
                self.sync_to_database(session_id)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error adding turn: {e}")
            # 폴백: 메모리만 사용
            return self.conversation_manager.add_turn(session_id, user_query, bot_response, question_type)
    
    def get_or_create_session(self, session_id: str, user_id: Optional[str] = None) -> ConversationContext:
        """
        세션 가져오기 또는 생성
        
        Args:
            session_id: 세션 ID
            user_id: 사용자 ID
            
        Returns:
            ConversationContext: 대화 맥락
        """
        try:
            # 1. 메모리에서 확인
            context = self.conversation_manager.sessions.get(session_id)
            if context:
                return context
            
            # 2. DB에서 로드 시도
            context = self.load_from_database(session_id)
            if context:
                # 메모리에 캐시
                self.conversation_manager.sessions[session_id] = context
                self._update_sync_info(session_id)
                return context
            
            # 3. 새 세션 생성
            context = ConversationContext(
                session_id=session_id,
                turns=[],
                entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
                topic_stack=[],
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            # 메모리에 저장
            self.conversation_manager.sessions[session_id] = context
            
            # DB에 저장 (사용자 ID 포함)
            if user_id:
                self._save_session_to_db(context, user_id)
            
            self._update_sync_info(session_id)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting or creating session: {e}")
            # 폴백: 기본 세션 생성
            return ConversationContext(
                session_id=session_id,
                turns=[],
                entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
                topic_stack=[],
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
    
    def sync_to_database(self, session_id: str) -> bool:
        """
        메모리에서 DB로 동기화
        
        Args:
            session_id: 세션 ID
            
        Returns:
            bool: 동기화 성공 여부
        """
        try:
            context = self.conversation_manager.sessions.get(session_id)
            if not context:
                self.logger.warning(f"Session {session_id} not found in memory")
                return False
            
            # 세션 데이터 변환
            session_data = self._convert_context_to_session_data(context)
            
            # 사용자 ID 추가
            user_id = self.session_users.get(session_id)
            if user_id:
                session_data["metadata"]["user_id"] = user_id
            
            # DB에 저장
            success = self.conversation_store.save_session(session_data)
            
            if success:
                # 동기화 정보 업데이트
                sync_info = self.sync_info.get(session_id)
                if sync_info:
                    sync_info.last_sync_time = datetime.now()
                    sync_info.sync_count += 1
                    sync_info.needs_sync = False
                
                self.logger.info(f"Session {session_id} synced to database")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error syncing session to database: {e}")
            return False
    
    def load_from_database(self, session_id: str) -> Optional[ConversationContext]:
        """
        DB에서 세션 로드
        
        Args:
            session_id: 세션 ID
            
        Returns:
            ConversationContext: 대화 맥락 (없으면 None)
        """
        try:
            session_data = self.conversation_store.load_session(session_id)
            if not session_data:
                return None
            
            # 세션 데이터를 ConversationContext로 변환
            context = self._convert_session_data_to_context(session_data)
            
            self.logger.info(f"Session {session_id} loaded from database")
            return context
            
        except Exception as e:
            self.logger.error(f"Error loading session from database: {e}")
            return None
    
    def get_relevant_context(self, 
                           session_id: str, 
                           current_query: str,
                           max_turns: int = 3) -> Optional[Dict[str, Any]]:
        """
        관련 대화 맥락 조회
        
        Args:
            session_id: 세션 ID
            current_query: 현재 질문
            max_turns: 최대 턴 수
            
        Returns:
            Dict[str, Any]: 관련 맥락 정보
        """
        try:
            # 메모리에서 먼저 확인
            context = self.conversation_manager.sessions.get(session_id)
            if not context:
                # DB에서 로드 시도
                context = self.load_from_database(session_id)
                if not context:
                    return None
            
            return self.conversation_manager.get_relevant_context(session_id, current_query, max_turns)
            
        except Exception as e:
            self.logger.error(f"Error getting relevant context: {e}")
            return None
    
    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        사용자별 세션 목록 조회
        
        Args:
            user_id: 사용자 ID
            limit: 최대 개수
            
        Returns:
            List[Dict[str, Any]]: 세션 목록
        """
        try:
            return self.conversation_store.get_user_sessions(user_id, limit)
        except Exception as e:
            self.logger.error(f"Error getting user sessions: {e}")
            return []
    
    def search_sessions(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        세션 검색
        
        Args:
            query: 검색 쿼리
            filters: 필터 조건
            
        Returns:
            List[Dict[str, Any]]: 검색 결과
        """
        try:
            return self.conversation_store.search_sessions(query, filters)
        except Exception as e:
            self.logger.error(f"Error searching sessions: {e}")
            return []
    
    def backup_session(self, session_id: str, backup_path: str) -> bool:
        """
        세션 백업
        
        Args:
            session_id: 세션 ID
            backup_path: 백업 경로
            
        Returns:
            bool: 백업 성공 여부
        """
        try:
            # 먼저 메모리에서 DB로 동기화
            self.sync_to_database(session_id)
            
            # 백업 실행
            return self.conversation_store.backup_session(session_id, backup_path)
            
        except Exception as e:
            self.logger.error(f"Error backing up session: {e}")
            return False
    
    def restore_session(self, backup_path: str) -> Optional[str]:
        """
        세션 복원
        
        Args:
            backup_path: 백업 파일 경로
            
        Returns:
            Optional[str]: 복원된 세션 ID
        """
        try:
            restored_session_id = self.conversation_store.restore_session(backup_path)
            
            if restored_session_id:
                # 복원된 세션을 메모리에 로드
                context = self.load_from_database(restored_session_id)
                if context:
                    self.conversation_manager.sessions[restored_session_id] = context
                    self._update_sync_info(restored_session_id)
            
            return restored_session_id
            
        except Exception as e:
            self.logger.error(f"Error restoring session: {e}")
            return None
    
    def cleanup_old_sessions(self, days: int = 30) -> int:
        """
        오래된 세션 정리
        
        Args:
            days: 보관 기간 (일)
            
        Returns:
            int: 정리된 세션 수
        """
        try:
            # DB에서 정리
            cleaned_count = self.conversation_store.cleanup_old_sessions(days)
            
            # 메모리에서도 정리
            current_time = datetime.now()
            sessions_to_remove = []
            
            for session_id, context in self.conversation_manager.sessions.items():
                age_hours = (current_time - context.last_updated).total_seconds() / 3600
                if age_hours > days * 24:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.conversation_manager.sessions[session_id]
                if session_id in self.sync_info:
                    del self.sync_info[session_id]
            
            self.logger.info(f"Cleaned up {len(sessions_to_remove)} sessions from memory")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old sessions: {e}")
            return 0
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        세션 통계 조회
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        try:
            # 메모리 통계
            memory_stats = self.conversation_manager.get_session_stats()
            
            # DB 통계
            db_stats = self.conversation_store.get_statistics()
            
            # 동기화 통계
            sync_stats = {
                "total_synced_sessions": len(self.sync_info),
                "sessions_needing_sync": sum(1 for info in self.sync_info.values() if info.needs_sync),
                "avg_sync_count": sum(info.sync_count for info in self.sync_info.values()) / len(self.sync_info) if self.sync_info else 0
            }
            
            return {
                "memory_stats": memory_stats,
                "database_stats": db_stats,
                "sync_stats": sync_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session stats: {e}")
            return {}
    
    def force_sync_all(self) -> int:
        """
        모든 세션 강제 동기화
        
        Returns:
            int: 동기화된 세션 수
        """
        try:
            synced_count = 0
            
            for session_id in self.conversation_manager.sessions.keys():
                if self.sync_to_database(session_id):
                    synced_count += 1
            
            self.logger.info(f"Force synced {synced_count} sessions")
            return synced_count
            
        except Exception as e:
            self.logger.error(f"Error force syncing all sessions: {e}")
            return 0
    
    def _update_sync_info(self, session_id: str):
        """동기화 정보 업데이트"""
        try:
            if session_id not in self.sync_info:
                self.sync_info[session_id] = SessionSyncInfo(
                    session_id=session_id,
                    last_sync_time=datetime.now(),
                    sync_count=0
                )
            
            sync_info = self.sync_info[session_id]
            sync_info.needs_sync = True
            
        except Exception as e:
            self.logger.error(f"Error updating sync info: {e}")
    
    def _should_sync(self, session_id: str) -> bool:
        """동기화 필요 여부 확인"""
        try:
            sync_info = self.sync_info.get(session_id)
            if not sync_info:
                return True
            
            # 턴 수 기반 동기화
            context = self.conversation_manager.sessions.get(session_id)
            if context and len(context.turns) % self.sync_interval == 0:
                return True
            
            # 시간 기반 동기화 (5분마다)
            time_since_sync = (datetime.now() - sync_info.last_sync_time).total_seconds()
            if time_since_sync > 300:  # 5분
                return True
            
            return sync_info.needs_sync
            
        except Exception as e:
            self.logger.error(f"Error checking sync requirement: {e}")
            return True
    
    def _convert_context_to_session_data(self, context: ConversationContext) -> Dict[str, Any]:
        """ConversationContext를 세션 데이터로 변환"""
        try:
            return {
                "session_id": context.session_id,
                "created_at": context.created_at.isoformat(),
                "last_updated": context.last_updated.isoformat(),
                "topic_stack": list(context.topic_stack),
                "metadata": {},
                "turns": [
                    {
                        "user_query": turn.user_query,
                        "bot_response": turn.bot_response,
                        "timestamp": turn.timestamp.isoformat(),
                        "question_type": turn.question_type,
                        "entities": turn.entities or {}
                    }
                    for turn in context.turns
                ],
                "entities": {
                    entity_type: list(entity_set)
                    for entity_type, entity_set in context.entities.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error converting context to session data: {e}")
            return {}
    
    def _convert_session_data_to_context(self, session_data: Dict[str, Any]) -> ConversationContext:
        """세션 데이터를 ConversationContext로 변환"""
        try:
            turns = []
            for turn_data in session_data.get("turns", []):
                turn = ConversationTurn(
                    user_query=turn_data["user_query"],
                    bot_response=turn_data["bot_response"],
                    timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                    question_type=turn_data.get("question_type"),
                    entities=turn_data.get("entities", {})
                )
                turns.append(turn)
            
            entities = {}
            for entity_type, entity_list in session_data.get("entities", {}).items():
                entities[entity_type] = set(entity_list)
            
            return ConversationContext(
                session_id=session_data["session_id"],
                turns=turns,
                entities=entities,
                topic_stack=session_data.get("topic_stack", []),
                created_at=datetime.fromisoformat(session_data["created_at"]),
                last_updated=datetime.fromisoformat(session_data["last_updated"])
            )
            
        except Exception as e:
            self.logger.error(f"Error converting session data to context: {e}")
            return ConversationContext(
                session_id=session_data.get("session_id", "unknown"),
                turns=[],
                entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
                topic_stack=[],
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
    
    def _save_session_to_db(self, context: ConversationContext, user_id: str):
        """세션을 DB에 저장"""
        try:
            session_data = self._convert_context_to_session_data(context)
            session_data["metadata"]["user_id"] = user_id
            self.conversation_store.save_session(session_data)
        except Exception as e:
            self.logger.error(f"Error saving session to DB: {e}")


# 테스트 함수
def test_integrated_session_manager():
    """통합 세션 관리자 테스트"""
    manager = IntegratedSessionManager("test_integrated_conversations.db")
    
    session_id = "test_integrated_session_001"
    user_id = "test_user_001"
    
    print("=== 통합 세션 관리자 테스트 ===")
    
    # 테스트 턴들 추가
    test_turns = [
        ("손해배상 청구 방법을 알려주세요", "민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...", "legal_advice"),
        ("계약 해지 절차는 어떻게 되나요?", "계약 해지 절차는 다음과 같습니다...", "procedure_guide"),
        ("위의 손해배상 사건에서 과실비율은 어떻게 정해지나요?", "과실비율은 교통사고의 경우...", "legal_advice"),
    ]
    
    for i, (query, response, q_type) in enumerate(test_turns, 1):
        print(f"\n턴 {i} 추가:")
        print(f"질문: {query}")
        
        context = manager.add_turn(session_id, query, response, q_type, user_id)
        
        print(f"엔티티 수: {sum(len(entities) for entities in context.entities.values())}")
        print(f"주제 스택: {context.topic_stack}")
    
    # 관련 맥락 조회
    print(f"\n=== 관련 맥락 조회 ===")
    current_query = "손해배상 관련 판례를 더 찾아주세요"
    relevant_context = manager.get_relevant_context(session_id, current_query, max_turns=2)
    
    if relevant_context:
        print(f"세션 ID: {relevant_context['session_id']}")
        print(f"관련 턴 수: {len(relevant_context['relevant_turns'])}")
        print(f"엔티티: {relevant_context['entities']}")
        print(f"현재 주제: {relevant_context['current_topics']}")
    
    # 사용자 세션 목록 조회
    print(f"\n=== 사용자 세션 목록 ===")
    user_sessions = manager.get_user_sessions(user_id)
    print(f"사용자 {user_id}의 세션 수: {len(user_sessions)}")
    for session in user_sessions:
        print(f"- {session['session_id']}: {session['turn_count']}턴")
    
    # 통계 조회
    stats = manager.get_session_stats()
    print(f"\n=== 통계 ===")
    print(f"메모리 통계: {stats.get('memory_stats', {})}")
    print(f"DB 통계: {stats.get('database_stats', {})}")
    print(f"동기화 통계: {stats.get('sync_stats', {})}")
    
    # 세션 백업
    print(f"\n=== 세션 백업 ===")
    backup_success = manager.backup_session(session_id, "backup_test")
    print(f"백업 성공: {backup_success}")
    
    # 테스트 데이터 정리
    manager.conversation_store.delete_session(session_id)
    print("\n테스트 데이터 정리 완료")


if __name__ == "__main__":
    test_integrated_session_manager()

