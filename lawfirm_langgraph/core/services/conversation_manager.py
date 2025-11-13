# -*- coding: utf-8 -*-
"""
대화 맥락 관리 시스템
이전 대화를 기억하고 연속된 질문에 대응하는 시스템
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """대화 턴"""
    user_query: str
    bot_response: str
    timestamp: datetime
    question_type: Optional[str] = None
    entities: Optional[Dict[str, List[str]]] = None


@dataclass
class ConversationContext:
    """대화 맥락"""
    session_id: str
    turns: List[ConversationTurn]
    entities: Dict[str, Set[str]]
    topic_stack: List[str]
    created_at: datetime
    last_updated: datetime
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = {
                "laws": set(),
                "articles": set(),
                "precedents": set(),
                "legal_terms": set()
            }
        if self.topic_stack is None:
            self.topic_stack = []


class ConversationManager:
    """대화 맥락 관리자"""
    
    def __init__(self, max_context_turns: int = 10, max_session_age_hours: int = 24):
        """대화 맥락 관리자 초기화"""
        self.logger = logging.getLogger(__name__)
        self.max_context_turns = max_context_turns
        self.max_session_age_hours = max_session_age_hours
        
        # 세션 저장소 (메모리 기반)
        self.sessions: Dict[str, ConversationContext] = {}
        
        # 법률 엔티티 패턴
        self.entity_patterns = {
            "laws": [
                r'([가-힣]+법)',  # 민법, 형법, 상법 등
                r'([가-힣]+법률)',  # 특별법률
                r'([가-힣]+보호법)',  # 주택임대차보호법 등
            ],
            "articles": [
                r'제(\d+)조',  # 제750조
                r'제(\d+)항',  # 제1항
                r'제(\d+)호',  # 제1호
                r'법률\s*제(\d+)호',  # 법률 제20883호
            ],
            "precedents": [
                r'(\d{4}[가나다라마바사아자차카타파하]\d+)',  # 2023다12345
                r'(대법원\s*\d{4}\.\d+\.\d+)',  # 대법원 2023.1.1
                r'(고등법원\s*\d{4}\.\d+\.\d+)',  # 고등법원 2023.1.1
                r'(지방법원\s*\d{4}\.\d+\.\d+)',  # 지방법원 2023.1.1
            ],
            "legal_terms": [
                r'(손해배상)', r'(계약)', r'(임대차)', r'(불법행위)',
                r'(소송)', r'(상속)', r'(이혼)', r'(교통사고)',
                r'(근로)', r'(부동산)', r'(금융)', r'(지적재산권)',
                r'(세금)', r'(환경)', r'(의료)'
            ]
        }
        
        # 주제 키워드
        self.topic_keywords = {
            "계약": ["계약", "계약서", "계약해지", "계약위반", "계약금", "위약금"],
            "손해배상": ["손해배상", "배상", "보상", "위자료", "불법행위"],
            "임대차": ["임대차", "전세", "월세", "보증금", "임대료"],
            "소송": ["소송", "소장", "답변서", "증거", "판결", "항소", "상고"],
            "상속": ["상속", "상속재산", "상속분", "유언", "유류분"],
            "이혼": ["이혼", "위자료", "재산분할", "양육비", "면접교섭권"],
            "교통사고": ["교통사고", "자동차사고", "과실비율", "자동차보험"],
            "근로": ["근로", "임금", "퇴직금", "해고", "근로기준법"],
            "부동산": ["부동산", "등기", "소유권이전", "매매계약"],
            "금융": ["대출", "이자", "연체", "담보", "보증"]
        }
    
    def add_turn(self, 
                 session_id: str, 
                 user_query: str, 
                 bot_response: str,
                 question_type: Optional[str] = None) -> ConversationContext:
        """
        대화 턴 추가
        
        Args:
            session_id: 세션 ID
            user_query: 사용자 질문
            bot_response: 봇 응답
            question_type: 질문 유형
            
        Returns:
            ConversationContext: 업데이트된 대화 맥락
        """
        try:
            self.logger.info(f"Adding turn to session {session_id}")
            
            # 세션 가져오기 또는 생성
            context = self.sessions.get(session_id)
            if not context:
                context = ConversationContext(
                    session_id=session_id,
                    turns=[],
                    entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
                    topic_stack=[],
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
                self.sessions[session_id] = context
            
            # 법률 엔티티 추출
            entities = self._extract_legal_entities(user_query, bot_response)
            
            # 대화 턴 생성
            turn = ConversationTurn(
                user_query=user_query,
                bot_response=bot_response,
                timestamp=datetime.now(),
                question_type=question_type,
                entities=entities
            )
            
            # 턴 추가
            context.turns.append(turn)
            
            # 엔티티 업데이트
            self._update_entities(context, entities)
            
            # 주제 스택 업데이트
            self._update_topic_stack(context, user_query, bot_response)
            
            # 컨텍스트 크기 제한
            self._limit_context_size(context)
            
            # 마지막 업데이트 시간 갱신
            context.last_updated = datetime.now()
            
            self.logger.info(f"Turn added: {len(context.turns)} turns, "
                           f"{sum(len(entities) for entities in context.entities.values())} entities")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error adding turn: {e}")
            return self.sessions.get(session_id, ConversationContext(
                session_id=session_id,
                turns=[],
                entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
                topic_stack=[],
                created_at=datetime.now(),
                last_updated=datetime.now()
            ))
    
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
            context = self.sessions.get(session_id)
            if not context or not context.turns:
                return None
            
            # 현재 질문과 관련된 턴들 찾기
            relevant_turns = self._find_relevant_turns(context, current_query, max_turns)
            
            # 엔티티 정보 정리
            entities_info = {
                "laws": list(context.entities["laws"]),
                "articles": list(context.entities["articles"]),
                "precedents": list(context.entities["precedents"]),
                "legal_terms": list(context.entities["legal_terms"])
            }
            
            # 주제 정보
            current_topics = self._identify_topics(current_query)
            
            return {
                "session_id": session_id,
                "relevant_turns": relevant_turns,
                "entities": entities_info,
                "current_topics": current_topics,
                "topic_stack": context.topic_stack,
                "context_age": (datetime.now() - context.created_at).total_seconds() / 3600,  # 시간
                "turn_count": len(context.turns)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting relevant context: {e}")
            return None
    
    def _extract_legal_entities(self, user_query: str, bot_response: str) -> Dict[str, List[str]]:
        """법률 엔티티 추출"""
        try:
            text = f"{user_query} {bot_response}"
            entities = {"laws": [], "articles": [], "precedents": [], "legal_terms": []}
            
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    entities[entity_type].extend(matches)
            
            # 중복 제거
            for entity_type in entities:
                entities[entity_type] = list(set(entities[entity_type]))
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error extracting legal entities: {e}")
            return {"laws": [], "articles": [], "precedents": [], "legal_terms": []}
    
    def _update_entities(self, context: ConversationContext, entities: Dict[str, List[str]]):
        """엔티티 정보 업데이트"""
        try:
            for entity_type, entity_list in entities.items():
                if entity_type in context.entities:
                    context.entities[entity_type].update(entity_list)
                    
        except Exception as e:
            self.logger.error(f"Error updating entities: {e}")
    
    def _update_topic_stack(self, context: ConversationContext, user_query: str, bot_response: str):
        """주제 스택 업데이트"""
        try:
            # 현재 질문에서 주제 식별
            current_topics = self._identify_topics(user_query)
            
            # 새로운 주제가 있으면 스택에 추가
            for topic in current_topics:
                if topic not in context.topic_stack:
                    context.topic_stack.append(topic)
            
            # 스택 크기 제한 (최근 5개 주제만 유지)
            if len(context.topic_stack) > 5:
                context.topic_stack = context.topic_stack[-5:]
                
        except Exception as e:
            self.logger.error(f"Error updating topic stack: {e}")
    
    def _identify_topics(self, text: str) -> List[str]:
        """텍스트에서 주제 식별"""
        try:
            identified_topics = []
            text_lower = text.lower()
            
            for topic, keywords in self.topic_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        identified_topics.append(topic)
                        break
            
            return list(set(identified_topics))
            
        except Exception as e:
            self.logger.error(f"Error identifying topics: {e}")
            return []
    
    def _find_relevant_turns(self, 
                           context: ConversationContext, 
                           current_query: str,
                           max_turns: int) -> List[Dict[str, Any]]:
        """관련 턴들 찾기"""
        try:
            if not context.turns:
                return []
            
            # 최근 턴들부터 검사
            recent_turns = context.turns[-max_turns*2:]  # 여유분 확보
            
            relevant_turns = []
            current_topics = self._identify_topics(current_query)
            
            for turn in reversed(recent_turns):
                # 주제 기반 관련성 검사
                turn_topics = self._identify_topics(turn.user_query)
                if any(topic in current_topics for topic in turn_topics):
                    relevant_turns.append({
                        "user_query": turn.user_query,
                        "bot_response": turn.bot_response[:200] + "..." if len(turn.bot_response) > 200 else turn.bot_response,
                        "timestamp": turn.timestamp.isoformat(),
                        "question_type": turn.question_type,
                        "topics": turn_topics
                    })
                    
                    if len(relevant_turns) >= max_turns:
                        break
            
            # 시간순으로 정렬
            relevant_turns.reverse()
            
            return relevant_turns
            
        except Exception as e:
            self.logger.error(f"Error finding relevant turns: {e}")
            return []
    
    def _limit_context_size(self, context: ConversationContext):
        """컨텍스트 크기 제한"""
        try:
            if len(context.turns) > self.max_context_turns:
                # 오래된 턴들 제거
                context.turns = context.turns[-self.max_context_turns:]
                
        except Exception as e:
            self.logger.error(f"Error limiting context size: {e}")
    
    def cleanup_old_sessions(self):
        """오래된 세션 정리"""
        try:
            current_time = datetime.now()
            sessions_to_remove = []
            
            for session_id, context in self.sessions.items():
                age_hours = (current_time - context.last_updated).total_seconds() / 3600
                if age_hours > self.max_session_age_hours:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.sessions[session_id]
                self.logger.info(f"Cleaned up old session: {session_id}")
            
            self.logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old sessions: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """세션 통계 조회"""
        try:
            total_sessions = len(self.sessions)
            total_turns = sum(len(context.turns) for context in self.sessions.values())
            total_entities = sum(
                sum(len(entities) for entities in context.entities.values())
                for context in self.sessions.values()
            )
            
            return {
                "total_sessions": total_sessions,
                "total_turns": total_turns,
                "total_entities": total_entities,
                "avg_turns_per_session": total_turns / total_sessions if total_sessions > 0 else 0,
                "avg_entities_per_session": total_entities / total_sessions if total_sessions > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session stats: {e}")
            return {}
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 내보내기"""
        try:
            context = self.sessions.get(session_id)
            if not context:
                return None
            
            return {
                "session_id": context.session_id,
                "created_at": context.created_at.isoformat(),
                "last_updated": context.last_updated.isoformat(),
                "turns": [
                    {
                        "user_query": turn.user_query,
                        "bot_response": turn.bot_response,
                        "timestamp": turn.timestamp.isoformat(),
                        "question_type": turn.question_type,
                        "entities": turn.entities
                    }
                    for turn in context.turns
                ],
                "entities": {
                    entity_type: list(entity_set)
                    for entity_type, entity_set in context.entities.items()
                },
                "topic_stack": context.topic_stack
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting session: {e}")
            return None


# 테스트 함수
def test_conversation_manager():
    """대화 맥락 관리자 테스트"""
    manager = ConversationManager()
    
    session_id = "test_session_001"
    
    print("=== 대화 맥락 관리자 테스트 ===")
    
    # 대화 턴들 추가
    test_turns = [
        ("손해배상 청구 방법을 알려주세요", "민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...", "legal_advice"),
        ("계약 해지 절차는 어떻게 되나요?", "계약 해지 절차는 다음과 같습니다...", "procedure_guide"),
        ("위의 손해배상 사건에서 과실비율은 어떻게 정해지나요?", "과실비율은 교통사고의 경우...", "legal_advice"),
        ("그 판례는 어떤 사건이었나요?", "해당 판례는 2023다12345 손해배상 사건으로...", "precedent_search")
    ]
    
    for i, (query, response, q_type) in enumerate(test_turns, 1):
        print(f"\n턴 {i} 추가:")
        print(f"질문: {query}")
        
        context = manager.add_turn(session_id, query, response, q_type)
        
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
        
        print(f"\n관련 턴들:")
        for i, turn in enumerate(relevant_context['relevant_turns'], 1):
            print(f"{i}. {turn['user_query']} -> {turn['topics']}")
    
    # 세션 통계
    stats = manager.get_session_stats()
    print(f"\n=== 세션 통계 ===")
    print(f"총 세션 수: {stats.get('total_sessions', 0)}")
    print(f"총 턴 수: {stats.get('total_turns', 0)}")
    print(f"총 엔티티 수: {stats.get('total_entities', 0)}")


if __name__ == "__main__":
    test_conversation_manager()
