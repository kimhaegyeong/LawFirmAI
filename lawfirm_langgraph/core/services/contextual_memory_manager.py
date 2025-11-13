# -*- coding: utf-8 -*-
"""
맥락적 메모리 관리자
중요한 사실과 정보를 장기 기억으로 저장하고 관리합니다.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

from ..data.conversation_store import ConversationStore
from .conversation_manager import ConversationContext, ConversationTurn

logger = logging.getLogger(__name__)


class MemoryType:
    """메모리 유형"""
    FACT = "fact"
    PREFERENCE = "preference"
    CASE_DETAIL = "case_detail"
    LEGAL_KNOWLEDGE = "legal_knowledge"
    USER_CONTEXT = "user_context"
    RELATIONSHIP = "relationship"


@dataclass
class MemoryItem:
    """메모리 아이템"""
    memory_id: str
    session_id: str
    user_id: str
    memory_type: str
    content: str
    importance_score: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    related_entities: List[str]
    tags: List[str]
    confidence: float


@dataclass
class MemorySearchResult:
    """메모리 검색 결과"""
    memory: MemoryItem
    relevance_score: float
    match_reason: str


class ContextualMemoryManager:
    """맥락적 메모리 관리자"""
    
    def __init__(self, conversation_store: Optional[ConversationStore] = None):
        """
        맥락적 메모리 관리자 초기화
        
        Args:
            conversation_store: 대화 저장소 (None이면 새로 생성)
        """
        self.logger = logging.getLogger(__name__)
        self.conversation_store = conversation_store or ConversationStore()
        
        # 메모리 중요도 계산을 위한 키워드 패턴
        self.importance_keywords = {
            "high": [
                "중요", "핵심", "필수", "반드시", "꼭", "절대", "결정적",
                "법적", "법률", "법령", "조문", "판례", "법원", "법정",
                "계약", "손해배상", "불법행위", "소유권", "상속", "이혼"
            ],
            "medium": [
                "관련", "연관", "비슷한", "유사한", "참고", "참조",
                "절차", "방법", "과정", "단계", "순서", "조건", "요건"
            ],
            "low": [
                "일반", "보통", "평범한", "흔한", "자주", "가끔", "때때로"
            ]
        }
        
        # 메모리 유형별 추출 패턴
        self.memory_extraction_patterns = {
            MemoryType.FACT: [
                r"(\w+)는\s+(\w+)이다",
                r"(\w+)의\s+특징은\s+(\w+)",
                r"(\w+)에\s+따르면\s+(\w+)",
                r"(\w+)는\s+(\w+)를\s+의미한다",
                r"(\w+)에\s+대해\s+설명",
                r"(\w+)의\s+내용",
                r"(\w+)에\s+관한\s+정보",
                r"(\w+)에\s+대해\s+자세히",
                r"(\w+)에\s+대한\s+설명",
                r"(\w+)에\s+대한\s+내용"
            ],
            MemoryType.PREFERENCE: [
                r"(\w+)을\s+선호한다",
                r"(\w+)를\s+좋아한다",
                r"(\w+)을\s+원한다",
                r"(\w+)를\s+싫어한다",
                r"(\w+)을\s+피한다"
            ],
            MemoryType.CASE_DETAIL: [
                r"(\w+)사건",
                r"(\w+)판례",
                r"(\w+)사안",
                r"(\w+)문제",
                r"(\w+)분쟁",
                r"(\w+)에\s+대한\s+사건",
                r"(\w+)관련\s+판례"
            ],
            MemoryType.LEGAL_KNOWLEDGE: [
                r"(\w+)법\s+제(\d+)조",
                r"(\w+)법령",
                r"(\w+)규정",
                r"(\w+)조항",
                r"(\w+)법리",
                r"(\w+)법\s+내용",
                r"(\w+)에\s+대한\s+법률",
                r"(\w+)법\s+제(\d+)조에\s+대해",
                r"(\w+)법\s+제(\d+)조는"
            ],
            MemoryType.USER_CONTEXT: [
                r"나는\s+(\w+)",
                r"내\s+상황은\s+(\w+)",
                r"내\s+경우는\s+(\w+)",
                r"우리\s+(\w+)"
            ]
        }
        
        # 메모리 통합을 위한 유사도 임계값
        self.similarity_threshold = 0.8
        self.importance_decay_rate = 0.1  # 시간에 따른 중요도 감소율
        
        self.logger.info("ContextualMemoryManager initialized")
    
    def store_important_facts(self, session_id: str, user_id: str, 
                             facts: Dict[str, Any]) -> bool:
        """
        중요한 사실들을 메모리에 저장
        
        Args:
            session_id: 세션 ID
            user_id: 사용자 ID
            facts: 저장할 사실들
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # facts가 None이거나 빈 딕셔너리인 경우 처리
            if not facts or not isinstance(facts, dict):
                self.logger.warning(f"No facts to store for session {session_id}")
                return False
            
            stored_count = 0
            
            for fact_type, fact_content in facts.items():
                if isinstance(fact_content, list):
                    for item in fact_content:
                        if self._is_important_fact(item):
                            importance = self._calculate_importance_score(item)
                            
                            success = self._store_single_memory(
                                session_id, user_id, fact_type, 
                                item, importance
                            )
                            if success:
                                stored_count += 1
                else:
                    if self._is_important_fact(fact_content):
                        importance = self._calculate_importance_score(fact_content)
                        
                        success = self._store_single_memory(
                            session_id, user_id, fact_type, 
                            fact_content, importance
                        )
                        if success:
                            stored_count += 1
            
            self.logger.info(f"Stored {stored_count} important facts for session {session_id}")
            return stored_count > 0
            
        except Exception as e:
            self.logger.error(f"Error storing important facts: {e}")
            return False
    
    def retrieve_relevant_memory(self, session_id: str, query: str, 
                                user_id: Optional[str] = None) -> List[MemorySearchResult]:
        """
        관련 메모리 검색
        
        Args:
            session_id: 세션 ID
            query: 검색 쿼리
            user_id: 사용자 ID (선택사항)
            
        Returns:
            List[MemorySearchResult]: 관련 메모리 검색 결과
        """
        try:
            # 메모리 검색
            memories = self._search_memories(session_id, user_id)
            
            # 관련성 점수 계산
            relevant_memories = []
            for memory in memories:
                relevance_score = self._calculate_relevance_score(memory, query)
                # 디버깅: 관련성 점수 출력
                self.logger.debug(f"Memory: {memory.content[:30]}... Score: {relevance_score}")
                if relevance_score > 0.0:  # 모든 메모리 포함
                    match_reason = self._get_match_reason(memory, query)
                    relevant_memories.append(MemorySearchResult(
                        memory=memory,
                        relevance_score=relevance_score,
                        match_reason=match_reason
                    ))
            
            # 관련성 점수순으로 정렬
            relevant_memories.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # 접근 기록 업데이트
            for result in relevant_memories[:5]:  # 상위 5개만 업데이트
                self._update_memory_access(result.memory.memory_id)
            
            self.logger.info(f"Retrieved {len(relevant_memories)} relevant memories for query: {query[:50]}...")
            return relevant_memories
            
        except Exception as e:
            self.logger.error(f"Error retrieving relevant memory: {e}")
            return []
    
    def update_memory_importance(self, memory_id: str, importance: float) -> bool:
        """
        메모리 중요도 업데이트
        
        Args:
            memory_id: 메모리 ID
            importance: 새로운 중요도 점수
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                UPDATE contextual_memories 
                SET importance_score = ?, last_accessed = ?
                WHERE memory_id = ?
                """, (importance, datetime.now().isoformat(), memory_id))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    self.logger.info(f"Updated memory importance: {memory_id} -> {importance}")
                    return True
                else:
                    self.logger.warning(f"Memory not found: {memory_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error updating memory importance: {e}")
            return False
    
    def consolidate_memories(self, user_id: str) -> int:
        """
        사용자의 메모리 통합 (중복 제거 및 통합)
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            int: 통합된 메모리 수
        """
        try:
            # 사용자의 모든 메모리 조회
            memories = self._get_user_memories(user_id)
            
            # 유사한 메모리 그룹화
            memory_groups = self._group_similar_memories(memories)
            
            consolidated_count = 0
            
            for group in memory_groups:
                if len(group) > 1:
                    # 가장 중요한 메모리를 기준으로 통합
                    primary_memory = max(group, key=lambda m: m.importance_score)
                    
                    # 다른 메모리들의 정보를 통합
                    consolidated_content = self._consolidate_memory_content(group)
                    consolidated_entities = list(set(
                        entity for memory in group for entity in memory.related_entities
                    ))
                    consolidated_tags = list(set(
                        tag for memory in group for tag in memory.tags
                    ))
                    
                    # 통합된 메모리로 업데이트
                    success = self._update_memory_content(
                        primary_memory.memory_id,
                        consolidated_content,
                        consolidated_entities,
                        consolidated_tags
                    )
                    
                    if success:
                        # 중복 메모리들 삭제
                        for memory in group:
                            if memory.memory_id != primary_memory.memory_id:
                                self._delete_memory(memory.memory_id)
                                consolidated_count += 1
            
            self.logger.info(f"Consolidated {consolidated_count} memories for user {user_id}")
            return consolidated_count
            
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {e}")
            return 0
    
    def extract_facts_from_conversation(self, turn: ConversationTurn) -> List[Dict[str, Any]]:
        """
        대화 턴에서 사실 추출
        
        Args:
            turn: 대화 턴
            
        Returns:
            List[Dict[str, Any]]: 추출된 사실들
        """
        try:
            extracted_facts = []
            
            # 사용자 질문에서 사실 추출
            user_facts = self._extract_facts_from_text(turn.user_query)
            if user_facts:
                extracted_facts.extend(user_facts)
            
            # 봇 응답에서 사실 추출
            bot_facts = self._extract_facts_from_text(turn.bot_response)
            if bot_facts:
                extracted_facts.extend(bot_facts)
            
            # 엔티티 정보 활용
            if turn.entities:
                for entity_type, entities in turn.entities.items():
                    for entity in entities:
                        fact = {
                            "type": f"entity_{entity_type}",
                            "content": f"{entity_type}: {entity}",
                            "importance": 0.7,
                            "entities": [entity],
                            "tags": [entity_type]
                        }
                        extracted_facts.append(fact)
            
            self.logger.debug(f"Extracted {len(extracted_facts)} facts from conversation turn")
            return extracted_facts
            
        except Exception as e:
            self.logger.error(f"Error extracting facts from conversation: {e}")
            return []
    
    def get_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        사용자 메모리 통계 조회
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            Dict[str, Any]: 메모리 통계
        """
        try:
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                # 전체 메모리 수
                cursor.execute("""
                SELECT COUNT(*) FROM contextual_memories WHERE user_id = ?
                """, (user_id,))
                total_memories = cursor.fetchone()[0]
                
                # 메모리 유형별 통계
                cursor.execute("""
                SELECT memory_type, COUNT(*), AVG(importance_score)
                FROM contextual_memories 
                WHERE user_id = ?
                GROUP BY memory_type
                """, (user_id,))
                type_stats = {}
                for row in cursor.fetchall():
                    type_stats[row[0]] = {
                        "count": row[1],
                        "avg_importance": row[2]
                    }
                
                # 최근 접근 메모리
                cursor.execute("""
                SELECT memory_id, COALESCE(content, memory_content) as content, last_accessed, access_count
                FROM contextual_memories 
                WHERE user_id = ?
                ORDER BY last_accessed DESC
                LIMIT 5
                """, (user_id,))
                recent_memories = []
                for row in cursor.fetchall():
                    recent_memories.append({
                        "memory_id": row[0],
                        "content": row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                        "last_accessed": row[2],
                        "access_count": row[3]
                    })
                
                # 중요도 분포
                cursor.execute("""
                SELECT 
                    CASE 
                        WHEN importance_score >= 0.8 THEN 'high'
                        WHEN importance_score >= 0.5 THEN 'medium'
                        ELSE 'low'
                    END as importance_level,
                    COUNT(*)
                FROM contextual_memories 
                WHERE user_id = ?
                GROUP BY importance_level
                """, (user_id,))
                importance_distribution = {}
                for row in cursor.fetchall():
                    importance_distribution[row[0]] = row[1]
                
                return {
                    "user_id": user_id,
                    "total_memories": total_memories,
                    "type_statistics": type_stats,
                    "recent_memories": recent_memories,
                    "importance_distribution": importance_distribution,
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting memory statistics: {e}")
            return {}
    
    def cleanup_old_memories(self, days: int = 30) -> int:
        """
        오래된 메모리 정리
        
        Args:
            days: 보관 기간 (일)
            
        Returns:
            int: 정리된 메모리 수
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                # 중요도가 낮고 오래된 메모리 삭제
                cursor.execute("""
                DELETE FROM contextual_memories 
                WHERE created_at < ? AND importance_score < 0.3 AND access_count < 3
                """, (cutoff_date.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} old memories")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old memories: {e}")
            return 0
    
    def _is_important_fact(self, text: str) -> bool:
        """중요한 사실인지 판단"""
        text_lower = text.lower()
        
        # 중요 키워드 포함 여부
        for importance_level, keywords in self.importance_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return True
        
        # 길이 기준 (너무 짧거나 너무 긴 것은 제외)
        if len(text.strip()) < 5 or len(text.strip()) > 1000:  # 더 관대한 기준
            return False
        
        # 법률 관련 키워드 포함 여부
        legal_keywords = ["법", "조문", "조항", "판례", "법원", "법정", "법률", "법령", "민법", "제750조"]
        if any(keyword in text_lower for keyword in legal_keywords):
            return True
        
        # 질문 형태도 사실로 간주
        question_keywords = ["에 대해", "에 대한", "에 관해", "에 관한", "자세히", "설명"]
        if any(keyword in text_lower for keyword in question_keywords):
            return True
        
        return True  # 기본적으로 모든 텍스트를 사실로 간주
    
    def _calculate_importance_score(self, text: str) -> float:
        """중요도 점수 계산"""
        text_lower = text.lower()
        score = 0.5  # 기본 점수
        
        # 중요 키워드에 따른 점수 조정
        for keyword in self.importance_keywords["high"]:
            if keyword in text_lower:
                score += 0.2
        
        for keyword in self.importance_keywords["medium"]:
            if keyword in text_lower:
                score += 0.1
        
        # 법률 관련 키워드 추가 점수
        legal_keywords = ["법", "조문", "조항", "판례", "법원", "법정"]
        for keyword in legal_keywords:
            if keyword in text_lower:
                score += 0.1
        
        # 길이에 따른 조정
        if len(text) > 100:
            score += 0.1
        
        return min(1.0, score)
    
    def _generate_memory_id(self, session_id: str, user_id: str, content: str) -> str:
        """메모리 ID 생성"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        return f"mem_{user_id}_{session_id}_{content_hash}"
    
    def _store_single_memory(self, session_id: str, user_id: str,
                           memory_type: str, content: str, importance: float) -> bool:
        """단일 메모리 저장"""
        try:
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                # 중복 확인 (content와 session_id로)
                cursor.execute("""
                SELECT memory_id FROM contextual_memories 
                WHERE content = ? AND session_id = ?
                """, (content, session_id))
                
                existing = cursor.fetchone()
                if existing:
                    # 기존 메모리 업데이트
                    cursor.execute("""
                    UPDATE contextual_memories 
                    SET importance_score = ?, last_accessed = ?, access_count = access_count + 1
                    WHERE memory_id = ?
                    """, (importance, datetime.now().isoformat(), existing[0]))
                else:
                    # 새 메모리 저장
                    entities = self._extract_entities_from_text(content)
                    tags = self._extract_tags_from_text(content)
                    
                    cursor.execute("""
                    INSERT INTO contextual_memories 
                    (session_id, user_id, memory_type, memory_content, content,
                     importance_score, created_at, last_accessed, access_count, 
                     related_entities, tags, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session_id, user_id, memory_type, content, content,
                        importance, datetime.now().isoformat(), datetime.now().isoformat(), 0,
                        json.dumps(entities), json.dumps(tags), 0.8
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing single memory: {e}")
            return False
    
    def _search_memories(self, session_id: str, user_id: Optional[str] = None) -> List[MemoryItem]:
        """메모리 검색"""
        try:
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                if user_id:
                    cursor.execute("""
                    SELECT * FROM contextual_memories 
                    WHERE user_id = ? AND (session_id = ? OR session_id IS NULL)
                    ORDER BY importance_score DESC, last_accessed DESC
                    """, (user_id, session_id))
                else:
                    cursor.execute("""
                    SELECT * FROM contextual_memories 
                    WHERE session_id = ?
                    ORDER BY importance_score DESC, last_accessed DESC
                    """, (session_id,))
                
                memories = []
                for row in cursor.fetchall():
                    memory = MemoryItem(
                        memory_id=row["memory_id"],
                        session_id=row["session_id"],
                        user_id=row["user_id"],
                        memory_type=row["memory_type"],
                        content=row["memory_content"],
                        importance_score=row["importance_score"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        last_accessed=datetime.fromisoformat(row["last_accessed"]),
                        access_count=row["access_count"],
                        related_entities=json.loads(row["related_entities"]) if row["related_entities"] else [],
                        tags=json.loads(row["tags"]) if row["tags"] else [],
                        confidence=row["confidence"] if "confidence" in row.keys() else 0.8
                    )
                    memories.append(memory)
                
                return memories
                
        except Exception as e:
            self.logger.error(f"Error searching memories: {e}")
            return []
    
    def _calculate_relevance_score(self, memory: MemoryItem, query: str) -> float:
        """관련성 점수 계산"""
        query_lower = query.lower()
        content_lower = memory.content.lower()
        
        score = 0.0
        
        # 직접 키워드 매칭
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        common_words = query_words.intersection(content_words)
        
        if common_words:
            score += len(common_words) / len(query_words) * 0.4
        
        # 부분 문자열 매칭 (더 관대한 기준)
        for word in query_words:
            if word in content_lower:
                score += 0.2
        
        # 엔티티 매칭
        for entity in memory.related_entities:
            if entity.lower() in query_lower:
                score += 0.3
        
        # 태그 매칭
        for tag in memory.tags:
            if tag.lower() in query_lower:
                score += 0.2
        
        # 기본 점수 (모든 메모리에 대해 최소 점수 부여)
        score += 0.1
        
        # 중요도 가중치
        score *= memory.importance_score
        
        # 접근 빈도 가중치
        if memory.access_count > 0:
            score *= (1 + memory.access_count * 0.1)
        
        return min(1.0, score)
    
    def _get_match_reason(self, memory: MemoryItem, query: str) -> str:
        """매칭 이유 설명"""
        query_lower = query.lower()
        content_lower = memory.content.lower()
        
        reasons = []
        
        # 키워드 매칭
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        common_words = query_words.intersection(content_words)
        
        if common_words:
            reasons.append(f"공통 키워드: {', '.join(list(common_words)[:3])}")
        
        # 엔티티 매칭
        matching_entities = [entity for entity in memory.related_entities 
                           if entity.lower() in query_lower]
        if matching_entities:
            reasons.append(f"관련 엔티티: {', '.join(matching_entities[:2])}")
        
        # 태그 매칭
        matching_tags = [tag for tag in memory.tags if tag.lower() in query_lower]
        if matching_tags:
            reasons.append(f"관련 태그: {', '.join(matching_tags[:2])}")
        
        return "; ".join(reasons) if reasons else "일반적 관련성"
    
    def _update_memory_access(self, memory_id: str):
        """메모리 접근 기록 업데이트"""
        try:
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                UPDATE contextual_memories 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE memory_id = ?
                """, (datetime.now().isoformat(), memory_id))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating memory access: {e}")
    
    def _extract_facts_from_text(self, text: str) -> List[Dict[str, Any]]:
        """텍스트에서 사실 추출"""
        facts = []
        
        for memory_type, patterns in self.memory_extraction_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        content = f"{match[0]} {match[1]}"
                    else:
                        content = match
                    
                    if self._is_important_fact(content):
                        fact = {
                            "type": memory_type,
                            "content": content,
                            "importance": self._calculate_importance_score(content),
                            "entities": self._extract_entities_from_text(content),
                            "tags": self._extract_tags_from_text(content)
                        }
                        facts.append(fact)
        
        return facts
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """텍스트에서 엔티티 추출"""
        entities = []
        
        # 법률 관련 엔티티 패턴
        legal_patterns = [
            r'(\w+)법\s+제(\d+)조',
            r'(\w+)법령',
            r'(\w+)규정',
            r'(\w+)조항',
            r'(\w+)판례',
            r'(\w+)사건'
        ]
        
        for pattern in legal_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    entities.extend([m for m in match if m])
                else:
                    entities.append(match)
        
        return list(set(entities))
    
    def _extract_tags_from_text(self, text: str) -> List[str]:
        """텍스트에서 태그 추출"""
        tags = []
        text_lower = text.lower()
        
        # 법률 분야 태그
        legal_fields = ["민법", "형법", "상법", "근로기준법", "부동산", "금융", "지적재산권", "세법"]
        for field in legal_fields:
            if field in text_lower:
                tags.append(field)
        
        # 일반 태그
        general_tags = ["계약", "손해배상", "불법행위", "소유권", "상속", "이혼", "절차", "법원"]
        for tag in general_tags:
            if tag in text_lower:
                tags.append(tag)
        
        return list(set(tags))
    
    def _get_user_memories(self, user_id: str) -> List[MemoryItem]:
        """사용자의 모든 메모리 조회"""
        try:
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT * FROM contextual_memories 
                WHERE user_id = ?
                ORDER BY created_at DESC
                """, (user_id,))
                
                memories = []
                for row in cursor.fetchall():
                    memory = MemoryItem(
                        memory_id=row["memory_id"],
                        session_id=row["session_id"],
                        user_id=row["user_id"],
                        memory_type=row["memory_type"],
                        content=row["memory_content"],
                        importance_score=row["importance_score"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        last_accessed=datetime.fromisoformat(row["last_accessed"]),
                        access_count=row["access_count"],
                        related_entities=json.loads(row["related_entities"]) if row["related_entities"] else [],
                        tags=json.loads(row["tags"]) if row["tags"] else [],
                        confidence=row["confidence"] if "confidence" in row.keys() else 0.8
                    )
                    memories.append(memory)
                
                return memories
                
        except Exception as e:
            self.logger.error(f"Error getting user memories: {e}")
            return []
    
    def _group_similar_memories(self, memories: List[MemoryItem]) -> List[List[MemoryItem]]:
        """유사한 메모리 그룹화"""
        groups = []
        used_memories = set()
        
        for i, memory1 in enumerate(memories):
            if memory1.memory_id in used_memories:
                continue
            
            group = [memory1]
            used_memories.add(memory1.memory_id)
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if memory2.memory_id in used_memories:
                    continue
                
                similarity = self._calculate_similarity(memory1, memory2)
                if similarity > self.similarity_threshold:
                    group.append(memory2)
                    used_memories.add(memory2.memory_id)
            
            groups.append(group)
        
        return groups
    
    def _calculate_similarity(self, memory1: MemoryItem, memory2: MemoryItem) -> float:
        """메모리 간 유사도 계산"""
        # 내용 유사도
        content1_words = set(memory1.content.lower().split())
        content2_words = set(memory2.content.lower().split())
        
        if not content1_words or not content2_words:
            return 0.0
        
        content_similarity = len(content1_words.intersection(content2_words)) / len(content1_words.union(content2_words))
        
        # 엔티티 유사도
        entity1_set = set(memory1.related_entities)
        entity2_set = set(memory2.related_entities)
        
        if entity1_set or entity2_set:
            entity_similarity = len(entity1_set.intersection(entity2_set)) / len(entity1_set.union(entity2_set))
        else:
            entity_similarity = 0.0
        
        # 태그 유사도
        tag1_set = set(memory1.tags)
        tag2_set = set(memory2.tags)
        
        if tag1_set or tag2_set:
            tag_similarity = len(tag1_set.intersection(tag2_set)) / len(tag1_set.union(tag2_set))
        else:
            tag_similarity = 0.0
        
        # 가중 평균
        return (content_similarity * 0.5 + entity_similarity * 0.3 + tag_similarity * 0.2)
    
    def _consolidate_memory_content(self, memories: List[MemoryItem]) -> str:
        """메모리 내용 통합"""
        if len(memories) == 1:
            return memories[0].content
        
        # 가장 중요한 메모리를 기준으로 하고, 다른 메모리의 추가 정보를 병합
        primary_memory = max(memories, key=lambda m: m.importance_score)
        consolidated = primary_memory.content
        
        # 다른 메모리들의 고유한 정보 추가
        for memory in memories:
            if memory.memory_id != primary_memory.memory_id:
                # 고유한 엔티티나 태그가 있으면 추가
                unique_entities = set(memory.related_entities) - set(primary_memory.related_entities)
                if unique_entities:
                    consolidated += f" (관련: {', '.join(list(unique_entities)[:3])})"
        
        return consolidated
    
    def _update_memory_content(self, memory_id: str, content: str, 
                              entities: List[str], tags: List[str]) -> bool:
        """메모리 내용 업데이트"""
        try:
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                UPDATE contextual_memories 
                SET memory_content = ?, related_entities = ?, tags = ?, last_accessed = ?
                WHERE memory_id = ?
                """, (
                    content, 
                    json.dumps(entities), 
                    json.dumps(tags), 
                    datetime.now().isoformat(),
                    memory_id
                ))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"Error updating memory content: {e}")
            return False
    
    def _delete_memory(self, memory_id: str) -> bool:
        """메모리 삭제"""
        try:
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                DELETE FROM contextual_memories WHERE memory_id = ?
                """, (memory_id,))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"Error deleting memory: {e}")
            return False


# 테스트 함수
def test_contextual_memory_manager():
    """맥락적 메모리 관리자 테스트"""
    manager = ContextualMemoryManager()
    
    print("=== 맥락적 메모리 관리자 테스트 ===")
    
    session_id = "test_session_memory"
    user_id = "test_user_memory"
    
    # 1. 중요한 사실 저장
    print("\n1. 중요한 사실 저장 테스트")
    facts = {
        "legal_knowledge": [
            "민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다",
            "계약 해제 시 위약금은 손해배상액을 초과할 수 없습니다"
        ],
        "user_context": [
            "사용자는 부동산 관련 문제를 자주 문의합니다",
            "사용자는 법률 초보자입니다"
        ],
        "preference": [
            "사용자는 상세한 설명을 선호합니다",
            "사용자는 판례 중심의 답변을 원합니다"
        ]
    }
    
    success = manager.store_important_facts(session_id, user_id, facts)
    print(f"사실 저장 결과: {success}")
    
    # 2. 관련 메모리 검색
    print("\n2. 관련 메모리 검색 테스트")
    query = "손해배상 관련 질문입니다"
    relevant_memories = manager.retrieve_relevant_memory(session_id, query, user_id)
    
    print(f"검색된 메모리 수: {len(relevant_memories)}")
    for i, result in enumerate(relevant_memories[:3]):
        print(f"  {i+1}. {result.memory.content[:50]}... (관련성: {result.relevance_score:.2f})")
        print(f"     매칭 이유: {result.match_reason}")
    
    # 3. 메모리 통합
    print("\n3. 메모리 통합 테스트")
    consolidated_count = manager.consolidate_memories(user_id)
    print(f"통합된 메모리 수: {consolidated_count}")
    
    # 4. 대화에서 사실 추출
    print("\n4. 대화에서 사실 추출 테스트")
    from core.services.conversation_manager import ConversationTurn
    
    turn = ConversationTurn(
        user_query="민법 제750조에 대해 자세히 설명해주세요",
        bot_response="민법 제750조는 불법행위로 인한 손해배상 책임을 규정하는 중요한 조문입니다...",
        timestamp=datetime.now(),
        question_type="law_inquiry"
    )
    
    extracted_facts = manager.extract_facts_from_conversation(turn)
    print(f"추출된 사실 수: {len(extracted_facts)}")
    for fact in extracted_facts[:3]:
        print(f"  - {fact['type']}: {fact['content'][:50]}... (중요도: {fact['importance']:.2f})")
    
    # 5. 메모리 통계
    print("\n5. 메모리 통계 테스트")
    stats = manager.get_memory_statistics(user_id)
    print(f"총 메모리 수: {stats.get('total_memories', 0)}")
    print(f"유형별 통계: {stats.get('type_statistics', {})}")
    print(f"중요도 분포: {stats.get('importance_distribution', {})}")
    
    # 6. 메모리 중요도 업데이트
    print("\n6. 메모리 중요도 업데이트 테스트")
    if relevant_memories:
        memory_id = relevant_memories[0].memory.memory_id
        success = manager.update_memory_importance(memory_id, 0.9)
        print(f"중요도 업데이트 결과: {success}")
    
    # 7. 오래된 메모리 정리
    print("\n7. 오래된 메모리 정리 테스트")
    cleaned_count = manager.cleanup_old_memories(days=1)  # 테스트를 위해 1일로 설정
    print(f"정리된 메모리 수: {cleaned_count}")
    
    print("\n테스트 완료")


if __name__ == "__main__":
    test_contextual_memory_manager()
