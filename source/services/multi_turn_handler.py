# -*- coding: utf-8 -*-
"""
다중 턴 질문 처리기
대명사 해결 및 불완전한 질문을 완성하는 기능
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .conversation_manager import ConversationContext, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class QuestionResolution:
    """질문 해결 결과"""
    original_query: str
    resolved_query: str
    is_multi_turn: bool
    resolved_entities: List[str]
    confidence: float
    reasoning: str


class MultiTurnQuestionHandler:
    """다중 턴 질문 처리기"""
    
    def __init__(self):
        """다중 턴 질문 처리기 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 대명사 패턴 정의
        self.pronoun_patterns = {
            "그것": r"그것|그거|그것을|그것이|그것의",
            "이것": r"이것|이거|이것을|이것이|이것의",
            "위의": r"위의|위에\s+말한|위에서\s+언급한",
            "해당": r"해당|해당\s+사안|해당\s+사건|해당\s+문제",
            "앞서": r"앞서|앞에서|이전에|전에",
            "그": r"그\s+[가-힣]+|그\s+사안|그\s+사건|그\s+문제",
            "이": r"이\s+[가-힣]+|이\s+사안|이\s+사건|이\s+문제",
            "저": r"저\s+[가-힣]+|저\s+사안|저\s+사건|저\s+문제"
        }
        
        # 질문 유형별 패턴
        self.question_patterns = {
            "incomplete": [
                r"그것은\s*$",  # "그것은"으로 끝남
                r"이것은\s*$",  # "이것은"으로 끝남
                r"어떻게\s*$",  # "어떻게"로 끝남
                r"왜\s*$",     # "왜"로 끝남
                r"언제\s*$",   # "언제"로 끝남
                r"어디서\s*$", # "어디서"로 끝남
            ],
            "follow_up": [
                r"더\s+자세히",  # 더 자세히
                r"추가로",       # 추가로
                r"또한",         # 또한
                r"그리고",       # 그리고
                r"또\s+다른",    # 또 다른
            ],
            "clarification": [
                r"정확히\s+말하면",  # 정확히 말하면
                r"구체적으로",       # 구체적으로
                r"예를\s+들어",      # 예를 들어
                r"즉",              # 즉
            ]
        }
        
        # 법률 엔티티 참조 패턴
        self.entity_reference_patterns = {
            "법령": r"그\s+법|해당\s+법|위의\s+법",
            "조문": r"그\s+조문|해당\s+조문|위의\s+조문",
            "판례": r"그\s+판례|해당\s+판례|위의\s+판례",
            "사건": r"그\s+사건|해당\s+사건|위의\s+사건"
        }
        
        self.logger.info("MultiTurnQuestionHandler initialized")
    
    def detect_multi_turn_question(self, query: str, context: ConversationContext) -> bool:
        """
        다중 턴 질문 감지
        
        Args:
            query: 현재 질문
            context: 대화 맥락
            
        Returns:
            bool: 다중 턴 질문 여부
        """
        try:
            query_lower = query.lower().strip()
            
            # 대명사 패턴 확인
            for pronoun_type, pattern in self.pronoun_patterns.items():
                if re.search(pattern, query_lower):
                    self.logger.debug(f"Detected pronoun pattern: {pronoun_type}")
                    return True
            
            # 불완전한 질문 패턴 확인
            for pattern_type, patterns in self.question_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        self.logger.debug(f"Detected {pattern_type} pattern")
                        return True
            
            # 엔티티 참조 패턴 확인
            for entity_type, pattern in self.entity_reference_patterns.items():
                if re.search(pattern, query_lower):
                    self.logger.debug(f"Detected entity reference: {entity_type}")
                    return True
            
            # 맥락 기반 감지 (이전 턴과의 연관성)
            if self._has_contextual_reference(query, context):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting multi-turn question: {e}")
            return False
    
    def resolve_pronouns(self, query: str, context: ConversationContext) -> str:
        """
        대명사 해결
        
        Args:
            query: 현재 질문
            context: 대화 맥락
            
        Returns:
            str: 대명사가 해결된 질문
        """
        try:
            resolved_query = query
            
            # 각 대명사 패턴에 대해 해결 시도
            for pronoun_type, pattern in self.pronoun_patterns.items():
                matches = re.finditer(pattern, resolved_query, re.IGNORECASE)
                
                for match in matches:
                    replacement = self._find_pronoun_replacement(
                        pronoun_type, match.group(), context
                    )
                    
                    if replacement:
                        resolved_query = resolved_query.replace(
                            match.group(), replacement
                        )
                        self.logger.debug(f"Resolved '{match.group()}' to '{replacement}'")
            
            return resolved_query
            
        except Exception as e:
            self.logger.error(f"Error resolving pronouns: {e}")
            return query
    
    def build_complete_query(self, query: str, context: ConversationContext) -> Dict[str, Any]:
        """
        완전한 질문 구성
        
        Args:
            query: 현재 질문
            context: 대화 맥락
            
        Returns:
            Dict[str, Any]: 완성된 질문 정보
        """
        try:
            # 1. 대명사 해결
            resolved_query = self.resolve_pronouns(query, context)
            
            # 2. 참조 엔티티 추출
            referenced_entities = self.extract_reference_entities(query)
            
            # 3. 맥락 정보 추가
            context_info = self._extract_context_info(context, referenced_entities)
            
            # 4. 질문 유형 분석
            question_type = self._analyze_question_type(resolved_query)
            
            # 5. 신뢰도 계산
            confidence = self._calculate_resolution_confidence(
                query, resolved_query, context
            )
            
            # 6. 추론 과정 생성
            reasoning = self._generate_reasoning(query, resolved_query, referenced_entities)
            
            return {
                "original_query": query,
                "resolved_query": resolved_query,
                "referenced_entities": referenced_entities,
                "context_info": context_info,
                "question_type": question_type,
                "confidence": confidence,
                "reasoning": reasoning,
                "is_multi_turn": query != resolved_query
            }
            
        except Exception as e:
            self.logger.error(f"Error building complete query: {e}")
            return {
                "original_query": query,
                "resolved_query": query,
                "referenced_entities": [],
                "context_info": {},
                "question_type": "unknown",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "is_multi_turn": False
            }
    
    def extract_reference_entities(self, query: str) -> List[str]:
        """
        참조 엔티티 추출
        
        Args:
            query: 질문
            
        Returns:
            List[str]: 참조된 엔티티 목록
        """
        try:
            referenced_entities = []
            query_lower = query.lower()
            
            # 법률 엔티티 참조 패턴 확인
            for entity_type, pattern in self.entity_reference_patterns.items():
                if re.search(pattern, query_lower):
                    referenced_entities.append(entity_type)
            
            # 구체적인 엔티티 값 추출
            entity_patterns = {
                "법령": r"([가-힣]+법)",
                "조문": r"제(\d+)조",
                "판례": r"(\d{4}[가나다라마바사아자차카타파하]\d+)",
                "사건": r"([가-힣]+사건)"
            }
            
            for entity_type, pattern in entity_patterns.items():
                matches = re.findall(pattern, query)
                referenced_entities.extend([f"{entity_type}:{match}" for match in matches])
            
            return list(set(referenced_entities))
            
        except Exception as e:
            self.logger.error(f"Error extracting reference entities: {e}")
            return []
    
    def _has_contextual_reference(self, query: str, context: ConversationContext) -> bool:
        """맥락적 참조 여부 확인"""
        try:
            if not context.turns:
                return False
            
            # 최근 턴들과의 연관성 확인
            recent_turns = context.turns[-3:]  # 최근 3턴
            
            query_lower = query.lower()
            
            for turn in recent_turns:
                # 공통 키워드 확인
                turn_text = f"{turn.user_query} {turn.bot_response}".lower()
                
                # 법률 용어 공통성 확인
                legal_terms = ["손해배상", "계약", "소송", "판례", "법령", "조문"]
                common_terms = [term for term in legal_terms 
                              if term in query_lower and term in turn_text]
                
                if common_terms:
                    return True
                
                # 엔티티 공통성 확인
                if turn.entities:
                    for entity_type, entities in turn.entities.items():
                        for entity in entities:
                            if entity.lower() in query_lower:
                                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking contextual reference: {e}")
            return False
    
    def _find_pronoun_replacement(self, pronoun_type: str, pronoun: str, 
                                 context: ConversationContext) -> Optional[str]:
        """대명사 대체어 찾기"""
        try:
            if not context.turns:
                return None
            
            # 최근 턴에서 관련 엔티티 찾기
            recent_turns = context.turns[-2:]  # 최근 2턴
            
            for turn in reversed(recent_turns):
                # 엔티티에서 대체어 찾기
                if turn.entities:
                    for entity_type, entities in turn.entities.items():
                        if entities:
                            # 가장 최근에 언급된 엔티티 선택
                            if isinstance(entities, (list, tuple)):
                                return entities[-1]
                            elif isinstance(entities, set):
                                return list(entities)[-1]
                            elif isinstance(entities, dict):
                                return list(entities.keys())[-1]
                            else:
                                return str(entities)
                
                # 질문/답변에서 명사구 추출
                text = f"{turn.user_query} {turn.bot_response}"
                nouns = self._extract_nouns(text)
                
                if nouns:
                    return nouns[-1]  # 가장 최근 명사
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding pronoun replacement: {e}")
            return None
    
    def _extract_nouns(self, text: str) -> List[str]:
        """텍스트에서 명사 추출"""
        try:
            # 간단한 명사 패턴 (법률 용어 중심)
            noun_patterns = [
                r"([가-힣]+법)",           # 민법, 형법 등
                r"([가-힣]+사건)",         # 손해배상사건 등
                r"([가-힣]+계약)",         # 매매계약 등
                r"([가-힣]+소송)",         # 손해배상소송 등
                r"제(\d+)조",             # 제750조 등
                r"(\d{4}[가나다라마바사아자차카타파하]\d+)",  # 판례번호
            ]
            
            nouns = []
            for pattern in noun_patterns:
                matches = re.findall(pattern, text)
                nouns.extend(matches)
            
            return nouns
            
        except Exception as e:
            self.logger.error(f"Error extracting nouns: {e}")
            return []
    
    def _extract_context_info(self, context: ConversationContext, 
                             referenced_entities: List[str]) -> Dict[str, Any]:
        """맥락 정보 추출"""
        try:
            context_info = {
                "recent_topics": list(context.topic_stack[-3:]),  # 최근 주제
                "total_turns": len(context.turns),
                "session_age": (datetime.now() - context.created_at).total_seconds() / 3600,
                "referenced_entities": referenced_entities
            }
            
            # 최근 턴의 엔티티 정보
            if context.turns:
                recent_turn = context.turns[-1]
                if recent_turn.entities:
                    context_info["recent_entities"] = {
                        entity_type: list(entities)
                        for entity_type, entities in recent_turn.entities.items()
                        if entities
                    }
            
            return context_info
            
        except Exception as e:
            self.logger.error(f"Error extracting context info: {e}")
            return {}
    
    def _analyze_question_type(self, query: str) -> str:
        """질문 유형 분석"""
        try:
            query_lower = query.lower()
            
            # 질문 유형별 키워드
            question_types = {
                "legal_advice": ["어떻게", "방법", "절차", "해결"],
                "precedent_search": ["판례", "사건", "법원", "판결"],
                "law_inquiry": ["법령", "조문", "법률", "항"],
                "term_explanation": ["의미", "정의", "뜻", "해석"],
                "procedure_guide": ["신청", "제출", "처리", "기간"]
            }
            
            for q_type, keywords in question_types.items():
                if any(keyword in query_lower for keyword in keywords):
                    return q_type
            
            return "general_question"
            
        except Exception as e:
            self.logger.error(f"Error analyzing question type: {e}")
            return "unknown"
    
    def _calculate_resolution_confidence(self, original_query: str, 
                                        resolved_query: str, 
                                        context: ConversationContext) -> float:
        """해결 신뢰도 계산"""
        try:
            confidence = 0.0
            
            # 기본 신뢰도
            if original_query == resolved_query:
                confidence = 1.0
            else:
                confidence = 0.5
            
            # 맥락 정보에 따른 조정
            if context.turns:
                confidence += 0.2
            
            if context.topic_stack:
                confidence += 0.1
            
            # 엔티티 정보에 따른 조정
            total_entities = sum(len(entities) for entities in context.entities.values())
            if total_entities > 0:
                confidence += min(0.2, total_entities * 0.05)
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating resolution confidence: {e}")
            return 0.0
    
    def _generate_reasoning(self, original_query: str, resolved_query: str, 
                           referenced_entities: List[str]) -> str:
        """추론 과정 생성"""
        try:
            reasoning_parts = []
            
            if original_query != resolved_query:
                reasoning_parts.append(f"대명사 해결: '{original_query}' → '{resolved_query}'")
            
            if referenced_entities:
                reasoning_parts.append(f"참조 엔티티: {', '.join(referenced_entities)}")
            
            if not reasoning_parts:
                reasoning_parts.append("다중 턴 질문이 아님")
            
            return "; ".join(reasoning_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning: {e}")
            return f"Error: {str(e)}"


# 테스트 함수
def test_multi_turn_handler():
    """다중 턴 질문 처리기 테스트"""
    handler = MultiTurnQuestionHandler()
    
    # 테스트용 대화 맥락 생성
    from .conversation_manager import ConversationTurn
    
    test_turns = [
        ConversationTurn(
            user_query="손해배상 청구 방법을 알려주세요",
            bot_response="민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...",
            timestamp=datetime.now(),
            question_type="legal_advice",
            entities={"laws": ["민법"], "articles": ["제750조"], "legal_terms": ["손해배상"]}
        ),
        ConversationTurn(
            user_query="계약 해지 절차는 어떻게 되나요?",
            bot_response="계약 해지 절차는 다음과 같습니다...",
            timestamp=datetime.now(),
            question_type="procedure_guide",
            entities={"legal_terms": ["계약", "해지"]}
        )
    ]
    
    context = ConversationContext(
        session_id="test_session",
        turns=test_turns,
        entities={"laws": {"민법"}, "articles": {"제750조"}, "precedents": set(), "legal_terms": {"손해배상", "계약", "해지"}},
        topic_stack=["손해배상", "계약"],
        created_at=datetime.now(),
        last_updated=datetime.now()
    )
    
    print("=== 다중 턴 질문 처리기 테스트 ===")
    
    # 테스트 쿼리들
    test_queries = [
        "그것에 대해 더 자세히 알려주세요",
        "위의 손해배상 사건에서 과실비율은 어떻게 정해지나요?",
        "그 판례는 어떤 사건이었나요?",
        "이것의 법적 근거는 무엇인가요?",
        "일반적인 질문입니다"
    ]
    
    for query in test_queries:
        print(f"\n질문: {query}")
        
        # 다중 턴 질문 감지
        is_multi_turn = handler.detect_multi_turn_question(query, context)
        print(f"다중 턴 질문: {is_multi_turn}")
        
        if is_multi_turn:
            # 완전한 질문 구성
            result = handler.build_complete_query(query, context)
            print(f"원본 질문: {result['original_query']}")
            print(f"해결된 질문: {result['resolved_query']}")
            print(f"참조 엔티티: {result['referenced_entities']}")
            print(f"질문 유형: {result['question_type']}")
            print(f"신뢰도: {result['confidence']:.2f}")
            print(f"추론: {result['reasoning']}")
        
        print("-" * 50)


if __name__ == "__main__":
    test_multi_turn_handler()

