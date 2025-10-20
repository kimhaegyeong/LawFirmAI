# -*- coding: utf-8 -*-
"""
컨텍스트 압축기
긴 대화를 요약하여 토큰 제한 문제 해결
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import Counter

from .conversation_manager import ConversationContext, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """압축 결과"""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    compressed_text: str
    preserved_entities: List[str]
    preserved_topics: List[str]
    summary: str


class ContextCompressor:
    """컨텍스트 압축기"""
    
    def __init__(self, max_tokens: int = 2000, compression_threshold: float = 0.8):
        """
        컨텍스트 압축기 초기화
        
        Args:
            max_tokens: 최대 토큰 수
            compression_threshold: 압축 임계값 (이 비율을 초과하면 압축)
        """
        self.logger = logging.getLogger(__name__)
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        
        # 토큰 계산을 위한 단어당 평균 토큰 수 (한국어 기준)
        self.tokens_per_word = 1.3
        
        # 중요도 가중치
        self.importance_weights = {
            "legal_entities": 3.0,    # 법률 엔티티
            "question_types": 2.0,    # 질문 유형
            "recent_turns": 2.5,     # 최근 턴
            "topic_keywords": 1.5,   # 주제 키워드
            "user_queries": 2.0,     # 사용자 질문
            "bot_responses": 1.0      # 봇 응답
        }
        
        # 법률 엔티티 패턴
        self.entity_patterns = {
            "laws": r"([가-힣]+법)",
            "articles": r"제(\d+)조",
            "precedents": r"(\d{4}[가나다라마바사아자차카타파하]\d+)",
            "legal_terms": r"(손해배상|계약|소송|판례|법령|조문|항|호|목)"
        }
        
        # 중요 키워드 패턴
        self.important_keywords = [
            "손해배상", "계약", "소송", "판례", "법령", "조문",
            "민법", "형법", "상법", "근로기준법", "상속", "이혼",
            "교통사고", "부동산", "금융", "지적재산권"
        ]
        
        self.logger.info(f"ContextCompressor initialized with max_tokens={max_tokens}")
    
    def compress_long_conversation(self, context: ConversationContext, 
                                   max_tokens: Optional[int] = None) -> CompressionResult:
        """
        긴 대화 압축
        
        Args:
            context: 대화 맥락
            max_tokens: 최대 토큰 수 (None이면 기본값 사용)
            
        Returns:
            CompressionResult: 압축 결과
        """
        try:
            if max_tokens is None:
                max_tokens = self.max_tokens
            
            # 현재 토큰 수 계산
            current_tokens = self.calculate_tokens(context)
            
            # 압축 필요 여부 확인
            if current_tokens <= max_tokens:
                return CompressionResult(
                    original_tokens=current_tokens,
                    compressed_tokens=current_tokens,
                    compression_ratio=1.0,
                    compressed_text=self._context_to_text(context),
                    preserved_entities=self._extract_all_entities(context),
                    preserved_topics=list(context.topic_stack),
                    summary="압축 불필요"
                )
            
            # 핵심 정보 추출
            key_info = self.extract_key_information(context.turns)
            
            # 관련 컨텍스트 유지
            relevant_turns = self.maintain_relevant_context(context, "")
            
            # 압축된 텍스트 생성
            compressed_text = self._generate_compressed_text(key_info, relevant_turns)
            
            # 압축된 토큰 수 계산
            compressed_tokens = self.calculate_tokens_from_text(compressed_text)
            
            # 압축률이 1.0을 초과하지 않도록 보장
            if compressed_tokens > current_tokens:
                # 압축이 효과적이지 않은 경우 원본 반환
                compressed_text = self._context_to_text(context)
                compressed_tokens = current_tokens
            
            # 요약 생성
            summary = self._generate_summary(context)
            
            return CompressionResult(
                original_tokens=current_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=compressed_tokens / current_tokens,
                compressed_text=compressed_text,
                preserved_entities=key_info["entities"],
                preserved_topics=key_info["topics"],
                summary=summary
            )
            
        except Exception as e:
            self.logger.error(f"Error compressing conversation: {e}")
            return CompressionResult(
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=0.0,
                compressed_text="",
                preserved_entities=[],
                preserved_topics=[],
                summary=f"Error: {str(e)}"
            )
    
    def extract_key_information(self, turns: List[ConversationTurn]) -> Dict[str, Any]:
        """
        핵심 정보 추출
        
        Args:
            turns: 대화 턴 목록
            
        Returns:
            Dict[str, Any]: 핵심 정보
        """
        try:
            key_info = {
                "entities": [],
                "topics": [],
                "question_types": [],
                "important_queries": [],
                "key_responses": []
            }
            
            # 엔티티 수집
            all_entities = set()
            for turn in turns:
                if turn.entities:
                    for entity_type, entities in turn.entities.items():
                        all_entities.update(entities)
            key_info["entities"] = list(all_entities)
            
            # 질문 유형 수집
            question_types = [turn.question_type for turn in turns if turn.question_type]
            key_info["question_types"] = list(set(question_types))
            
            # 중요 질문 추출 (엔티티가 많은 질문)
            important_queries = []
            for turn in turns:
                entity_count = sum(len(entities) for entities in (turn.entities or {}).values())
                if entity_count > 0:
                    important_queries.append({
                        "query": turn.user_query,
                        "entity_count": entity_count,
                        "timestamp": turn.timestamp
                    })
            
            # 엔티티 수 기준으로 정렬
            important_queries.sort(key=lambda x: x["entity_count"], reverse=True)
            key_info["important_queries"] = important_queries[:3]  # 상위 3개
            
            # 중요 응답 추출 (긴 응답 중에서)
            key_responses = []
            for turn in turns:
                if len(turn.bot_response) > 100:  # 100자 이상
                    key_responses.append({
                        "response": turn.bot_response[:200] + "...",  # 200자로 제한
                        "timestamp": turn.timestamp
                    })
            
            key_info["key_responses"] = key_responses[:2]  # 상위 2개
            
            return key_info
            
        except Exception as e:
            self.logger.error(f"Error extracting key information: {e}")
            return {"entities": [], "topics": [], "question_types": [], 
                   "important_queries": [], "key_responses": []}
    
    def maintain_relevant_context(self, context: ConversationContext, 
                                 current_query: str) -> List[ConversationTurn]:
        """
        관련 컨텍스트 유지
        
        Args:
            context: 대화 맥락
            current_query: 현재 질문
            
        Returns:
            List[ConversationTurn]: 관련 턴 목록
        """
        try:
            if not context.turns:
                return []
            
            # 최근 턴들 (항상 유지)
            recent_turns = context.turns[-2:]  # 최근 2턴
            
            # 중요도 기반 턴 선택
            scored_turns = []
            for turn in context.turns[:-2]:  # 최근 2턴 제외
                score = self._calculate_turn_importance(turn, context)
                scored_turns.append((score, turn))
            
            # 중요도 순으로 정렬
            scored_turns.sort(key=lambda x: x[0], reverse=True)
            
            # 상위 턴들 선택 (토큰 제한 고려)
            selected_turns = []
            current_tokens = sum(self.calculate_tokens_from_text(turn.user_query + " " + turn.bot_response) 
                               for turn in recent_turns)
            
            for score, turn in scored_turns:
                turn_tokens = self.calculate_tokens_from_text(turn.user_query + " " + turn.bot_response)
                if current_tokens + turn_tokens <= self.max_tokens * 0.7:  # 70% 제한
                    selected_turns.append(turn)
                    current_tokens += turn_tokens
                else:
                    break
            
            # 최근 턴 + 선택된 턴들
            return recent_turns + selected_turns
            
        except Exception as e:
            self.logger.error(f"Error maintaining relevant context: {e}")
            return context.turns[-3:] if context.turns else []  # 폴백: 최근 3턴
    
    def calculate_tokens(self, context: ConversationContext) -> int:
        """
        컨텍스트의 토큰 수 계산
        
        Args:
            context: 대화 맥락
            
        Returns:
            int: 토큰 수
        """
        try:
            total_tokens = 0
            
            # 턴별 토큰 계산
            for turn in context.turns:
                turn_text = f"{turn.user_query} {turn.bot_response}"
                total_tokens += self.calculate_tokens_from_text(turn_text)
            
            # 엔티티 토큰 계산
            for entity_type, entities in context.entities.items():
                entity_text = " ".join(entities)
                total_tokens += self.calculate_tokens_from_text(entity_text)
            
            # 주제 스택 토큰 계산
            topic_text = " ".join(context.topic_stack)
            total_tokens += self.calculate_tokens_from_text(topic_text)
            
            return total_tokens
            
        except Exception as e:
            self.logger.error(f"Error calculating tokens: {e}")
            return 0
    
    def calculate_tokens_from_text(self, text: str) -> int:
        """
        텍스트의 토큰 수 계산
        
        Args:
            text: 텍스트
            
        Returns:
            int: 토큰 수
        """
        try:
            if not text:
                return 0
            
            # 간단한 토큰 계산 (단어 수 기반)
            words = len(text.split())
            
            # 한국어 특성 고려 (조사, 어미 등)
            korean_chars = len([c for c in text if ord('가') <= ord(c) <= ord('힣')])
            korean_adjustment = korean_chars * 0.1  # 한국어 문자당 추가 토큰
            
            tokens = int(words * self.tokens_per_word + korean_adjustment)
            
            return max(1, tokens)  # 최소 1토큰
            
        except Exception as e:
            self.logger.error(f"Error calculating tokens from text: {e}")
            return len(text.split()) if text else 0
    
    def _calculate_turn_importance(self, turn: ConversationTurn, 
                                   context: ConversationContext) -> float:
        """턴 중요도 계산"""
        try:
            score = 0.0
            
            # 엔티티 기반 점수
            if turn.entities:
                entity_count = sum(len(entities) for entities in turn.entities.values())
                score += entity_count * self.importance_weights["legal_entities"]
            
            # 질문 유형 기반 점수
            if turn.question_type:
                score += self.importance_weights["question_types"]
            
            # 키워드 기반 점수
            turn_text = f"{turn.user_query} {turn.bot_response}".lower()
            keyword_count = sum(1 for keyword in self.important_keywords 
                              if keyword in turn_text)
            score += keyword_count * self.importance_weights["topic_keywords"]
            
            # 시간 기반 점수 (최근일수록 높음)
            time_score = (turn.timestamp - context.created_at).total_seconds() / 3600
            score += max(0, 10 - time_score) * 0.1  # 시간이 지날수록 감소
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating turn importance: {e}")
            return 0.0
    
    def _generate_compressed_text(self, key_info: Dict[str, Any], 
                                 relevant_turns: List[ConversationTurn]) -> str:
        """압축된 텍스트 생성"""
        try:
            compressed_parts = []
            
            # 핵심 엔티티 정보
            if key_info["entities"]:
                compressed_parts.append(f"핵심 법률 엔티티: {', '.join(key_info['entities'][:5])}")
            
            # 질문 유형 정보
            if key_info["question_types"]:
                compressed_parts.append(f"질문 유형: {', '.join(key_info['question_types'])}")
            
            # 중요 질문들
            if key_info["important_queries"]:
                compressed_parts.append("주요 질문:")
                for i, query_info in enumerate(key_info["important_queries"], 1):
                    compressed_parts.append(f"{i}. {query_info['query']}")
            
            # 관련 턴들
            if relevant_turns:
                compressed_parts.append("관련 대화:")
                for turn in relevant_turns[-3:]:  # 최근 3턴만
                    compressed_parts.append(f"Q: {turn.user_query}")
                    compressed_parts.append(f"A: {turn.bot_response[:100]}...")
            
            return "\n".join(compressed_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating compressed text: {e}")
            return ""
    
    def _generate_summary(self, context: ConversationContext) -> str:
        """대화 요약 생성"""
        try:
            if not context.turns:
                return "대화 내용 없음"
            
            # 주요 주제 추출
            topics = list(context.topic_stack)
            if topics:
                topic_summary = f"주요 주제: {', '.join(topics)}"
            else:
                topic_summary = "주제 정보 없음"
            
            # 엔티티 요약
            total_entities = sum(len(entities) for entities in context.entities.values())
            entity_summary = f"총 {total_entities}개 법률 엔티티 언급"
            
            # 턴 수 요약
            turn_summary = f"총 {len(context.turns)}턴의 대화"
            
            # 세션 기간 요약
            duration_hours = (context.last_updated - context.created_at).total_seconds() / 3600
            duration_summary = f"대화 기간: {duration_hours:.1f}시간"
            
            return f"{topic_summary}; {entity_summary}; {turn_summary}; {duration_summary}"
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return f"요약 생성 오류: {str(e)}"
    
    def _extract_all_entities(self, context: ConversationContext) -> List[str]:
        """모든 엔티티 추출"""
        try:
            all_entities = []
            for entity_type, entities in context.entities.items():
                all_entities.extend(list(entities))
            return list(set(all_entities))
        except Exception as e:
            self.logger.error(f"Error extracting all entities: {e}")
            return []
    
    def _context_to_text(self, context: ConversationContext) -> str:
        """컨텍스트를 텍스트로 변환"""
        try:
            text_parts = []
            
            for turn in context.turns:
                text_parts.append(f"Q: {turn.user_query}")
                text_parts.append(f"A: {turn.bot_response}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Error converting context to text: {e}")
            return ""


# 테스트 함수
def test_context_compressor():
    """컨텍스트 압축기 테스트"""
    compressor = ContextCompressor(max_tokens=1000)
    
    # 테스트용 대화 맥락 생성
    from .conversation_manager import ConversationTurn
    
    test_turns = [
        ConversationTurn(
            user_query="손해배상 청구 방법을 알려주세요",
            bot_response="민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다. 손해배상은 불법행위로 인한 손해를 배상받는 제도입니다. 손해의 발생, 가해자의 고의 또는 과실, 인과관계, 손해의 발생이 필요합니다.",
            timestamp=datetime.now(),
            question_type="legal_advice",
            entities={"laws": ["민법"], "articles": ["제750조"], "legal_terms": ["손해배상", "불법행위"]}
        ),
        ConversationTurn(
            user_query="계약 해지 절차는 어떻게 되나요?",
            bot_response="계약 해지 절차는 다음과 같습니다. 1) 해지 사유 확인 2) 해지 통지 3) 손해배상 청구 4) 소송 제기 등이 있습니다. 민법 제543조에 따라 계약은 당사자 일방의 의사표시로 해지할 수 있습니다.",
            timestamp=datetime.now(),
            question_type="procedure_guide",
            entities={"laws": ["민법"], "articles": ["제543조"], "legal_terms": ["계약", "해지"]}
        ),
        ConversationTurn(
            user_query="위의 손해배상 사건에서 과실비율은 어떻게 정해지나요?",
            bot_response="과실비율은 교통사고의 경우 보험회사에서 정한 기준표를 참고하여 결정됩니다. 대법원 2023다12345 판례에 따르면 과실비율은 사고 상황, 도로 상황, 차량 상태 등을 종합적으로 고려하여 결정됩니다.",
            timestamp=datetime.now(),
            question_type="legal_advice",
            entities={"precedents": ["2023다12345"], "legal_terms": ["과실비율", "교통사고"]}
        )
    ]
    
    context = ConversationContext(
        session_id="test_session",
        turns=test_turns,
        entities={"laws": {"민법"}, "articles": {"제750조", "제543조"}, 
                 "precedents": {"2023다12345"}, 
                 "legal_terms": {"손해배상", "불법행위", "계약", "해지", "과실비율", "교통사고"}},
        topic_stack=["손해배상", "계약", "교통사고"],
        created_at=datetime.now(),
        last_updated=datetime.now()
    )
    
    print("=== 컨텍스트 압축기 테스트 ===")
    
    # 토큰 수 계산
    original_tokens = compressor.calculate_tokens(context)
    print(f"원본 토큰 수: {original_tokens}")
    
    # 압축 실행
    compression_result = compressor.compress_long_conversation(context, max_tokens=500)
    
    print(f"\n압축 결과:")
    print(f"원본 토큰: {compression_result.original_tokens}")
    print(f"압축 토큰: {compression_result.compressed_tokens}")
    print(f"압축 비율: {compression_result.compression_ratio:.2f}")
    print(f"보존된 엔티티: {compression_result.preserved_entities}")
    print(f"보존된 주제: {compression_result.preserved_topics}")
    print(f"요약: {compression_result.summary}")
    
    print(f"\n압축된 텍스트:")
    print(compression_result.compressed_text)
    
    # 핵심 정보 추출 테스트
    print(f"\n=== 핵심 정보 추출 ===")
    key_info = compressor.extract_key_information(context.turns)
    print(f"엔티티: {key_info['entities']}")
    print(f"질문 유형: {key_info['question_types']}")
    print(f"중요 질문: {[q['query'] for q in key_info['important_queries']]}")


if __name__ == "__main__":
    test_context_compressor()

