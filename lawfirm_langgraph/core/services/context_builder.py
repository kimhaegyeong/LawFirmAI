# -*- coding: utf-8 -*-
"""
컨텍스트 윈도우 최적화 시스템
효율적인 컨텍스트 관리 및 토큰 사용량 최적화
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .question_classifier import QuestionType, QuestionClassification

logger = logging.getLogger(__name__)


class ContextPriority(Enum):
    """컨텍스트 우선순위"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ContextItem:
    """컨텍스트 아이템"""
    content: str
    priority: ContextPriority
    source_type: str  # law, precedent, general
    relevance_score: float
    token_count: int
    metadata: Dict[str, Any]


@dataclass
class ContextWindow:
    """컨텍스트 윈도우"""
    items: List[ContextItem]
    total_tokens: int
    max_tokens: int
    utilization_rate: float
    priority_distribution: Dict[str, int]


class ContextBuilder:
    """컨텍스트 윈도우 최적화기"""
    
    def __init__(self, 
                 max_context_tokens: int = 4000,
                 max_law_tokens: int = 1500,
                 max_precedent_tokens: int = 1500,
                 max_general_tokens: int = 1000):
        """컨텍스트 빌더 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 토큰 제한 설정
        self.max_context_tokens = max_context_tokens
        self.max_law_tokens = max_law_tokens
        self.max_precedent_tokens = max_precedent_tokens
        self.max_general_tokens = max_general_tokens
        
        # 질문 유형별 토큰 할당 비율
        self.question_type_allocations = {
            QuestionType.PRECEDENT_SEARCH: {
                "law": 0.3,
                "precedent": 0.6,
                "general": 0.1
            },
            QuestionType.LAW_INQUIRY: {
                "law": 0.7,
                "precedent": 0.2,
                "general": 0.1
            },
            QuestionType.LEGAL_ADVICE: {
                "law": 0.4,
                "precedent": 0.4,
                "general": 0.2
            },
            QuestionType.PROCEDURE_GUIDE: {
                "law": 0.5,
                "precedent": 0.3,
                "general": 0.2
            },
            QuestionType.TERM_EXPLANATION: {
                "law": 0.6,
                "precedent": 0.2,
                "general": 0.2
            },
            QuestionType.GENERAL_QUESTION: {
                "law": 0.3,
                "precedent": 0.3,
                "general": 0.4
            }
        }
        
        # 우선순위 점수 매핑
        self.priority_scores = {
            ContextPriority.HIGH: 3.0,
            ContextPriority.MEDIUM: 2.0,
            ContextPriority.LOW: 1.0
        }
    
    def build_optimized_context(self,
                               query: str,
                               question_classification: QuestionClassification,
                               search_results: Dict[str, List[Dict[str, Any]]],
                               conversation_history: Optional[List[Dict[str, Any]]] = None) -> ContextWindow:
        """
        최적화된 컨텍스트 윈도우 구성
        
        Args:
            query: 사용자 질문
            question_classification: 질문 분류 결과
            search_results: 검색 결과
            conversation_history: 대화 이력
            
        Returns:
            ContextWindow: 최적화된 컨텍스트 윈도우
        """
        try:
            self.logger.info(f"Building optimized context for query: {query[:50]}...")
            
            # 1. 컨텍스트 아이템 생성
            context_items = self._create_context_items(
                query, question_classification, search_results, conversation_history
            )
            
            # 2. 질문 유형별 토큰 할당 계산
            token_allocations = self._calculate_token_allocations(question_classification)
            
            # 3. 우선순위 기반 필터링
            filtered_items = self._filter_by_priority(context_items, token_allocations)
            
            # 4. 토큰 제한 내에서 최적화
            optimized_items = self._optimize_within_token_limits(filtered_items, token_allocations)
            
            # 5. 컨텍스트 윈도우 구성
            context_window = self._build_context_window(optimized_items)
            
            self.logger.info(f"Context window built: {len(context_window.items)} items, "
                           f"{context_window.total_tokens} tokens, "
                           f"{context_window.utilization_rate:.1%} utilization")
            
            return context_window
            
        except Exception as e:
            self.logger.error(f"Error building optimized context: {e}")
            return self._create_fallback_context(query, search_results)
    
    def _create_context_items(self,
                             query: str,
                             question_classification: QuestionClassification,
                             search_results: Dict[str, List[Dict[str, Any]]],
                             conversation_history: Optional[List[Dict[str, Any]]]) -> List[ContextItem]:
        """컨텍스트 아이템 생성"""
        try:
            items = []
            
            # 법률 검색 결과 처리
            law_results = search_results.get("law_results", [])
            for law in law_results:
                content = self._format_law_content(law)
                priority = self._calculate_law_priority(law, question_classification)
                relevance_score = law.get("similarity", 0.0)
                token_count = self._estimate_tokens(content)
                
                items.append(ContextItem(
                    content=content,
                    priority=priority,
                    source_type="law",
                    relevance_score=relevance_score,
                    token_count=token_count,
                    metadata={"law_id": law.get("law_id"), "article_number": law.get("article_number")}
                ))
            
            # 판례 검색 결과 처리
            precedent_results = search_results.get("precedent_results", [])
            for precedent in precedent_results:
                content = self._format_precedent_content(precedent)
                priority = self._calculate_precedent_priority(precedent, question_classification)
                relevance_score = precedent.get("similarity", 0.0)
                token_count = self._estimate_tokens(content)
                
                items.append(ContextItem(
                    content=content,
                    priority=priority,
                    source_type="precedent",
                    relevance_score=relevance_score,
                    token_count=token_count,
                    metadata={"case_id": precedent.get("case_id"), "case_number": precedent.get("case_number")}
                ))
            
            # 일반 검색 결과 처리
            general_results = search_results.get("results", [])
            for result in general_results:
                if result.get("type") not in ["law", "precedent"]:
                    content = self._format_general_content(result)
                    priority = self._calculate_general_priority(result, question_classification)
                    relevance_score = result.get("similarity", 0.0)
                    token_count = self._estimate_tokens(content)
                    
                    items.append(ContextItem(
                        content=content,
                        priority=priority,
                        source_type="general",
                        relevance_score=relevance_score,
                        token_count=token_count,
                        metadata={"type": result.get("type")}
                    ))
            
            # 대화 이력 처리
            if conversation_history:
                history_items = self._process_conversation_history(conversation_history, question_classification)
                items.extend(history_items)
            
            return items
            
        except Exception as e:
            self.logger.error(f"Error creating context items: {e}")
            return []
    
    def _format_law_content(self, law: Dict[str, Any]) -> str:
        """법률 내용 포맷팅"""
        try:
            law_name = law.get("law_name", "법률명 없음")
            article_number = law.get("article_number", "")
            content = law.get("content", "")
            
            formatted = f"**{law_name} {article_number}**\n{content}"
            
            # 내용이 너무 길면 자르기
            if len(formatted) > 800:
                formatted = formatted[:800] + "..."
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting law content: {e}")
            return f"법률 정보: {law.get('law_name', '알 수 없음')}"
    
    def _format_precedent_content(self, precedent: Dict[str, Any]) -> str:
        """판례 내용 포맷팅"""
        try:
            case_name = precedent.get("case_name", "사건명 없음")
            case_number = precedent.get("case_number", "")
            court = precedent.get("court", "")
            decision_date = precedent.get("decision_date", "")
            summary = precedent.get("summary", "")
            
            formatted = f"**{case_name}** ({case_number})\n"
            formatted += f"법원: {court}\n"
            formatted += f"판결일: {decision_date}\n"
            formatted += f"판결요지: {summary}"
            
            # 내용이 너무 길면 자르기
            if len(formatted) > 800:
                formatted = formatted[:800] + "..."
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting precedent content: {e}")
            return f"판례 정보: {precedent.get('case_name', '알 수 없음')}"
    
    def _format_general_content(self, result: Dict[str, Any]) -> str:
        """일반 내용 포맷팅"""
        try:
            title = result.get("title", result.get("case_name", "제목 없음"))
            content = result.get("content", result.get("summary", ""))
            
            formatted = f"**{title}**\n{content}"
            
            # 내용이 너무 길면 자르기
            if len(formatted) > 600:
                formatted = formatted[:600] + "..."
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting general content: {e}")
            return f"정보: {result.get('title', '알 수 없음')}"
    
    def _calculate_law_priority(self, law: Dict[str, Any], question_classification: QuestionClassification) -> ContextPriority:
        """법률 우선순위 계산"""
        try:
            question_type = question_classification.question_type
            similarity = law.get("similarity", 0.0)
            
            # 질문 유형별 기본 우선순위
            if question_type in [QuestionType.LAW_INQUIRY, QuestionType.TERM_EXPLANATION]:
                base_priority = ContextPriority.HIGH
            elif question_type in [QuestionType.LEGAL_ADVICE, QuestionType.PROCEDURE_GUIDE]:
                base_priority = ContextPriority.MEDIUM
            else:
                base_priority = ContextPriority.LOW
            
            # 유사도에 따른 조정
            if similarity >= 0.8:
                return ContextPriority.HIGH
            elif similarity >= 0.6:
                return ContextPriority.MEDIUM
            else:
                return ContextPriority.LOW
                
        except Exception as e:
            self.logger.error(f"Error calculating law priority: {e}")
            return ContextPriority.LOW
    
    def _calculate_precedent_priority(self, precedent: Dict[str, Any], question_classification: QuestionClassification) -> ContextPriority:
        """판례 우선순위 계산"""
        try:
            question_type = question_classification.question_type
            similarity = precedent.get("similarity", 0.0)
            
            # 질문 유형별 기본 우선순위
            if question_type == QuestionType.PRECEDENT_SEARCH:
                base_priority = ContextPriority.HIGH
            elif question_type in [QuestionType.LEGAL_ADVICE, QuestionType.PROCEDURE_GUIDE]:
                base_priority = ContextPriority.MEDIUM
            else:
                base_priority = ContextPriority.LOW
            
            # 유사도에 따른 조정
            if similarity >= 0.8:
                return ContextPriority.HIGH
            elif similarity >= 0.6:
                return ContextPriority.MEDIUM
            else:
                return ContextPriority.LOW
                
        except Exception as e:
            self.logger.error(f"Error calculating precedent priority: {e}")
            return ContextPriority.LOW
    
    def _calculate_general_priority(self, result: Dict[str, Any], question_classification: QuestionClassification) -> ContextPriority:
        """일반 결과 우선순위 계산"""
        try:
            similarity = result.get("similarity", 0.0)
            
            if similarity >= 0.7:
                return ContextPriority.MEDIUM
            else:
                return ContextPriority.LOW
                
        except Exception as e:
            self.logger.error(f"Error calculating general priority: {e}")
            return ContextPriority.LOW
    
    def _process_conversation_history(self, 
                                     history: List[Dict[str, Any]], 
                                     question_classification: QuestionClassification) -> List[ContextItem]:
        """대화 이력 처리"""
        try:
            items = []
            
            # 최근 대화만 처리 (최대 3개)
            recent_history = history[-3:] if len(history) > 3 else history
            
            for i, turn in enumerate(recent_history):
                if "user_message" in turn and "assistant_response" in turn:
                    content = f"이전 질문: {turn['user_message']}\n답변: {turn['assistant_response'][:200]}..."
                    priority = ContextPriority.LOW  # 대화 이력은 낮은 우선순위
                    relevance_score = 0.5  # 기본 관련성 점수
                    token_count = self._estimate_tokens(content)
                    
                    items.append(ContextItem(
                        content=content,
                        priority=priority,
                        source_type="general",
                        relevance_score=relevance_score,
                        token_count=token_count,
                        metadata={"turn_number": i, "type": "conversation_history"}
                    ))
            
            return items
            
        except Exception as e:
            self.logger.error(f"Error processing conversation history: {e}")
            return []
    
    def _calculate_token_allocations(self, question_classification: QuestionClassification) -> Dict[str, int]:
        """질문 유형별 토큰 할당 계산"""
        try:
            question_type = question_classification.question_type
            allocations = self.question_type_allocations.get(question_type, {
                "law": 0.3,
                "precedent": 0.3,
                "general": 0.4
            })
            
            return {
                "law": int(self.max_context_tokens * allocations["law"]),
                "precedent": int(self.max_context_tokens * allocations["precedent"]),
                "general": int(self.max_context_tokens * allocations["general"])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating token allocations: {e}")
            return {
                "law": self.max_law_tokens,
                "precedent": self.max_precedent_tokens,
                "general": self.max_general_tokens
            }
    
    def _filter_by_priority(self, 
                           items: List[ContextItem], 
                           token_allocations: Dict[str, int]) -> List[ContextItem]:
        """우선순위 기반 필터링"""
        try:
            # 우선순위별로 정렬
            sorted_items = sorted(items, key=lambda x: (
                self.priority_scores[x.priority],
                x.relevance_score
            ), reverse=True)
            
            # 소스 타입별로 그룹화
            filtered_items = []
            used_tokens = {"law": 0, "precedent": 0, "general": 0}
            
            for item in sorted_items:
                source_type = item.source_type
                
                # 토큰 제한 확인
                if used_tokens[source_type] + item.token_count <= token_allocations[source_type]:
                    filtered_items.append(item)
                    used_tokens[source_type] += item.token_count
                else:
                    # 토큰 제한 초과 시 우선순위가 높은 경우만 부분 포함
                    if item.priority == ContextPriority.HIGH and used_tokens[source_type] < token_allocations[source_type]:
                        # 내용을 자르기
                        remaining_tokens = token_allocations[source_type] - used_tokens[source_type]
                        if remaining_tokens > 50:  # 최소 50 토큰은 확보
                            truncated_content = self._truncate_content(item.content, remaining_tokens)
                            truncated_item = ContextItem(
                                content=truncated_content,
                                priority=item.priority,
                                source_type=item.source_type,
                                relevance_score=item.relevance_score,
                                token_count=remaining_tokens,
                                metadata=item.metadata
                            )
                            filtered_items.append(truncated_item)
                            used_tokens[source_type] += remaining_tokens
            
            return filtered_items
            
        except Exception as e:
            self.logger.error(f"Error filtering by priority: {e}")
            return items[:10]  # 기본적으로 상위 10개만 반환
    
    def _optimize_within_token_limits(self, 
                                    items: List[ContextItem], 
                                    token_allocations: Dict[str, int]) -> List[ContextItem]:
        """토큰 제한 내에서 최적화"""
        try:
            total_used_tokens = sum(item.token_count for item in items)
            
            if total_used_tokens <= self.max_context_tokens:
                return items
            
            # 전체 토큰 제한 초과 시 추가 최적화
            optimized_items = []
            remaining_tokens = self.max_context_tokens
            
            # 우선순위 순으로 정렬
            sorted_items = sorted(items, key=lambda x: (
                self.priority_scores[x.priority],
                x.relevance_score
            ), reverse=True)
            
            for item in sorted_items:
                if remaining_tokens >= item.token_count:
                    optimized_items.append(item)
                    remaining_tokens -= item.token_count
                elif remaining_tokens > 50:  # 최소 50 토큰 확보
                    # 내용을 자르기
                    truncated_content = self._truncate_content(item.content, remaining_tokens)
                    truncated_item = ContextItem(
                        content=truncated_content,
                        priority=item.priority,
                        source_type=item.source_type,
                        relevance_score=item.relevance_score,
                        token_count=remaining_tokens,
                        metadata=item.metadata
                    )
                    optimized_items.append(truncated_item)
                    break
            
            return optimized_items
            
        except Exception as e:
            self.logger.error(f"Error optimizing within token limits: {e}")
            return items
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """내용을 토큰 제한에 맞게 자르기"""
        try:
            # 대략적인 토큰 수 계산 (한국어 기준)
            estimated_chars = max_tokens * 2
            
            if len(content) <= estimated_chars:
                return content
            
            # 문장 단위로 자르기
            truncated = content[:estimated_chars]
            
            # 마지막 문장이 잘렸으면 제거
            if not truncated.endswith(('.', '!', '?', '다', '요')):
                last_sentence_end = max(
                    truncated.rfind('.'),
                    truncated.rfind('!'),
                    truncated.rfind('?'),
                    truncated.rfind('다'),
                    truncated.rfind('요')
                )
                if last_sentence_end > estimated_chars * 0.7:  # 70% 이상이면 유지
                    truncated = truncated[:last_sentence_end + 1]
            
            return truncated + "..."
            
        except Exception as e:
            self.logger.error(f"Error truncating content: {e}")
            return content[:max_tokens * 2] + "..."
    
    def _build_context_window(self, items: List[ContextItem]) -> ContextWindow:
        """컨텍스트 윈도우 구성"""
        try:
            total_tokens = sum(item.token_count for item in items)
            utilization_rate = total_tokens / self.max_context_tokens
            
            # 우선순위별 분포 계산
            priority_distribution = {
                "high": sum(1 for item in items if item.priority == ContextPriority.HIGH),
                "medium": sum(1 for item in items if item.priority == ContextPriority.MEDIUM),
                "low": sum(1 for item in items if item.priority == ContextPriority.LOW)
            }
            
            return ContextWindow(
                items=items,
                total_tokens=total_tokens,
                max_tokens=self.max_context_tokens,
                utilization_rate=utilization_rate,
                priority_distribution=priority_distribution
            )
            
        except Exception as e:
            self.logger.error(f"Error building context window: {e}")
            return ContextWindow(
                items=[],
                total_tokens=0,
                max_tokens=self.max_context_tokens,
                utilization_rate=0.0,
                priority_distribution={"high": 0, "medium": 0, "low": 0}
            )
    
    def _estimate_tokens(self, text: str) -> int:
        """토큰 수 추정 (한국어 기준)"""
        try:
            # 한국어 기준 대략적인 토큰 수 추정
            return len(text) // 2
        except Exception as e:
            self.logger.error(f"Error estimating tokens: {e}")
            return len(text) // 2
    
    def _create_fallback_context(self, query: str, search_results: Dict[str, List[Dict[str, Any]]]) -> ContextWindow:
        """오류 시 기본 컨텍스트 생성"""
        try:
            items = []
            
            # 검색 결과에서 상위 3개만 선택
            law_results = search_results.get("law_results", [])[:2]
            precedent_results = search_results.get("precedent_results", [])[:2]
            
            for law in law_results:
                content = f"법률: {law.get('law_name', '')} {law.get('article_number', '')}"
                items.append(ContextItem(
                    content=content,
                    priority=ContextPriority.MEDIUM,
                    source_type="law",
                    relevance_score=law.get("similarity", 0.0),
                    token_count=self._estimate_tokens(content),
                    metadata={"law_id": law.get("law_id")}
                ))
            
            for precedent in precedent_results:
                content = f"판례: {precedent.get('case_name', '')} ({precedent.get('case_number', '')})"
                items.append(ContextItem(
                    content=content,
                    priority=ContextPriority.MEDIUM,
                    source_type="precedent",
                    relevance_score=precedent.get("similarity", 0.0),
                    token_count=self._estimate_tokens(content),
                    metadata={"case_id": precedent.get("case_id")}
                ))
            
            return ContextWindow(
                items=items,
                total_tokens=sum(item.token_count for item in items),
                max_tokens=self.max_context_tokens,
                utilization_rate=sum(item.token_count for item in items) / self.max_context_tokens,
                priority_distribution={"high": 0, "medium": len(items), "low": 0}
            )
            
        except Exception as e:
            self.logger.error(f"Error creating fallback context: {e}")
            return ContextWindow(
                items=[],
                total_tokens=0,
                max_tokens=self.max_context_tokens,
                utilization_rate=0.0,
                priority_distribution={"high": 0, "medium": 0, "low": 0}
            )
    
    def format_context_for_llm(self, context_window: ContextWindow) -> str:
        """LLM용 컨텍스트 포맷팅"""
        try:
            if not context_window.items:
                return "관련 정보를 찾을 수 없습니다."
            
            formatted_parts = []
            
            # 소스 타입별로 그룹화
            law_items = [item for item in context_window.items if item.source_type == "law"]
            precedent_items = [item for item in context_window.items if item.source_type == "precedent"]
            general_items = [item for item in context_window.items if item.source_type == "general"]
            
            if law_items:
                formatted_parts.append("## 관련 법률")
                for item in law_items:
                    formatted_parts.append(item.content)
                formatted_parts.append("")
            
            if precedent_items:
                formatted_parts.append("## 관련 판례")
                for item in precedent_items:
                    formatted_parts.append(item.content)
                formatted_parts.append("")
            
            if general_items:
                formatted_parts.append("## 기타 정보")
                for item in general_items:
                    formatted_parts.append(item.content)
                formatted_parts.append("")
            
            return "\n".join(formatted_parts)
            
        except Exception as e:
            self.logger.error(f"Error formatting context for LLM: {e}")
            return "컨텍스트 포맷팅 오류"


# 테스트 함수
def test_context_builder():
    """컨텍스트 빌더 테스트"""
    builder = ContextBuilder(max_context_tokens=2000)
    
    # 테스트 데이터
    test_query = "손해배상 청구 방법"
    test_question_classification = QuestionClassification(
        question_type=QuestionType.LEGAL_ADVICE,
        law_weight=0.5,
        precedent_weight=0.5,
        confidence=0.8,
        keywords=["손해배상", "청구"],
        patterns=[]
    )
    
    test_search_results = {
        "law_results": [
            {
                "law_name": "민법",
                "article_number": "제750조",
                "content": "불법행위로 인한 손해배상에 관한 규정",
                "similarity": 0.9,
                "law_id": "law_001"
            },
            {
                "law_name": "민법",
                "article_number": "제751조",
                "content": "정신적 피해에 대한 위자료 청구",
                "similarity": 0.8,
                "law_id": "law_002"
            }
        ],
        "precedent_results": [
            {
                "case_name": "손해배상청구 사건",
                "case_number": "2023다12345",
                "court": "서울중앙지방법원",
                "decision_date": "2023.05.15",
                "summary": "불법행위로 인한 손해배상 청구권 인정",
                "similarity": 0.85,
                "case_id": "case_001"
            }
        ],
        "results": []
    }
    
    print("=== 컨텍스트 빌더 테스트 ===")
    print(f"질문: {test_query}")
    print(f"질문 유형: {test_question_classification.question_type.value}")
    
    try:
        context_window = builder.build_optimized_context(
            query=test_query,
            question_classification=test_question_classification,
            search_results=test_search_results
        )
        
        print(f"\n컨텍스트 윈도우 결과:")
        print(f"- 총 아이템 수: {len(context_window.items)}")
        print(f"- 총 토큰 수: {context_window.total_tokens}")
        print(f"- 최대 토큰 수: {context_window.max_tokens}")
        print(f"- 활용률: {context_window.utilization_rate:.1%}")
        print(f"- 우선순위 분포: {context_window.priority_distribution}")
        
        print(f"\n컨텍스트 아이템:")
        for i, item in enumerate(context_window.items, 1):
            print(f"{i}. [{item.source_type}] {item.priority.value} 우선순위")
            print(f"   토큰: {item.token_count}, 관련성: {item.relevance_score:.2f}")
            print(f"   내용: {item.content[:100]}...")
            print()
        
        # LLM용 포맷팅 테스트
        formatted_context = builder.format_context_for_llm(context_window)
        print(f"LLM용 포맷팅된 컨텍스트:")
        print(formatted_context[:500] + "..." if len(formatted_context) > 500 else formatted_context)
        
    except Exception as e:
        print(f"테스트 실패: {e}")


if __name__ == "__main__":
    test_context_builder()
