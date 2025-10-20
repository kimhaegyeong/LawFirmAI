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
                        # 문법적 완성도 개선
                        improved_replacement = self._improve_grammatical_completeness(
                            match.group(), replacement, resolved_query, context
                        )
                        resolved_query = resolved_query.replace(
                            match.group(), improved_replacement
                        )
                        self.logger.debug(f"Resolved '{match.group()}' to '{improved_replacement}'")
            
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
        """대명사 대체어 찾기 (확장된 컨텍스트 윈도우 버전)"""
        try:
            if not context.turns:
                return None
            
            # 확장된 컨텍스트 윈도우 (최근 5턴)
            recent_turns = context.turns[-5:] if len(context.turns) >= 5 else context.turns
            
            # 향상된 컨텍스트 정보 추출
            enhanced_context = self._extract_enhanced_context(context)
            
            replacement_candidates = []
            
            # 1. 최근 턴에서 직접 추출 (우선순위 조정)
            for turn in reversed(recent_turns):
                # 1순위: 복합 법률 용어 우선 추출 (가장 높은 우선순위)
                complex_terms = self._extract_complex_legal_terms(turn.user_query)
                if complex_terms:
                    # 복합 용어에 우선순위 태그 추가
                    for term in complex_terms:
                        replacement_candidates.append(f"HIGH_PRIORITY:{term}")
                
                # 2순위: 전체 질문에서 핵심 법률 용어 추출
                enhanced_entities = self._extract_enhanced_entities(turn.user_query)
                if enhanced_entities:
                    for entity in enhanced_entities:
                        replacement_candidates.append(f"MEDIUM_PRIORITY:{entity}")
                
                # 3순위: 엔티티에서 대체어 찾기
                if turn.entities:
                    for entity_type in ["laws", "articles", "legal_terms", "precedents"]:
                        entities = turn.entities.get(entity_type, [])
                        if entities:
                            if isinstance(entities, (list, tuple)):
                                for entity in entities:
                                    replacement_candidates.append(f"LOW_PRIORITY:{entity}")
                            elif isinstance(entities, set):
                                for entity in list(entities):
                                    replacement_candidates.append(f"LOW_PRIORITY:{entity}")
                            elif isinstance(entities, dict):
                                for entity in list(entities.keys()):
                                    replacement_candidates.append(f"LOW_PRIORITY:{entity}")
                            else:
                                replacement_candidates.append(f"LOW_PRIORITY:{str(entities)}")
                
                # 4순위: 질문/답변에서 명사구 추출
                text = f"{turn.user_query} {turn.bot_response}"
                nouns = self._extract_nouns(text)
                if nouns:
                    for noun in nouns:
                        replacement_candidates.append(f"LOWEST_PRIORITY:{noun}")
            
            # 2. 향상된 컨텍스트에서 추가 후보 추출
            context_candidates = self._extract_from_enhanced_context(enhanced_context)
            replacement_candidates.extend(context_candidates)
            
            # 3. 도메인 특화 패턴에서 후보 추출
            domain_candidates = self._extract_domain_candidates(context)
            replacement_candidates.extend(domain_candidates)
            
            # 우선순위별로 후보 정리
            priority_candidates = {
                "HIGH_PRIORITY": [],
                "MEDIUM_PRIORITY": [],
                "LOW_PRIORITY": [],
                "LOWEST_PRIORITY": []
            }
            
            seen = set()
            for candidate in replacement_candidates:
                if candidate and len(candidate.strip()) > 0:
                    if ":" in candidate:
                        priority, term = candidate.split(":", 1)
                        if priority in priority_candidates and term not in seen:
                            priority_candidates[priority].append(term.strip())
                            seen.add(term.strip())
                    else:
                        if candidate not in seen:
                            priority_candidates["LOWEST_PRIORITY"].append(candidate.strip())
                            seen.add(candidate.strip())
            
            # 우선순위 순서로 후보 선택
            final_candidates = []
            for priority in ["HIGH_PRIORITY", "MEDIUM_PRIORITY", "LOW_PRIORITY", "LOWEST_PRIORITY"]:
                final_candidates.extend(priority_candidates[priority])
            
            if final_candidates:
                return self._select_best_replacement_improved(final_candidates, pronoun_type, context, priority_candidates)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding pronoun replacement: {e}")
            return None
    
    def _extract_complex_legal_terms(self, text: str) -> List[str]:
        """복합 법률 용어 추출 (강화된 버전)"""
        try:
            # 강화된 복합 법률 용어 패턴들
            complex_patterns = [
                # 손해배상 관련 (최우선)
                r"(손해배상\s*청구\s*방법)",              # 손해배상 청구 방법
                r"(손해배상\s*청구)",                     # 손해배상 청구
                r"(손해배상\s*소송)",                     # 손해배상 소송
                r"(손해배상\s*사건)",                     # 손해배상 사건
                r"(손해배상\s*절차)",                     # 손해배상 절차
                r"(손해배상\s*과실비율)",                 # 손해배상 과실비율
                r"(손해배상\s*불법행위)",                 # 손해배상 불법행위
                
                # 계약 관련 (높은 우선순위)
                r"(매매계약서)",                         # 매매계약서
                r"(임대차계약서)",                       # 임대차계약서
                r"(매매계약)",                           # 매매계약
                r"(임대차계약)",                         # 임대차계약
                r"([가-힣]+계약서)",                     # 기타 계약서
                r"([가-힣]+계약\s*[가-힣]*)",           # 기타 계약
                r"(계약의\s*위험요소)",                  # 계약의 위험요소
                r"(계약서의\s*주의조항)",                # 계약서의 주의조항
                
                # 소송 관련 (높은 우선순위)
                r"(손해배상\s*소송\s*절차)",             # 손해배상 소송 절차
                r"(손해배상\s*소송에\s*필요한\s*서류)",   # 손해배상 소송에 필요한 서류
                r"(민사소송)",                           # 민사소송
                r"(형사소송)",                           # 형사소송
                r"([가-힣]+소송\s*절차)",                # 기타 소송 절차
                r"([가-힣]+소송에\s*필요한\s*[가-힣]+)", # 기타 소송에 필요한 서류
                r"(소송절차)",                           # 소송절차
                r"(소멸시효)",                           # 소멸시효
                
                # 법령 관련 (높은 우선순위)
                r"(민법\s*제\s*\d+조)",                 # 민법 제750조
                r"(형법\s*제\s*\d+조)",                 # 형법 제XXX조
                r"(상법\s*제\s*\d+조)",                 # 상법 제XXX조
                r"([가-힣]+법\s*제\s*\d+조)",           # 기타 법 제XXX조
                r"(제\s*\d+조)",                        # 제750조
                r"(조문)",                              # 조문
                r"(적용범위)",                          # 적용범위
                r"(예외사항)",                          # 예외사항
                r"(법령)",                              # 법령
                
                # 판례 관련
                r"(손해배상\s*판례)",                   # 손해배상 판례
                r"(손해배상\s*사건)",                   # 손해배상 사건
                r"([가-힣]+판례)",                      # 기타 판례
                r"([가-힣]+사건)",                      # 기타 사건
                r"(\d{4}[가나다라마바사아자차카타파하]\d+)",  # 판례번호
                r"(대법원)",                            # 대법원
                r"(고등법원)",                          # 고등법원
                r"(지방법원)",                          # 지방법원
                
                # 절차 관련
                r"(신청절차)",                          # 신청절차
                r"(제출절차)",                          # 제출절차
                r"(처리절차)",                          # 처리절차
                r"([가-힣]+절차)",                      # 기타 절차
                r"(신청)",                              # 신청
                r"(제출)",                              # 제출
                r"(처리)",                              # 처리
                
                # 위험요소 및 주의사항
                r"(위험요소)",                          # 위험요소
                r"(주의조항)",                          # 주의조항
                r"(주의사항)",                          # 주의사항
                r"(개선안)",                            # 개선안
                r"(권고사항)",                          # 권고사항
                r"(개선방안)",                          # 개선방안
                
                # 기타 법률 용어
                r"([가-힣]+의\s*[가-힣]+)",            # 계약의 위험요소 등
                r"([가-힣]+에\s*필요한\s*[가-힣]+)",   # 소송에 필요한 서류 등
                r"([가-힣]+에서\s*[가-힣]+)",          # 계약에서 주의해야 할 등
                r"([가-힣]+를\s*[가-힣]+)",            # 계약을 검토해주세요 등
            ]
            
            terms = []
            for pattern in complex_patterns:
                matches = re.findall(pattern, text)
                terms.extend(matches)
            
            # 중복 제거 및 우선순위 정렬
            unique_terms = []
            seen = set()
            
            # 손해배상 관련 용어 우선
            damage_terms = [term for term in terms if "손해배상" in term]
            for term in damage_terms:
                if term not in seen:
                    unique_terms.append(term)
                    seen.add(term)
            
            # 계약 관련 용어 우선
            contract_terms = [term for term in terms if any(keyword in term for keyword in ["계약", "계약서"])]
            for term in contract_terms:
                if term not in seen:
                    unique_terms.append(term)
                    seen.add(term)
            
            # 소송 관련 용어 우선
            lawsuit_terms = [term for term in terms if "소송" in term]
            for term in lawsuit_terms:
                if term not in seen:
                    unique_terms.append(term)
                    seen.add(term)
            
            # 법령 관련 용어 우선
            legal_terms = [term for term in terms if any(keyword in term for keyword in ["법", "조", "조문"])]
            for term in legal_terms:
                if term not in seen:
                    unique_terms.append(term)
                    seen.add(term)
            
            # 나머지 용어들
            for term in terms:
                if term not in seen:
                    unique_terms.append(term)
                    seen.add(term)
            
            return unique_terms
            
        except Exception as e:
            self.logger.error(f"Error extracting complex legal terms: {e}")
            return []
    
    def _select_best_replacement(self, candidates: List[str], pronoun_type: str, 
                                context: ConversationContext) -> str:
        """개선된 대체어 선택 시스템 (가중치 기반)"""
        try:
            if not candidates:
                return None
            
            # 법률 도메인 특화 가중치
            legal_weights = {
                "복합_법률_용어": 15,      # 손해배상 청구 방법
                "법령_조문": 12,          # 민법 제750조
                "계약서_유형": 10,        # 매매계약서
                "소송_유형": 10,          # 손해배상 소송
                "법률_용어": 8,           # 민법, 형법
                "일반_명사": 3,           # 방법, 절차
                "최근성_보너스": 5,       # 최근 언급
                "길이_보너스": 2          # 구체성
            }
            
            scored_candidates = []
            
            for candidate in candidates:
                score = 0
                
                # 1. 복합 법률 용어 우선순위 (가장 높음)
                if self._is_complex_legal_term(candidate):
                    score += legal_weights["복합_법률_용어"]
                
                # 2. 법령 조문 우선순위
                elif self._is_legal_article(candidate):
                    score += legal_weights["법령_조문"]
                
                # 3. 계약서 유형 우선순위
                elif self._is_contract_type(candidate):
                    score += legal_weights["계약서_유형"]
                
                # 4. 소송 유형 우선순위
                elif self._is_lawsuit_type(candidate):
                    score += legal_weights["소송_유형"]
                
                # 5. 법률 용어 우선순위
                elif candidate.endswith("법"):
                    score += legal_weights["법률_용어"]
                
                # 6. 일반 명사 (낮은 우선순위)
                else:
                    score += legal_weights["일반_명사"]
                
                # 7. 최근성 보너스
                recency_bonus = self._calculate_recency_bonus(candidate, context)
                score += recency_bonus * legal_weights["최근성_보너스"]
                
                # 8. 길이 보너스 (구체성)
                length_bonus = len(candidate.split()) * legal_weights["길이_보너스"]
                score += length_bonus
                
                # 9. 의미적 유사성 보너스
                semantic_bonus = self._calculate_semantic_similarity(candidate, context) * 10
                score += semantic_bonus
                
                # 10. 도메인 특화 패턴 보너스
                domain_bonus = self._calculate_domain_pattern_bonus(candidate, context) * 8
                score += domain_bonus
                
                scored_candidates.append((candidate, score))
            
            # 점수순으로 정렬
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 디버깅을 위한 로그
            self.logger.debug(f"Candidate scores: {scored_candidates[:3]}")
            
            return scored_candidates[0][0] if scored_candidates else candidates[0]
            
        except Exception as e:
            self.logger.error(f"Error in improved replacement selection: {e}")
            return candidates[0] if candidates else None

    def _select_best_replacement_improved(self, candidates: List[str], pronoun_type: str, 
                                        context: ConversationContext, priority_candidates: Dict[str, List[str]]) -> str:
        """개선된 대체어 선택 시스템 (우선순위 기반)"""
        try:
            if not candidates:
                return None
            
            # 우선순위별 가중치
            priority_weights = {
                "HIGH_PRIORITY": 100,    # 복합 법률 용어
                "MEDIUM_PRIORITY": 80,   # 향상된 엔티티
                "LOW_PRIORITY": 60,      # 기존 엔티티
                "LOWEST_PRIORITY": 40    # 명사구
            }
            
            scored_candidates = []
            
            for candidate in candidates:
                score = 0
                
                # 1. 우선순위 점수 (가장 중요)
                candidate_priority = None
                for priority, terms in priority_candidates.items():
                    if candidate in terms:
                        candidate_priority = priority
                        break
                
                if candidate_priority:
                    score += priority_weights.get(candidate_priority, 40)
                
                # 2. 복합 법률 용어 보너스
                if self._is_complex_legal_term(candidate):
                    score += 50
                
                # 3. 법령 조문 보너스
                elif self._is_legal_article(candidate):
                    score += 40
                
                # 4. 계약서/소송 유형 보너스
                elif self._is_contract_type(candidate) or self._is_lawsuit_type(candidate):
                    score += 35
                
                # 5. 법률 용어 보너스
                elif candidate.endswith("법"):
                    score += 30
                
                # 6. 최근성 보너스
                recency_bonus = self._calculate_recency_bonus(candidate, context)
                score += recency_bonus * 20
                
                # 7. 의미적 유사성 보너스
                semantic_bonus = self._calculate_semantic_similarity(candidate, context) * 30
                score += semantic_bonus
                
                # 8. 길이 보너스 (구체성)
                length_bonus = len(candidate.split()) * 5
                score += length_bonus
                
                # 9. 문맥적 연관성 보너스
                contextual_bonus = self._calculate_contextual_relevance(candidate, context) * 25
                score += contextual_bonus
                
                scored_candidates.append((candidate, score))
            
            # 점수순으로 정렬
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 디버깅을 위한 로그
            self.logger.debug(f"Improved candidate scores: {scored_candidates[:3]}")
            
            return scored_candidates[0][0] if scored_candidates else candidates[0]
            
        except Exception as e:
            self.logger.error(f"Error in improved replacement selection: {e}")
            return candidates[0] if candidates else None

    def _calculate_contextual_relevance(self, candidate: str, context: ConversationContext) -> float:
        """문맥적 연관성 계산"""
        try:
            if not context.turns:
                return 0.0
            
            relevance_score = 0.0
            
            # 최근 턴들과의 연관성 계산
            recent_turns = context.turns[-3:] if len(context.turns) >= 3 else context.turns
            
            for i, turn in enumerate(reversed(recent_turns)):
                # 질문과의 연관성
                query_relevance = self._calculate_text_relevance(candidate, turn.user_query)
                relevance_score += query_relevance * (3 - i)  # 최근일수록 높은 가중치
                
                # 답변과의 연관성
                response_relevance = self._calculate_text_relevance(candidate, turn.bot_response)
                relevance_score += response_relevance * (3 - i) * 0.5  # 답변은 낮은 가중치
            
            return min(1.0, relevance_score / len(recent_turns))
            
        except Exception as e:
            self.logger.error(f"Error calculating contextual relevance: {e}")
            return 0.0

    def _calculate_text_relevance(self, candidate: str, text: str) -> float:
        """텍스트와의 연관성 계산"""
        try:
            if not candidate or not text:
                return 0.0
            
            # 공통 단어 기반 연관성
            candidate_words = set(candidate.split())
            text_words = set(text.split())
            
            common_words = candidate_words.intersection(text_words)
            total_words = candidate_words.union(text_words)
            
            if total_words:
                word_similarity = len(common_words) / len(total_words)
            else:
                word_similarity = 0.0
            
            # 부분 문자열 매칭
            substring_match = 0.0
            if candidate in text:
                substring_match = 0.8
            elif any(word in text for word in candidate_words if len(word) > 1):
                substring_match = 0.4
            
            # 최종 연관성 점수
            relevance = max(word_similarity, substring_match)
            
            return relevance
            
        except Exception as e:
            self.logger.error(f"Error calculating text relevance: {e}")
            return 0.0

    def _is_complex_legal_term(self, candidate: str) -> bool:
        """복합 법률 용어 여부 확인"""
        complex_patterns = [
            r".*청구.*방법.*",      # 손해배상 청구 방법
            r".*계약서.*",          # 매매계약서
            r".*소송.*",            # 손해배상 소송
            r".*사건.*",            # 손해배상 사건
            r".*절차.*",            # 소송 절차
            r".*과실비율.*",        # 과실비율
            r".*위험요소.*",        # 위험요소
            r".*주의조항.*",        # 주의조항
            r".*소멸시효.*",        # 소멸시효
            r".*적용범위.*",        # 적용범위
            r".*예외사항.*",        # 예외사항
        ]
        
        for pattern in complex_patterns:
            if re.match(pattern, candidate):
                return True
        return False

    def _is_legal_article(self, candidate: str) -> bool:
        """법령 조문 여부 확인"""
        return "제" in candidate and "조" in candidate

    def _is_contract_type(self, candidate: str) -> bool:
        """계약서 유형 여부 확인"""
        return "계약서" in candidate

    def _is_lawsuit_type(self, candidate: str) -> bool:
        """소송 유형 여부 확인"""
        return "소송" in candidate

    def _calculate_recency_bonus(self, candidate: str, context: ConversationContext) -> float:
        """최근성 보너스 계산"""
        try:
            bonus = 0.0
            recent_turns = context.turns[-3:] if len(context.turns) >= 3 else context.turns
            
            for i, turn in enumerate(reversed(recent_turns)):
                if candidate in turn.user_query or candidate in turn.bot_response:
                    bonus += (3 - i) * 0.5  # 최근일수록 높은 보너스
                    break
            
            return bonus
            
        except Exception as e:
            self.logger.error(f"Error calculating recency bonus: {e}")
            return 0.0

    def _calculate_semantic_similarity(self, candidate: str, context: ConversationContext) -> float:
        """의미적 유사성 계산"""
        try:
            if not context.turns:
                return 0.0
            
            # 법률 도메인 특화 유사성 계산
            similarity_score = 0.0
            
            # 1. 법률 용어 사전 기반 매칭
            legal_term_similarity = self._calculate_legal_term_similarity(candidate, context)
            similarity_score += legal_term_similarity * 0.4
            
            # 2. 문맥적 유사성 계산
            contextual_similarity = self._calculate_contextual_similarity(candidate, context)
            similarity_score += contextual_similarity * 0.3
            
            # 3. 구문적 유사성 계산
            syntactic_similarity = self._calculate_syntactic_similarity(candidate, context)
            similarity_score += syntactic_similarity * 0.3
            
            return min(1.0, similarity_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0

    def _calculate_legal_term_similarity(self, candidate: str, context: ConversationContext) -> float:
        """법률 용어 유사성 계산"""
        try:
            # 법률 용어 계층 구조 기반 유사성
            legal_hierarchy = {
                "손해배상": ["청구", "소송", "사건", "절차", "과실비율", "불법행위"],
                "계약": ["계약서", "매매", "임대차", "위험요소", "주의조항", "해지"],
                "민법": ["제750조", "적용범위", "예외사항", "판례", "조문"],
                "소송": ["절차", "서류", "소멸시효", "제기", "진행"],
                "법령": ["조문", "항", "호", "적용", "해석"]
            }
            
            # 계층 구조에서의 거리 계산
            for main_term, related_terms in legal_hierarchy.items():
                if main_term in candidate:
                    for related_term in related_terms:
                        if related_term in context.turns[-1].user_query:
                            return 0.9  # 높은 유사성
                        elif any(related_term in turn.user_query for turn in context.turns[-2:]):
                            return 0.7  # 중간 유사성
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating legal term similarity: {e}")
            return 0.0

    def _calculate_contextual_similarity(self, candidate: str, context: ConversationContext) -> float:
        """문맥적 유사성 계산"""
        try:
            if not context.turns:
                return 0.0
            
            # 최근 턴과의 문맥적 유사성
            recent_turn = context.turns[-1]
            recent_text = f"{recent_turn.user_query} {recent_turn.bot_response}"
            
            # 공통 키워드 기반 유사성
            common_keywords = 0
            candidate_words = set(candidate.split())
            recent_words = set(recent_text.split())
            
            common_keywords = len(candidate_words.intersection(recent_words))
            total_words = len(candidate_words.union(recent_words))
            
            if total_words > 0:
                return common_keywords / total_words
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating contextual similarity: {e}")
            return 0.0

    def _calculate_syntactic_similarity(self, candidate: str, context: ConversationContext) -> float:
        """구문적 유사성 계산"""
        try:
            if not context.turns:
                return 0.0
            
            # 구문 패턴 기반 유사성
            syntactic_patterns = [
                r"([가-힣]+의\s*[가-힣]+)",      # 명사 + 의 + 명사
                r"([가-힣]+에\s*[가-힣]+)",      # 명사 + 에 + 명사
                r"([가-힣]+를\s*[가-힣]+)",      # 명사 + 를 + 명사
                r"([가-힣]+에서\s*[가-힣]+)",    # 명사 + 에서 + 명사
            ]
            
            candidate_patterns = []
            for pattern in syntactic_patterns:
                if re.search(pattern, candidate):
                    candidate_patterns.append(pattern)
            
            if not candidate_patterns:
                return 0.0
            
            # 최근 턴에서 동일한 패턴 찾기
            recent_turn = context.turns[-1]
            recent_text = f"{recent_turn.user_query} {recent_turn.bot_response}"
            
            matching_patterns = 0
            for pattern in candidate_patterns:
                if re.search(pattern, recent_text):
                    matching_patterns += 1
            
            return matching_patterns / len(candidate_patterns)
            
        except Exception as e:
            self.logger.error(f"Error calculating syntactic similarity: {e}")
            return 0.0

    def _calculate_domain_pattern_bonus(self, candidate: str, context: ConversationContext) -> float:
        """도메인 특화 패턴 보너스 계산"""
        try:
            # 도메인 특화 패턴들
            domain_patterns = self._extract_domain_specific_patterns(candidate)
            
            if not domain_patterns:
                return 0.0
            
            # 패턴 신뢰도 계산
            total_confidence = 0.0
            for pattern in domain_patterns:
                confidence = self._calculate_pattern_confidence(pattern, "general")
                total_confidence += confidence
            
            return total_confidence / len(domain_patterns) if domain_patterns else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating domain pattern bonus: {e}")
            return 0.0

    def _extract_domain_specific_patterns(self, text: str) -> List[str]:
        """도메인 특화 패턴 추출 (확장된 버전)"""
        try:
            # 법률 도메인 특화 패턴들 (확장)
            domain_patterns = {
                "damage_compensation": [
                    r"([가-힣]+손해배상\s*[가-힣]*)",           # 손해배상 청구
                    r"([가-힣]+과실비율)",                      # 과실비율
                    r"([가-힣]+불법행위)",                      # 불법행위
                    r"([가-힣]+청구\s*방법)",                   # 청구 방법
                    r"([가-힣]+손해배상\s*사건)",               # 손해배상 사건
                ],
                "contract_law": [
                    r"([가-힣]+계약서)",                        # 매매계약서
                    r"([가-힣]+계약\s*[가-힣]*)",              # 계약 해지
                    r"([가-힣]+위험요소)",                      # 위험요소
                    r"([가-힣]+주의조항)",                      # 주의조항
                    r"([가-힣]+계약의\s*[가-힣]+)",            # 계약의 위험요소
                    r"([가-힣]+계약서의\s*[가-힣]+)",          # 계약서의 주의조항
                    r"([가-힣]+매매계약)",                     # 매매계약
                    r"([가-힣]+임대차계약)",                   # 임대차계약
                ],
                "legal_provisions": [
                    r"([가-힣]+법\s*제\s*\d+조)",              # 민법 제750조
                    r"([가-힣]+제\s*\d+조)",                   # 제750조
                    r"([가-힣]+적용범위)",                      # 적용범위
                    r"([가-힣]+예외사항)",                      # 예외사항
                    r"([가-힣]+조문)",                         # 조문
                    r"([가-힣]+법령)",                         # 법령
                    r"([가-힣]+민법)",                         # 민법
                    r"([가-힣]+형법)",                         # 형법
                    r"([가-힣]+상법)",                         # 상법
                ],
                "litigation": [
                    r"([가-힣]+소송\s*[가-힣]*)",              # 손해배상 소송
                    r"([가-힣]+절차)",                         # 소송 절차
                    r"([가-힣]+서류)",                         # 필요 서류
                    r"([가-힣]+소멸시효)",                     # 소멸시효
                    r"([가-힣]+소송의\s*[가-힣]+)",            # 소송의 절차
                    r"([가-힣]+소송에\s*필요한\s*[가-힣]+)",  # 소송에 필요한 서류
                    r"([가-힣]+소송절차)",                     # 소송절차
                    r"([가-힣]+민사소송)",                     # 민사소송
                    r"([가-힣]+형사소송)",                     # 형사소송
                ],
                "precedents": [
                    r"([가-힣]+판례)",                         # 판례
                    r"([가-힣]+사건)",                         # 사건
                    r"(\d{4}[가나다라마바사아자차카타파하]\d+)",  # 판례번호
                    r"([가-힣]+대법원)",                       # 대법원
                    r"([가-힣]+고등법원)",                     # 고등법원
                    r"([가-힣]+지방법원)",                     # 지방법원
                ],
                "procedures": [
                    r"([가-힣]+신청)",                         # 신청
                    r"([가-힣]+제출)",                         # 제출
                    r"([가-힣]+처리)",                         # 처리
                    r"([가-힣]+절차)",                         # 절차
                    r"([가-힣]+신청절차)",                     # 신청절차
                    r"([가-힣]+제출절차)",                     # 제출절차
                ],
                "improvements": [
                    r"([가-힣]+개선안)",                       # 개선안
                    r"([가-힣]+권고사항)",                     # 권고사항
                    r"([가-힣]+주의사항)",                     # 주의사항
                    r"([가-힣]+개선방안)",                     # 개선방안
                ]
            }
            
            extracted_patterns = []
            
            for domain, patterns in domain_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        extracted_patterns.append({
                            "domain": domain,
                            "pattern": match,
                            "confidence": self._calculate_pattern_confidence(match, domain)
                        })
            
            # 신뢰도순으로 정렬
            extracted_patterns.sort(key=lambda x: x["confidence"], reverse=True)
            
            return [pattern["pattern"] for pattern in extracted_patterns]
            
        except Exception as e:
            self.logger.error(f"Error extracting domain-specific patterns: {e}")
            return []

    def _calculate_pattern_confidence(self, pattern: str, domain: str) -> float:
        """패턴 신뢰도 계산 (개선된 버전)"""
        try:
            confidence = 0.5  # 기본 신뢰도
            
            # 도메인별 가중치 (확장)
            domain_weights = {
                "damage_compensation": 0.9,
                "contract_law": 0.8,
                "legal_provisions": 0.95,
                "litigation": 0.85,
                "precedents": 0.8,
                "procedures": 0.7,
                "improvements": 0.6
            }
            
            confidence *= domain_weights.get(domain, 0.5)
            
            # 패턴 길이에 따른 보너스
            if len(pattern.split()) > 2:
                confidence += 0.1
            
            # 복합 용어 보너스
            if any(keyword in pattern for keyword in ["계약서", "청구", "소송", "사건", "절차"]):
                confidence += 0.15
            
            # 법령 조문 보너스
            if "제" in pattern and "조" in pattern:
                confidence += 0.2
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern confidence: {e}")
            return 0.0

    def _improve_grammatical_completeness(self, pronoun: str, replacement: str, 
                                         query: str, context: ConversationContext) -> str:
        """문법적 완성도 개선"""
        try:
            if not replacement:
                return replacement
            
            # 조사 처리 개선
            improved_replacement = self._adjust_particles(pronoun, replacement, query)
            
            # 어미 처리 개선
            improved_replacement = self._adjust_endings(pronoun, improved_replacement, query)
            
            # 문맥적 자연스러움 개선
            improved_replacement = self._improve_contextual_naturalness(
                pronoun, improved_replacement, query, context
            )
            
            return improved_replacement
            
        except Exception as e:
            self.logger.error(f"Error improving grammatical completeness: {e}")
            return replacement

    def _adjust_particles(self, pronoun: str, replacement: str, query: str) -> str:
        """조사 조정"""
        try:
            # 조사 매핑 규칙
            particle_rules = {
                "그것의": "의",
                "이것의": "의", 
                "그의": "의",
                "이의": "의",
                "그것에": "에",
                "이것에": "에",
                "그에": "에",
                "이에": "에",
                "그것을": "을",
                "이것을": "을",
                "그를": "을",
                "이를": "을",
                "그것에서": "에서",
                "이것에서": "에서",
                "그에서": "에서",
                "이에서": "에서"
            }
            
            # 대명사에서 조사 추출
            pronoun_particle = None
            for key, particle in particle_rules.items():
                if pronoun.startswith(key.replace(particle, "")):
                    pronoun_particle = particle
                    break
            
            if pronoun_particle:
                # 대체어에 적절한 조사 추가
                if not replacement.endswith(pronoun_particle):
                    # 조사가 없으면 추가
                    if not any(replacement.endswith(p) for p in ["의", "에", "을", "에서", "는", "은", "가", "이"]):
                        improved = replacement + pronoun_particle
                        return improved
                    else:
                        # 기존 조사를 적절한 조사로 변경
                        return self._replace_particle(replacement, pronoun_particle)
            
            return replacement
            
        except Exception as e:
            self.logger.error(f"Error adjusting particles: {e}")
            return replacement

    def _replace_particle(self, text: str, target_particle: str) -> str:
        """조사 교체"""
        try:
            particles = ["의", "에", "을", "에서", "는", "은", "가", "이", "로", "으로"]
            
            for particle in particles:
                if text.endswith(particle):
                    return text[:-len(particle)] + target_particle
            
            return text + target_particle
            
        except Exception as e:
            self.logger.error(f"Error replacing particle: {e}")
            return text

    def _adjust_endings(self, pronoun: str, replacement: str, query: str) -> str:
        """어미 조정"""
        try:
            # 문장 끝 어미 패턴
            ending_patterns = [
                r"([가-힣]+)을\s*알려주세요$",
                r"([가-힣]+)을\s*찾아주세요$",
                r"([가-힣]+)을\s*제시해주세요$",
                r"([가-힣]+)을\s*검토해주세요$",
                r"([가-힣]+)은\s*무엇인가요\?$",
                r"([가-힣]+)은\s*어떻게\s*[가-힣]+나요\?$",
                r"([가-힣]+)은\s*언제까지인가요\?$"
            ]
            
            for pattern in ending_patterns:
                if re.search(pattern, query):
                    # 문장 패턴에 맞게 어미 조정
                    if "알려주세요" in query and not replacement.endswith("을"):
                        if not any(replacement.endswith(p) for p in ["을", "를"]):
                            replacement += "을"
                    elif "찾아주세요" in query and not replacement.endswith("을"):
                        if not any(replacement.endswith(p) for p in ["을", "를"]):
                            replacement += "을"
                    elif "제시해주세요" in query and not replacement.endswith("을"):
                        if not any(replacement.endswith(p) for p in ["을", "를"]):
                            replacement += "을"
                    elif "검토해주세요" in query and not replacement.endswith("을"):
                        if not any(replacement.endswith(p) for p in ["을", "를"]):
                            replacement += "을"
                    elif "무엇인가요" in query and not replacement.endswith("은"):
                        if not any(replacement.endswith(p) for p in ["은", "는"]):
                            replacement += "은"
                    elif "어떻게" in query and not replacement.endswith("은"):
                        if not any(replacement.endswith(p) for p in ["은", "는"]):
                            replacement += "은"
                    elif "언제까지" in query and not replacement.endswith("은"):
                        if not any(replacement.endswith(p) for p in ["은", "는"]):
                            replacement += "은"
            
            return replacement
            
        except Exception as e:
            self.logger.error(f"Error adjusting endings: {e}")
            return replacement

    def _improve_contextual_naturalness(self, pronoun: str, replacement: str, 
                                      query: str, context: ConversationContext) -> str:
        """문맥적 자연스러움 개선"""
        try:
            # 불필요한 중복 제거
            if replacement in query:
                return replacement
            
            # 문맥에서 자연스러운 형태로 조정
            contextual_adjustments = {
                "손해배상 청구 방법": "손해배상 청구",
                "매매계약서를 검토해주세요": "매매계약서",
                "민법 제750조의 내용": "민법 제750조",
                "손해배상 소송을 제기하려고 합니다": "손해배상 소송"
            }
            
            for full_form, short_form in contextual_adjustments.items():
                if replacement == full_form:
                    # 문맥에 따라 적절한 형태 선택
                    if "의" in pronoun or "에" in pronoun or "을" in pronoun:
                        return short_form
                    else:
                        return replacement
            
            return replacement
            
        except Exception as e:
            self.logger.error(f"Error improving contextual naturalness: {e}")
            return replacement

    def _extract_enhanced_entities(self, text: str) -> List[str]:
        """향상된 엔티티 추출 (복합 용어 우선)"""
        try:
            entities = []
            
            # 1. 복합 법률 용어 우선 추출 (이미 구현된 메서드 활용)
            complex_terms = self._extract_complex_legal_terms(text)
            entities.extend(complex_terms)
            
            # 2. 문맥적 법률 용어 추출
            contextual_terms = self._extract_contextual_legal_terms(text)
            entities.extend(contextual_terms)
            
            # 3. 핵심 법률 개념 추출
            core_concepts = self._extract_core_legal_concepts(text)
            entities.extend(core_concepts)
            
            # 중복 제거 및 우선순위 정렬
            unique_entities = []
            seen = set()
            
            # 복합 용어 우선
            for term in complex_terms:
                if term not in seen:
                    unique_entities.append(term)
                    seen.add(term)
            
            # 문맥적 용어 추가
            for term in contextual_terms:
                if term not in seen:
                    unique_entities.append(term)
                    seen.add(term)
            
            # 핵심 개념 추가
            for term in core_concepts:
                if term not in seen:
                    unique_entities.append(term)
                    seen.add(term)
            
            return unique_entities
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced entities: {e}")
            return []

    def _extract_contextual_legal_terms(self, text: str) -> List[str]:
        """문맥적 법률 용어 추출"""
        try:
            contextual_patterns = [
                # 손해배상 관련 문맥
                r"([가-힣]+손해배상\s*[가-힣]*)",           # 손해배상 청구, 손해배상 소송
                r"손해배상\s*([가-힣]+)",                    # 손해배상 청구, 손해배상 소송
                
                # 계약 관련 문맥
                r"([가-힣]+계약서)",                        # 매매계약서, 임대차계약서
                r"([가-힣]+계약\s*[가-힣]*)",              # 매매계약, 임대차계약
                
                # 소송 관련 문맥
                r"([가-힣]+소송\s*[가-힣]*)",              # 손해배상 소송, 민사소송
                r"소송\s*([가-힣]+)",                       # 소송 절차, 소송 제기
                
                # 법령 관련 문맥
                r"([가-힣]+법\s*제\s*\d+조)",              # 민법 제750조
                r"제\s*(\d+조)",                           # 제750조
                
                # 판례 관련 문맥
                r"([가-힣]+판례)",                         # 손해배상 판례
                r"([가-힣]+사건)",                         # 손해배상 사건
            ]
            
            terms = []
            for pattern in contextual_patterns:
                matches = re.findall(pattern, text)
                terms.extend(matches)
            
            return terms
            
        except Exception as e:
            self.logger.error(f"Error extracting contextual legal terms: {e}")
            return []

    def _extract_core_legal_concepts(self, text: str) -> List[str]:
        """핵심 법률 개념 추출"""
        try:
            # 법률 도메인 핵심 개념 사전
            core_concepts = {
                "손해배상": ["손해배상", "손해배상청구", "손해배상소송", "손해배상사건"],
                "계약": ["계약", "계약서", "매매계약", "임대차계약", "매매계약서", "임대차계약서"],
                "소송": ["소송", "민사소송", "형사소송", "소송절차", "소송제기"],
                "법령": ["민법", "형법", "상법", "법령", "조문"],
                "판례": ["판례", "사건", "대법원", "고등법원", "지방법원"],
                "절차": ["절차", "신청", "제출", "처리", "신청절차"],
                "개선": ["개선안", "권고사항", "주의사항", "개선방안"]
            }
            
            found_concepts = []
            
            for concept_type, concepts in core_concepts.items():
                for concept in concepts:
                    if concept in text:
                        # 문맥에서 더 구체적인 용어 찾기
                        context_pattern = rf"([가-힣]*{concept}[가-힣]*)"
                        matches = re.findall(context_pattern, text)
                        if matches:
                            found_concepts.extend(matches)
                        else:
                            found_concepts.append(concept)
            
            return found_concepts
            
        except Exception as e:
            self.logger.error(f"Error extracting core legal concepts: {e}")
            return []

    def _extract_enhanced_context(self, context: ConversationContext) -> Dict[str, Any]:
        """향상된 컨텍스트 추출"""
        try:
            enhanced_context = {
                "recent_topics": [],
                "legal_entities": {},
                "conversation_flow": [],
                "semantic_clusters": [],
                "temporal_weights": {}
            }
            
            # 확장된 컨텍스트 윈도우 (최근 5턴)
            recent_turns = context.turns[-5:] if len(context.turns) >= 5 else context.turns
            
            # 1. 법률 엔티티 클러스터링
            legal_entities = self._cluster_legal_entities(recent_turns)
            enhanced_context["legal_entities"] = legal_entities
            
            # 2. 대화 흐름 분석
            conversation_flow = self._analyze_conversation_flow(recent_turns)
            enhanced_context["conversation_flow"] = conversation_flow
            
            # 3. 의미적 클러스터 생성
            semantic_clusters = self._create_semantic_clusters(recent_turns)
            enhanced_context["semantic_clusters"] = semantic_clusters
            
            # 4. 시간적 가중치 계산
            temporal_weights = self._calculate_temporal_weights(recent_turns)
            enhanced_context["temporal_weights"] = temporal_weights
            
            return enhanced_context
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced context: {e}")
            return {}

    def _cluster_legal_entities(self, turns: List[ConversationTurn]) -> Dict[str, List[str]]:
        """법률 엔티티 클러스터링"""
        try:
            clusters = {
                "damage_compensation": [],    # 손해배상 관련
                "contract": [],               # 계약 관련
                "legal_provisions": [],       # 법령 조문 관련
                "lawsuit": [],               # 소송 관련
                "precedents": [],            # 판례 관련
                "procedures": [],            # 절차 관련
                "improvements": []           # 개선사항 관련
            }
            
            for turn in turns:
                if turn.entities:
                    for entity_type, entities in turn.entities.items():
                        for entity in entities:
                            entity_str = str(entity)
                            
                            # 손해배상 관련 클러스터
                            if any(keyword in entity_str for keyword in ["손해배상", "청구", "과실비율", "불법행위"]):
                                clusters["damage_compensation"].append(entity_str)
                            
                            # 계약 관련 클러스터
                            elif any(keyword in entity_str for keyword in ["계약", "매매", "임대차", "계약서"]):
                                clusters["contract"].append(entity_str)
                            
                            # 법령 조문 관련 클러스터
                            elif "제" in entity_str and "조" in entity_str:
                                clusters["legal_provisions"].append(entity_str)
                            
                            # 소송 관련 클러스터
                            elif "소송" in entity_str:
                                clusters["lawsuit"].append(entity_str)
                            
                            # 판례 관련 클러스터
                            elif any(keyword in entity_str for keyword in ["판례", "사건", "대법원", "고등법원"]):
                                clusters["precedents"].append(entity_str)
                            
                            # 절차 관련 클러스터
                            elif any(keyword in entity_str for keyword in ["절차", "신청", "제출", "처리"]):
                                clusters["procedures"].append(entity_str)
                            
                            # 개선사항 관련 클러스터
                            elif any(keyword in entity_str for keyword in ["개선안", "권고사항", "주의사항"]):
                                clusters["improvements"].append(entity_str)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error clustering legal entities: {e}")
            return {}

    def _analyze_conversation_flow(self, turns: List[ConversationTurn]) -> List[str]:
        """대화 흐름 분석"""
        try:
            flow_patterns = []
            
            for i, turn in enumerate(turns):
                # 질문 유형 분석
                question_type = self._classify_question_type(turn.user_query)
                flow_patterns.append(question_type)
            
            return flow_patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation flow: {e}")
            return []

    def _classify_question_type(self, query: str) -> str:
        """질문 유형 분류"""
        try:
            if "방법" in query or "어떻게" in query:
                return "method"
            elif "무엇" in query or "뭐" in query:
                return "what"
            elif "언제" in query:
                return "when"
            elif "어디" in query:
                return "where"
            elif "왜" in query or "이유" in query:
                return "why"
            elif "누구" in query:
                return "who"
            elif "검토" in query or "검토해" in query:
                return "review"
            elif "찾아" in query or "검색" in query:
                return "search"
            else:
                return "general"
                
        except Exception as e:
            self.logger.error(f"Error classifying question type: {e}")
            return "general"

    def _create_semantic_clusters(self, turns: List[ConversationTurn]) -> List[Dict[str, Any]]:
        """의미적 클러스터 생성"""
        try:
            clusters = []
            
            # 법률 도메인별 클러스터
            domain_clusters = {
                "civil_law": ["민법", "계약", "손해배상", "불법행위"],
                "contract_law": ["계약서", "매매", "임대차", "위험요소"],
                "litigation": ["소송", "절차", "서류", "소멸시효"],
                "legal_provisions": ["제750조", "적용범위", "예외사항"],
                "precedents": ["판례", "사건", "대법원", "고등법원"]
            }
            
            for domain, keywords in domain_clusters.items():
                cluster_entities = []
                for turn in turns:
                    for keyword in keywords:
                        if keyword in turn.user_query or keyword in turn.bot_response:
                            cluster_entities.append(keyword)
                
                if cluster_entities:
                    clusters.append({
                        "domain": domain,
                        "entities": list(set(cluster_entities)),
                        "strength": len(cluster_entities) / len(turns)
                    })
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error creating semantic clusters: {e}")
            return []

    def _calculate_temporal_weights(self, turns: List[ConversationTurn]) -> Dict[str, float]:
        """시간적 가중치 계산"""
        try:
            weights = {}
            
            for i, turn in enumerate(turns):
                # 최근일수록 높은 가중치
                weight = (len(turns) - i) / len(turns)
                weights[f"turn_{i}"] = weight
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal weights: {e}")
            return {}

    def _extract_from_enhanced_context(self, enhanced_context: Dict[str, Any]) -> List[str]:
        """향상된 컨텍스트에서 후보 추출"""
        try:
            candidates = []
            
            # 법률 엔티티 클러스터에서 추출
            legal_entities = enhanced_context.get("legal_entities", {})
            for cluster_name, entities in legal_entities.items():
                candidates.extend(entities)
            
            # 의미적 클러스터에서 추출
            semantic_clusters = enhanced_context.get("semantic_clusters", [])
            for cluster in semantic_clusters:
                candidates.extend(cluster.get("entities", []))
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error extracting from enhanced context: {e}")
            return []

    def _extract_domain_candidates(self, context: ConversationContext) -> List[str]:
        """도메인 특화 후보 추출"""
        try:
            candidates = []
            
            # 최근 턴에서 도메인 특화 패턴 추출
            recent_turns = context.turns[-3:] if len(context.turns) >= 3 else context.turns
            
            for turn in recent_turns:
                domain_patterns = self._extract_domain_specific_patterns(turn.user_query)
                candidates.extend(domain_patterns)
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error extracting domain candidates: {e}")
            return []

    def _extract_nouns(self, text: str) -> List[str]:
        """텍스트에서 명사 추출 (개선된 버전)"""
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

