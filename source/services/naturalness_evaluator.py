# -*- coding: utf-8 -*-
"""
자연스러움 평가 시스템
답변의 자연스러움과 대화 품질을 평가하는 시스템
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class NaturalnessLevel(Enum):
    """자연스러움 레벨"""
    VERY_NATURAL = "very_natural"
    NATURAL = "natural"
    MODERATE = "moderate"
    UNNATURAL = "unnatural"
    VERY_UNNATURAL = "very_unnatural"


class EvaluationCategory(Enum):
    """평가 카테고리"""
    FORMALITY = "formality"
    CONVERSATION_FLOW = "conversation_flow"
    EMOTIONAL_APPROPRIATENESS = "emotional_appropriateness"
    PERSONALIZATION = "personalization"
    READABILITY = "readability"


@dataclass
class NaturalnessMetrics:
    """자연스러움 메트릭 데이터 클래스"""
    formality_score: float
    conversation_flow_score: float
    emotional_appropriateness: float
    personalization_score: float
    readability_score: float
    overall_naturalness: float
    evaluation_timestamp: str
    detailed_analysis: Dict[str, Any]


class NaturalnessEvaluator:
    """답변 자연스러움 평가"""
    
    def __init__(self):
        """자연스러움 평가기 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 자연스러움 평가 기준
        self.evaluation_criteria = {
            "formality": {
                "natural_patterns": [
                    r"~예요", r"~해요", r"~되요", r"~이에요",
                    r"질문하신", r"말씀하신", r"문의하신",
                    r"~하시면 됩니다", r"~해 보세요", r"~하시면 돼요"
                ],
                "formal_patterns": [
                    r"~입니다", r"~합니다", r"~됩니다", r"~이옵니다",
                    r"귀하", r"당신", r"~하시기 바랍니다",
                    r"~하여야 합니다", r"~하시어야 합니다"
                ],
                "weights": {
                    "natural": 1.0,
                    "formal": 0.3,
                    "neutral": 0.7
                }
            },
            "conversation_flow": {
                "connectors": [
                    "그럼", "그렇다면", "그러면", "그래서",
                    "또한", "그리고", "추가로", "한 가지 더",
                    "앞서 말씀하신", "이전에 언급한", "방금 말한"
                ],
                "transitions": [
                    "이제", "다음으로", "또 다른", "추가로",
                    "정리하면", "요약하면", "핵심은", "중요한 포인트는"
                ],
                "weights": {
                    "has_connectors": 0.8,
                    "has_transitions": 0.6,
                    "smooth_flow": 1.0
                }
            },
            "emotional_appropriateness": {
                "empathy_expressions": [
                    "이해하시는 마음이에요", "답답하시겠어요", "속상하시겠어요",
                    "걱정하지 마세요", "괜찮을 거예요", "안심하세요"
                ],
                "encouraging_expressions": [
                    "좋은 질문이네요", "잘 물어보셨어요", "정확한 포인트입니다",
                    "핵심을 짚으셨네요", "훌륭한 질문이에요"
                ],
                "supportive_expressions": [
                    "차근차근 설명드릴게요", "이해하기 쉽게 말씀드릴게요",
                    "단계별로 알아보죠", "천천히 설명해드릴게요"
                ],
                "weights": {
                    "has_empathy": 0.9,
                    "has_encouragement": 0.7,
                    "has_support": 0.8
                }
            },
            "personalization": {
                "personal_references": [
                    "질문하신", "말씀하신", "문의하신", "상황을 보면",
                    "귀하의 경우", "이런 상황에서는", "이런 경우에는"
                ],
                "contextual_adaptations": [
                    "상황에 따라", "경우에 따라", "상황을 고려하면",
                    "이런 경우", "이런 상황에서"
                ],
                "weights": {
                    "has_personal_ref": 0.8,
                    "has_contextual": 0.7,
                    "adaptive": 0.9
                }
            },
            "readability": {
                "sentence_length_optimal": (10, 30),  # 최적 문장 길이
                "paragraph_length_optimal": (2, 5),   # 최적 문단 길이
                "complex_words": [
                    "법률", "법령", "조문", "판례", "소송", "계약",
                    "손해배상", "불법행위", "소유권", "임대차"
                ],
                "simplification_indicators": [
                    "쉽게 말하면", "간단히 말하면", "즉,", "다시 말해",
                    "예를 들어", "구체적으로", "실제로는"
                ],
                "weights": {
                    "optimal_length": 0.8,
                    "has_simplification": 0.9,
                    "appropriate_complexity": 0.7
                }
            }
        }
        
        # 평가 가중치
        self.category_weights = {
            EvaluationCategory.FORMALITY: 0.25,
            EvaluationCategory.CONVERSATION_FLOW: 0.20,
            EvaluationCategory.EMOTIONAL_APPROPRIATENESS: 0.20,
            EvaluationCategory.PERSONALIZATION: 0.15,
            EvaluationCategory.READABILITY: 0.20
        }
        
        # 자연스러움 임계값
        self.naturalness_thresholds = {
            NaturalnessLevel.VERY_NATURAL: 0.9,
            NaturalnessLevel.NATURAL: 0.7,
            NaturalnessLevel.MODERATE: 0.5,
            NaturalnessLevel.UNNATURAL: 0.3,
            NaturalnessLevel.VERY_UNNATURAL: 0.0
        }
        
        self.logger.info("NaturalnessEvaluator initialized")
    
    def evaluate_naturalness(self, answer: str, context: Dict[str, Any]) -> NaturalnessMetrics:
        """
        답변의 자연스러움 평가
        
        Args:
            answer: 평가할 답변
            context: 대화 맥락 정보
            
        Returns:
            NaturalnessMetrics: 자연스러움 평가 결과
        """
        try:
            if not answer or not isinstance(answer, str):
                return self._create_default_metrics()
            
            # 각 카테고리별 평가
            formality_score = self._evaluate_formality(answer, context)
            conversation_flow_score = self._evaluate_conversation_flow(answer, context)
            emotional_appropriateness = self._evaluate_emotional_appropriateness(answer, context)
            personalization_score = self._evaluate_personalization(answer, context)
            readability_score = self._evaluate_readability(answer, context)
            
            # 전체 자연스러움 점수 계산
            overall_naturalness = self._calculate_overall_score({
                EvaluationCategory.FORMALITY: formality_score,
                EvaluationCategory.CONVERSATION_FLOW: conversation_flow_score,
                EvaluationCategory.EMOTIONAL_APPROPRIATENESS: emotional_appropriateness,
                EvaluationCategory.PERSONALIZATION: personalization_score,
                EvaluationCategory.READABILITY: readability_score
            })
            
            # 상세 분석 생성
            detailed_analysis = self._generate_detailed_analysis(
                answer, formality_score, conversation_flow_score,
                emotional_appropriateness, personalization_score, readability_score
            )
            
            return NaturalnessMetrics(
                formality_score=formality_score,
                conversation_flow_score=conversation_flow_score,
                emotional_appropriateness=emotional_appropriateness,
                personalization_score=personalization_score,
                readability_score=readability_score,
                overall_naturalness=overall_naturalness,
                evaluation_timestamp=datetime.now().isoformat(),
                detailed_analysis=detailed_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating naturalness: {e}")
            return self._create_default_metrics()
    
    def _evaluate_formality(self, answer: str, context: Dict[str, Any]) -> float:
        """형식성 평가"""
        try:
            criteria = self.evaluation_criteria["formality"]
            natural_patterns = criteria["natural_patterns"]
            formal_patterns = criteria["formal_patterns"]
            weights = criteria["weights"]
            
            # 패턴 매칭
            natural_count = sum(len(re.findall(pattern, answer)) for pattern in natural_patterns)
            formal_count = sum(len(re.findall(pattern, answer)) for pattern in formal_patterns)
            
            # 점수 계산
            if natural_count > formal_count:
                score = weights["natural"]
            elif formal_count > natural_count:
                score = weights["formal"]
            else:
                score = weights["neutral"]
            
            # 사용자 선호도 반영
            user_preference = context.get("user_preference", "medium")
            if user_preference == "casual" and natural_count > 0:
                score += 0.1
            elif user_preference == "formal" and formal_count > 0:
                score += 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Error evaluating formality: {e}")
            return 0.5
    
    def _evaluate_conversation_flow(self, answer: str, context: Dict[str, Any]) -> float:
        """대화 흐름 평가"""
        try:
            criteria = self.evaluation_criteria["conversation_flow"]
            connectors = criteria["connectors"]
            transitions = criteria["transitions"]
            weights = criteria["weights"]
            
            score = 0.0
            
            # 연결어 사용 확인
            has_connectors = any(connector in answer for connector in connectors)
            if has_connectors:
                score += weights["has_connectors"]
            
            # 전환 표현 사용 확인
            has_transitions = any(transition in answer for transition in transitions)
            if has_transitions:
                score += weights["has_transitions"]
            
            # 문장 간 연결성 평가
            sentences = answer.split(".")
            if len(sentences) > 1:
                # 문장 간 자연스러운 연결 확인
                smooth_connections = 0
                for i in range(len(sentences) - 1):
                    if any(connector in sentences[i+1] for connector in connectors):
                        smooth_connections += 1
                
                if smooth_connections > 0:
                    score += weights["smooth_flow"] * (smooth_connections / len(sentences))
            
            # 이전 대화 맥락 반영
            previous_topic = context.get("previous_topic", "")
            if previous_topic and any(ref in answer for ref in ["앞서", "이전", "방금", "말씀하신"]):
                score += 0.2
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Error evaluating conversation flow: {e}")
            return 0.5
    
    def _evaluate_emotional_appropriateness(self, answer: str, context: Dict[str, Any]) -> float:
        """감정적 적절성 평가"""
        try:
            criteria = self.evaluation_criteria["emotional_appropriateness"]
            empathy_expressions = criteria["empathy_expressions"]
            encouraging_expressions = criteria["encouraging_expressions"]
            supportive_expressions = criteria["supportive_expressions"]
            weights = criteria["weights"]
            
            score = 0.0
            
            # 공감 표현 확인
            has_empathy = any(expression in answer for expression in empathy_expressions)
            if has_empathy:
                score += weights["has_empathy"]
            
            # 격려 표현 확인
            has_encouragement = any(expression in answer for expression in encouraging_expressions)
            if has_encouragement:
                score += weights["has_encouragement"]
            
            # 지원적 표현 확인
            has_support = any(expression in answer for expression in supportive_expressions)
            if has_support:
                score += weights["has_support"]
            
            # 사용자 감정 상태 반영
            user_emotion = context.get("user_emotion", "neutral")
            if user_emotion in ["anxious", "confused"] and has_support:
                score += 0.2
            elif user_emotion in ["urgent", "angry"] and has_empathy:
                score += 0.2
            elif user_emotion == "positive" and has_encouragement:
                score += 0.2
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Error evaluating emotional appropriateness: {e}")
            return 0.5
    
    def _evaluate_personalization(self, answer: str, context: Dict[str, Any]) -> float:
        """개인화 평가"""
        try:
            criteria = self.evaluation_criteria["personalization"]
            personal_references = criteria["personal_references"]
            contextual_adaptations = criteria["contextual_adaptations"]
            weights = criteria["weights"]
            
            score = 0.0
            
            # 개인적 참조 확인
            has_personal_ref = any(ref in answer for ref in personal_references)
            if has_personal_ref:
                score += weights["has_personal_ref"]
            
            # 맥락적 적응 확인
            has_contextual = any(adaptation in answer for adaptation in contextual_adaptations)
            if has_contextual:
                score += weights["has_contextual"]
            
            # 사용자 프로필 반영
            user_type = context.get("user_type", "general")
            expertise_level = context.get("expertise_level", "beginner")
            
            if user_type == "general" and expertise_level == "beginner":
                # 일반인/초보자에게는 더 개인화된 설명 필요
                if has_personal_ref and has_contextual:
                    score += 0.2
            
            # 질문 유형에 따른 적응
            question_type = context.get("question_type", "general")
            if question_type in ["contract", "practical"] and has_contextual:
                score += 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Error evaluating personalization: {e}")
            return 0.5
    
    def _evaluate_readability(self, answer: str, context: Dict[str, Any]) -> float:
        """가독성 평가"""
        try:
            criteria = self.evaluation_criteria["readability"]
            sentence_length_optimal = criteria["sentence_length_optimal"]
            paragraph_length_optimal = criteria["paragraph_length_optimal"]
            complex_words = criteria["complex_words"]
            simplification_indicators = criteria["simplification_indicators"]
            weights = criteria["weights"]
            
            score = 0.0
            
            # 문장 길이 평가
            sentences = [s.strip() for s in answer.split(".") if s.strip()]
            if sentences:
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                min_optimal, max_optimal = sentence_length_optimal
                
                if min_optimal <= avg_sentence_length <= max_optimal:
                    score += weights["optimal_length"]
                elif avg_sentence_length < min_optimal:
                    score += weights["optimal_length"] * 0.7  # 너무 짧음
                else:
                    score += weights["optimal_length"] * 0.5  # 너무 김
            
            # 문단 길이 평가
            paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
            if paragraphs:
                avg_paragraph_length = sum(len(p.split(".")) for p in paragraphs) / len(paragraphs)
                min_optimal, max_optimal = paragraph_length_optimal
                
                if min_optimal <= avg_paragraph_length <= max_optimal:
                    score += 0.1
            
            # 단순화 지표 확인
            has_simplification = any(indicator in answer for indicator in simplification_indicators)
            if has_simplification:
                score += weights["has_simplification"]
            
            # 복잡한 단어 사용 적절성
            complex_word_count = sum(1 for word in complex_words if word in answer)
            total_words = len(answer.split())
            
            if total_words > 0:
                complex_word_ratio = complex_word_count / total_words
                if 0.05 <= complex_word_ratio <= 0.15:  # 적절한 비율
                    score += weights["appropriate_complexity"]
                elif complex_word_ratio > 0.15:  # 너무 복잡
                    score += weights["appropriate_complexity"] * 0.5
            
            # 사용자 전문성 수준 반영
            expertise_level = context.get("expertise_level", "beginner")
            if expertise_level == "beginner" and has_simplification:
                score += 0.2
            elif expertise_level == "expert" and complex_word_count > 0:
                score += 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Error evaluating readability: {e}")
            return 0.5
    
    def _calculate_overall_score(self, category_scores: Dict[EvaluationCategory, float]) -> float:
        """전체 점수 계산"""
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for category, score in category_scores.items():
                weight = self.category_weights.get(category, 0.2)
                weighted_sum += score * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return 0.5
    
    def _generate_detailed_analysis(self, answer: str, formality_score: float, 
                                 conversation_flow_score: float, emotional_appropriateness: float,
                                 personalization_score: float, readability_score: float) -> Dict[str, Any]:
        """상세 분석 생성"""
        try:
            analysis = {
                "strengths": [],
                "weaknesses": [],
                "suggestions": [],
                "naturalness_level": self._determine_naturalness_level(
                    (formality_score + conversation_flow_score + emotional_appropriateness + 
                     personalization_score + readability_score) / 5
                )
            }
            
            # 강점 분석
            if formality_score > 0.7:
                analysis["strengths"].append("자연스러운 말투 사용")
            if conversation_flow_score > 0.7:
                analysis["strengths"].append("대화 흐름이 자연스러움")
            if emotional_appropriateness > 0.7:
                analysis["strengths"].append("감정적으로 적절한 표현")
            if personalization_score > 0.7:
                analysis["strengths"].append("개인화된 답변")
            if readability_score > 0.7:
                analysis["strengths"].append("읽기 쉬운 구성")
            
            # 약점 분석
            if formality_score < 0.4:
                analysis["weaknesses"].append("격식적인 표현이 많음")
            if conversation_flow_score < 0.4:
                analysis["weaknesses"].append("대화 연결이 부자연스러움")
            if emotional_appropriateness < 0.4:
                analysis["weaknesses"].append("감정적 공감 부족")
            if personalization_score < 0.4:
                analysis["weaknesses"].append("개인화 부족")
            if readability_score < 0.4:
                analysis["weaknesses"].append("가독성 부족")
            
            # 개선 제안
            if formality_score < 0.5:
                analysis["suggestions"].append("더 자연스러운 말투로 변경하세요")
            if conversation_flow_score < 0.5:
                analysis["suggestions"].append("대화 연결어를 추가하세요")
            if emotional_appropriateness < 0.5:
                analysis["suggestions"].append("공감적 표현을 추가하세요")
            if personalization_score < 0.5:
                analysis["suggestions"].append("개인화된 표현을 사용하세요")
            if readability_score < 0.5:
                analysis["suggestions"].append("더 읽기 쉽게 구성하세요")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating detailed analysis: {e}")
            return {"strengths": [], "weaknesses": [], "suggestions": [], "naturalness_level": "moderate"}
    
    def _determine_naturalness_level(self, overall_score: float) -> str:
        """자연스러움 레벨 결정"""
        try:
            for level, threshold in self.naturalness_thresholds.items():
                if overall_score >= threshold:
                    return level.value
            
            return NaturalnessLevel.VERY_UNNATURAL.value
            
        except Exception as e:
            self.logger.error(f"Error determining naturalness level: {e}")
            return NaturalnessLevel.MODERATE.value
    
    def _create_default_metrics(self) -> NaturalnessMetrics:
        """기본 메트릭 생성"""
        return NaturalnessMetrics(
            formality_score=0.5,
            conversation_flow_score=0.5,
            emotional_appropriateness=0.5,
            personalization_score=0.5,
            readability_score=0.5,
            overall_naturalness=0.5,
            evaluation_timestamp=datetime.now().isoformat(),
            detailed_analysis={
                "strengths": [],
                "weaknesses": [],
                "suggestions": [],
                "naturalness_level": "moderate"
            }
        )
    
    def get_improvement_recommendations(self, metrics: NaturalnessMetrics) -> List[str]:
        """개선 권장사항 생성"""
        try:
            recommendations = []
            
            if metrics.formality_score < 0.5:
                recommendations.append("격식적인 표현을 자연스러운 표현으로 변경하세요")
            
            if metrics.conversation_flow_score < 0.5:
                recommendations.append("대화 연결어와 전환 표현을 추가하세요")
            
            if metrics.emotional_appropriateness < 0.5:
                recommendations.append("공감적이고 격려하는 표현을 포함하세요")
            
            if metrics.personalization_score < 0.5:
                recommendations.append("사용자에게 맞춤화된 표현을 사용하세요")
            
            if metrics.readability_score < 0.5:
                recommendations.append("문장 길이를 조절하고 단순화 지표를 추가하세요")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting improvement recommendations: {e}")
            return []
    
    def compare_naturalness(self, answer1: str, answer2: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """두 답변의 자연스러움 비교"""
        try:
            metrics1 = self.evaluate_naturalness(answer1, context)
            metrics2 = self.evaluate_naturalness(answer2, context)
            
            return {
                "answer1": {
                    "overall_score": metrics1.overall_naturalness,
                    "level": metrics1.detailed_analysis["naturalness_level"],
                    "strengths": metrics1.detailed_analysis["strengths"],
                    "weaknesses": metrics1.detailed_analysis["weaknesses"]
                },
                "answer2": {
                    "overall_score": metrics2.overall_naturalness,
                    "level": metrics2.detailed_analysis["naturalness_level"],
                    "strengths": metrics2.detailed_analysis["strengths"],
                    "weaknesses": metrics2.detailed_analysis["weaknesses"]
                },
                "comparison": {
                    "better_answer": "answer1" if metrics1.overall_naturalness > metrics2.overall_naturalness else "answer2",
                    "score_difference": abs(metrics1.overall_naturalness - metrics2.overall_naturalness),
                    "improvement_suggestions": self._get_comparison_suggestions(metrics1, metrics2)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing naturalness: {e}")
            return {"error": str(e)}
    
    def _get_comparison_suggestions(self, metrics1: NaturalnessMetrics, metrics2: NaturalnessMetrics) -> List[str]:
        """비교 기반 개선 제안"""
        suggestions = []
        
        # 각 카테고리별 비교
        categories = [
            ("formality", metrics1.formality_score, metrics2.formality_score),
            ("conversation_flow", metrics1.conversation_flow_score, metrics2.conversation_flow_score),
            ("emotional_appropriateness", metrics1.emotional_appropriateness, metrics2.emotional_appropriateness),
            ("personalization", metrics1.personalization_score, metrics2.personalization_score),
            ("readability", metrics1.readability_score, metrics2.readability_score)
        ]
        
        for category, score1, score2 in categories:
            if score1 > score2 + 0.2:
                suggestions.append(f"첫 번째 답변의 {category}가 더 우수합니다")
            elif score2 > score1 + 0.2:
                suggestions.append(f"두 번째 답변의 {category}가 더 우수합니다")
        
        return suggestions
