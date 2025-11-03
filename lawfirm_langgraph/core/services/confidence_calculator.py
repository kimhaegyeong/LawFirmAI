# -*- coding: utf-8 -*-
"""
Confidence Calculator
신뢰도 계산기
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """신뢰도 레벨"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class ConfidenceInfo:
    """신뢰도 정보"""
    confidence: float
    level: ConfidenceLevel
    factors: Dict[str, float]
    explanation: str
    
    @property
    def reliability_level(self) -> str:
        """신뢰도 레벨을 문자열로 반환"""
        return self.level.value
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return {
            "confidence": self.confidence,
            "level": self.level.value,
            "factors": self.factors,
            "explanation": self.explanation,
            "reliability_level": self.reliability_level
        }


class ConfidenceCalculator:
    """신뢰도 계산기"""
    
    def __init__(self):
        """계산기 초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("ConfidenceCalculator initialized")
        
        # 신뢰도 계산 가중치 (개선된 버전)
        self.weights = {
            'answer_length': 0.10,      # 답변 길이 (10%) - 감소
            'source_quality': 0.25,     # 소스 품질 (25%) - 신규 추가
            'question_type': 0.20,      # 질문 유형 (20%) - 감소
            'answer_quality': 0.35,     # 답변 품질 (35%) - 증가
            'source_count': 0.10        # 소스 수 (10%) - 감소
        }
    
    def calculate_confidence(self, 
                           answer: str,
                           sources: List[Dict[str, Any]],
                           question_type: str = "general") -> ConfidenceInfo:
        """
        신뢰도 계산
        
        Args:
            answer: 생성된 답변
            sources: 참조 소스
            question_type: 질문 유형
            
        Returns:
            ConfidenceInfo: 신뢰도 정보
        """
        factors = {}
        
        # 답변 길이 기반 신뢰도
        factors['answer_length'] = self._calculate_length_confidence(answer)
        
        # 소스 품질 기반 신뢰도 (개선된 버전)
        factors['source_quality'] = self._calculate_source_quality_confidence(sources)
        
        # 소스 수 기반 신뢰도 (보조 지표)
        factors['source_count'] = self._calculate_source_confidence(sources)
        
        # 질문 유형 기반 신뢰도
        factors['question_type'] = self._calculate_type_confidence(question_type)
        
        # 답변 품질 기반 신뢰도 (고도화된 버전)
        factors['answer_quality'] = self._calculate_enhanced_quality_confidence(answer)
        
        # 전체 신뢰도 계산 (가중 평균 적용)
        overall_confidence = (
            factors['answer_length'] * self.weights['answer_length'] +
            factors['source_quality'] * self.weights['source_quality'] +
            factors['source_count'] * self.weights['source_count'] +
            factors['question_type'] * self.weights['question_type'] +
            factors['answer_quality'] * self.weights['answer_quality']
        )
        
        # 신뢰도 보정 메커니즘 적용
        overall_confidence = self._apply_confidence_boost(overall_confidence, answer, sources)
        
        # 신뢰도 레벨 결정
        level = self._determine_confidence_level(overall_confidence)
        
        # 설명 생성
        explanation = self._generate_explanation(factors, overall_confidence)
        
        return ConfidenceInfo(
            confidence=overall_confidence,
            level=level,
            factors=factors,
            explanation=explanation
        )
    
    def _calculate_length_confidence(self, answer: str) -> float:
        """답변 길이 기반 신뢰도"""
        length = len(answer)
        
        if length < 50:
            return 0.3
        elif length < 100:
            return 0.5
        elif length < 200:
            return 0.7
        elif length < 500:
            return 0.9
        else:
            return 1.0
    
    def _calculate_source_confidence(self, sources: List[Dict[str, Any]]) -> float:
        """소스 수 기반 신뢰도"""
        source_count = len(sources)
        
        if source_count == 0:
            return 0.1
        elif source_count == 1:
            return 0.6
        elif source_count <= 3:
            return 0.8
        elif source_count <= 5:
            return 0.9
        else:
            return 0.7  # 너무 많은 소스는 오히려 혼란
    
    def _calculate_type_confidence(self, question_type: str) -> float:
        """질문 유형 기반 신뢰도 (상향 조정된 버전)"""
        type_confidence_map = {
            "precedent_search": 0.95,    # 판례 검색 - 높은 신뢰도
            "law_inquiry": 0.90,         # 법률 문의 - 높은 신뢰도
            "legal_advice": 0.88,        # 법적 조언 - 상향 조정
            "procedure_guide": 0.85,     # 절차 안내 - 상향 조정
            "term_explanation": 0.88,    # 용어 해설 - 상향 조정
            "general_question": 0.80,    # 일반 질문 - 상향 조정
            "analysis_request": 0.82,    # 분석 요청 - 상향 조정
            "comparison_request": 0.78   # 비교 요청 - 상향 조정
        }
        
        return type_confidence_map.get(question_type, 0.75)  # 기본값 상향
    
    def _calculate_quality_confidence(self, answer: str) -> float:
        """답변 품질 기반 신뢰도 (법률 분야 강화 - 답변 품질 향상)"""
        quality_score = 0.0
        
        # 1. 법률 용어 포함 여부 (강화)
        legal_terms = [
            "법", "조항", "판례", "법원", "판결", "소송", "계약", "손해배상",
            "민법", "형법", "상법", "노동법", "행정법", "헌법", "소송법",
            "불법행위", "채권", "채무", "소유권", "물권", "인권", "가족법",
            "대법원", "하급심", "판례", "법령", "규정", "조문", "항목"
        ]
        legal_term_count = sum(1 for term in legal_terms if term in answer)
        if legal_term_count >= 5:
            quality_score += 0.25
        elif legal_term_count >= 3:
            quality_score += 0.20
        elif legal_term_count >= 2:
            quality_score += 0.15
        elif legal_term_count >= 1:
            quality_score += 0.10
        
        # 2. 구체적인 설명 포함 여부 (강화)
        explanation_indicators = [
            "따라서", "결론적으로", "판단컨대", "요약하면", "종합하면",
            "첫째", "둘째", "셋째", "1.", "2.", "3.", "•", "-",
            "단계별", "구체적으로", "실제로", "예를 들어", "즉"
        ]
        explanation_count = sum(1 for indicator in explanation_indicators if indicator in answer)
        if explanation_count >= 3:
            quality_score += 0.20
        elif explanation_count >= 2:
            quality_score += 0.15
        elif explanation_count >= 1:
            quality_score += 0.10
        
        # 3. 법적 근거 제시 여부 (강화)
        evidence_indicators = [
            "법률에 따르면", "판례에 따르면", "법원은", "대법원은",
            "제", "조", "항", "호", "법령", "규정", "조문",
            "민법 제", "형법 제", "상법 제", "근로기준법 제"
        ]
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in answer)
        if evidence_count >= 3:
            quality_score += 0.25
        elif evidence_count >= 2:
            quality_score += 0.20
        elif evidence_count >= 1:
            quality_score += 0.15
        
        # 4. 답변 구조화 품질 (신규)
        structure_indicators = [
            "상황 정리", "법적 분석", "판례 검토", "실무 조언", "주의사항",
            "##", "###", "**", "1.", "2.", "3.", "•", "-"
        ]
        structure_count = sum(1 for indicator in structure_indicators if indicator in answer)
        if structure_count >= 4:
            quality_score += 0.15
        elif structure_count >= 2:
            quality_score += 0.10
        elif structure_count >= 1:
            quality_score += 0.05
        
        # 5. 실무적 조언 포함 여부 (신규)
        practical_indicators = [
            "실행 가능", "구체적", "단계별", "절차", "방법", "조치",
            "권장", "고려", "주의", "필요", "서류", "증거"
        ]
        practical_count = sum(1 for indicator in practical_indicators if indicator in answer)
        if practical_count >= 3:
            quality_score += 0.15
        elif practical_count >= 2:
            quality_score += 0.10
        elif practical_count >= 1:
            quality_score += 0.05
        
        return min(quality_score, 1.0)
    
    def _calculate_source_quality_confidence(self, sources: List[Dict[str, Any]]) -> float:
        """소스 품질 기반 신뢰도 (개선된 버전)"""
        if not sources:
            return 0.3  # 소스가 없어도 기본 신뢰도 제공
        
        # 소스 품질 평가
        quality_scores = []
        for source in sources:
            score = 0.0
            
            # 소스 타입별 가중치
            source_type = source.get('search_type', 'unknown')
            type_weights = {
                'exact_law': 1.0,
                'exact_precedent': 0.9,
                'fts_optimized': 0.8,
                'vector_search': 0.7,
                'exact_assembly_law': 0.9,
                'exact_constitutional': 0.8,
                'unknown': 0.5
            }
            score += type_weights.get(source_type, 0.5)
            
            # 소스 신뢰성 평가
            if 'case_number' in source or 'law_name' in source or 'case_id' in source:
                score += 0.2  # 구체적인 식별자 있음
            
            # 소스 완성도 평가
            if 'full_text' in source and source['full_text']:
                score += 0.1  # 전체 텍스트 있음
            
            if 'summary' in source and source['summary']:
                score += 0.1  # 요약 있음
            
            quality_scores.append(min(score, 1.0))
        
        return sum(quality_scores) / len(quality_scores)
    
    def _calculate_enhanced_quality_confidence(self, answer: str) -> float:
        """답변 품질 기반 신뢰도 (고도화된 버전)"""
        quality_score = 0.0
        
        # 1. 법률 전문성 평가 (40점)
        legal_expertise = self._evaluate_legal_expertise(answer)
        quality_score += legal_expertise * 0.4
        
        # 2. 구조화 수준 평가 (30점)
        structure_score = self._evaluate_structure(answer)
        quality_score += structure_score * 0.3
        
        # 3. 근거 제시 수준 평가 (20점)
        evidence_score = self._evaluate_evidence(answer)
        quality_score += evidence_score * 0.2
        
        # 4. 명확성 평가 (10점)
        clarity_score = self._evaluate_clarity(answer)
        quality_score += clarity_score * 0.1
        
        return min(quality_score, 1.0)
    
    def _evaluate_legal_expertise(self, answer: str) -> float:
        """법률 전문성 평가"""
        score = 0.0
        
        # 법률 용어 사용률
        legal_terms = ["법", "조항", "판례", "법원", "판결", "소송", "계약", "손해배상"]
        term_count = sum(1 for term in legal_terms if term in answer)
        score += min(term_count * 0.1, 0.3)
        
        # 구체적 법령 참조
        if any(word in answer for word in ["제", "조", "항", "호"]):
            score += 0.3
        
        # 판례 인용
        if any(word in answer for word in ["대법원", "고등법원", "지방법원"]):
            score += 0.2
        
        # 법적 논리 구조
        if any(word in answer for word in ["따라서", "결론적으로", "판단컨대"]):
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_structure(self, answer: str) -> float:
        """구조화 수준 평가"""
        score = 0.0
        
        # 번호나 불릿 포인트 사용
        if any(indicator in answer for indicator in ["1.", "2.", "3.", "•", "-", "첫째", "둘째", "셋째"]):
            score += 0.4
        
        # 논리적 연결어 사용
        if any(indicator in answer for indicator in ["따라서", "결론적으로", "요약하면", "종합하면"]):
            score += 0.3
        
        # 문단 구분
        if "\n\n" in answer or "문단" in answer:
            score += 0.3
        
        return min(score, 1.0)
    
    def _evaluate_evidence(self, answer: str) -> float:
        """근거 제시 수준 평가"""
        score = 0.0
        
        # 법적 근거 제시
        if any(indicator in answer for indicator in ["법률에 따르면", "판례에 따르면", "법원은", "대법원은"]):
            score += 0.5
        
        # 구체적 조문 참조
        if any(indicator in answer for indicator in ["제", "조", "항", "호", "법령", "규정", "조문"]):
            score += 0.3
        
        # 판례 번호나 사건명
        if any(indicator in answer for indicator in ["2023", "2022", "2021", "2020", "다", "가", "나"]):
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_clarity(self, answer: str) -> float:
        """명확성 평가"""
        score = 0.0
        
        # 적절한 길이 (너무 짧지도 길지도 않음)
        length = len(answer)
        if 100 <= length <= 1000:
            score += 0.4
        elif 50 <= length < 100 or 1000 < length <= 2000:
            score += 0.2
        
        # 명확한 설명
        if any(indicator in answer for indicator in ["의미는", "정의는", "내용은", "절차는"]):
            score += 0.3
        
        # 구체적 예시
        if any(indicator in answer for indicator in ["예를 들어", "예시로", "구체적으로", "실제로"]):
            score += 0.3
        
        return min(score, 1.0)
    
    def _apply_confidence_boost(self, base_confidence: float, answer: str, sources: List[Dict[str, Any]]) -> float:
        """신뢰도 보정 메커니즘"""
        boosted_confidence = base_confidence
        
        # 법률 전문성 보정
        if self._has_high_legal_expertise(answer):
            boosted_confidence += 0.1
        
        # 소스 품질 보정
        if self._has_high_quality_sources(sources):
            boosted_confidence += 0.1
        
        # 답변 완성도 보정
        if self._is_complete_answer(answer):
            boosted_confidence += 0.05
        
        return min(boosted_confidence, 1.0)
    
    def _has_high_legal_expertise(self, answer: str) -> bool:
        """높은 법률 전문성 여부 판단"""
        legal_indicators = ["제", "조", "항", "대법원", "판례", "법률에 따르면"]
        return sum(1 for indicator in legal_indicators if indicator in answer) >= 3
    
    def _has_high_quality_sources(self, sources: List[Dict[str, Any]]) -> bool:
        """높은 품질의 소스 여부 판단"""
        if not sources:
            return False
        
        high_quality_count = 0
        for source in sources:
            source_type = source.get('search_type', 'unknown')
            if source_type in ['exact_law', 'exact_precedent', 'exact_assembly_law']:
                high_quality_count += 1
        
        return high_quality_count >= len(sources) * 0.5
    
    def _is_complete_answer(self, answer: str) -> bool:
        """완성된 답변 여부 판단"""
        # 적절한 길이와 구조를 가진 답변
        return (100 <= len(answer) <= 2000 and 
                any(indicator in answer for indicator in ["따라서", "결론적으로", "요약하면"]) and
                any(indicator in answer for indicator in ["법", "조항", "판례"]))
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """신뢰도 점수에 따른 레벨 결정"""
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_explanation(self, factors: Dict[str, float], overall_confidence: float) -> str:
        """신뢰도 설명 생성"""
        explanations = []
        
        if factors.get('answer_length', 0) > 0.7:
            explanations.append("답변 길이가 충분합니다")
        
        if factors.get('source_count', 0) > 0.7:
            explanations.append("충분한 출처가 참조되었습니다")
        
        if factors.get('question_type', 0) > 0.7:
            explanations.append("질문 유형에 적합한 답변입니다")
        
        if factors.get('answer_quality', 0) > 0.7:
            explanations.append("답변 품질이 양호합니다")
        
        if not explanations:
            explanations.append("제공된 정보가 제한적입니다")
        
        return f"신뢰도: {overall_confidence:.2f}. " + ", ".join(explanations)
    
    def calculate_enhanced_confidence(self, 
                                    answer: str,
                                    sources: List[Dict[str, Any]],
                                    question_type: str = "general",
                                    domain: str = "general") -> ConfidenceInfo:
        """향상된 신뢰도 계산 (답변 품질 향상 버전)"""
        try:
            factors = {}
            
            # 기존 신뢰도 계산
            factors['answer_length'] = self._calculate_length_confidence(answer)
            factors['source_count'] = self._calculate_source_confidence(sources)
            factors['question_type'] = self._calculate_type_confidence(question_type)
            factors['answer_quality'] = self._calculate_quality_confidence(answer)
            
            # 소스 품질 계산 (기존 메서드 사용)
            if hasattr(self, '_calculate_source_quality_confidence'):
                factors['source_quality'] = self._calculate_source_quality_confidence(sources)
            else:
                factors['source_quality'] = factors['source_count']
            
            # 도메인별 신뢰도 조정
            domain_multiplier = self._get_domain_multiplier(domain)
            
            # 전체 신뢰도 계산 (가중 평균 적용)
            overall_confidence = (
                factors['answer_length'] * self.weights['answer_length'] +
                factors['source_quality'] * self.weights['source_quality'] +
                factors['question_type'] * self.weights['question_type'] +
                factors['answer_quality'] * self.weights['answer_quality'] +
                factors['source_count'] * self.weights['source_count']
            ) * domain_multiplier
            
            # 신뢰도 레벨 결정
            level = self._determine_confidence_level(overall_confidence)
            
            # 향상된 설명 생성
            explanation = self._generate_enhanced_explanation(factors, overall_confidence, domain)
            
            return ConfidenceInfo(
                confidence=overall_confidence,
                level=level,
                factors=factors,
                explanation=explanation
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced confidence: {e}")
            # 폴백: 기본 신뢰도 계산
            return self.calculate_confidence(answer, sources, question_type)
    
    def _get_domain_multiplier(self, domain: str) -> float:
        """도메인별 신뢰도 배수"""
        domain_multipliers = {
            "민사법": 1.1,      # 민사법은 높은 신뢰도
            "형사법": 1.05,     # 형사법은 중간 신뢰도
            "가족법": 1.0,      # 가족법은 기본 신뢰도
            "상사법": 1.1,      # 상사법은 높은 신뢰도
            "행정법": 0.95,     # 행정법은 약간 낮은 신뢰도
            "노동법": 1.0,      # 노동법은 기본 신뢰도
            "부동산법": 1.05,   # 부동산법은 중간 신뢰도
            "지적재산권법": 0.9, # 지적재산권법은 낮은 신뢰도
            "세법": 0.9,        # 세법은 낮은 신뢰도
            "민사소송법": 1.0,  # 민사소송법은 기본 신뢰도
            "형사소송법": 1.0,  # 형사소송법은 기본 신뢰도
            "general": 1.0      # 일반은 기본 신뢰도
        }
        return domain_multipliers.get(domain, 1.0)
    
    def _generate_enhanced_explanation(self, factors: Dict[str, float], 
                                     overall_confidence: float, domain: str) -> str:
        """향상된 신뢰도 설명 생성"""
        explanations = []
        
        # 답변 품질 분석
        if factors.get('answer_quality', 0) > 0.8:
            explanations.append("답변 품질이 매우 우수합니다")
        elif factors.get('answer_quality', 0) > 0.6:
            explanations.append("답변 품질이 양호합니다")
        else:
            explanations.append("답변 품질 개선이 필요합니다")
        
        # 소스 품질 분석
        if factors.get('source_quality', 0) > 0.8:
            explanations.append("참조 소스의 품질이 우수합니다")
        elif factors.get('source_quality', 0) > 0.6:
            explanations.append("참조 소스가 적절합니다")
        else:
            explanations.append("더 많은 참조 소스가 필요합니다")
        
        # 질문 유형 분석
        if factors.get('question_type', 0) > 0.8:
            explanations.append("질문 유형에 적합한 답변입니다")
        else:
            explanations.append("질문 유형별 특화가 필요합니다")
        
        # 도메인 특화 분석
        if domain != "general":
            explanations.append(f"{domain} 분야에 특화된 답변입니다")
        
        # 전체 평가
        if overall_confidence > 0.8:
            explanations.append("전반적으로 신뢰할 수 있는 답변입니다")
        elif overall_confidence > 0.6:
            explanations.append("기본적인 신뢰도를 갖춘 답변입니다")
        else:
            explanations.append("추가 검토가 필요한 답변입니다")
        
        return f"신뢰도: {overall_confidence:.2f}. " + ", ".join(explanations)