# -*- coding: utf-8 -*-
"""
신뢰도 기반 답변 시스템
검색 결과 품질을 기반으로 답변 신뢰도를 계산하는 시스템
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ReliabilityLevel(Enum):
    """신뢰도 수준"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


@dataclass
class ConfidenceInfo:
    """신뢰도 정보"""
    confidence: float                    # 전체 신뢰도 (0.0 ~ 1.0)
    reliability_level: ReliabilityLevel  # 신뢰도 수준
    similarity_score: float              # 검색 결과 유사도 점수
    matching_score: float                # 법률/판례 매칭 정확도
    answer_quality: float                # 답변 품질 점수
    warnings: List[str]                  # 경고 메시지
    recommendations: List[str]            # 권장사항


class ConfidenceCalculator:
    """신뢰도 계산기"""
    
    def __init__(self):
        """신뢰도 계산기 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 신뢰도 계산 가중치
        self.weights = {
            "similarity": 0.4,      # 검색 결과 유사도
            "matching": 0.3,        # 법률/판례 매칭 정확도
            "answer_quality": 0.3   # 답변 품질
        }
        
        # 신뢰도 임계값
        self.thresholds = {
            ReliabilityLevel.HIGH: 0.8,
            ReliabilityLevel.MEDIUM: 0.6,
            ReliabilityLevel.LOW: 0.4,
            ReliabilityLevel.VERY_LOW: 0.0
        }
        
        # 법률 키워드 패턴
        self.legal_keywords = [
            "법률", "법령", "조문", "조항", "법조문", "법규",
            "민법", "형법", "상법", "노동법", "행정법", "헌법",
            "판례", "사건", "판결", "대법원", "고등법원", "지방법원",
            "손해배상", "계약", "불법행위", "채무", "권리", "의무"
        ]
        
        # 답변 품질 지표
        self.quality_indicators = {
            "structure": ["1.", "2.", "3.", "•", "-", "가.", "나.", "다."],
            "legal_terms": self.legal_keywords,
            "references": ["제", "조", "항", "호", "사건번호", "판례"],
            "length_range": (50, 2000)  # 최소/최대 길이
        }
    
    def calculate_confidence(self, 
                           query: str, 
                           retrieved_docs: List[Dict[str, Any]], 
                           answer: str) -> ConfidenceInfo:
        """
        신뢰도 계산
        
        Args:
            query: 사용자 질문
            retrieved_docs: 검색된 문서들
            answer: 생성된 답변
            
        Returns:
            ConfidenceInfo: 신뢰도 정보
        """
        try:
            self.logger.info(f"Calculating confidence for query: {query[:100]}...")
            
            # 1. 검색 결과 유사도 점수 계산
            similarity_score = self._calc_similarity_score(retrieved_docs)
            
            # 2. 법률/판례 매칭 정확도 계산
            matching_score = self._calc_matching_score(query, retrieved_docs)
            
            # 3. 답변 품질 점수 계산
            answer_quality = self._calc_answer_quality(answer, query)
            
            # 4. 가중 평균으로 최종 신뢰도 계산
            confidence = (
                similarity_score * self.weights["similarity"] +
                matching_score * self.weights["matching"] +
                answer_quality * self.weights["answer_quality"]
            )
            
            # 5. 신뢰도 수준 결정
            reliability_level = self._get_reliability_level(confidence)
            
            # 6. 경고 및 권장사항 생성
            warnings = self._generate_warnings(confidence, similarity_score, matching_score, answer_quality)
            recommendations = self._generate_recommendations(confidence, reliability_level)
            
            result = ConfidenceInfo(
                confidence=confidence,
                reliability_level=reliability_level,
                similarity_score=similarity_score,
                matching_score=matching_score,
                answer_quality=answer_quality,
                warnings=warnings,
                recommendations=recommendations
            )
            
            self.logger.info(f"Confidence calculated: {confidence:.3f} ({reliability_level.value})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return ConfidenceInfo(
                confidence=0.0,
                reliability_level=ReliabilityLevel.VERY_LOW,
                similarity_score=0.0,
                matching_score=0.0,
                answer_quality=0.0,
                warnings=[f"신뢰도 계산 오류: {e}"],
                recommendations=["전문가 상담을 권장합니다."]
            )
    
    def _calc_similarity_score(self, retrieved_docs: List[Dict[str, Any]]) -> float:
        """검색 결과 유사도 점수 계산"""
        try:
            if not retrieved_docs:
                return 0.0
            
            # 각 문서의 유사도 점수 수집
            similarities = []
            for doc in retrieved_docs:
                similarity = doc.get("similarity", 0.0)
                if isinstance(similarity, (int, float)) and 0 <= similarity <= 1:
                    similarities.append(similarity)
            
            if not similarities:
                return 0.0
            
            # 평균 유사도 계산
            avg_similarity = sum(similarities) / len(similarities)
            
            # 상위 결과들의 가중치 적용
            top_similarities = sorted(similarities, reverse=True)[:3]
            if len(top_similarities) >= 2:
                weighted_avg = (top_similarities[0] * 0.5 + 
                              top_similarities[1] * 0.3 + 
                              top_similarities[2] * 0.2)
                return max(avg_similarity, weighted_avg)
            
            return avg_similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity score: {e}")
            return 0.0
    
    def _calc_matching_score(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """법률/판례 매칭 정확도 계산"""
        try:
            if not retrieved_docs:
                return 0.0
            
            # 질문에서 법률 키워드 추출
            query_keywords = self._extract_legal_keywords(query)
            
            # 검색 결과에서 법률 정보 추출
            doc_keywords = []
            for doc in retrieved_docs:
                doc_text = self._extract_doc_text(doc)
                doc_keywords.extend(self._extract_legal_keywords(doc_text))
            
            # 키워드 매칭 점수 계산
            if not query_keywords:
                return 0.5  # 키워드가 없으면 중간 점수
            
            matched_keywords = set(query_keywords) & set(doc_keywords)
            matching_ratio = len(matched_keywords) / len(query_keywords)
            
            # 문서 타입별 가중치 적용
            type_bonus = self._calc_type_matching_bonus(query, retrieved_docs)
            
            final_score = min(matching_ratio + type_bonus, 1.0)
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error calculating matching score: {e}")
            return 0.0
    
    def _calc_answer_quality(self, answer: str, query: str) -> float:
        """답변 품질 점수 계산"""
        try:
            if not answer or len(answer.strip()) == 0:
                return 0.0
            
            quality_score = 0.0
            
            # 1. 길이 적절성 (20%)
            length_score = self._calc_length_score(answer)
            quality_score += length_score * 0.2
            
            # 2. 구조화 정도 (30%)
            structure_score = self._calc_structure_score(answer)
            quality_score += structure_score * 0.3
            
            # 3. 법률 용어 사용 (25%)
            legal_term_score = self._calc_legal_term_score(answer)
            quality_score += legal_term_score * 0.25
            
            # 4. 참조 정보 포함 (25%)
            reference_score = self._calc_reference_score(answer)
            quality_score += reference_score * 0.25
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating answer quality: {e}")
            return 0.0
    
    def _extract_legal_keywords(self, text: str) -> List[str]:
        """텍스트에서 법률 키워드 추출"""
        try:
            if not text:
                return []
            
            keywords = []
            text_lower = text.lower()
            
            for keyword in self.legal_keywords:
                if keyword in text_lower:
                    keywords.append(keyword)
            
            return keywords
            
        except Exception as e:
            self.logger.error(f"Error extracting legal keywords: {e}")
            return []
    
    def _extract_doc_text(self, doc: Dict[str, Any]) -> str:
        """문서에서 텍스트 추출"""
        try:
            # 다양한 필드에서 텍스트 추출
            text_fields = [
                "content", "text", "summary", "description",
                "case_name", "law_name", "article_content",
                "judgment_summary", "judgment_gist"
            ]
            
            text_parts = []
            for field in text_fields:
                if field in doc and doc[field]:
                    text_parts.append(str(doc[field]))
            
            return " ".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Error extracting doc text: {e}")
            return ""
    
    def _calc_type_matching_bonus(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """문서 타입별 매칭 보너스 계산"""
        try:
            bonus = 0.0
            
            # 질문 유형 추정
            if any(keyword in query.lower() for keyword in ["판례", "사건", "판결"]):
                # 판례 관련 질문
                precedent_count = len([doc for doc in retrieved_docs if doc.get("type") == "precedent"])
                if precedent_count > 0:
                    bonus += 0.2
            
            elif any(keyword in query.lower() for keyword in ["법률", "조문", "법령"]):
                # 법률 관련 질문
                law_count = len([doc for doc in retrieved_docs if doc.get("type") == "law"])
                if law_count > 0:
                    bonus += 0.2
            
            return min(bonus, 0.3)  # 최대 0.3 보너스
            
        except Exception as e:
            self.logger.error(f"Error calculating type matching bonus: {e}")
            return 0.0
    
    def _calc_length_score(self, answer: str) -> float:
        """답변 길이 점수 계산"""
        try:
            length = len(answer.strip())
            min_length, max_length = self.quality_indicators["length_range"]
            
            if length < min_length:
                return length / min_length * 0.5  # 너무 짧으면 낮은 점수
            elif length > max_length:
                return max_length / length * 0.8  # 너무 길면 약간 감점
            else:
                return 1.0  # 적절한 길이
            
        except Exception as e:
            self.logger.error(f"Error calculating length score: {e}")
            return 0.5
    
    def _calc_structure_score(self, answer: str) -> float:
        """답변 구조화 점수 계산"""
        try:
            structure_indicators = self.quality_indicators["structure"]
            
            # 구조화 지표 개수 계산
            structure_count = 0
            for indicator in structure_indicators:
                structure_count += answer.count(indicator)
            
            # 구조화 점수 계산 (0~1)
            if structure_count == 0:
                return 0.3  # 구조화 없음
            elif structure_count <= 2:
                return 0.6  # 약간 구조화
            elif structure_count <= 5:
                return 0.8  # 잘 구조화
            else:
                return 1.0  # 매우 잘 구조화
            
        except Exception as e:
            self.logger.error(f"Error calculating structure score: {e}")
            return 0.5
    
    def _calc_legal_term_score(self, answer: str) -> float:
        """법률 용어 사용 점수 계산"""
        try:
            legal_terms = self.quality_indicators["legal_terms"]
            answer_lower = answer.lower()
            
            # 사용된 법률 용어 개수
            used_terms = sum(1 for term in legal_terms if term in answer_lower)
            
            # 점수 계산 (0~1)
            if used_terms == 0:
                return 0.2  # 법률 용어 없음
            elif used_terms <= 2:
                return 0.5  # 약간 사용
            elif used_terms <= 5:
                return 0.8  # 적절히 사용
            else:
                return 1.0  # 많이 사용
            
        except Exception as e:
            self.logger.error(f"Error calculating legal term score: {e}")
            return 0.5
    
    def _calc_reference_score(self, answer: str) -> float:
        """참조 정보 포함 점수 계산"""
        try:
            reference_patterns = self.quality_indicators["references"]
            
            # 참조 패턴 개수 계산
            reference_count = 0
            for pattern in reference_patterns:
                reference_count += len(re.findall(pattern, answer))
            
            # 점수 계산 (0~1)
            if reference_count == 0:
                return 0.3  # 참조 없음
            elif reference_count <= 2:
                return 0.6  # 약간 참조
            elif reference_count <= 5:
                return 0.8  # 적절히 참조
            else:
                return 1.0  # 많이 참조
            
        except Exception as e:
            self.logger.error(f"Error calculating reference score: {e}")
            return 0.5
    
    def _get_reliability_level(self, confidence: float) -> ReliabilityLevel:
        """신뢰도 수준 결정"""
        try:
            if confidence >= self.thresholds[ReliabilityLevel.HIGH]:
                return ReliabilityLevel.HIGH
            elif confidence >= self.thresholds[ReliabilityLevel.MEDIUM]:
                return ReliabilityLevel.MEDIUM
            elif confidence >= self.thresholds[ReliabilityLevel.LOW]:
                return ReliabilityLevel.LOW
            else:
                return ReliabilityLevel.VERY_LOW
                
        except Exception as e:
            self.logger.error(f"Error getting reliability level: {e}")
            return ReliabilityLevel.VERY_LOW
    
    def _generate_warnings(self, 
                          confidence: float, 
                          similarity_score: float, 
                          matching_score: float, 
                          answer_quality: float) -> List[str]:
        """경고 메시지 생성"""
        warnings = []
        
        try:
            if confidence < 0.4:
                warnings.append("신뢰도가 낮습니다. 전문가 상담을 권장합니다.")
            
            if similarity_score < 0.3:
                warnings.append("검색 결과와 질문의 유사도가 낮습니다.")
            
            if matching_score < 0.3:
                warnings.append("관련 법률/판례 정보가 부족할 수 있습니다.")
            
            if answer_quality < 0.4:
                warnings.append("답변의 품질이 낮을 수 있습니다.")
            
            return warnings
            
        except Exception as e:
            self.logger.error(f"Error generating warnings: {e}")
            return ["신뢰도 계산 중 오류가 발생했습니다."]
    
    def _generate_recommendations(self, 
                                confidence: float, 
                                reliability_level: ReliabilityLevel) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        try:
            if reliability_level == ReliabilityLevel.HIGH:
                recommendations.append("높은 신뢰도의 답변입니다.")
            elif reliability_level == ReliabilityLevel.MEDIUM:
                recommendations.append("보통 신뢰도의 답변입니다. 추가 확인을 권장합니다.")
            elif reliability_level == ReliabilityLevel.LOW:
                recommendations.append("낮은 신뢰도의 답변입니다. 전문가 상담을 권장합니다.")
            else:
                recommendations.append("매우 낮은 신뢰도의 답변입니다. 반드시 전문가와 상담하세요.")
            
            if confidence < 0.6:
                recommendations.append("더 구체적인 질문을 하시면 더 정확한 답변을 받을 수 있습니다.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["전문가 상담을 권장합니다."]


# 테스트 함수
def test_confidence_calculator():
    """신뢰도 계산기 테스트"""
    calculator = ConfidenceCalculator()
    
    # 테스트 데이터
    test_query = "손해배상 청구 방법"
    test_docs = [
        {
            "type": "law",
            "similarity": 0.8,
            "content": "민법 제750조 불법행위로 인한 손해배상",
            "law_name": "민법",
            "article_number": "제750조"
        },
        {
            "type": "precedent",
            "similarity": 0.7,
            "case_name": "손해배상청구 사건",
            "case_number": "2023다12345",
            "summary": "불법행위로 인한 손해배상청구에 관한 판례"
        }
    ]
    test_answer = """손해배상 청구 방법은 다음과 같습니다:

1. 불법행위 성립 요건 확인
   - 가해행위, 손해 발생, 인과관계, 고의 또는 과실

2. 적용 법률
   - 민법 제750조 (불법행위로 인한 손해배상)

3. 관련 판례
   - 2023다12345 손해배상청구 사건

4. 청구 절차
   - 소장 작성 및 제출
   - 증거 자료 준비
   - 법원에서 소송 진행"""
    
    print("=== 신뢰도 계산기 테스트 ===")
    print(f"질문: {test_query}")
    print(f"검색 문서 수: {len(test_docs)}")
    print(f"답변 길이: {len(test_answer)}")
    
    # 신뢰도 계산
    confidence_info = calculator.calculate_confidence(test_query, test_docs, test_answer)
    
    print(f"\n신뢰도 결과:")
    print(f"- 전체 신뢰도: {confidence_info.confidence:.3f}")
    print(f"- 신뢰도 수준: {confidence_info.reliability_level.value}")
    print(f"- 유사도 점수: {confidence_info.similarity_score:.3f}")
    print(f"- 매칭 점수: {confidence_info.matching_score:.3f}")
    print(f"- 답변 품질: {confidence_info.answer_quality:.3f}")
    
    print(f"\n경고:")
    for warning in confidence_info.warnings:
        print(f"- {warning}")
    
    print(f"\n권장사항:")
    for recommendation in confidence_info.recommendations:
        print(f"- {recommendation}")


if __name__ == "__main__":
    test_confidence_calculator()
