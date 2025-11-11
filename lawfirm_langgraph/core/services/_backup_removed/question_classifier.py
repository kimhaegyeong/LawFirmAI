# -*- coding: utf-8 -*-
"""
질문 유형 분류기
사용자 질문을 분석하여 질문 유형을 분류하고 검색 가중치를 결정
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """질문 유형"""
    PRECEDENT_SEARCH = "precedent_search"
    LAW_INQUIRY = "law_inquiry"
    LEGAL_ADVICE = "legal_advice"
    PROCEDURE_GUIDE = "procedure_guide"
    TERM_EXPLANATION = "term_explanation"
    GENERAL_QUESTION = "general_question"


@dataclass
class QuestionClassification:
    """질문 분류 결과"""
    question_type: QuestionType
    law_weight: float
    precedent_weight: float
    confidence: float
    keywords: List[str]
    patterns: List[str]


class QuestionClassifier:
    """질문 유형 분류기"""

    def __init__(self):
        """질문 분류기 초기화"""
        self.logger = logging.getLogger(__name__)

        # 질문 유형별 키워드 패턴
        self.question_patterns = {
            QuestionType.PRECEDENT_SEARCH: {
                "keywords": [
                    "판례", "사건", "법원", "판결", "대법원", "지방법원", "고등법원",
                    "판시사항", "판결요지", "참고판례", "유사사건", "선례", "사례"
                ],
                "patterns": [
                    r".*판례.*찾아.*",
                    r".*사건.*있.*",
                    r".*법원.*판결.*",
                    r".*유사.*사례.*",
                    r".*참고.*판례.*"
                ],
                "law_weight": 0.2,
                "precedent_weight": 0.8
            },

            QuestionType.LAW_INQUIRY: {
                "keywords": [
                    "법률", "조문", "법령", "규정", "법조문", "법률해석", "법적근거",
                    "민법", "형법", "상법", "노동법", "행정법", "헌법", "소송법"
                ],
                "patterns": [
                    r".*법률.*무엇.*",
                    r".*조문.*내용.*",
                    r".*법령.*규정.*",
                    r".*법적.*근거.*",
                    r".*법률.*해석.*"
                ],
                "law_weight": 0.8,
                "precedent_weight": 0.2
            },

            QuestionType.LEGAL_ADVICE: {
                "keywords": [
                    "조언", "상담", "해결방법", "어떻게", "해야", "방법", "절차",
                    "권리", "의무", "책임", "손해배상", "소송", "분쟁해결"
                ],
                "patterns": [
                    r".*어떻게.*해야.*",
                    r".*방법.*알려.*",
                    r".*조언.*해주.*",
                    r".*해결.*방법.*",
                    r".*어떤.*절차.*"
                ],
                "law_weight": 0.5,
                "precedent_weight": 0.5
            },

            QuestionType.PROCEDURE_GUIDE: {
                "keywords": [
                    "절차", "신청", "제출", "서류", "기간", "비용", "처리",
                    "소송절차", "신고절차", "등기절차", "허가절차", "승인절차"
                ],
                "patterns": [
                    r".*절차.*어떻게.*",
                    r".*신청.*방법.*",
                    r".*서류.*무엇.*",
                    r".*기간.*얼마.*",
                    r".*비용.*얼마.*"
                ],
                "law_weight": 0.6,
                "precedent_weight": 0.4
            },

            QuestionType.TERM_EXPLANATION: {
                "keywords": [
                    "의미", "정의", "뜻", "개념", "용어", "해설", "설명",
                    "무엇인가", "무엇을", "어떤", "정의가"
                ],
                "patterns": [
                    r".*의미.*무엇.*",
                    r".*정의.*알려.*",
                    r".*뜻.*무엇.*",
                    r".*개념.*설명.*",
                    r".*용어.*해설.*"
                ],
                "law_weight": 0.7,
                "precedent_weight": 0.3
            },

            QuestionType.GENERAL_QUESTION: {
                "keywords": [
                    "질문", "궁금", "알고", "싶", "문의", "확인", "찾아"
                ],
                "patterns": [
                    r".*궁금.*",
                    r".*알고.*싶.*",
                    r".*문의.*",
                    r".*확인.*"
                ],
                "law_weight": 0.4,
                "precedent_weight": 0.4
            }
        }

        # 법률 분야별 키워드 (현재 지원 도메인만)
        # 지원 도메인: 민사법, 형사법, 행정법, 지식재산권법
        self.legal_domains = {
            "민사": ["계약", "손해배상", "임대차", "상속", "부동산", "채권", "채무", "민법"],
            "형사": ["범죄", "형벌", "벌금", "징역", "교통사고", "절도", "사기", "폭행", "형법"],
            "행정": ["허가", "신고", "행정처분", "행정소송", "행정심판", "공무원", "행정법"],
            "지적재산": ["특허", "상표", "저작권", "디자인", "영업비밀", "지적재산권", "지식재산권"]
        }

    def classify_question(self, question: str) -> QuestionClassification:
        """
        질문 분류

        Args:
            question: 사용자 질문

        Returns:
            QuestionClassification: 분류 결과
        """
        try:
            self.logger.info(f"Classifying question: {question[:50]}...")

            # 질문 전처리
            processed_question = self._preprocess_question(question)

            # 각 질문 유형별 점수 계산
            scores = {}
            matched_keywords = {}
            matched_patterns = {}

            for question_type, config in self.question_patterns.items():
                score = 0
                keywords = []
                patterns = []

                # 키워드 매칭
                for keyword in config["keywords"]:
                    if keyword in processed_question:
                        score += 1
                        keywords.append(keyword)

                # 패턴 매칭
                for pattern in config["patterns"]:
                    if re.search(pattern, processed_question):
                        score += 2  # 패턴 매칭은 더 높은 점수
                        patterns.append(pattern)

                scores[question_type] = score
                matched_keywords[question_type] = keywords
                matched_patterns[question_type] = patterns

            # 최고 점수 질문 유형 선택
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]

            # 신뢰도 계산
            total_possible_score = len(self.question_patterns[best_type]["keywords"]) + len(self.question_patterns[best_type]["patterns"]) * 2
            confidence = min(best_score / total_possible_score, 1.0) if total_possible_score > 0 else 0.0

            # 신뢰도가 낮으면 일반 질문으로 분류
            if confidence < 0.3:
                best_type = QuestionType.GENERAL_QUESTION
                confidence = 0.5

            # 법률/판례 가중치 결정
            config = self.question_patterns[best_type]
            law_weight = config["law_weight"]
            precedent_weight = config["precedent_weight"]

            # 법률 분야별 가중치 조정
            domain_weights = self._get_domain_weights(processed_question)
            if domain_weights:
                law_weight = max(law_weight, domain_weights["law"])
                precedent_weight = max(precedent_weight, domain_weights["precedent"])

            result = QuestionClassification(
                question_type=best_type,
                law_weight=law_weight,
                precedent_weight=precedent_weight,
                confidence=confidence,
                keywords=matched_keywords[best_type],
                patterns=matched_patterns[best_type]
            )

            self.logger.info(f"Question classified as: {best_type.value} "
                           f"(confidence: {confidence:.2f}, law_weight: {law_weight}, precedent_weight: {precedent_weight})")

            return result

        except Exception as e:
            self.logger.error(f"Error classifying question: {e}")
            # 오류 시 기본 분류 반환
            return QuestionClassification(
                question_type=QuestionType.GENERAL_QUESTION,
                law_weight=0.5,
                precedent_weight=0.5,
                confidence=0.3,
                keywords=[],
                patterns=[]
            )

    def _preprocess_question(self, question: str) -> str:
        """질문 전처리"""
        try:
            # 소문자 변환
            processed = question.lower()

            # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
            processed = re.sub(r'[^\w\s가-힣]', ' ', processed)

            # 연속된 공백 제거
            processed = re.sub(r'\s+', ' ', processed).strip()

            return processed

        except Exception as e:
            self.logger.error(f"Error preprocessing question: {e}")
            return question

    def _get_domain_weights(self, question: str) -> Optional[Dict[str, float]]:
        """법률 분야별 가중치 계산"""
        try:
            domain_scores = {}

            for domain, keywords in self.legal_domains.items():
                score = 0
                for keyword in keywords:
                    if keyword in question:
                        score += 1

                if score > 0:
                    domain_scores[domain] = score

            if not domain_scores:
                return None

            # 가장 높은 점수의 분야 선택
            best_domain = max(domain_scores, key=domain_scores.get)
            best_score = domain_scores[best_domain]

            # 분야별 특성에 따른 가중치 조정 (지원 도메인만)
            if best_domain in ["민사", "형사"]:
                # 민사/형사는 판례가 중요
                return {"law": 0.4, "precedent": 0.6}
            elif best_domain == "행정":
                # 행정은 법률이 중요
                return {"law": 0.7, "precedent": 0.3}
            elif best_domain == "지적재산":
                # 지적재산은 균형
                return {"law": 0.5, "precedent": 0.5}
            else:
                # 기본값 (지원되지 않는 도메인은 기본 가중치)
                return {"law": 0.5, "precedent": 0.5}

        except Exception as e:
            self.logger.error(f"Error calculating domain weights: {e}")
            return None

    def get_question_type_description(self, question_type: QuestionType) -> str:
        """질문 유형 설명 반환"""
        descriptions = {
            QuestionType.PRECEDENT_SEARCH: "판례 검색 - 관련 판례나 사건을 찾는 질문",
            QuestionType.LAW_INQUIRY: "법률 문의 - 법률 조문이나 법령에 대한 질문",
            QuestionType.LEGAL_ADVICE: "법적 조언 - 구체적인 해결방법이나 조언을 요청하는 질문",
            QuestionType.PROCEDURE_GUIDE: "절차 안내 - 법적 절차나 신청 방법에 대한 질문",
            QuestionType.TERM_EXPLANATION: "용어 해설 - 법률 용어나 개념에 대한 설명 요청",
            QuestionType.GENERAL_QUESTION: "일반 질문 - 기타 법률 관련 일반적인 질문"
        }
        return descriptions.get(question_type, "알 수 없는 질문 유형")

    def get_supported_question_types(self) -> List[Dict[str, Any]]:
        """지원하는 질문 유형 목록 반환"""
        try:
            types = []
            for question_type in QuestionType:
                config = self.question_patterns[question_type]
                types.append({
                    "type": question_type.value,
                    "description": self.get_question_type_description(question_type),
                    "keywords": config["keywords"][:5],  # 상위 5개 키워드만
                    "law_weight": config["law_weight"],
                    "precedent_weight": config["precedent_weight"]
                })
            return types

        except Exception as e:
            self.logger.error(f"Error getting supported question types: {e}")
            return []


# 테스트 함수
def test_question_classifier():
    """질문 분류기 테스트"""
    classifier = QuestionClassifier()

    # 테스트 질문들
    test_questions = [
        "손해배상 관련 판례를 찾아주세요",
        "민법 제750조의 내용이 무엇인가요?",
        "이혼 절차는 어떻게 진행하나요?",
        "불법행위의 정의를 알려주세요",
        "계약 해제 방법을 조언해주세요",
        "법률에 대해 궁금한 것이 있습니다"
    ]

    print("=== 질문 분류기 테스트 ===")

    for question in test_questions:
        print(f"\n질문: {question}")

        try:
            result = classifier.classify_question(question)

            print(f"분류 결과:")
            print(f"- 질문 유형: {result.question_type.value}")
            print(f"- 설명: {classifier.get_question_type_description(result.question_type)}")
            print(f"- 신뢰도: {result.confidence:.2f}")
            print(f"- 법률 가중치: {result.law_weight}")
            print(f"- 판례 가중치: {result.precedent_weight}")
            print(f"- 매칭된 키워드: {result.keywords}")
            print(f"- 매칭된 패턴: {result.patterns}")

        except Exception as e:
            print(f"분류 실패: {e}")

    # 지원하는 질문 유형 목록
    print(f"\n=== 지원하는 질문 유형 ===")
    supported_types = classifier.get_supported_question_types()
    for type_info in supported_types:
        print(f"- {type_info['type']}: {type_info['description']}")
        print(f"  키워드: {', '.join(type_info['keywords'])}")
        print(f"  가중치: 법률 {type_info['law_weight']}, 판례 {type_info['precedent_weight']}")


if __name__ == "__main__":
    test_question_classifier()
