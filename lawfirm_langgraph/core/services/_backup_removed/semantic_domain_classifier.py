# -*- coding: utf-8 -*-
"""
의미 기반 도메인 분류 시스템
법률 질문의 의미를 분석하여 도메인을 분류하는 시스템
"""

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .question_classifier import QuestionType
from .unified_prompt_manager import LegalDomain, ModelType

logger = logging.getLogger(__name__)


@dataclass
class LegalTerm:
    """법률 용어 정보"""
    term: str
    domain: LegalDomain
    weight: float
    synonyms: List[str]
    related_terms: List[str]
    context_keywords: List[str]


@dataclass
class DomainScore:
    """도메인 점수 정보"""
    domain: LegalDomain
    score: float
    matched_terms: List[str]
    confidence: float
    reasoning: str


class LegalTermsDatabase:
    """법률 용어 데이터베이스"""

    def __init__(self, terms_file: str = "data/legal_terms_database.json"):
        """법률 용어 데이터베이스 초기화"""
        self.terms_file = Path(terms_file)
        self.terms_by_domain = {}
        self.terms_index = {}
        self.synonyms_index = {}
        self.related_terms_index = {}

        # 데이터베이스 로드
        self._load_legal_terms_database()

        logger.info(f"Legal terms database loaded: {len(self.terms_index)} terms")

    def _load_legal_terms_database(self):
        """법률 용어 데이터베이스 로드"""
        try:
            if self.terms_file.exists():
                with open(self.terms_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._parse_terms_data(data)
            else:
                # 기본 법률 용어 데이터베이스 생성
                self._create_default_terms_database()

        except Exception as e:
            logger.error(f"Error loading legal terms database: {e}")
            self._create_default_terms_database()

    def _create_default_terms_database(self):
        """기본 법률 용어 데이터베이스 생성"""
        default_terms = {
            "민사법": {
                "계약": {
                    "weight": 0.9,
                    "synonyms": ["계약서", "계약관계", "계약체결", "계약해지", "계약위반"],
                    "related_terms": ["쌍무계약", "편무계약", "유상계약", "무상계약"],
                    "context_keywords": ["체결", "해지", "위반", "이행", "채무", "채권"]
                },
                "손해배상": {
                    "weight": 0.8,
                    "synonyms": ["손해", "배상", "피해", "손해배상청구"],
                    "related_terms": ["불법행위", "과실", "고의", "과실상계"],
                    "context_keywords": ["사고", "과실", "고의", "피해자", "가해자"]
                },
                "소유권": {
                    "weight": 0.7,
                    "synonyms": ["소유", "소유자", "소유물", "소유권이전"],
                    "related_terms": ["점유", "소유권보전", "소유권이전등기"],
                    "context_keywords": ["부동산", "동산", "등기", "이전"]
                },
                "상속": {
                    "weight": 0.8,
                    "synonyms": ["상속분", "상속인", "상속재산", "상속포기"],
                    "related_terms": ["유언", "유증", "상속회복청구권"],
                    "context_keywords": ["사망", "유족", "재산", "분할"]
                },
                "불법행위": {
                    "weight": 0.8,
                    "synonyms": ["불법행위책임", "불법행위손해배상"],
                    "related_terms": ["과실", "고의", "과실상계", "손해배상"],
                    "context_keywords": ["사고", "과실", "고의", "피해", "손해"]
                }
            },
            "형사법": {
                "살인": {
                    "weight": 0.9,
                    "synonyms": ["살인죄", "살인범", "살인사건", "살인미수"],
                    "related_terms": ["존속살인", "영아살인", "살인교사"],
                    "context_keywords": ["사망", "생명", "범죄", "처벌"]
                },
                "절도": {
                    "weight": 0.8,
                    "synonyms": ["절도죄", "절도범", "절취", "절취죄"],
                    "related_terms": ["강도", "사기", "횡령", "배임"],
                    "context_keywords": ["재물", "타인", "점유", "불법"]
                },
                "사기": {
                    "weight": 0.8,
                    "synonyms": ["사기죄", "사기범", "사기사건"],
                    "related_terms": ["횡령", "배임", "절도", "강도"],
                    "context_keywords": ["기망", "착오", "재물", "이득"]
                },
                "강도": {
                    "weight": 0.8,
                    "synonyms": ["강도죄", "강도범", "강도사건"],
                    "related_terms": ["강도살인", "강도강간", "강도상해"],
                    "context_keywords": ["폭행", "협박", "재물", "강제"]
                }
            },
            "가족법": {
                "이혼": {
                    "weight": 0.9,
                    "synonyms": ["이혼신고", "이혼절차", "이혼소송"],
                    "related_terms": ["협의이혼", "재판이혼", "이혼사유"],
                    "context_keywords": ["부부", "혼인", "가정", "자녀"]
                },
                "혼인": {
                    "weight": 0.8,
                    "synonyms": ["결혼", "혼인신고", "혼인신고서"],
                    "related_terms": ["혼인무효", "혼인취소", "혼인신고"],
                    "context_keywords": ["부부", "가족", "신고", "절차"]
                },
                "친자": {
                    "weight": 0.7,
                    "synonyms": ["친자관계", "친자확인", "친자부인"],
                    "related_terms": ["친생자", "양자", "계부모"],
                    "context_keywords": ["부모", "자녀", "혈연", "관계"]
                }
            },
            "상사법": {
                "회사": {
                    "weight": 0.9,
                    "synonyms": ["주식회사", "유한회사", "합명회사", "합자회사"],
                    "related_terms": ["회사법", "정관", "주주", "이사"],
                    "context_keywords": ["설립", "해산", "합병", "분할"]
                },
                "주식": {
                    "weight": 0.8,
                    "synonyms": ["주식", "주권", "주주", "주주총회"],
                    "related_terms": ["신주발행", "주식양도", "주식매수청구권"],
                    "context_keywords": ["투자", "자본", "이익", "배당"]
                },
                "어음": {
                    "weight": 0.7,
                    "synonyms": ["어음", "수표", "어음법", "수표법"],
                    "related_terms": ["어음할인", "어음보증", "어음양도"],
                    "context_keywords": ["지급", "할인", "양도", "보증"]
                }
            },
            "노동법": {
                "근로": {
                    "weight": 0.9,
                    "synonyms": ["근로계약", "근로자", "근로시간", "근로조건"],
                    "related_terms": ["임금", "휴가", "해고", "부당해고"],
                    "context_keywords": ["고용", "직장", "업무", "시간"]
                },
                "해고": {
                    "weight": 0.8,
                    "synonyms": ["해고", "부당해고", "해고사유", "해고절차"],
                    "related_terms": ["징계", "정리해고", "해고제한"],
                    "context_keywords": ["고용", "직장", "징계", "사유"]
                },
                "임금": {
                    "weight": 0.8,
                    "synonyms": ["임금", "급여", "보수", "상여금"],
                    "related_terms": ["최저임금", "임금지급", "임금채권"],
                    "context_keywords": ["지급", "계산", "차감", "보장"]
                }
            },
            "부동산법": {
                "부동산": {
                    "weight": 0.9,
                    "synonyms": ["부동산", "토지", "건물", "아파트"],
                    "related_terms": ["부동산등기", "소유권이전", "매매계약"],
                    "context_keywords": ["매매", "임대", "등기", "소유권"]
                },
                "등기": {
                    "weight": 0.8,
                    "synonyms": ["등기", "등기부", "등기부등본"],
                    "related_terms": ["소유권이전등기", "저당권등기", "가등기"],
                    "context_keywords": ["부동산", "소유권", "이전", "절차"]
                }
            },
            "지적재산권법": {
                "특허": {
                    "weight": 0.9,
                    "synonyms": ["특허", "특허권", "특허출원", "특허침해"],
                    "related_terms": ["발명", "실용신안", "디자인"],
                    "context_keywords": ["발명", "기술", "출원", "등록"]
                },
                "상표": {
                    "weight": 0.8,
                    "synonyms": ["상표", "상표권", "상표출원", "상표침해"],
                    "related_terms": ["서비스표", "단체표장", "지리적표시"],
                    "context_keywords": ["브랜드", "출원", "등록", "사용"]
                },
                "저작권": {
                    "weight": 0.8,
                    "synonyms": ["저작권", "저작물", "저작권침해"],
                    "related_terms": ["저작인격권", "저작재산권", "공연권"],
                    "context_keywords": ["창작", "표현", "복제", "배포"]
                }
            },
            "세법": {
                "소득세": {
                    "weight": 0.9,
                    "synonyms": ["소득세", "소득세법", "소득세신고"],
                    "related_terms": ["종합소득세", "퇴직소득세", "양도소득세"],
                    "context_keywords": ["소득", "신고", "납부", "세액"]
                },
                "법인세": {
                    "weight": 0.8,
                    "synonyms": ["법인세", "법인세법", "법인세신고"],
                    "related_terms": ["법인", "사업소득", "이익"],
                    "context_keywords": ["법인", "사업", "이익", "세액"]
                },
                "부가가치세": {
                    "weight": 0.7,
                    "synonyms": ["부가가치세", "부가세", "VAT"],
                    "related_terms": ["공급", "매입", "매출"],
                    "context_keywords": ["공급", "매입", "매출", "세액"]
                }
            },
            "민사소송법": {
                "소송": {
                    "weight": 0.9,
                    "synonyms": ["소송", "민사소송", "소송제기", "소송절차"],
                    "related_terms": ["소장", "답변서", "준비서면"],
                    "context_keywords": ["법원", "재판", "절차", "증거"]
                },
                "재판": {
                    "weight": 0.8,
                    "synonyms": ["재판", "판결", "판정", "심리"],
                    "related_terms": ["법원", "법관", "변호사"],
                    "context_keywords": ["법원", "절차", "증거", "판결"]
                }
            },
            "형사소송법": {
                "수사": {
                    "weight": 0.9,
                    "synonyms": ["수사", "수사기관", "수사절차"],
                    "related_terms": ["검사", "경찰", "수사권"],
                    "context_keywords": ["범죄", "수사", "기소", "재판"]
                },
                "기소": {
                    "weight": 0.8,
                    "synonyms": ["기소", "기소권", "기소절차"],
                    "related_terms": ["검사", "공소", "공소제기"],
                    "context_keywords": ["범죄", "수사", "재판", "처벌"]
                }
            }
        }

        # 데이터베이스 저장
        self._save_terms_database(default_terms)
        self._parse_terms_data(default_terms)

    def _parse_terms_data(self, data: Dict[str, Any]):
        """용어 데이터 파싱"""
        self.terms_by_domain = {}
        self.terms_index = {}
        self.synonyms_index = {}
        self.related_terms_index = {}

        for domain_name, terms in data.items():
            domain = LegalDomain(domain_name)
            self.terms_by_domain[domain] = {}

            for term, info in terms.items():
                legal_term = LegalTerm(
                    term=term,
                    domain=domain,
                    weight=info["weight"],
                    synonyms=info["synonyms"],
                    related_terms=info["related_terms"],
                    context_keywords=info["context_keywords"]
                )

                # 도메인별 용어 저장
                self.terms_by_domain[domain][term] = legal_term

                # 전체 용어 인덱스
                self.terms_index[term] = legal_term

                # 동의어 인덱스
                for synonym in info["synonyms"]:
                    self.synonyms_index[synonym] = legal_term

                # 관련 용어 인덱스
                for related_term in info["related_terms"]:
                    self.related_terms_index[related_term] = legal_term

    def _save_terms_database(self, data: Dict[str, Any]):
        """용어 데이터베이스 저장"""
        try:
            self.terms_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.terms_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving terms database: {e}")

    def get_domain_scores(self, query: str) -> Dict[LegalDomain, DomainScore]:
        """질문의 도메인별 점수 계산"""
        query_lower = query.lower()
        domain_scores = {}

        # 법조문 패턴 인식 (우선 처리)
        law_patterns = {
            r'민법\s+제\d+조': LegalDomain.CIVIL_LAW,
            r'형법\s+제\d+조': LegalDomain.CRIMINAL_LAW,
            r'상법\s+제\d+조': LegalDomain.COMMERCIAL_LAW,
            r'근로기준법\s+제\d+조': LegalDomain.LABOR_LAW,
            r'부동산등기법\s+제\d+조': LegalDomain.PROPERTY_LAW,
            r'특허법\s+제\d+조': LegalDomain.INTELLECTUAL_PROPERTY,
            r'소득세법\s+제\d+조': LegalDomain.TAX_LAW,
            r'민사소송법\s+제\d+조': LegalDomain.CIVIL_PROCEDURE,
            r'형사소송법\s+제\d+조': LegalDomain.CRIMINAL_PROCEDURE
        }

        # 법조문 패턴 매칭
        for pattern, domain in law_patterns.items():
            if re.search(pattern, query):
                if domain not in domain_scores:
                    domain_scores[domain] = DomainScore(
                        domain=domain,
                        score=0.0,
                        matched_terms=[],
                        confidence=0.0,
                        reasoning=""
                    )

                match = re.search(pattern, query)
                matched_text = match.group(0)
                domain_scores[domain].score += 2.0  # 법조문은 높은 가중치
                domain_scores[domain].matched_terms.append(matched_text)
                domain_scores[domain].confidence = 0.9  # 법조문은 높은 신뢰도
                domain_scores[domain].reasoning = f"법조문 패턴 매칭: '{matched_text}'"

        for domain in LegalDomain:
            if domain == LegalDomain.GENERAL:
                continue

            if domain not in domain_scores:
                domain_scores[domain] = DomainScore(
                    domain=domain,
                    score=0.0,
                    matched_terms=[],
                    confidence=0.0,
                    reasoning="매칭된 용어 없음"
                )

            matched_terms = []
            total_score = 0.0
            reasoning_parts = []

            # 도메인별 용어 검사 (기존 점수에 추가)
            if domain in self.terms_by_domain:
                for term, legal_term in self.terms_by_domain[domain].items():
                    term_score = 0.0

                    # 직접 매칭
                    if term in query_lower:
                        term_score += legal_term.weight
                        matched_terms.append(term)
                        reasoning_parts.append(f"'{term}' 직접 매칭")

                    # 동의어 매칭
                    for synonym in legal_term.synonyms:
                        if synonym in query_lower:
                            term_score += legal_term.weight * 0.8
                            matched_terms.append(synonym)
                            reasoning_parts.append(f"'{synonym}' 동의어 매칭")

                    # 관련 용어 매칭
                    for related_term in legal_term.related_terms:
                        if related_term in query_lower:
                            term_score += legal_term.weight * 0.6
                            matched_terms.append(related_term)
                            reasoning_parts.append(f"'{related_term}' 관련 용어 매칭")

                    # 문맥 키워드 매칭
                    context_matches = sum(1 for keyword in legal_term.context_keywords if keyword in query_lower)
                    if context_matches > 0:
                        term_score += legal_term.weight * 0.4 * (context_matches / len(legal_term.context_keywords))
                        reasoning_parts.append(f"문맥 키워드 {context_matches}개 매칭")

                    total_score += term_score

            # 기존 점수에 추가
            domain_scores[domain].score += total_score
            domain_scores[domain].matched_terms.extend(matched_terms)

            # 신뢰도 재계산 (법조문 패턴이 있으면 높은 신뢰도 유지)
            if domain_scores[domain].confidence < 0.9:  # 법조문 패턴이 아닌 경우만 재계산
                confidence = min(1.0, domain_scores[domain].score / 2.0) if domain_scores[domain].score > 0 else 0.0
                domain_scores[domain].confidence = confidence

            # 추론 정보 업데이트
            if reasoning_parts:
                if domain_scores[domain].reasoning:
                    domain_scores[domain].reasoning += "; " + "; ".join(reasoning_parts)
                else:
                    domain_scores[domain].reasoning = "; ".join(reasoning_parts)

        return domain_scores


class LegalContextAnalyzer:
    """법률 문맥 분석기"""

    def __init__(self):
        """문맥 분석기 초기화"""
        self.legal_entity_patterns = [
            r'개인|자연인|법인|회사|국가|지방자치단체',
            r'피해자|가해자|원고|피고|원심|피심',
            r'채권자|채무자|매도인|매수인|임대인|임차인'
        ]

        self.legal_action_patterns = [
            r'계약체결|계약해지|계약위반|계약이행',
            r'손해배상청구|소송제기|고소|고발',
            r'해고|징계|임금지급|근로계약',
            r'등기신청|소유권이전|매매계약'
        ]

        self.temporal_indicators = [
            r'언제|언제부터|언제까지|기간|시효',
            r'과거|현재|미래|이전|이후',
            r'일정|예정|계획|진행|완료'
        ]

    def analyze_context(self, query: str) -> Dict[str, Any]:
        """문맥 분석"""
        context = {
            "legal_entities": self._extract_legal_entities(query),
            "legal_actions": self._extract_legal_actions(query),
            "temporal_indicators": self._extract_temporal_indicators(query),
            "question_type_indicators": self._extract_question_type_indicators(query)
        }
        return context

    def _extract_legal_entities(self, query: str) -> List[str]:
        """법적 주체 추출"""
        entities = []
        for pattern in self.legal_entity_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        return list(set(entities))

    def _extract_legal_actions(self, query: str) -> List[str]:
        """법적 행위 추출"""
        actions = []
        for pattern in self.legal_action_patterns:
            matches = re.findall(pattern, query)
            actions.extend(matches)
        return list(set(actions))

    def _extract_temporal_indicators(self, query: str) -> List[str]:
        """시간적 지표 추출"""
        indicators = []
        for pattern in self.temporal_indicators:
            matches = re.findall(pattern, query)
            indicators.extend(matches)
        return list(set(indicators))

    def _extract_question_type_indicators(self, query: str) -> List[str]:
        """질문 유형 지표 추출"""
        indicators = []

        # 질문 유형별 키워드
        question_keywords = {
            "what": ["무엇", "어떤", "무슨"],
            "how": ["어떻게", "방법", "절차"],
            "when": ["언제", "시기", "기간"],
            "why": ["왜", "이유", "원인"],
            "who": ["누구", "누가", "어떤 사람"]
        }

        for q_type, keywords in question_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    indicators.append(q_type)
                    break

        return indicators


class SemanticDomainClassifier:
    """의미 기반 도메인 분류기"""

    def __init__(self):
        """의미 기반 도메인 분류기 초기화"""
        self.terms_database = LegalTermsDatabase()
        self.context_analyzer = LegalContextAnalyzer()

        # 도메인별 가중치 (경험적 설정)
        self.domain_weights = {
            LegalDomain.CIVIL_LAW: 1.0,
            LegalDomain.CRIMINAL_LAW: 1.0,
            LegalDomain.FAMILY_LAW: 1.0,
            LegalDomain.COMMERCIAL_LAW: 1.0,
            LegalDomain.LABOR_LAW: 1.0,
            LegalDomain.PROPERTY_LAW: 1.0,
            LegalDomain.INTELLECTUAL_PROPERTY: 1.0,
            LegalDomain.TAX_LAW: 1.0,
            LegalDomain.CIVIL_PROCEDURE: 1.0,
            LegalDomain.CRIMINAL_PROCEDURE: 1.0
        }

        try:
            logger.info("Semantic domain classifier initialized")
        except Exception:
            # 로깅 오류를 무시하고 계속 진행
            pass

    def classify_domain(self, query: str, question_type: Optional[QuestionType] = None) -> Tuple[LegalDomain, float, str]:
        """의미 기반 도메인 분류"""
        try:
            # 1. 용어 기반 점수 계산
            domain_scores = self.terms_database.get_domain_scores(query)

            # 2. 문맥 분석
            context = self.context_analyzer.analyze_context(query)

            # 3. 문맥 가중치 적용
            adjusted_scores = self._apply_context_weights(domain_scores, context)

            # 4. 질문 유형 가중치 적용
            if question_type:
                adjusted_scores = self._apply_question_type_weights(adjusted_scores, question_type)

            # 5. 최종 도메인 결정
            final_domain, confidence, reasoning = self._determine_final_domain(adjusted_scores)

            return final_domain, confidence, reasoning

        except Exception as e:
            logger.error(f"Error in semantic domain classification: {e}")
            return LegalDomain.GENERAL, 0.0, f"분류 오류: {str(e)}"

    def _apply_context_weights(self, domain_scores: Dict[LegalDomain, DomainScore],
                             context: Dict[str, Any]) -> Dict[LegalDomain, DomainScore]:
        """문맥 가중치 적용"""
        adjusted_scores = {}

        for domain, score in domain_scores.items():
            if domain == LegalDomain.GENERAL:
                continue

            adjusted_score = score.score
            adjusted_reasoning = score.reasoning

            # 법적 행위에 따른 가중치
            if context["legal_actions"]:
                action_boost = self._calculate_action_boost(domain, context["legal_actions"])
                adjusted_score += action_boost
                if action_boost > 0:
                    adjusted_reasoning += f"; 법적 행위 가중치 +{action_boost:.2f}"

            # 법적 주체에 따른 가중치
            if context["legal_entities"]:
                entity_boost = self._calculate_entity_boost(domain, context["legal_entities"])
                adjusted_score += entity_boost
                if entity_boost > 0:
                    adjusted_reasoning += f"; 법적 주체 가중치 +{entity_boost:.2f}"

            # 신뢰도 재계산
            adjusted_confidence = min(1.0, adjusted_score / 2.0) if adjusted_score > 0 else 0.0

            adjusted_scores[domain] = DomainScore(
                domain=domain,
                score=adjusted_score,
                matched_terms=score.matched_terms,
                confidence=adjusted_confidence,
                reasoning=adjusted_reasoning
            )

        return adjusted_scores

    def _apply_question_type_weights(self, domain_scores: Dict[LegalDomain, DomainScore],
                                   question_type: QuestionType) -> Dict[LegalDomain, DomainScore]:
        """질문 유형 가중치 적용"""
        # 질문 유형별 도메인 선호도
        question_type_preferences = {
            QuestionType.PRECEDENT_SEARCH: [LegalDomain.CIVIL_LAW, LegalDomain.CRIMINAL_LAW],
            QuestionType.LAW_INQUIRY: [LegalDomain.GENERAL],
            QuestionType.LEGAL_ADVICE: [LegalDomain.CIVIL_LAW, LegalDomain.FAMILY_LAW],
            QuestionType.PROCEDURE_GUIDE: [LegalDomain.CIVIL_PROCEDURE, LegalDomain.CRIMINAL_PROCEDURE],
            QuestionType.TERM_EXPLANATION: [LegalDomain.GENERAL]
        }

        preferred_domains = question_type_preferences.get(question_type, [])

        for domain, score in domain_scores.items():
            if domain in preferred_domains:
                # 선호 도메인에 가중치 적용
                score.score += 0.5
                score.reasoning += f"; {question_type.value} 질문 유형 가중치 +0.5"

        return domain_scores

    def _calculate_action_boost(self, domain: LegalDomain, actions: List[str]) -> float:
        """법적 행위에 따른 가중치 계산"""
        action_weights = {
            LegalDomain.CIVIL_LAW: ["계약체결", "계약해지", "손해배상청구", "소유권이전"],
            LegalDomain.CRIMINAL_LAW: ["고소", "고발", "소송제기"],
            LegalDomain.LABOR_LAW: ["해고", "징계", "임금지급", "근로계약"],
            LegalDomain.PROPERTY_LAW: ["등기신청", "매매계약", "소유권이전"]
        }

        if domain in action_weights:
            matching_actions = [action for action in actions if action in action_weights[domain]]
            return len(matching_actions) * 0.3

        return 0.0

    def _calculate_entity_boost(self, domain: LegalDomain, entities: List[str]) -> float:
        """법적 주체에 따른 가중치 계산"""
        entity_weights = {
            LegalDomain.CIVIL_LAW: ["채권자", "채무자", "매도인", "매수인"],
            LegalDomain.CRIMINAL_LAW: ["피해자", "가해자"],
            LegalDomain.LABOR_LAW: ["근로자", "사용자"],
            LegalDomain.COMMERCIAL_LAW: ["회사", "주주", "이사"]
        }

        if domain in entity_weights:
            matching_entities = [entity for entity in entities if entity in entity_weights[domain]]
            return len(matching_entities) * 0.2

        return 0.0

    def _determine_final_domain(self, domain_scores: Dict[LegalDomain, DomainScore]) -> Tuple[LegalDomain, float, str]:
        """최종 도메인 결정"""
        if not domain_scores:
            return LegalDomain.GENERAL, 0.0, "분류 가능한 도메인 없음"

        # 점수 기준으로 정렬
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1].score, reverse=True)

        best_domain, best_score = sorted_domains[0]

        # 신뢰도가 낮으면 GENERAL로 분류
        if best_score.confidence < 0.3:
            return LegalDomain.GENERAL, best_score.confidence, f"신뢰도 부족 ({best_score.confidence:.2f}): {best_score.reasoning}"

        # 상위 3개 도메인 정보
        top_domains = sorted_domains[:3]
        reasoning = f"최고 점수: {best_domain.value} ({best_score.score:.2f}, 신뢰도: {best_score.confidence:.2f})"

        if len(top_domains) > 1:
            reasoning += f" | 2위: {top_domains[1][0].value} ({top_domains[1][1].score:.2f})"
        if len(top_domains) > 2:
            reasoning += f" | 3위: {top_domains[2][0].value} ({top_domains[2][1].score:.2f})"

        reasoning += f" | 상세: {best_score.reasoning}"

        return best_domain, best_score.confidence, reasoning


# 전역 인스턴스 (안전한 초기화)
_semantic_domain_classifier = None

def get_semantic_domain_classifier():
    """전역 SemanticDomainClassifier 인스턴스를 가져오거나 생성"""
    global _semantic_domain_classifier
    if _semantic_domain_classifier is None:
        try:
            _semantic_domain_classifier = SemanticDomainClassifier()
        except Exception as e:
            logger.warning(f"Failed to initialize semantic_domain_classifier: {e}")
            # 더미 객체 반환 또는 None 반환
            _semantic_domain_classifier = None
    return _semantic_domain_classifier

# 하위 호환성을 위한 전역 변수 (지연 초기화)
def _get_semantic_domain_classifier_global():
    """하위 호환성을 위한 전역 변수 접근"""
    return get_semantic_domain_classifier()

# 속성 접근 방식으로 사용 가능하도록 설정
class _SemanticDomainClassifierProxy:
    """전역 semantic_domain_classifier에 대한 프록시"""
    def __getattr__(self, name):
        instance = get_semantic_domain_classifier()
        if instance is None:
            raise RuntimeError("SemanticDomainClassifier is not available")
        return getattr(instance, name)

# 전역 변수로 프록시 설정 (하위 호환성)
semantic_domain_classifier = _SemanticDomainClassifierProxy()
