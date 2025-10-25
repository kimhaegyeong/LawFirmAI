#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합된 하이브리드 질문 분류 시스템
기존 시스템과 호환되는 버전
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .unified_question_types import UnifiedQuestionType
from .question_type_adapter import QuestionTypeAdapter

@dataclass
class IntegratedClassificationResult:
    """통합된 분류 결과"""
    question_type: UnifiedQuestionType
    confidence: float
    method: str  # 'rule_based', 'ml_based', 'hybrid'
    features: Dict[str, Any] = None
    reasoning: str = ""
    
    # 기존 시스템과의 호환성을 위한 속성들
    @property
    def question_type_value(self) -> str:
        return self.question_type.value
    
    @property
    def law_weight(self) -> float:
        """법률 가중치 (기존 시스템 호환)"""
        return QuestionTypeAdapter.get_law_weight(self.question_type)
    
    @property
    def precedent_weight(self) -> float:
        """판례 가중치 (기존 시스템 호환)"""
        return QuestionTypeAdapter.get_precedent_weight(self.question_type)

class IntegratedRuleBasedClassifier:
    """통합된 규칙 기반 분류기"""
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self.keyword_weights = self._load_keyword_weights()
        self.conflict_rules = self._load_conflict_rules()
    
    def _load_patterns(self) -> Dict[UnifiedQuestionType, List[str]]:
        """패턴 로딩 - 기존 시스템과 통합"""
        return {
            UnifiedQuestionType.LAW_INQUIRY: [
                r'제\d+조', r'제\d+항', r'제\d+호',
                r'\w+법\s+제\d+조', r'\w+법\s+제\d+항',
                r'민법', r'형법', r'상법', r'노동법', r'행정법'
            ],
            UnifiedQuestionType.PRECEDENT_SEARCH: [
                r'판례를?\s+찾아주세요', r'관련\s+판례', r'유사한\s+판례',
                r'최근\s+판례', r'대법원\s+판례', r'고등법원\s+판례',
                r'사건', r'판결', r'선례'
            ],
            UnifiedQuestionType.CONTRACT_REVIEW: [
                r'계약서를?\s+검토', r'계약서를?\s+수정', r'계약서를?\s+작성',
                r'계약\s+조항', r'불리한\s+조항', r'계약서의?\s+문제점'
            ],
            UnifiedQuestionType.DIVORCE_PROCEDURE: [
                r'이혼\s+절차', r'이혼\s+방법', r'이혼\s+신청',
                r'협의이혼\s+절차', r'재판이혼\s+절차'
            ],
            UnifiedQuestionType.INHERITANCE_PROCEDURE: [
                r'상속\s+절차', r'상속\s+신청', r'유산\s+분할',
                r'상속인\s+확인', r'상속세\s+신고', r'유언\s+검인'
            ],
            UnifiedQuestionType.CRIMINAL_CASE: [
                r'\w+죄\s+구성요건', r'\w+죄\s+처벌', r'\w+죄\s+형량',
                r'\w+범죄\s+처벌', r'형사\s+사건', r'범죄\s+구성요건'
            ],
            UnifiedQuestionType.LABOR_DISPUTE: [
                r'노동\s+분쟁', r'근로\s+분쟁', r'임금\s+체불',
                r'부당해고\s+구제', r'해고\s+대응', r'근로시간\s+규정'
            ],
            UnifiedQuestionType.PROCEDURE_GUIDE: [
                r'\w+\s+절차', r'\w+\s+신청', r'소액사건\s+절차',
                r'민사조정\s+신청', r'소송\s+제기', r'어떻게\s+신청'
            ],
            UnifiedQuestionType.TERM_EXPLANATION: [
                r'무엇이\s+\w+인가요?', r'\w+의\s+의미는?', r'\w+의\s+정의는?',
                r'\w+의\s+개념은?', r'\w+이\s+무엇인가요?', r'\w+란\s+무엇인가요?'
            ],
            UnifiedQuestionType.LEGAL_ADVICE: [
                r'어떻게\s+대응해야', r'권리\s+구제', r'의무\s+이행',
                r'법률\s+상담', r'법률\s+자문', r'변호사\s+상담',
                r'법적\s+대응', r'법적\s+조언', r'법적\s+보호'
            ],
            UnifiedQuestionType.GENERAL_QUESTION: [
                r'안녕', r'도움', r'질문', r'궁금'
            ]
        }
    
    def _load_keyword_weights(self) -> Dict[str, Dict[str, float]]:
        """키워드 가중치 로딩"""
        return {
            UnifiedQuestionType.LEGAL_ADVICE: {
                '법적': 3.0, '권리': 2.5, '의무': 2.5, '구제': 2.5,
                '보호': 2.0, '해결': 2.0, '대응': 2.0, '상담': 2.0,
                '자문': 2.0, '조언': 2.0, '도움': 1.5, '지원': 1.5
            },
            UnifiedQuestionType.PROCEDURE_GUIDE: {
                '절차': 3.0, '신청': 2.5, '제출': 2.0, '처리': 2.0,
                '소송': 2.0, '민사조정': 2.0, '소액사건': 2.0
            },
            UnifiedQuestionType.TERM_EXPLANATION: {
                '의미': 3.0, '정의': 3.0, '개념': 3.0, '설명': 2.5,
                '무엇': 2.0, '뜻': 2.0, '내용': 1.5
            }
        }
    
    def _load_conflict_rules(self) -> Dict[str, List[str]]:
        """충돌 해결 규칙"""
        return {
            'high_priority': ['계약서', '이혼', '상속', '형사'],
            'exclude_patterns': ['안녕', '감사', '고마워']
        }
    
    def classify(self, question: str) -> IntegratedClassificationResult:
        """통합된 규칙 기반 분류"""
        question_lower = question.lower().strip()
        
        if not question_lower:
            return IntegratedClassificationResult(
                question_type=UnifiedQuestionType.GENERAL_QUESTION,
                confidence=1.0,
                method='rule_based',
                reasoning="빈 질문"
            )
        
        # 점수 계산
        scores = self._calculate_scores(question_lower)
        
        # 최고 점수 질문 유형 선택
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])[0]
            best_score = scores[best_type]
        else:
            best_type = UnifiedQuestionType.GENERAL_QUESTION
            best_score = 0
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(question_lower, best_type, scores)
        
        return IntegratedClassificationResult(
            question_type=best_type,
            confidence=confidence,
            method='rule_based',
            features={'scores': scores},
            reasoning=f"규칙 기반 분류: {best_type.value}"
        )
    
    def _calculate_scores(self, question_lower: str) -> Dict[UnifiedQuestionType, float]:
        """각 질문 유형별 점수 계산"""
        scores = {}
        
        for question_type, patterns in self.patterns.items():
            score = 0.0
            
            # 패턴 매칭 점수
            pattern_matches = sum(1 for pattern in patterns if re.search(pattern, question_lower))
            score += pattern_matches * 2.0
            
            # 키워드 가중치 점수
            if question_type in self.keyword_weights:
                for keyword, weight in self.keyword_weights[question_type].items():
                    if keyword in question_lower:
                        score += weight
            
            scores[question_type] = score
        
        return scores
    
    def _calculate_confidence(self, question_lower: str, question_type: UnifiedQuestionType, scores: Dict[UnifiedQuestionType, float]) -> float:
        """신뢰도 계산"""
        if not scores:
            return 0.5
        
        max_score = max(scores.values())
        total_score = sum(scores.values())
        
        if total_score == 0:
            return 0.5
        
        # 점수 기반 신뢰도
        score_confidence = min(max_score / total_score, 1.0)
        
        # 질문 길이 기반 신뢰도
        length_confidence = min(len(question_lower) / 20, 1.0)
        
        # 최종 신뢰도
        confidence = (score_confidence * 0.7 + length_confidence * 0.3)
        
        return min(max(confidence, 0.1), 1.0)

class IntegratedMLBasedClassifier:
    """통합된 ML 기반 분류기"""
    
    def __init__(self, model_path: str = "models/integrated_question_classifier.pkl"):
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_trained = False
        
        # 모델 로드 시도
        self._load_model()
    
    def _load_model(self):
        """저장된 모델 로드"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.vectorizer = model_data['vectorizer']
                self.label_encoder = model_data['label_encoder']
                self.is_trained = True
                print(f"통합 ML 모델 로드 완료: {self.model_path}")
        except Exception as e:
            print(f"통합 ML 모델 로드 실패: {e}")
            self.is_trained = False
    
    def train(self, training_data: List[Tuple[str, UnifiedQuestionType]]):
        """모델 훈련"""
        if not training_data:
            print("훈련 데이터가 없습니다.")
            return
        
        # 데이터 준비
        questions = [data[0] for data in training_data]
        labels = [data[1].value for data in training_data]
        
        # TF-IDF 벡터화
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words=None,
            min_df=2,
            max_df=0.95
        )
        
        X = self.vectorizer.fit_transform(questions)
        
        # 라벨 인코딩
        unique_labels = list(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        y = [self.label_encoder[label] for label in labels]
        
        # 모델 훈련 (앙상블)
        models = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            model.fit(X, y)
            score = model.score(X, y)
            print(f"{name} 정확도: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        self.model = best_model
        self.is_trained = True
        
        # 모델 저장
        self._save_model()
        
        print(f"통합 ML 모델 훈련 완료. 최고 정확도: {best_score:.3f}")
    
    def _save_model(self):
        """모델 저장"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder
            }
            
            joblib.dump(model_data, self.model_path)
            print(f"통합 ML 모델 저장 완료: {self.model_path}")
        except Exception as e:
            print(f"통합 ML 모델 저장 실패: {e}")
    
    def classify(self, question: str) -> IntegratedClassificationResult:
        """통합된 ML 기반 분류"""
        if not self.is_trained:
            return IntegratedClassificationResult(
                question_type=UnifiedQuestionType.GENERAL_QUESTION,
                confidence=0.5,
                method='ml_based',
                reasoning="ML 모델이 훈련되지 않음"
            )
        
        try:
            # 질문 벡터화
            X = self.vectorizer.transform([question])
            
            # 예측
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # 라벨 디코딩
            predicted_label = None
            for label, idx in self.label_encoder.items():
                if idx == prediction:
                    predicted_label = label
                    break
            
            if predicted_label is None:
                predicted_label = UnifiedQuestionType.GENERAL_QUESTION.value
            
            # 신뢰도 계산
            confidence = max(probabilities)
            
            return IntegratedClassificationResult(
                question_type=UnifiedQuestionType(predicted_label),
                confidence=confidence,
                method='ml_based',
                features={'probabilities': probabilities.tolist()},
                reasoning=f"ML 기반 분류: {predicted_label}"
            )
            
        except Exception as e:
            return IntegratedClassificationResult(
                question_type=UnifiedQuestionType.GENERAL_QUESTION,
                confidence=0.5,
                method='ml_based',
                reasoning=f"ML 분류 오류: {e}"
            )

class IntegratedHybridQuestionClassifier:
    """통합된 하이브리드 질문 분류기"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.rule_based = IntegratedRuleBasedClassifier()
        self.ml_based = IntegratedMLBasedClassifier()
        
        # 성능 통계
        self.stats = {
            'rule_based_calls': 0,
            'ml_based_calls': 0,
            'hybrid_calls': 0,
            'total_calls': 0
        }
    
    def classify(self, question: str) -> IntegratedClassificationResult:
        """통합된 하이브리드 분류"""
        self.stats['total_calls'] += 1
        
        # 규칙 기반 분류
        rule_result = self.rule_based.classify(question)
        self.stats['rule_based_calls'] += 1
        
        # 신뢰도가 임계값 이상이면 규칙 기반 결과 반환
        if rule_result.confidence >= self.confidence_threshold:
            # 키워드 정보 추가
            rule_result.features = self._extract_keywords_and_features(question, rule_result)
            return rule_result
        
        # ML 모델이 훈련되어 있으면 ML 기반 분류 시도
        if self.ml_based.is_trained:
            ml_result = self.ml_based.classify(question)
            self.stats['ml_based_calls'] += 1
            
            # ML 결과의 신뢰도가 더 높으면 ML 결과 반환
            if ml_result.confidence > rule_result.confidence:
                ml_result.method = 'hybrid'
                ml_result.reasoning = f"하이브리드 분류: ML 신뢰도({ml_result.confidence:.3f}) > 규칙 신뢰도({rule_result.confidence:.3f})"
                ml_result.features = self._extract_keywords_and_features(question, ml_result)
                self.stats['hybrid_calls'] += 1
                return ml_result
        
        # 규칙 기반 결과 반환 (신뢰도가 낮아도)
        rule_result.method = 'hybrid'
        rule_result.reasoning = f"하이브리드 분류: 규칙 기반 결과 사용 (신뢰도: {rule_result.confidence:.3f})"
        rule_result.features = self._extract_keywords_and_features(question, rule_result)
        self.stats['hybrid_calls'] += 1
        return rule_result
    
    def _extract_keywords_and_features(self, question: str, result: IntegratedClassificationResult) -> Dict[str, Any]:
        """질문에서 키워드와 특징 추출"""
        try:
            question_lower = question.lower()
            
            # 법률 도메인 키워드 정의 (하이브리드 분류기 내부로 이동)
            LEGAL_DOMAIN_KEYWORDS = {
                "civil_law": {
                    "primary": ["민법", "계약", "손해배상", "불법행위", "채권", "채무", "소유권", "물권"],
                    "secondary": ["계약서", "위약금", "손해", "배상", "채권자", "채무자", "소유자"],
                    "exclude": ["형법", "형사", "범죄", "처벌"]
                },
                "criminal_law": {
                    "primary": ["형법", "범죄", "처벌", "형량", "구성요건", "고의", "과실"],
                    "secondary": ["사기", "절도", "강도", "살인", "상해", "폭행", "협박"],
                    "exclude": ["민법", "계약", "손해배상"]
                },
                "family_law": {
                    "primary": ["이혼", "상속", "양육권", "친권", "위자료", "재산분할", "유언"],
                    "secondary": ["협의이혼", "재판이혼", "상속인", "상속세", "유산", "양육비"],
                    "exclude": ["회사", "상법", "주식"]
                },
                "commercial_law": {
                    "primary": ["상법", "회사", "주식", "이사", "주주", "회사설립", "합병"],
                    "secondary": ["주식회사", "유한회사", "합명회사", "합자회사", "자본금", "정관"],
                    "exclude": ["이혼", "상속", "가족"]
                },
                "labor_law": {
                    "primary": ["노동법", "근로", "임금", "해고", "근로시간", "휴게시간", "연차"],
                    "secondary": ["근로계약서", "임금체불", "부당해고", "노동위원회", "최저임금"],
                    "exclude": ["이혼", "상속", "범죄"]
                },
                "real_estate": {
                    "primary": ["부동산", "매매", "임대차", "등기", "소유권이전", "전세", "월세"],
                    "secondary": ["부동산등기법", "매매계약서", "임대차계약서", "등기부등본"],
                    "exclude": ["이혼", "상속", "범죄"]
                },
                "general": {
                    "primary": ["법률", "법령", "조문", "법원", "판례", "소송"],
                    "secondary": ["법적", "법률적", "법적근거", "법적효력"],
                    "exclude": []
                }
            }
            
            # 키워드 추출
            extracted_keywords = []
            domain_scores = {}
            
            for domain, keywords_config in LEGAL_DOMAIN_KEYWORDS.items():
                score = 0.0
                domain_keywords = []
                
                # 주요 키워드 매칭
                for keyword in keywords_config.get("primary", []):
                    if keyword in question_lower:
                        score += 3.0
                        domain_keywords.append(keyword)
                        extracted_keywords.append(keyword)
                
                # 보조 키워드 매칭
                for keyword in keywords_config.get("secondary", []):
                    if keyword in question_lower:
                        score += 1.0
                        domain_keywords.append(keyword)
                        extracted_keywords.append(keyword)
                
                # 제외 키워드 매칭
                for keyword in keywords_config.get("exclude", []):
                    if keyword in question_lower:
                        score -= 2.0
                
                domain_scores[domain] = max(0, score)
            
            # 법률 조문 패턴 검색
            import re
            statute_patterns = [
                r'(민법|형법|상법|노동법|가족법|행정법|헌법|민사소송법|형사소송법|노동기준법|가족관계등록법)\s*제\s*(\d+)\s*조',
                r'(민법|형법|상법|노동법|가족법|행정법|헌법|민사소송법|형사소송법|노동기준법|가족관계등록법)제\s*(\d+)\s*조',
                r'(민법|형법|상법|노동법|가족법|행정법|헌법|민사소송법|형사소송법|노동기준법|가족관계등록법)\s+(\d+)\s*조',
                r'제\s*(\d+)\s*조',
                r'(\d+)\s*조'
            ]
            
            statute_match = None
            statute_law = None
            statute_article = None
            
            for pattern in statute_patterns:
                match = re.search(pattern, question)
                if match:
                    statute_match = match
                    groups = match.groups()
                    
                    if len(groups) == 2:
                        statute_law = groups[0]
                        statute_article = groups[1]
                    elif len(groups) == 1:
                        statute_article = groups[0]
                    break
            
            # 중복 제거
            extracted_keywords = list(set(extracted_keywords))
            
            return {
                "keywords": extracted_keywords,
                "domain_scores": domain_scores,
                "statute_match": statute_match.group(0) if statute_match else None,
                "statute_law": statute_law,
                "statute_article": statute_article,
                "question_length": len(question),
                "word_count": len(question.split()),
                "has_question_mark": "?" in question,
                "has_legal_terms": len(extracted_keywords) > 0
            }
            
        except Exception as e:
            return {
                "keywords": [],
                "domain_scores": {},
                "statute_match": None,
                "statute_law": None,
                "statute_article": None,
                "error": str(e)
            }
    
    def get_domain_from_question_type(self, question_type: UnifiedQuestionType) -> str:
        """질문 유형에서 도메인 추출"""
        domain_mapping = {
            UnifiedQuestionType.CONTRACT_REVIEW: "civil_law",
            UnifiedQuestionType.DIVORCE_PROCEDURE: "family_law", 
            UnifiedQuestionType.INHERITANCE_PROCEDURE: "family_law",
            UnifiedQuestionType.CRIMINAL_CASE: "criminal_law",
            UnifiedQuestionType.LABOR_DISPUTE: "labor_law",
            UnifiedQuestionType.PRECEDENT_SEARCH: "general",
            UnifiedQuestionType.LAW_INQUIRY: "general",
            UnifiedQuestionType.LEGAL_ADVICE: "general",
            UnifiedQuestionType.PROCEDURE_GUIDE: "general",
            UnifiedQuestionType.TERM_EXPLANATION: "general",
            UnifiedQuestionType.GENERAL_QUESTION: "general"
        }
        return domain_mapping.get(question_type, "general")
    
    def get_enhanced_domain_analysis(self, question: str, result: IntegratedClassificationResult) -> Dict[str, Any]:
        """향상된 도메인 분석"""
        try:
            # 기본 도메인 매핑
            base_domain = self.get_domain_from_question_type(result.question_type)
            
            # 키워드 기반 도메인 점수 계산
            features = result.features or {}
            domain_scores = features.get("domain_scores", {})
            
            # 가장 높은 점수의 도메인 찾기
            if domain_scores:
                best_keyword_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
                best_score = domain_scores[best_keyword_domain]
                
                # 키워드 기반 도메인이 더 높은 점수를 가지면 사용
                if best_score > 0:
                    final_domain = best_keyword_domain
                    domain_confidence = min(1.0, best_score / 10.0)  # 점수를 0-1 범위로 정규화
                else:
                    final_domain = base_domain
                    domain_confidence = result.confidence
            else:
                final_domain = base_domain
                domain_confidence = result.confidence
            
            # 도메인별 특화 정보
            domain_info = self._get_domain_specific_info(final_domain, question)
            
            return {
                "domain": final_domain,
                "domain_confidence": domain_confidence,
                "base_domain": base_domain,
                "domain_scores": domain_scores,
                "domain_info": domain_info,
                "method": "hybrid_domain_analysis"
            }
            
        except Exception as e:
            return {
                "domain": "general",
                "domain_confidence": 0.5,
                "base_domain": "general",
                "domain_scores": {},
                "domain_info": {},
                "error": str(e)
            }
    
    def _get_domain_specific_info(self, domain: str, question: str) -> Dict[str, Any]:
        """도메인별 특화 정보 제공"""
        domain_info = {
            "civil_law": {
                "description": "민사법 관련 질문",
                "common_topics": ["계약", "손해배상", "불법행위", "소유권", "채권채무"],
                "suggested_actions": ["계약서 검토", "손해배상 청구", "소유권 확인"]
            },
            "criminal_law": {
                "description": "형사법 관련 질문", 
                "common_topics": ["범죄", "처벌", "형량", "구성요건"],
                "suggested_actions": ["변호사 상담", "법정 대리", "형사 절차 안내"]
            },
            "family_law": {
                "description": "가족법 관련 질문",
                "common_topics": ["이혼", "상속", "양육권", "재산분할"],
                "suggested_actions": ["이혼 절차", "상속 등기", "양육권 조정"]
            },
            "commercial_law": {
                "description": "상법 관련 질문",
                "common_topics": ["회사", "주식", "이사", "회사설립"],
                "suggested_actions": ["회사 설립", "주식 발행", "이사 선임"]
            },
            "labor_law": {
                "description": "노동법 관련 질문",
                "common_topics": ["근로", "임금", "해고", "근로시간"],
                "suggested_actions": ["근로계약서 작성", "임금 체불 신고", "부당해고 구제"]
            },
            "real_estate": {
                "description": "부동산법 관련 질문",
                "common_topics": ["매매", "임대차", "등기", "소유권이전"],
                "suggested_actions": ["부동산 매매", "임대차 계약", "등기 신청"]
            },
            "general": {
                "description": "일반 법률 질문",
                "common_topics": ["법률", "법령", "조문", "판례"],
                "suggested_actions": ["법령 검색", "판례 조회", "법률 상담"]
            }
        }
        
        return domain_info.get(domain, domain_info["general"])
    
    def train_ml_model(self, training_data: List[Tuple[str, UnifiedQuestionType]]):
        """ML 모델 훈련"""
        self.ml_based.train(training_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        total = self.stats['total_calls']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'rule_based_percentage': (self.stats['rule_based_calls'] / total) * 100,
            'ml_based_percentage': (self.stats['ml_based_calls'] / total) * 100,
            'hybrid_percentage': (self.stats['hybrid_calls'] / total) * 100
        }
    
    def adjust_threshold(self, new_threshold: float):
        """신뢰도 임계값 조정"""
        self.confidence_threshold = max(0.1, min(1.0, new_threshold))
        print(f"신뢰도 임계값 조정: {self.confidence_threshold}")
    
    # 기존 시스템과의 호환성을 위한 메서드들
    def classify_question(self, question: str) -> IntegratedClassificationResult:
        """기존 시스템 호환 메서드"""
        return self.classify(question)
    
    def get_question_type_description(self, question_type: UnifiedQuestionType) -> str:
        """질문 유형 설명 반환"""
        return question_type.get_description()
