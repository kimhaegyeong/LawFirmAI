#!/usr/bin/env python3
"""
하이브리드 질문 분류 시스템
규칙 기반 + ML 기반 분류기를 결합한 시스템
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import json

# 프로젝트 루트 디렉토리를 Python 경로에 추가
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.services.answer_structure_enhancer import QuestionType

@dataclass
class ClassificationResult:
    """분류 결과를 담는 데이터 클래스"""
    question_type: QuestionType
    confidence: float
    method: str  # 'rule_based', 'ml_based', 'hybrid'
    features: Dict[str, Any] = None
    reasoning: str = ""

class RuleBasedClassifier:
    """개선된 규칙 기반 분류기"""
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self.keyword_weights = self._load_keyword_weights()
        self.conflict_rules = self._load_conflict_rules()
    
    def _load_patterns(self) -> Dict[QuestionType, List[str]]:
        """패턴 로딩"""
        return {
            QuestionType.LAW_INQUIRY: [
                r'제\d+조', r'제\d+항', r'제\d+호',
                r'\w+법\s+제\d+조', r'\w+법\s+제\d+항'
            ],
            QuestionType.PRECEDENT_SEARCH: [
                r'판례를?\s+찾아주세요', r'관련\s+판례', r'유사한\s+판례',
                r'최근\s+판례', r'대법원\s+판례', r'고등법원\s+판례'
            ],
            QuestionType.CONTRACT_REVIEW: [
                r'계약서를?\s+검토', r'계약서를?\s+수정', r'계약서를?\s+작성',
                r'계약\s+조항', r'불리한\s+조항', r'계약서의?\s+문제점'
            ],
            QuestionType.DIVORCE_PROCEDURE: [
                r'이혼\s+절차', r'이혼\s+방법', r'이혼\s+신청',
                r'협의이혼\s+절차', r'재판이혼\s+절차'
            ],
            QuestionType.INHERITANCE_PROCEDURE: [
                r'상속\s+절차', r'상속\s+신청', r'유산\s+분할',
                r'상속인\s+확인', r'상속세\s+신고', r'유언\s+검인'
            ],
            QuestionType.CRIMINAL_CASE: [
                r'\w+죄\s+구성요건', r'\w+죄\s+처벌', r'\w+죄\s+형량',
                r'\w+범죄\s+처벌', r'형사\s+사건', r'범죄\s+구성요건'
            ],
            QuestionType.LABOR_DISPUTE: [
                r'노동\s+분쟁', r'근로\s+분쟁', r'임금\s+체불',
                r'부당해고\s+구제', r'해고\s+대응', r'근로시간\s+규정'
            ],
            QuestionType.PROCEDURE_GUIDE: [
                r'\w+\s+절차', r'\w+\s+신청', r'소액사건\s+절차',
                r'민사조정\s+신청', r'소송\s+제기', r'어떻게\s+신청'
            ],
            QuestionType.TERM_EXPLANATION: [
                r'무엇이\s+\w+인가요?', r'\w+의\s+의미는?', r'\w+의\s+정의는?',
                r'\w+의\s+개념은?', r'\w+이\s+무엇인가요?', r'\w+란\s+무엇인가요?'
            ],
            QuestionType.LEGAL_ADVICE: [
                r'어떻게\s+대응해야', r'권리\s+구제', r'의무\s+이행',
                r'법률\s+상담', r'법률\s+자문', r'변호사\s+상담',
                r'법적\s+대응', r'법적\s+조언', r'법적\s+보호',
                r'법적\s+해결', r'법적\s+구제', r'조언을?\s+구하고',
                r'도움이?\s+필요', r'지원을?\s+받고'
            ]
        }
    
    def _load_keyword_weights(self) -> Dict[str, Dict[str, float]]:
        """키워드 가중치 로딩"""
        return {
            QuestionType.LEGAL_ADVICE: {
                '법적': 3.0, '권리': 2.5, '의무': 2.5, '구제': 2.5,
                '보호': 2.0, '해결': 2.0, '대응': 2.0, '상담': 2.0,
                '자문': 2.0, '조언': 2.0, '도움': 1.5, '지원': 1.5,
                '방법': 1.0, '어떻게': 1.0
            },
            QuestionType.PROCEDURE_GUIDE: {
                '절차': 3.0, '신청': 2.5, '제출': 2.0, '처리': 2.0,
                '소송': 2.0, '민사조정': 2.0, '소액사건': 2.0,
                '방법': 1.5, '어떻게': 1.5, '어디서': 1.5
            },
            QuestionType.TERM_EXPLANATION: {
                '의미': 3.0, '정의': 3.0, '개념': 3.0, '설명': 2.5,
                '무엇': 2.0, '뜻': 2.0, '내용': 1.5
            }
        }
    
    def _load_conflict_rules(self) -> Dict[str, List[str]]:
        """충돌 해결 규칙"""
        return {
            'legal_advice_vs_procedure_guide': {
                'legal_advice_keywords': ['법적', '권리', '의무', '구제', '보호', '해결', '대응'],
                'procedure_guide_keywords': ['절차', '신청', '제출', '처리'],
                'resolution': 'legal_advice'  # 법률 자문 우선
            },
            'term_explanation_vs_contract_review': {
                'term_explanation_patterns': [r'\w+의\s+내용은?', r'\w+이\s+어떤\s+것인가요?'],
                'contract_review_keywords': ['계약서', '검토', '수정', '작성'],
                'resolution': 'term_explanation'  # 용어 설명 우선
            }
        }
    
    def classify(self, question: str) -> ClassificationResult:
        """규칙 기반 분류"""
        question_lower = question.lower().strip()
        
        if not question_lower:
            return ClassificationResult(
                question_type=QuestionType.GENERAL_QUESTION,
                confidence=1.0,
                method='rule_based',
                reasoning="빈 질문"
            )
        
        # 점수 계산
        scores = self._calculate_scores(question_lower)
        
        # 충돌 해결
        final_type = self._resolve_conflicts(question_lower, scores)
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(question_lower, final_type, scores)
        
        return ClassificationResult(
            question_type=final_type,
            confidence=confidence,
            method='rule_based',
            features={'scores': scores},
            reasoning=f"규칙 기반 분류: {final_type.value}"
        )
    
    def _calculate_scores(self, question_lower: str) -> Dict[QuestionType, float]:
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
    
    def _resolve_conflicts(self, question_lower: str, scores: Dict[QuestionType, float]) -> QuestionType:
        """충돌 해결"""
        # 법률 자문 vs 절차 안내 충돌
        if (scores.get(QuestionType.LEGAL_ADVICE, 0) > 0 and 
            scores.get(QuestionType.PROCEDURE_GUIDE, 0) > 0):
            
            legal_keywords = self.conflict_rules['legal_advice_vs_procedure_guide']['legal_advice_keywords']
            if any(keyword in question_lower for keyword in legal_keywords):
                return QuestionType.LEGAL_ADVICE
        
        # 용어 설명 vs 계약서 검토 충돌
        if (scores.get(QuestionType.TERM_EXPLANATION, 0) > 0 and 
            scores.get(QuestionType.CONTRACT_REVIEW, 0) > 0):
            
            term_patterns = self.conflict_rules['term_explanation_vs_contract_review']['term_explanation_patterns']
            if any(re.search(pattern, question_lower) for pattern in term_patterns):
                return QuestionType.TERM_EXPLANATION
        
        # 최고 점수 반환
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return QuestionType.GENERAL_QUESTION
    
    def _calculate_confidence(self, question_lower: str, question_type: QuestionType, scores: Dict[QuestionType, float]) -> float:
        """신뢰도 계산"""
        if not scores:
            return 0.5
        
        max_score = max(scores.values())
        total_score = sum(scores.values())
        
        if total_score == 0:
            return 0.5
        
        # 점수 기반 신뢰도
        score_confidence = min(max_score / total_score, 1.0)
        
        # 질문 길이 기반 신뢰도 (너무 짧으면 신뢰도 낮음)
        length_confidence = min(len(question_lower) / 20, 1.0)
        
        # 최종 신뢰도
        confidence = (score_confidence * 0.7 + length_confidence * 0.3)
        
        return min(max(confidence, 0.1), 1.0)

class MLBasedClassifier:
    """ML 기반 분류기"""
    
    def __init__(self, model_path: str = "models/question_classifier.pkl"):
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
                print(f"ML 모델 로드 완료: {self.model_path}")
        except Exception as e:
            print(f"ML 모델 로드 실패: {e}")
            self.is_trained = False
    
    def train(self, training_data: List[Tuple[str, QuestionType]]):
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
            stop_words=None,  # 한국어는 불용어 처리 생략
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
        
        print(f"ML 모델 훈련 완료. 최고 정확도: {best_score:.3f}")
    
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
            print(f"ML 모델 저장 완료: {self.model_path}")
        except Exception as e:
            print(f"ML 모델 저장 실패: {e}")
    
    def classify(self, question: str) -> ClassificationResult:
        """ML 기반 분류"""
        if not self.is_trained:
            return ClassificationResult(
                question_type=QuestionType.GENERAL_QUESTION,
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
                predicted_label = QuestionType.GENERAL_QUESTION.value
            
            # 신뢰도 계산
            confidence = max(probabilities)
            
            return ClassificationResult(
                question_type=QuestionType(predicted_label),
                confidence=confidence,
                method='ml_based',
                features={'probabilities': probabilities.tolist()},
                reasoning=f"ML 기반 분류: {predicted_label}"
            )
            
        except Exception as e:
            return ClassificationResult(
                question_type=QuestionType.GENERAL_QUESTION,
                confidence=0.5,
                method='ml_based',
                reasoning=f"ML 분류 오류: {e}"
            )

class HybridQuestionClassifier:
    """하이브리드 질문 분류기"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.rule_based = RuleBasedClassifier()
        self.ml_based = MLBasedClassifier()
        
        # 성능 통계
        self.stats = {
            'rule_based_calls': 0,
            'ml_based_calls': 0,
            'hybrid_calls': 0,
            'total_calls': 0
        }
    
    def classify(self, question: str) -> ClassificationResult:
        """하이브리드 분류"""
        self.stats['total_calls'] += 1
        
        # 규칙 기반 분류
        rule_result = self.rule_based.classify(question)
        self.stats['rule_based_calls'] += 1
        
        # 신뢰도가 임계값 이상이면 규칙 기반 결과 반환
        if rule_result.confidence >= self.confidence_threshold:
            return rule_result
        
        # ML 모델이 훈련되어 있으면 ML 기반 분류 시도
        if self.ml_based.is_trained:
            ml_result = self.ml_based.classify(question)
            self.stats['ml_based_calls'] += 1
            
            # ML 결과의 신뢰도가 더 높으면 ML 결과 반환
            if ml_result.confidence > rule_result.confidence:
                ml_result.method = 'hybrid'
                ml_result.reasoning = f"하이브리드 분류: ML 신뢰도({ml_result.confidence:.3f}) > 규칙 신뢰도({rule_result.confidence:.3f})"
                self.stats['hybrid_calls'] += 1
                return ml_result
        
        # 규칙 기반 결과 반환 (신뢰도가 낮아도)
        rule_result.method = 'hybrid'
        rule_result.reasoning = f"하이브리드 분류: 규칙 기반 결과 사용 (신뢰도: {rule_result.confidence:.3f})"
        self.stats['hybrid_calls'] += 1
        return rule_result
    
    def train_ml_model(self, training_data: List[Tuple[str, QuestionType]]):
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

def create_training_data() -> List[Tuple[str, QuestionType]]:
    """훈련 데이터 생성"""
    training_data = []
    
    # 법률 문의
    law_inquiry_examples = [
        "민법 제123조의 내용이 무엇인가요?",
        "형법 제250조 처벌 기준은?",
        "근로기준법 제15조 의미는?",
        "상법 제123조 해석해주세요",
        "헌법 제10조 내용은?",
        "특허법 제25조 규정은?",
        "부동산등기법 제123조는?",
        "민법 제456항의 의미는?",
        "형법 제789호 규정은?",
        "근로기준법 제12조 제3항은?"
    ]
    
    # 판례 검색
    precedent_examples = [
        "대법원 판례를 찾아주세요",
        "관련 판례가 있나요?",
        "고등법원 판결을 알려주세요",
        "지방법원 판례 검색",
        "최근 판례를 찾아주세요",
        "유사한 판례가 있나요?",
        "대법원 판례 검색",
        "고등법원 판례 찾기",
        "지방법원 판례를 찾아주세요",
        "판례를 찾아주세요"
    ]
    
    # 계약서 검토
    contract_examples = [
        "계약서를 검토해주세요",
        "이 계약 조항이 불리한가요?",
        "계약서 수정이 필요한가요?",
        "계약서를 작성해주세요",
        "계약서를 체결하고 싶어요",
        "계약 조항을 확인해주세요",
        "계약 조건이 적절한가요?",
        "계약 내용을 검토해주세요",
        "불리한 조항이 있나요?",
        "계약서의 문제점은?"
    ]
    
    # 이혼 절차
    divorce_examples = [
        "이혼 절차를 알려주세요",
        "협의이혼 방법은?",
        "재판이혼 절차는?",
        "이혼절차 신청 방법",
        "이혼 어떻게 해야 하나요?",
        "이혼 어디서 신청하나요?",
        "이혼 비용은 얼마인가요?",
        "협의이혼 절차",
        "재판이혼 절차",
        "이혼 방법"
    ]
    
    # 상속 절차
    inheritance_examples = [
        "상속 절차를 알려주세요",
        "유산 분할 방법은?",
        "상속인 확인 방법",
        "상속세 신고 절차",
        "유언 검인 절차",
        "상속포기 방법",
        "상속 신청 방법",
        "유산 분할 절차",
        "상속인 확인 절차",
        "상속세 신고 방법"
    ]
    
    # 형사 사건
    criminal_examples = [
        "사기죄 구성요건은?",
        "절도 범죄 처벌은?",
        "강도 사건 대응 방법",
        "살인죄 형량은?",
        "형사 사건 절차",
        "절도죄 구성요건",
        "강도죄 처벌",
        "살인죄 형량",
        "사기범죄 처벌",
        "절도범죄 처벌"
    ]
    
    # 노동 분쟁
    labor_examples = [
        "노동 분쟁 해결 방법",
        "근로 시간 규정은?",
        "임금 체불 대응",
        "부당해고 구제 방법",
        "해고 통보 대응",
        "노동위원회 신청",
        "근로 분쟁 해결",
        "임금 지급 문제",
        "부당해고 구제",
        "해고 대응"
    ]
    
    # 절차 안내
    procedure_examples = [
        "소송 절차를 알려주세요",
        "민사조정 신청 방법",
        "소액사건 절차는?",
        "어떻게 신청하나요?",
        "어디서 신청하나요?",
        "신청 방법을 알려주세요",
        "신청 절차를 알려주세요",
        "처리 절차를 알려주세요",
        "진행 절차를 알려주세요",
        "소송 제기 방법"
    ]
    
    # 용어 설명
    term_examples = [
        "법인격의 의미는?",
        "소멸시효 정의는?",
        "무효와 취소의 개념",
        "무엇이 계약인가요?",
        "뜻을 설명해주세요",
        "계약의 의미는?",
        "계약의 정의는?",
        "계약의 개념은?",
        "계약이 무엇인가요?",
        "계약이란 무엇인가요?"
    ]
    
    # 법률 자문
    advice_examples = [
        "어떻게 대응해야 하나요?",
        "권리 구제 방법은?",
        "의무 이행 방법",
        "해야 할 일은?",
        "법률 상담을 받고 싶어요",
        "변호사 상담이 필요해요",
        "법적 대응 방법은?",
        "법적 조언을 구하고 싶어요",
        "어떤 조치를 취해야 하나요?",
        "권리 보호 방법은?",
        "의무 준수 방법",
        "법적 구제 방법은?",
        "법적 보호 방법은?",
        "법적 해결 방법은?",
        "법률 자문을 받고 싶어요",
        "변호사 자문이 필요해요",
        "법적 상담을 받고 싶어요",
        "법적 도움이 필요해요",
        "법적 지원을 받고 싶어요"
    ]
    
    # 일반 질문
    general_examples = [
        "안녕하세요",
        "도움이 필요합니다",
        "질문이 있습니다",
        "고마워요",
        "감사합니다",
        "좋은 하루 되세요",
        "수고하세요",
        "잘 부탁드립니다",
        "도와주세요",
        "궁금한 것이 있어요"
    ]
    
    # 훈련 데이터 구성
    all_examples = [
        (law_inquiry_examples, QuestionType.LAW_INQUIRY),
        (precedent_examples, QuestionType.PRECEDENT_SEARCH),
        (contract_examples, QuestionType.CONTRACT_REVIEW),
        (divorce_examples, QuestionType.DIVORCE_PROCEDURE),
        (inheritance_examples, QuestionType.INHERITANCE_PROCEDURE),
        (criminal_examples, QuestionType.CRIMINAL_CASE),
        (labor_examples, QuestionType.LABOR_DISPUTE),
        (procedure_examples, QuestionType.PROCEDURE_GUIDE),
        (term_examples, QuestionType.TERM_EXPLANATION),
        (advice_examples, QuestionType.LEGAL_ADVICE),
        (general_examples, QuestionType.GENERAL_QUESTION)
    ]
    
    for examples, question_type in all_examples:
        for example in examples:
            training_data.append((example, question_type))
    
    return training_data

if __name__ == "__main__":
    # 하이브리드 분류기 생성
    classifier = HybridQuestionClassifier(confidence_threshold=0.7)
    
    # 훈련 데이터 생성 및 ML 모델 훈련
    print("훈련 데이터 생성 중...")
    training_data = create_training_data()
    print(f"총 {len(training_data)}개의 훈련 데이터 생성")
    
    print("\nML 모델 훈련 중...")
    classifier.train_ml_model(training_data)
    
    # 테스트
    test_questions = [
        "어떻게 대응해야 하나요?",
        "법적 보호 방법은?",
        "법적 해결 방법은?",
        "어떤 방법이 있나요?",
        "민법 제123조의 내용이 무엇인가요?",
        "계약서를 검토해주세요",
        "무엇이 계약인가요?"
    ]
    
    print("\n하이브리드 분류 테스트:")
    print("=" * 60)
    
    for question in test_questions:
        result = classifier.classify(question)
        print(f"질문: {question}")
        print(f"분류: {result.question_type.value}")
        print(f"신뢰도: {result.confidence:.3f}")
        print(f"방법: {result.method}")
        print(f"이유: {result.reasoning}")
        print("-" * 40)
    
    # 통계 출력
    print("\n성능 통계:")
    stats = classifier.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
