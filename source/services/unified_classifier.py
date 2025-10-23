# -*- coding: utf-8 -*-
"""
Unified Classifier
모든 분류 기능을 통합한 단일 분류기
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .question_classifier import QuestionClassifier
from .simple_text_classifier import SimpleTextClassifier
from .bert_classifier import BERTClassifier

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """질문 유형"""
    GENERAL = "general"
    LEGAL_ADVICE = "legal_advice"
    PRECEDENT = "precedent"
    LAW_SEARCH = "law_search"
    CONTRACT = "contract"
    CRIMINAL = "criminal"
    CIVIL = "civil"
    FAMILY = "family"
    LABOR = "labor"
    TAX = "tax"
    REAL_ESTATE = "real_estate"
    INTELLECTUAL_PROPERTY = "intellectual_property"


class IntentType(Enum):
    """의도 유형"""
    QUESTION = "question"
    REQUEST = "request"
    COMPLAINT = "complaint"
    INFORMATION = "information"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"


@dataclass
class ClassificationResult:
    """분류 결과"""
    query: str
    question_type: QuestionType
    intent: IntentType
    confidence: float
    subcategories: List[str]
    classification_method: str
    processing_time: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class UnifiedClassifier:
    """통합 분류기 클래스"""
    
    def __init__(self,
                 question_classifier: Optional[QuestionClassifier] = None,
                 simple_text_classifier: Optional[SimpleTextClassifier] = None,
                 bert_classifier: Optional[BERTClassifier] = None,
                 enable_fallback: bool = True):
        """
        통합 분류기 초기화
        
        Args:
            question_classifier: 질문 분류기
            simple_text_classifier: 간단한 텍스트 분류기
            bert_classifier: BERT 분류기
            enable_fallback: 폴백 활성화
        """
        self.question_classifier = question_classifier
        self.simple_text_classifier = simple_text_classifier
        self.bert_classifier = bert_classifier
        self.enable_fallback = enable_fallback
        
        # 성능 통계
        self._stats = {
            'total_classifications': 0,
            'successful_classifications': 0,
            'failed_classifications': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'method_usage': {
                'question_classifier': 0,
                'simple_text_classifier': 0,
                'bert_classifier': 0,
                'fallback': 0
            }
        }
        
        logger.info("UnifiedClassifier initialized successfully")
    
    def classify(self, query: str) -> ClassificationResult:
        """
        통합 분류 수행
        
        Args:
            query: 분류할 질문
            
        Returns:
            ClassificationResult: 분류 결과
        """
        start_time = time.time()
        
        try:
            # 1순위: 질문 분류기
            if self.question_classifier:
                try:
                    result = self.question_classifier.classify_question(query)
                    classification_result = self._convert_question_classifier_result(
                        query, result, time.time() - start_time
                    )
                    self._update_stats(time.time() - start_time, success=True, method='question_classifier')
                    return classification_result
                except Exception as e:
                    logger.warning(f"Question classifier failed: {e}")
            
            # 2순위: BERT 분류기
            if self.bert_classifier:
                try:
                    result = self.bert_classifier.predict_proba(query)
                    classification_result = self._convert_bert_classifier_result(
                        query, result, time.time() - start_time
                    )
                    self._update_stats(time.time() - start_time, success=True, method='bert_classifier')
                    return classification_result
                except Exception as e:
                    logger.warning(f"BERT classifier failed: {e}")
            
            # 3순위: 간단한 텍스트 분류기
            if self.simple_text_classifier:
                try:
                    result = self.simple_text_classifier.predict_proba(query)
                    classification_result = self._convert_simple_classifier_result(
                        query, result, time.time() - start_time
                    )
                    self._update_stats(time.time() - start_time, success=True, method='simple_text_classifier')
                    return classification_result
                except Exception as e:
                    logger.warning(f"Simple text classifier failed: {e}")
            
            # 4순위: 폴백 분류
            if self.enable_fallback:
                classification_result = self._fallback_classification(query, time.time() - start_time)
                self._update_stats(time.time() - start_time, success=True, method='fallback')
                return classification_result
            
            # 모든 분류기 실패
            raise Exception("All classifiers failed")
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            self._update_stats(time.time() - start_time, success=False)
            
            # 에러 응답
            return ClassificationResult(
                query=query,
                question_type=QuestionType.GENERAL,
                intent=IntentType.QUESTION,
                confidence=0.1,
                subcategories=[],
                classification_method="error_fallback",
                processing_time=time.time() - start_time
            )
    
    def _convert_question_classifier_result(self, query: str, result: Any, processing_time: float) -> ClassificationResult:
        """질문 분류기 결과 변환"""
        try:
            question_type_value = getattr(result.question_type, 'value', 'general')
            question_type = QuestionType(question_type_value) if question_type_value in [e.value for e in QuestionType] else QuestionType.GENERAL
            
            confidence = getattr(result, 'confidence', 0.5)
            subcategories = getattr(result, 'subcategories', [])
            
            return ClassificationResult(
                query=query,
                question_type=question_type,
                intent=IntentType.QUESTION,
                confidence=confidence,
                subcategories=subcategories,
                classification_method="question_classifier",
                processing_time=processing_time
            )
        except Exception as e:
            logger.warning(f"Question classifier result conversion failed: {e}")
            return self._fallback_classification(query, processing_time)
    
    def _convert_bert_classifier_result(self, query: str, result: Dict[str, float], processing_time: float) -> ClassificationResult:
        """BERT 분류기 결과 변환"""
        try:
            # allowed/restricted 결과를 질문 유형으로 변환
            if result.get('allowed', 0) > result.get('restricted', 0):
                question_type = QuestionType.GENERAL
                confidence = result.get('allowed', 0.5)
            else:
                question_type = QuestionType.LEGAL_ADVICE
                confidence = result.get('restricted', 0.5)
            
            return ClassificationResult(
                query=query,
                question_type=question_type,
                intent=IntentType.QUESTION,
                confidence=confidence,
                subcategories=[],
                classification_method="bert_classifier",
                processing_time=processing_time
            )
        except Exception as e:
            logger.warning(f"BERT classifier result conversion failed: {e}")
            return self._fallback_classification(query, processing_time)
    
    def _convert_simple_classifier_result(self, query: str, result: Dict[str, float], processing_time: float) -> ClassificationResult:
        """간단한 분류기 결과 변환"""
        try:
            # allowed/restricted 결과를 질문 유형으로 변환
            if result.get('allowed', 0) > result.get('restricted', 0):
                question_type = QuestionType.GENERAL
                confidence = result.get('allowed', 0.5)
            else:
                question_type = QuestionType.LEGAL_ADVICE
                confidence = result.get('restricted', 0.5)
            
            return ClassificationResult(
                query=query,
                question_type=question_type,
                intent=IntentType.QUESTION,
                confidence=confidence,
                subcategories=[],
                classification_method="simple_text_classifier",
                processing_time=processing_time
            )
        except Exception as e:
            logger.warning(f"Simple classifier result conversion failed: {e}")
            return self._fallback_classification(query, processing_time)
    
    def _fallback_classification(self, query: str, processing_time: float) -> ClassificationResult:
        """폴백 분류"""
        try:
            # 키워드 기반 간단한 분류
            query_lower = query.lower()
            
            # 법률 관련 키워드
            legal_keywords = ['법률', '법', '변호사', '소송', '계약', '판례', '법원', '재판']
            if any(keyword in query_lower for keyword in legal_keywords):
                question_type = QuestionType.LEGAL_ADVICE
                confidence = 0.6
            else:
                question_type = QuestionType.GENERAL
                confidence = 0.4
            
            # 의도 분석
            if '?' in query or '질문' in query_lower or '궁금' in query_lower:
                intent = IntentType.QUESTION
            elif '요청' in query_lower or '부탁' in query_lower:
                intent = IntentType.REQUEST
            elif '불만' in query_lower or '항의' in query_lower:
                intent = IntentType.COMPLAINT
            else:
                intent = IntentType.INFORMATION
            
            return ClassificationResult(
                query=query,
                question_type=question_type,
                intent=intent,
                confidence=confidence,
                subcategories=[],
                classification_method="fallback",
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Fallback classification failed: {e}")
            return ClassificationResult(
                query=query,
                question_type=QuestionType.GENERAL,
                intent=IntentType.QUESTION,
                confidence=0.1,
                subcategories=[],
                classification_method="error_fallback",
                processing_time=processing_time
            )
    
    def _update_stats(self, processing_time: float, success: bool = True, method: str = "unknown"):
        """통계 업데이트"""
        self._stats['total_classifications'] += 1
        self._stats['total_processing_time'] += processing_time
        self._stats['avg_processing_time'] = self._stats['total_processing_time'] / self._stats['total_classifications']
        
        if success:
            self._stats['successful_classifications'] += 1
        else:
            self._stats['failed_classifications'] += 1
        
        if method in self._stats['method_usage']:
            self._stats['method_usage'][method] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return self._stats.copy()
    
    def get_available_methods(self) -> List[str]:
        """사용 가능한 분류 방법 반환"""
        methods = []
        if self.question_classifier:
            methods.append("question_classifier")
        if self.simple_text_classifier:
            methods.append("simple_text_classifier")
        if self.bert_classifier:
            methods.append("bert_classifier")
        if self.enable_fallback:
            methods.append("fallback")
        return methods
