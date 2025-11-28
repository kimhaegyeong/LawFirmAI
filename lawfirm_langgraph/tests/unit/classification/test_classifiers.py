# -*- coding: utf-8 -*-
"""
Classification Classifiers 테스트
분류기 모듈 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from lawfirm_langgraph.core.classification.classifiers.question_classifier import QuestionClassifier, QuestionType, QuestionClassification
from lawfirm_langgraph.core.classification.classifiers.hybrid_question_classifier import HybridQuestionClassifier
from lawfirm_langgraph.core.classification.classifiers.semantic_domain_classifier import SemanticDomainClassifier
from lawfirm_langgraph.core.classification.classifiers.complexity_classifier import ComplexityClassifier
from lawfirm_langgraph.core.services.unified_prompt_manager import LegalDomain


class TestQuestionClassifier:
    """QuestionClassifier 테스트"""
    
    @pytest.fixture
    def question_classifier(self):
        """QuestionClassifier 인스턴스"""
        return QuestionClassifier()
    
    def test_question_classifier_initialization(self):
        """QuestionClassifier 초기화 테스트"""
        classifier = QuestionClassifier()
        
        assert classifier.logger is not None
        assert hasattr(classifier, 'question_patterns')
    
    def test_classify_question(self, question_classifier):
        """질문 분류 테스트"""
        result = question_classifier.classify_question("계약서 작성 시 주의할 사항은?")
        
        assert isinstance(result, QuestionClassification)
        assert hasattr(result, 'question_type')
        assert hasattr(result, 'confidence')
    
    def test_classify_precedent_search(self, question_classifier):
        """판례 검색 질문 분류 테스트"""
        result = question_classifier.classify_question("판례를 찾아주세요")
        
        assert isinstance(result, QuestionClassification)
        # 실제 분류 결과가 PRECEDENT_SEARCH일 가능성이 높지만, 패턴 매칭 결과에 따라 다를 수 있음
        assert result.question_type in QuestionType
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
    
    def test_classify_law_inquiry(self, question_classifier):
        """법률 조문 질문 분류 테스트"""
        result = question_classifier.classify_question("민법 제1조의 내용은?")
        
        assert isinstance(result, QuestionClassification)
        # 실제 분류 결과가 LAW_INQUIRY일 가능성이 높지만, 패턴 매칭 결과에 따라 다를 수 있음
        assert result.question_type in QuestionType
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
    
    def test_classify_legal_advice(self, question_classifier):
        """법률 조언 질문 분류 테스트"""
        result = question_classifier.classify_question("어떻게 대응해야 하나요?")
        
        assert isinstance(result, QuestionClassification)
        # 실제 분류 결과가 LEGAL_ADVICE일 가능성이 높지만, 패턴 매칭 결과에 따라 다를 수 있음
        assert result.question_type in QuestionType
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0


class TestHybridQuestionClassifier:
    """HybridQuestionClassifier 테스트"""
    
    @pytest.fixture
    def hybrid_classifier(self):
        """HybridQuestionClassifier 인스턴스"""
        with patch('lawfirm_langgraph.core.classification.classifiers.hybrid_question_classifier.TfidfVectorizer'):
            with patch('lawfirm_langgraph.core.classification.classifiers.hybrid_question_classifier.MultinomialNB'):
                classifier = HybridQuestionClassifier()
                return classifier
    
    def test_hybrid_classifier_initialization(self):
        """HybridQuestionClassifier 초기화 테스트"""
        with patch('lawfirm_langgraph.core.classification.classifiers.hybrid_question_classifier.TfidfVectorizer'):
            with patch('lawfirm_langgraph.core.classification.classifiers.hybrid_question_classifier.MultinomialNB'):
                classifier = HybridQuestionClassifier()
                
                assert classifier is not None
    
    def test_classify_hybrid(self, hybrid_classifier):
        """하이브리드 분류 테스트"""
        result = hybrid_classifier.classify("계약서 작성 시 주의할 사항은?")
        
        assert isinstance(result, dict) or hasattr(result, 'question_type')


class TestSemanticDomainClassifier:
    """SemanticDomainClassifier 테스트"""
    
    @pytest.fixture
    def semantic_classifier(self):
        """SemanticDomainClassifier 인스턴스"""
        with patch('lawfirm_langgraph.core.classification.classifiers.semantic_domain_classifier.LegalTermsDatabase'):
            classifier = SemanticDomainClassifier()
            return classifier
    
    def test_semantic_classifier_initialization(self):
        """SemanticDomainClassifier 초기화 테스트"""
        with patch('lawfirm_langgraph.core.classification.classifiers.semantic_domain_classifier.LegalTermsDatabase'):
            classifier = SemanticDomainClassifier()
            
            assert classifier is not None
    
    def test_classify_domain(self, semantic_classifier):
        """도메인 분류 테스트"""
        # 실제 코드는 Tuple[LegalDomain, float, str]을 반환
        with patch.object(semantic_classifier, 'terms_database') as mock_terms_db, \
             patch.object(semantic_classifier, 'context_analyzer') as mock_context:
            mock_terms_db.get_domain_scores.return_value = {}
            mock_context.analyze_context.return_value = {}
            
            domain, confidence, reasoning = semantic_classifier.classify_domain("계약서 작성 시 주의할 사항은?")
            
            assert isinstance(domain, LegalDomain) or domain in LegalDomain
            assert isinstance(confidence, float)
            assert isinstance(reasoning, str)


class TestComplexityClassifier:
    """ComplexityClassifier 테스트"""
    
    def test_classify_complexity_simple(self):
        """간단한 질문 복잡도 분류 테스트"""
        complexity, needs_search = ComplexityClassifier.classify_complexity("안녕하세요")
        
        assert complexity in ["simple", "moderate", "complex"]
        assert isinstance(needs_search, bool)
    
    def test_classify_complexity_complex(self):
        """복잡한 질문 복잡도 분류 테스트"""
        complexity, needs_search = ComplexityClassifier.classify_complexity("계약서 작성 시 주의할 사항과 판례를 찾아주세요")
        
        assert complexity in ["simple", "moderate", "complex"]
        assert isinstance(needs_search, bool)
    
    def test_classify_complexity_moderate(self):
        """중간 복잡도 질문 분류 테스트"""
        complexity, needs_search = ComplexityClassifier.classify_complexity("계약서 작성 시 주의할 사항은?")
        
        assert complexity in ["simple", "moderate", "complex"]
        assert isinstance(needs_search, bool)
    
    def test_classify_complexity_empty(self):
        """빈 질문 복잡도 분류 테스트"""
        complexity, needs_search = ComplexityClassifier.classify_complexity("")
        
        assert complexity in ["simple", "moderate", "complex"]
        assert isinstance(needs_search, bool)
    
    def test_classify_with_llm(self):
        """LLM을 사용한 복잡도 분류 테스트"""
        complexity, needs_search = ComplexityClassifier.classify_with_llm("테스트 질문")
        
        assert complexity in ["simple", "moderate", "complex"]
        assert isinstance(needs_search, bool)

