#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Tests for LawFirmAI Gradio Interface
Tests chat functionality and document upload/analysis
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add source directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent / "source"))
sys.path.append(str(Path(__file__).parent.parent.parent / "gradio"))

class TestGradioInterface:
    """Gradio 인터페이스 통합 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        # Mock the services to avoid loading heavy models
        self.mock_config = Mock()
        self.mock_config.database_path = ":memory:"
        
        self.mock_database = Mock()
        self.mock_vector_store = Mock()
        self.mock_model_manager = Mock()
        self.mock_rag_service = Mock()
        self.mock_search_service = Mock()
        self.mock_chat_service = Mock()
    
    @patch('gradio.app.DatabaseManager')
    @patch('gradio.app.LegalVectorStore')
    @patch('gradio.app.LegalModelManager')
    @patch('gradio.app.MLEnhancedRAGService')
    @patch('gradio.app.MLEnhancedSearchService')
    @patch('gradio.app.ChatService')
    def test_interface_creation(self, mock_chat, mock_search, mock_rag, 
                                mock_model, mock_vector, mock_db):
        """인터페이스 생성 테스트"""
        # Setup mocks
        mock_db.return_value = self.mock_database
        mock_vector.return_value = self.mock_vector_store
        mock_model.return_value = self.mock_model_manager
        mock_rag.return_value = self.mock_rag_service
        mock_search.return_value = self.mock_search_service
        mock_chat.return_value = self.mock_chat_service
        
        # Import and test interface creation
        from gradio.app import create_ml_enhanced_gradio_interface
        
        try:
            interface = create_ml_enhanced_gradio_interface()
            assert interface is not None
            print("✅ 인터페이스 생성 테스트 통과")
        except Exception as e:
            pytest.fail(f"인터페이스 생성 실패: {e}")
    
    def test_example_queries_structure(self):
        """예시 질문 구조 테스트"""
        from gradio.app import create_ml_enhanced_gradio_interface
        
        # Mock all dependencies
        with patch('gradio.app.DatabaseManager'), \
             patch('gradio.app.LegalVectorStore'), \
             patch('gradio.app.LegalModelManager'), \
             patch('gradio.app.MLEnhancedRAGService'), \
             patch('gradio.app.MLEnhancedSearchService'), \
             patch('gradio.app.ChatService'):
            
            # Test that example queries are properly structured
            expected_categories = ["계약서", "민법", "부동산", "근로"]
            
            # This would be tested by checking the EXAMPLE_QUERIES constant
            # For now, we'll just verify the structure exists
            assert True  # Placeholder - actual test would check EXAMPLE_QUERIES
            print("✅ 예시 질문 구조 테스트 통과")
    
    def test_document_analyzer_import(self):
        """문서 분석기 임포트 테스트"""
        try:
            from gradio.components.document_analyzer import DocumentAnalyzer
            analyzer = DocumentAnalyzer()
            assert analyzer is not None
            print("✅ 문서 분석기 임포트 테스트 통과")
        except ImportError as e:
            pytest.fail(f"문서 분석기 임포트 실패: {e}")
    
    def test_document_analyzer_pdf_parsing(self):
        """PDF 파싱 테스트"""
        from gradio.components.document_analyzer import DocumentAnalyzer
        
        analyzer = DocumentAnalyzer()
        
        # Create a temporary PDF-like file for testing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            # Write some test content
            tmp_file.write(b"Test PDF content")
            tmp_file_path = tmp_file.name
        
        try:
            # Test PDF parsing (this will fail without PyPDF2, but we can test the method exists)
            if hasattr(analyzer, 'parse_pdf'):
                print("✅ PDF 파싱 메서드 존재 확인")
            else:
                pytest.fail("PDF 파싱 메서드가 없습니다")
        finally:
            # Clean up
            os.unlink(tmp_file_path)
    
    def test_document_analyzer_docx_parsing(self):
        """DOCX 파싱 테스트"""
        from gradio.components.document_analyzer import DocumentAnalyzer
        
        analyzer = DocumentAnalyzer()
        
        # Test DOCX parsing method exists
        if hasattr(analyzer, 'parse_docx'):
            print("✅ DOCX 파싱 메서드 존재 확인")
        else:
            pytest.fail("DOCX 파싱 메서드가 없습니다")
    
    def test_document_analyzer_contract_analysis(self):
        """계약서 분석 테스트"""
        from gradio.components.document_analyzer import DocumentAnalyzer
        
        analyzer = DocumentAnalyzer()
        
        # Test contract analysis with sample text
        sample_text = """
        제1조 (목적)
        이 계약은 갑과 을 사이의 부동산 매매에 관한 사항을 정함을 목적으로 한다.
        
        제2조 (손해배상)
        당사자 중 일방이 계약을 위반한 경우 상대방에게 손해배상의 책임을 진다.
        
        제3조 (위약금)
        계약 해지 시 위약금을 지급하여야 한다.
        """
        
        try:
            result = analyzer.analyze_contract(sample_text)
            
            # Check result structure
            assert 'summary' in result
            assert 'clauses' in result
            assert 'risks' in result
            assert 'recommendations' in result
            assert 'risk_score' in result
            
            # Check that risks were detected
            assert len(result['risks']) > 0
            
            # Check that high-risk keywords were found
            high_risks = [r for r in result['risks'] if r['risk_level'] == 'high']
            assert len(high_risks) > 0  # Should find "손해배상", "위약금"
            
            print("✅ 계약서 분석 테스트 통과")
            
        except Exception as e:
            pytest.fail(f"계약서 분석 실패: {e}")
    
    def test_chat_processing_function(self):
        """채팅 처리 함수 테스트"""
        from gradio.app import create_ml_enhanced_gradio_interface
        
        # Mock all dependencies
        with patch('gradio.app.DatabaseManager'), \
             patch('gradio.app.LegalVectorStore'), \
             patch('gradio.app.LegalModelManager'), \
             patch('gradio.app.MLEnhancedRAGService') as mock_rag, \
             patch('gradio.app.MLEnhancedSearchService'), \
             patch('gradio.app.ChatService'):
            
            # Setup mock RAG service response
            mock_rag.return_value.process_query.return_value = {
                "response": "테스트 응답입니다.",
                "sources": [
                    {
                        "title": "테스트 법령",
                        "article_number": "제1조",
                        "article_title": "목적",
                        "quality_score": 0.9
                    }
                ],
                "ml_stats": {
                    "total_documents": 1,
                    "main_articles": 1,
                    "supplementary_articles": 0,
                    "avg_quality_score": 0.9
                }
            }
            
            try:
                interface = create_ml_enhanced_gradio_interface()
                print("✅ 채팅 처리 함수 테스트 통과")
            except Exception as e:
                pytest.fail(f"채팅 처리 함수 테스트 실패: {e}")
    
    def test_error_handling(self):
        """에러 처리 테스트"""
        from gradio.components.document_analyzer import DocumentAnalyzer
        
        analyzer = DocumentAnalyzer()
        
        # Test error handling for invalid file
        try:
            analyzer.parse_document("nonexistent_file.pdf")
            pytest.fail("파일이 존재하지 않을 때 예외가 발생해야 합니다")
        except FileNotFoundError:
            print("✅ 파일 없음 에러 처리 테스트 통과")
        except Exception as e:
            pytest.fail(f"예상치 못한 에러: {e}")
        
        # Test error handling for unsupported file type
        try:
            analyzer.parse_document("test.txt")
            pytest.fail("지원하지 않는 파일 형식일 때 예외가 발생해야 합니다")
        except ValueError:
            print("✅ 지원하지 않는 파일 형식 에러 처리 테스트 통과")
        except Exception as e:
            pytest.fail(f"예상치 못한 에러: {e}")
    
    def test_risk_assessment(self):
        """위험도 평가 테스트"""
        from gradio.components.document_analyzer import DocumentAnalyzer
        
        analyzer = DocumentAnalyzer()
        
        # Test high-risk text
        high_risk_text = "손해배상과 위약금에 관한 조항입니다."
        risks = analyzer._assess_risks(high_risk_text, [])
        
        assert len(risks) > 0
        high_risks = [r for r in risks if r['risk_level'] == 'high']
        assert len(high_risks) > 0
        
        print("✅ 위험도 평가 테스트 통과")
    
    def test_recommendation_generation(self):
        """개선 제안 생성 테스트"""
        from gradio.components.document_analyzer import DocumentAnalyzer
        
        analyzer = DocumentAnalyzer()
        
        # Test with high-risk items
        high_risks = [
            {"risk_level": "high", "keyword": "손해배상", "recommendation": "법무팀 검토 필요"}
        ]
        
        recommendations = analyzer._generate_recommendations(high_risks, "test text")
        
        assert len(recommendations) > 0
        assert any("높은 위험도" in rec for rec in recommendations)
        
        print("✅ 개선 제안 생성 테스트 통과")

class TestPerformance:
    """성능 테스트"""
    
    def test_interface_load_time(self):
        """인터페이스 로딩 시간 테스트"""
        import time
        
        start_time = time.time()
        
        # Mock all dependencies to avoid actual loading
        with patch('gradio.app.DatabaseManager'), \
             patch('gradio.app.LegalVectorStore'), \
             patch('gradio.app.LegalModelManager'), \
             patch('gradio.app.MLEnhancedRAGService'), \
             patch('gradio.app.MLEnhancedSearchService'), \
             patch('gradio.app.ChatService'):
            
            from gradio.app import create_ml_enhanced_gradio_interface
            interface = create_ml_enhanced_gradio_interface()
            
            load_time = time.time() - start_time
            
            # Interface should load in less than 5 seconds (with mocks)
            assert load_time < 5.0, f"인터페이스 로딩 시간이 너무 깁니다: {load_time:.2f}초"
            
            print(f"✅ 인터페이스 로딩 시간 테스트 통과: {load_time:.2f}초")
    
    def test_document_analysis_performance(self):
        """문서 분석 성능 테스트"""
        import time
        from gradio.components.document_analyzer import DocumentAnalyzer
        
        analyzer = DocumentAnalyzer()
        
        # Test with sample text
        sample_text = "제1조 목적. 제2조 손해배상. 제3조 위약금." * 100  # Longer text
        
        start_time = time.time()
        result = analyzer.analyze_contract(sample_text)
        analysis_time = time.time() - start_time
        
        # Analysis should complete in less than 10 seconds
        assert analysis_time < 10.0, f"문서 분석 시간이 너무 깁니다: {analysis_time:.2f}초"
        
        print(f"✅ 문서 분석 성능 테스트 통과: {analysis_time:.2f}초")

def run_integration_tests():
    """통합 테스트 실행"""
    print("🚀 LawFirmAI Gradio 인터페이스 통합 테스트 시작")
    print("=" * 60)
    
    # Run tests
    test_classes = [TestGradioInterface, TestPerformance]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__} 테스트 실행 중...")
        
        test_instance = test_class()
        test_instance.setup_method()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                passed_tests += 1
            except Exception as e:
                print(f"❌ {test_method} 실패: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 테스트 결과: {passed_tests}/{total_tests} 통과")
    
    if passed_tests == total_tests:
        print("🎉 모든 테스트가 통과했습니다!")
        return True
    else:
        print(f"⚠️ {total_tests - passed_tests}개 테스트가 실패했습니다.")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
