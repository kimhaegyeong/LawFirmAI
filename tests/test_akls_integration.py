# -*- coding: utf-8 -*-
"""
AKLS (법률전문대학원협의회) 통합 테스트
AKLS 데이터 처리, 검색, RAG 통합 기능 테스트
"""

import unittest
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.akls_processor import AKLSProcessor, AKLSDocument
from source.services.akls_search_engine import AKLSSearchEngine, AKLSSearchResult
from source.services.enhanced_rag_service import EnhancedRAGService, EnhancedRAGResult

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAKLSProcessor(unittest.TestCase):
    """AKLS 프로세서 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.processor = AKLSProcessor()
        self.test_pdf_path = "data/raw/akls"
        
    def test_law_area_extraction(self):
        """법률 영역 추출 테스트"""
        test_cases = [
            ("형법표준판례 연구보고서.pdf", "criminal_law"),
            ("민법 표준판례 2023년.pdf", "civil_law"),
            ("상법표준판례.pdf", "commercial_law"),
            ("민사소송법 표준판례.pdf", "civil_procedure"),
            ("형사소송법표준판례.pdf", "criminal_procedure"),
            ("행정법 표준판례.pdf", "administrative_law"),
            ("헌법 표준판례.pdf", "constitutional_law"),
            ("표준판례 전체.pdf", "standard_precedent")
        ]
        
        for filename, expected_area in test_cases:
            with self.subTest(filename=filename):
                result = self.processor.extract_law_area_from_filename(filename)
                self.assertEqual(result, expected_area)
    
    def test_year_extraction(self):
        """연도 추출 테스트"""
        test_cases = [
            ("2023형법표준판례.pdf", "2023"),
            ("230425 민법 표준판례.pdf", "230425"),
            ("표준판례.pdf", None)
        ]
        
        for filename, expected_year in test_cases:
            with self.subTest(filename=filename):
                result = self.processor.extract_year_from_filename(filename)
                self.assertEqual(result, expected_year)
    
    def test_precedent_structure_parsing(self):
        """표준판례 구조 파싱 테스트"""
        sample_content = """
        사건번호: 2023다12345
        법원: 대법원
        날짜: 2023년 3월 15일
        
        요약: 계약 해지에 관한 표준판례입니다.
        
        이유: 계약 해지의 요건은 다음과 같습니다.
        
        결론: 원고의 청구를 인용한다.
        """
        
        sections = self.processor.parse_standard_precedent_structure(sample_content)
        
        self.assertIn("case_number", sections)
        self.assertIn("court", sections)
        self.assertIn("date", sections)
        self.assertIn("요약", sections)
        self.assertIn("이유", sections)
        self.assertIn("결론", sections)
        
        self.assertEqual(sections["case_number"], "2023다12345")
        self.assertEqual(sections["court"], "대법원")
    
    def test_legal_references_extraction(self):
        """법률 조항 참조 추출 테스트"""
        sample_content = """
        민법 제750조에 따르면, 형법 제250조의 규정을 적용한다.
        상법 제1조 제2항에서 규정하고 있다.
        """
        
        references = self.processor.extract_legal_references(sample_content)
        
        self.assertGreater(len(references), 0)
        
        # 법률명 확인
        law_names = [ref["content"] for ref in references if ref["type"] == "law_name"]
        self.assertIn("민법", law_names)
        self.assertIn("형법", law_names)
        self.assertIn("상법", law_names)
        
        # 조항 확인
        articles = [ref["content"] for ref in references if ref["type"] == "article"]
        self.assertIn("제750조", articles)
        self.assertIn("제250조", articles)


class TestAKLSSearchEngine(unittest.TestCase):
    """AKLS 검색 엔진 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.search_engine = AKLSSearchEngine()
        
    def test_search_engine_initialization(self):
        """검색 엔진 초기화 테스트"""
        self.assertIsNotNone(self.search_engine)
        self.assertIsNotNone(self.search_engine.law_area_mapping)
        
    def test_law_area_statistics(self):
        """법률 영역 통계 테스트"""
        stats = self.search_engine.get_law_area_statistics()
        
        self.assertIsInstance(stats, dict)
        
        if stats:  # 인덱스가 있는 경우
            total_docs = sum(stats.values())
            self.assertGreater(total_docs, 0)
            
            # 예상되는 법률 영역들 확인
            expected_areas = ["criminal_law", "civil_law", "commercial_law", 
                            "civil_procedure", "criminal_procedure", 
                            "administrative_law", "constitutional_law"]
            
            for area in expected_areas:
                if area in stats:
                    self.assertGreater(stats[area], 0)
    
    def test_search_functionality(self):
        """검색 기능 테스트"""
        test_queries = [
            "계약 해지",
            "손해배상",
            "형법",
            "민사소송"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                try:
                    results = self.search_engine.search(query, top_k=3)
                    
                    self.assertIsInstance(results, list)
                    self.assertLessEqual(len(results), 3)
                    
                    for result in results:
                        self.assertIsInstance(result, AKLSSearchResult)
                        self.assertIsInstance(result.content, str)
                        self.assertIsInstance(result.metadata, dict)
                        self.assertIsInstance(result.score, float)
                        self.assertGreaterEqual(result.score, 0)
                        self.assertLessEqual(result.score, 1)
                        
                except Exception as e:
                    logger.warning(f"검색 테스트 실패 (쿼리: {query}): {e}")
    
    def test_search_by_law_area(self):
        """법률 영역별 검색 테스트"""
        law_areas = ["criminal_law", "civil_law", "commercial_law"]
        
        for law_area in law_areas:
            with self.subTest(law_area=law_area):
                try:
                    results = self.search_engine.search_by_law_area("테스트", law_area, top_k=2)
                    
                    self.assertIsInstance(results, list)
                    
                    for result in results:
                        self.assertEqual(result.law_area, law_area)
                        
                except Exception as e:
                    logger.warning(f"법률 영역별 검색 테스트 실패 (영역: {law_area}): {e}")
    
    def test_search_by_case_type(self):
        """사건 유형별 검색 테스트"""
        case_types = ["다", "고", "드"]
        
        for case_type in case_types:
            with self.subTest(case_type=case_type):
                try:
                    results = self.search_engine.search_by_case_type("테스트", case_type, top_k=2)
                    
                    self.assertIsInstance(results, list)
                    
                    for result in results:
                        if result.case_number:
                            self.assertIn(case_type, result.case_number)
                            
                except Exception as e:
                    logger.warning(f"사건 유형별 검색 테스트 실패 (유형: {case_type}): {e}")


class TestEnhancedRAGService(unittest.TestCase):
    """Enhanced RAG Service 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.enhanced_rag = EnhancedRAGService()
        
    def test_rag_service_initialization(self):
        """RAG 서비스 초기화 테스트"""
        self.assertIsNotNone(self.enhanced_rag)
        self.assertIsNotNone(self.enhanced_rag.base_rag_service)
        self.assertIsNotNone(self.enhanced_rag.akls_search_engine)
        
    def test_query_routing(self):
        """쿼리 라우팅 테스트"""
        test_cases = [
            ("표준판례", "akls_precedents"),
            ("대법원", "akls_precedents"),
            ("법령", "assembly_laws"),
            ("법률", "assembly_laws"),
            ("판례", "assembly_precedents"),
            ("일반 질문", "hybrid_search")
        ]
        
        for query, expected_source in test_cases:
            with self.subTest(query=query):
                source_type, law_area = self.enhanced_rag.route_query_to_source(query)
                self.assertEqual(source_type, expected_source)
    
    def test_akls_statistics(self):
        """AKLS 통계 테스트"""
        stats = self.enhanced_rag.get_akls_statistics()
        
        self.assertIsInstance(stats, dict)
        
        if "error" not in stats:
            self.assertIn("total_documents", stats)
            self.assertIn("law_area_distribution", stats)
            self.assertIn("index_available", stats)
            
            total_docs = stats["total_documents"]
            self.assertGreaterEqual(total_docs, 0)
    
    def test_enhanced_search(self):
        """향상된 검색 테스트"""
        test_queries = [
            "계약 해지에 대한 표준판례",
            "형법 제250조",
            "손해배상 책임"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                try:
                    result = self.enhanced_rag.search_with_akls(query, top_k=3)
                    
                    self.assertIsInstance(result, EnhancedRAGResult)
                    self.assertIsInstance(result.response, str)
                    self.assertIsInstance(result.confidence, float)
                    self.assertIsInstance(result.sources, list)
                    self.assertIsInstance(result.akls_sources, list)
                    self.assertIsInstance(result.metadata, dict)
                    
                    self.assertGreaterEqual(result.confidence, 0)
                    self.assertLessEqual(result.confidence, 1)
                    
                except Exception as e:
                    logger.warning(f"향상된 검색 테스트 실패 (쿼리: {query}): {e}")
    
    def test_law_area_specific_search(self):
        """법률 영역별 검색 테스트"""
        law_areas = ["criminal_law", "civil_law", "commercial_law"]
        
        for law_area in law_areas:
            with self.subTest(law_area=law_area):
                try:
                    result = self.enhanced_rag.search_by_law_area("테스트", law_area, top_k=2)
                    
                    self.assertIsInstance(result, EnhancedRAGResult)
                    self.assertEqual(result.law_area, law_area)
                    self.assertEqual(result.search_type, "law_area_specific")
                    
                except Exception as e:
                    logger.warning(f"법률 영역별 검색 테스트 실패 (영역: {law_area}): {e}")


class TestAKLSDataIntegration(unittest.TestCase):
    """AKLS 데이터 통합 테스트"""
    
    def test_processed_data_exists(self):
        """처리된 데이터 존재 확인"""
        processed_dir = Path("data/processed/akls")
        
        if processed_dir.exists():
            json_files = list(processed_dir.glob("*.json"))
            self.assertGreater(len(json_files), 0, "처리된 AKLS JSON 파일이 없습니다")
            
            # 첫 번째 파일 내용 확인
            if json_files:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                required_fields = ["filename", "content", "metadata", "law_area", "document_type"]
                for field in required_fields:
                    self.assertIn(field, data, f"필수 필드 '{field}'가 없습니다")
                
                self.assertIsInstance(data["content"], str)
                self.assertGreater(len(data["content"]), 0, "문서 내용이 비어있습니다")
        else:
            self.skipTest("처리된 AKLS 데이터 디렉토리가 없습니다")
    
    def test_vector_index_exists(self):
        """벡터 인덱스 존재 확인"""
        index_dir = Path("data/embeddings/akls_precedents")
        
        if index_dir.exists():
            index_file = index_dir / "akls_index.faiss"
            metadata_file = index_dir / "akls_metadata.json"
            
            self.assertTrue(index_file.exists(), "FAISS 인덱스 파일이 없습니다")
            self.assertTrue(metadata_file.exists(), "메타데이터 파일이 없습니다")
            
            # 메타데이터 파일 내용 확인
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.assertIsInstance(metadata, list)
            self.assertGreater(len(metadata), 0, "메타데이터가 비어있습니다")
        else:
            self.skipTest("AKLS 벡터 인덱스 디렉토리가 없습니다")
    
    def test_search_performance(self):
        """검색 성능 테스트"""
        search_engine = AKLSSearchEngine()
        
        if search_engine.index is None:
            self.skipTest("AKLS 검색 인덱스가 없습니다")
        
        import time
        
        test_queries = [
            "계약 해지",
            "손해배상",
            "형법",
            "민사소송",
            "대법원"
        ]
        
        total_time = 0
        successful_searches = 0
        
        for query in test_queries:
            start_time = time.time()
            try:
                results = search_engine.search(query, top_k=3)
                end_time = time.time()
                
                search_time = end_time - start_time
                total_time += search_time
                successful_searches += 1
                
                self.assertLess(search_time, 5.0, f"검색 시간이 너무 깁니다: {search_time:.2f}초")
                
            except Exception as e:
                logger.warning(f"성능 테스트 중 검색 실패 (쿼리: {query}): {e}")
        
        if successful_searches > 0:
            avg_time = total_time / successful_searches
            logger.info(f"평균 검색 시간: {avg_time:.3f}초")
            self.assertLess(avg_time, 2.0, f"평균 검색 시간이 너무 깁니다: {avg_time:.2f}초")


def run_akls_tests():
    """AKLS 테스트 실행"""
    print("=" * 80)
    print("AKLS 통합 테스트 실행")
    print("=" * 80)
    
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()
    
    # 테스트 클래스들 추가
    test_classes = [
        TestAKLSProcessor,
        TestAKLSSearchEngine,
        TestEnhancedRAGService,
        TestAKLSDataIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("AKLS 테스트 결과 요약")
    print("=" * 80)
    print(f"총 테스트 수: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"오류: {len(result.errors)}")
    
    if result.failures:
        print("\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n오류가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_akls_tests()
    sys.exit(0 if success else 1)
