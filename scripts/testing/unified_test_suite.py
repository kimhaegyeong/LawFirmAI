#!/usr/bin/env python3
"""
통합 테스트 스위트 매니저
다양한 테스트 타입과 실행 전략을 지원하는 통합 시스템
"""

import sys
import os
from pathlib import Path
import json
import time
import asyncio
import multiprocessing
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import traceback
from enum import Enum
import logging

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from source.services.multi_stage_validation_system import MultiStageValidationSystem
    from source.services.chat_service import ChatService
    from source.utils.config import Config
    from source.services.semantic_search_engine import SemanticSearchEngine
except ImportError as e:
    print(f"Warning: Could not import source modules: {e}")

try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Warning: Could not import ML libraries: {e}")
    np = None
    faiss = None
    SentenceTransformer = None


class TestType(Enum):
    """테스트 타입"""
    PERFORMANCE = "performance"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    EDGE_CASE = "edge_case"
    LEGAL_RESTRICTION = "legal_restriction"
    MASSIVE = "massive"
    CONTINUOUS_LEARNING = "continuous_learning"
    VECTOR_EMBEDDING = "vector_embedding"
    SEMANTIC_SEARCH = "semantic_search"


class ExecutionMode(Enum):
    """실행 모드"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ASYNC = "async"
    MULTIPROCESS = "multiprocess"


@dataclass
class TestConfig:
    """테스트 설정"""
    test_type: TestType = TestType.VALIDATION
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_workers: int = 4
    batch_size: int = 100
    timeout_seconds: int = 300
    log_level: str = "INFO"
    enable_chat_service: bool = True
    use_improved_validation: bool = False
    save_results: bool = True
    results_dir: str = "results"


@dataclass
class TestResult:
    """테스트 결과"""
    test_id: str
    query: str
    response: str
    is_legal: bool
    confidence: float
    processing_time: float
    error: Optional[str] = None
    timestamp: str = ""


class UnifiedTestSuite:
    """통합 테스트 스위트"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.results = []
        self.start_time = None
        
        # 결과 저장 경로
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(f"test_suite_{self.config.test_type.value}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # 핸들러가 이미 있으면 제거
        if logger.handlers:
            logger.handlers.clear()
            
        # 파일 핸들러
        log_file = f"logs/unified_test_{self.config.test_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def run_tests(self, test_queries: List[str]) -> Dict[str, Any]:
        """테스트 실행"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.config.test_type.value} tests with {len(test_queries)} queries")
        
        results = {
            'test_type': self.config.test_type.value,
            'execution_mode': self.config.execution_mode.value,
            'start_time': self.start_time.isoformat(),
            'total_queries': len(test_queries),
            'test_results': [],
            'summary': {},
            'errors': []
        }
        
        try:
            # 실행 모드별 테스트 실행
            if self.config.execution_mode == ExecutionMode.SEQUENTIAL:
                results['test_results'] = self._run_sequential_tests(test_queries)
            elif self.config.execution_mode == ExecutionMode.PARALLEL:
                results['test_results'] = self._run_parallel_tests(test_queries)
            elif self.config.execution_mode == ExecutionMode.ASYNC:
                results['test_results'] = asyncio.run(self._run_async_tests(test_queries))
            elif self.config.execution_mode == ExecutionMode.MULTIPROCESS:
                results['test_results'] = self._run_multiprocess_tests(test_queries)
            
            # 결과 분석
            results['summary'] = self._analyze_results(results['test_results'])
            
            # 결과 저장
            if self.config.save_results:
                self._save_results(results)
            
            # 완료
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = duration
            
            self.logger.info(f"Tests completed successfully in {duration:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}", exc_info=True)
            results['errors'].append(str(e))
            return results
    
    def _run_sequential_tests(self, test_queries: List[str]) -> List[Dict[str, Any]]:
        """순차 테스트 실행"""
        self.logger.info("Running sequential tests")
        
        test_results = []
        
        for i, query in enumerate(test_queries):
            try:
                self.logger.info(f"Processing query {i+1}/{len(test_queries)}: {query[:50]}...")
                
                result = self._execute_single_test(query, i)
                test_results.append(asdict(result))
                
                # 진행률 출력
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Progress: {i+1}/{len(test_queries)} ({((i+1)/len(test_queries)*100):.1f}%)")
                
            except Exception as e:
                self.logger.error(f"Error processing query {i+1}: {e}")
                error_result = TestResult(
                    test_id=f"test_{i}",
                    query=query,
                    response="",
                    is_legal=False,
                    confidence=0.0,
                    processing_time=0.0,
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )
                test_results.append(asdict(error_result))
        
        return test_results
    
    def _run_parallel_tests(self, test_queries: List[str]) -> List[Dict[str, Any]]:
        """병렬 테스트 실행"""
        self.logger.info(f"Running parallel tests with {self.config.max_workers} workers")
        
        test_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 배치 단위로 처리
            for i in range(0, len(test_queries), self.config.batch_size):
                batch = test_queries[i:i + self.config.batch_size]
                
                # 배치 작업 제출
                future_to_query = {
                    executor.submit(self._execute_single_test, query, i + j): query
                    for j, query in enumerate(batch)
                }
                
                # 결과 수집
                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        result = future.result(timeout=self.config.timeout_seconds)
                        test_results.append(asdict(result))
                    except Exception as e:
                        self.logger.error(f"Error processing query: {e}")
                        error_result = TestResult(
                            test_id=f"test_{len(test_results)}",
                            query=query,
                            response="",
                            is_legal=False,
                            confidence=0.0,
                            processing_time=0.0,
                            error=str(e),
                            timestamp=datetime.now().isoformat()
                        )
                        test_results.append(asdict(error_result))
                
                self.logger.info(f"Completed batch {i//self.config.batch_size + 1}")
        
        return test_results
    
    async def _run_async_tests(self, test_queries: List[str]) -> List[Dict[str, Any]]:
        """비동기 테스트 실행"""
        self.logger.info("Running async tests")
        
        test_results = []
        
        # 세마포어로 동시 실행 수 제한
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_query_with_semaphore(query: str, index: int):
            async with semaphore:
                return await self._execute_single_test_async(query, index)
        
        # 모든 테스트 작업 생성
        tasks = [
            process_query_with_semaphore(query, i)
            for i, query in enumerate(test_queries)
        ]
        
        # 배치 단위로 실행
        for i in range(0, len(tasks), self.config.batch_size):
            batch_tasks = tasks[i:i + self.config.batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Async test error: {result}")
                    error_result = TestResult(
                        test_id=f"test_{len(test_results)}",
                        query="",
                        response="",
                        is_legal=False,
                        confidence=0.0,
                        processing_time=0.0,
                        error=str(result),
                        timestamp=datetime.now().isoformat()
                    )
                    test_results.append(asdict(error_result))
                else:
                    test_results.append(asdict(result))
            
            self.logger.info(f"Completed async batch {i//self.config.batch_size + 1}")
        
        return test_results
    
    def _run_multiprocess_tests(self, test_queries: List[str]) -> List[Dict[str, Any]]:
        """멀티프로세스 테스트 실행"""
        self.logger.info(f"Running multiprocess tests with {self.config.max_workers} processes")
        
        test_results = []
        
        # 배치 단위로 처리
        batches = [
            test_queries[i:i + self.config.batch_size]
            for i in range(0, len(test_queries), self.config.batch_size)
        ]
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 배치 작업 제출
            future_to_batch = {
                executor.submit(self._process_batch_multiprocess, batch, i): batch
                for i, batch in enumerate(batches)
            }
            
            # 결과 수집
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result(timeout=self.config.timeout_seconds)
                    test_results.extend(batch_results)
                except Exception as e:
                    self.logger.error(f"Error processing batch: {e}")
                    # 에러 발생 시 빈 결과 추가
                    for query in batch:
                        error_result = TestResult(
                            test_id=f"test_{len(test_results)}",
                            query=query,
                            response="",
                            is_legal=False,
                            confidence=0.0,
                            processing_time=0.0,
                            error=str(e),
                            timestamp=datetime.now().isoformat()
                        )
                        test_results.append(asdict(error_result))
        
        return test_results
    
    def _execute_single_test(self, query: str, index: int) -> TestResult:
        """단일 테스트 실행"""
        start_time = time.time()
        
        try:
            # 검증 시스템 초기화
            if self.config.use_improved_validation:
                try:
                    from source.services.improved_multi_stage_validation_system import ImprovedMultiStageValidationSystem
                    validation_system = ImprovedMultiStageValidationSystem()
                except ImportError:
                    validation_system = MultiStageValidationSystem()
            else:
                validation_system = MultiStageValidationSystem()
            
            # 테스트 타입별 실행
            if self.config.test_type == TestType.VALIDATION:
                result = self._run_validation_test(query, validation_system)
            elif self.config.test_type == TestType.PERFORMANCE:
                result = self._run_performance_test(query, validation_system)
            elif self.config.test_type == TestType.INTEGRATION:
                result = self._run_integration_test(query, validation_system)
            elif self.config.test_type == TestType.EDGE_CASE:
                result = self._run_edge_case_test(query, validation_system)
            elif self.config.test_type == TestType.LEGAL_RESTRICTION:
                result = self._run_legal_restriction_test(query, validation_system)
            elif self.config.test_type == TestType.MASSIVE:
                result = self._run_massive_test(query, validation_system)
            elif self.config.test_type == TestType.CONTINUOUS_LEARNING:
                result = self._run_continuous_learning_test(query, validation_system)
            elif self.config.test_type == TestType.VECTOR_EMBEDDING:
                result = self._run_vector_embedding_test(query, validation_system)
            elif self.config.test_type == TestType.SEMANTIC_SEARCH:
                result = self._run_semantic_search_test(query, validation_system)
            else:
                result = self._run_default_test(query, validation_system)
            
            processing_time = time.time() - start_time
            
            return TestResult(
                test_id=f"test_{index}",
                query=query,
                response=result.get('response', ''),
                is_legal=result.get('is_legal', False),
                confidence=result.get('confidence', 0.0),
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return TestResult(
                test_id=f"test_{index}",
                query=query,
                response="",
                is_legal=False,
                confidence=0.0,
                processing_time=processing_time,
                error=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    async def _execute_single_test_async(self, query: str, index: int) -> TestResult:
        """비동기 단일 테스트 실행"""
        start_time = time.time()
        
        try:
            # 비동기 검증 시스템 사용
            validation_system = MultiStageValidationSystem()
            
            # 비동기 테스트 실행
            result = await self._run_async_validation_test(query, validation_system)
            
            processing_time = time.time() - start_time
            
            return TestResult(
                test_id=f"test_{index}",
                query=query,
                response=result.get('response', ''),
                is_legal=result.get('is_legal', False),
                confidence=result.get('confidence', 0.0),
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return TestResult(
                test_id=f"test_{index}",
                query=query,
                response="",
                is_legal=False,
                confidence=0.0,
                processing_time=processing_time,
                error=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    def _process_batch_multiprocess(self, batch: List[str], batch_index: int) -> List[Dict[str, Any]]:
        """멀티프로세스 배치 처리"""
        results = []
        
        try:
            # 프로세스별 검증 시스템 초기화
            validation_system = MultiStageValidationSystem()
            
            for i, query in enumerate(batch):
                try:
                    result = self._run_validation_test(query, validation_system)
                    processing_time = 0.1  # 대략적인 처리 시간
                    
                    test_result = TestResult(
                        test_id=f"batch_{batch_index}_test_{i}",
                        query=query,
                        response=result.get('response', ''),
                        is_legal=result.get('is_legal', False),
                        confidence=result.get('confidence', 0.0),
                        processing_time=processing_time,
                        timestamp=datetime.now().isoformat()
                    )
                    
                    results.append(asdict(test_result))
                    
                except Exception as e:
                    error_result = TestResult(
                        test_id=f"batch_{batch_index}_test_{i}",
                        query=query,
                        response="",
                        is_legal=False,
                        confidence=0.0,
                        processing_time=0.0,
                        error=str(e),
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(asdict(error_result))
            
        except Exception as e:
            # 배치 전체 실패 시 빈 결과 반환
            for i, query in enumerate(batch):
                error_result = TestResult(
                    test_id=f"batch_{batch_index}_test_{i}",
                    query=query,
                    response="",
                    is_legal=False,
                    confidence=0.0,
                    processing_time=0.0,
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )
                results.append(asdict(error_result))
        
        return results
    
    def _run_validation_test(self, query: str, validation_system) -> Dict[str, Any]:
        """검증 테스트 실행"""
        try:
            result = validation_system.validate_query(query)
            return {
                'response': result.get('response', ''),
                'is_legal': result.get('is_legal', False),
                'confidence': result.get('confidence', 0.0)
            }
        except Exception as e:
            return {
                'response': '',
                'is_legal': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _run_performance_test(self, query: str, validation_system) -> Dict[str, Any]:
        """성능 테스트 실행"""
        start_time = time.time()
        
        try:
            result = validation_system.validate_query(query)
            processing_time = time.time() - start_time
            
            return {
                'response': result.get('response', ''),
                'is_legal': result.get('is_legal', False),
                'confidence': result.get('confidence', 0.0),
                'processing_time': processing_time
            }
        except Exception as e:
            return {
                'response': '',
                'is_legal': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _run_integration_test(self, query: str, validation_system) -> Dict[str, Any]:
        """통합 테스트 실행"""
        try:
            # 채팅 서비스와 함께 테스트
            if self.config.enable_chat_service:
                config = Config()
                chat_service = ChatService(config)
                
                # 채팅 서비스 응답
                chat_response = chat_service.process_message(query)
                
                # 검증 시스템 응답
                validation_result = validation_system.validate_query(query)
                
                return {
                    'response': chat_response,
                    'is_legal': validation_result.get('is_legal', False),
                    'confidence': validation_result.get('confidence', 0.0),
                    'chat_response': chat_response,
                    'validation_response': validation_result.get('response', '')
                }
            else:
                return self._run_validation_test(query, validation_system)
                
        except Exception as e:
            return {
                'response': '',
                'is_legal': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _run_edge_case_test(self, query: str, validation_system) -> Dict[str, Any]:
        """엣지 케이스 테스트 실행"""
        try:
            # 특수한 케이스들 테스트
            edge_cases = [
                query.upper(),  # 대문자
                query.lower(),  # 소문자
                query.strip(),  # 공백 제거
                query.replace(' ', ''),  # 공백 제거
            ]
            
            results = []
            for case in edge_cases:
                result = validation_system.validate_query(case)
                results.append(result)
            
            # 가장 일반적인 결과 반환
            return results[0] if results else {
                'response': '',
                'is_legal': False,
                'confidence': 0.0
            }
            
        except Exception as e:
            return {
                'response': '',
                'is_legal': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _run_legal_restriction_test(self, query: str, validation_system) -> Dict[str, Any]:
        """법적 제한 테스트 실행"""
        try:
            # 법적 제한 관련 키워드 확인
            restricted_keywords = [
                '범죄', '살인', '강도', '절도', '사기', '폭력',
                '마약', '도박', '매춘', '성매매', '아동', '청소년'
            ]
            
            has_restricted_keyword = any(keyword in query for keyword in restricted_keywords)
            
            result = validation_system.validate_query(query)
            
            return {
                'response': result.get('response', ''),
                'is_legal': result.get('is_legal', False),
                'confidence': result.get('confidence', 0.0),
                'has_restricted_keyword': has_restricted_keyword
            }
            
        except Exception as e:
            return {
                'response': '',
                'is_legal': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _run_massive_test(self, query: str, validation_system) -> Dict[str, Any]:
        """대규모 테스트 실행"""
        try:
            # 대규모 테스트를 위한 최적화된 처리
            result = validation_system.validate_query(query)
            
            return {
                'response': result.get('response', ''),
                'is_legal': result.get('is_legal', False),
                'confidence': result.get('confidence', 0.0),
                'test_type': 'massive'
            }
            
        except Exception as e:
            return {
                'response': '',
                'is_legal': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _run_continuous_learning_test(self, query: str, validation_system) -> Dict[str, Any]:
        """연속 학습 테스트 실행"""
        try:
            # 연속 학습을 위한 피드백 수집
            result = validation_system.validate_query(query)
            
            # 피드백 데이터 수집
            feedback_data = {
                'query': query,
                'response': result.get('response', ''),
                'is_legal': result.get('is_legal', False),
                'confidence': result.get('confidence', 0.0),
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'response': result.get('response', ''),
                'is_legal': result.get('is_legal', False),
                'confidence': result.get('confidence', 0.0),
                'feedback_data': feedback_data
            }
            
        except Exception as e:
            return {
                'response': '',
                'is_legal': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _run_vector_embedding_test(self, query: str, validation_system) -> Dict[str, Any]:
        """벡터 임베딩 테스트 실행"""
        try:
            if np is None or faiss is None or SentenceTransformer is None:
                return {
                    'response': '',
                    'is_legal': False,
                    'confidence': 0.0,
                    'error': 'Required ML libraries not available'
                }
            
            # 임베딩 파일 로드
            embeddings = np.load('data/embeddings/embeddings_jhgan_ko-sroberta-multitask.npy')
            
            # 모델 로드
            model = SentenceTransformer("jhgan/ko-sroberta-multitask")
            
            # 쿼리 임베딩 생성
            query_embedding = model.encode([query])
            
            # FAISS 인덱스 생성
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            
            # 벡터 정규화 및 인덱스에 추가
            embeddings_f32 = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings_f32)
            index.add(embeddings_f32)
            
            # 검색 수행
            query_embedding_normalized = query_embedding.copy().astype(np.float32)
            faiss.normalize_L2(query_embedding_normalized)
            
            scores, indices = index.search(query_embedding_normalized, k=3)
            
            return {
                'response': f"Vector embedding test completed. Found {len(indices[0])} similar vectors.",
                'is_legal': True,
                'confidence': float(scores[0][0]) if len(scores[0]) > 0 else 0.0,
                'vector_count': len(embeddings),
                'search_results': len(indices[0]),
                'top_score': float(scores[0][0]) if len(scores[0]) > 0 else 0.0
            }
            
        except Exception as e:
            return {
                'response': '',
                'is_legal': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _run_semantic_search_test(self, query: str, validation_system) -> Dict[str, Any]:
        """시맨틱 검색 테스트 실행"""
        try:
            if SemanticSearchEngine is None:
                return {
                    'response': '',
                    'is_legal': False,
                    'confidence': 0.0,
                    'error': 'SemanticSearchEngine not available'
                }
            
            # 시맨틱 검색 엔진 초기화
            search_engine = SemanticSearchEngine()
            
            # 검색 수행
            results = search_engine.search(query, k=3)
            
            if results:
                top_result = results[0]
                return {
                    'response': f"Semantic search completed. Found {len(results)} results.",
                    'is_legal': True,
                    'confidence': top_result.get('score', 0.0),
                    'search_results': len(results),
                    'top_title': top_result.get('title', 'No title')[:50],
                    'top_score': top_result.get('score', 0.0)
                }
            else:
                return {
                    'response': "Semantic search completed. No results found.",
                    'is_legal': True,
                    'confidence': 0.0,
                    'search_results': 0
                }
                
        except Exception as e:
            return {
                'response': '',
                'is_legal': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _run_default_test(self, query: str, validation_system) -> Dict[str, Any]:
        """기본 테스트 실행"""
        return self._run_validation_test(query, validation_system)
    
    async def _run_async_validation_test(self, query: str, validation_system) -> Dict[str, Any]:
        """비동기 검증 테스트 실행"""
        try:
            # 비동기 처리를 위한 지연
            await asyncio.sleep(0.01)
            
            result = validation_system.validate_query(query)
            return {
                'response': result.get('response', ''),
                'is_legal': result.get('is_legal', False),
                'confidence': result.get('confidence', 0.0)
            }
        except Exception as e:
            return {
                'response': '',
                'is_legal': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _analyze_results(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """결과 분석"""
        if not test_results:
            return {}
        
        total_tests = len(test_results)
        successful_tests = len([r for r in test_results if not r.get('error')])
        failed_tests = total_tests - successful_tests
        
        legal_queries = len([r for r in test_results if r.get('is_legal', False)])
        illegal_queries = total_tests - legal_queries
        
        avg_confidence = sum(r.get('confidence', 0.0) for r in test_results) / total_tests
        avg_processing_time = sum(r.get('processing_time', 0.0) for r in test_results) / total_tests
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
            'legal_queries': legal_queries,
            'illegal_queries': illegal_queries,
            'avg_confidence': avg_confidence,
            'avg_processing_time': avg_processing_time
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """결과 저장"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"unified_test_results_{self.config.test_type.value}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Results saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Test Suite Manager')
    parser.add_argument('--test-type', choices=['performance', 'validation', 'integration', 'edge_case', 'legal_restriction', 'massive', 'continuous_learning', 'vector_embedding', 'semantic_search'], 
                       default='validation', help='Test type')
    parser.add_argument('--execution-mode', choices=['sequential', 'parallel', 'async', 'multiprocess'], 
                       default='sequential', help='Execution mode')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum workers')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds')
    parser.add_argument('--queries-file', help='File containing test queries')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    # 테스트 쿼리 로드
    if args.queries_file:
        with open(args.queries_file, 'r', encoding='utf-8') as f:
            test_queries = [line.strip() for line in f if line.strip()]
    else:
        # 기본 테스트 쿼리
        test_queries = [
            "계약서 검토 요청",
            "법률 상담 문의",
            "판례 검색",
            "법령 해설",
            "법적 문제 해결"
        ]
    
    # 설정 생성
    config = TestConfig(
        test_type=TestType(args.test_type),
        execution_mode=ExecutionMode(args.execution_mode),
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout,
        log_level=args.log_level
    )
    
    # 테스트 스위트 생성 및 실행
    test_suite = UnifiedTestSuite(config)
    results = test_suite.run_tests(test_queries)
    
    # 결과 출력
    print(f"\n=== Test Results ===")
    print(f"Test Type: {results['test_type']}")
    print(f"Execution Mode: {results['execution_mode']}")
    print(f"Total Queries: {results['total_queries']}")
    print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
    print(f"Success Rate: {results['summary'].get('success_rate', 0):.1f}%")
    print(f"Avg Confidence: {results['summary'].get('avg_confidence', 0):.2f}")
    print(f"Avg Processing Time: {results['summary'].get('avg_processing_time', 0):.3f}s")
    
    if results['errors']:
        print(f"\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
