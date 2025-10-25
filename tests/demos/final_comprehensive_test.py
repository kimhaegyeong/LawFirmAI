# -*- coding: utf-8 -*-
"""
Final Comprehensive Answer Quality Test
최종 종합 답변 품질 테스트 - 개선된 콘솔 로그 가독성
"""

import asyncio
import sys
import os
import time
from typing import List, Dict, Any
from enum import Enum

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, '.')


class LogLevel(Enum):
    """로그 레벨 정의"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class ConsoleLogger:
    """개선된 콘솔 로거"""
    
    def __init__(self, log_level: LogLevel = LogLevel.INFO):
        self.log_level = log_level
        self.colors = {
            LogLevel.DEBUG: "\033[90m",      # 회색
            LogLevel.INFO: "\033[94m",       # 파란색
            LogLevel.WARNING: "\033[93m",    # 노란색
            LogLevel.ERROR: "\033[91m",      # 빨간색
            LogLevel.SUCCESS: "\033[92m",    # 초록색
            "RESET": "\033[0m"               # 리셋
        }
    
    def log(self, level: LogLevel, message: str, show_prefix: bool = True):
        """로그 메시지 출력"""
        if self._should_log(level):
            color = self.colors.get(level, "")
            reset = self.colors["RESET"]
            prefix = f"[{level.value}] " if show_prefix else ""
            print(f"{color}{prefix}{message}{reset}")
    
    def _should_log(self, level: LogLevel) -> bool:
        """로그 레벨에 따른 출력 여부 결정"""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.SUCCESS: 4
        }
        return level_order[level] >= level_order[self.log_level]
    
    def debug(self, message: str):
        self.log(LogLevel.DEBUG, message)
    
    def info(self, message: str):
        self.log(LogLevel.INFO, message)
    
    def warning(self, message: str):
        self.log(LogLevel.WARNING, message)
    
    def error(self, message: str):
        self.log(LogLevel.ERROR, message)
    
    def success(self, message: str):
        self.log(LogLevel.SUCCESS, message)


class ProgressTracker:
    """진행률 추적기"""
    
    def __init__(self, total: int, title: str = "진행률"):
        self.total = total
        self.current = 0
        self.title = title
        self.start_time = time.time()
    
    def update(self, message: str = ""):
        """진행률 업데이트"""
        self.current += 1
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        # 진행률 바 표시
        bar_length = 30
        filled_length = int(bar_length * self.current // self.total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        print(f"\r{self.title}: [{bar}] {percentage:.1f}% ({self.current}/{self.total}) {message}", end="")
        if self.current == self.total:
            print(f" 완료! ({elapsed:.2f}초)")


class SectionLogger:
    """섹션별 로그 관리"""
    
    def __init__(self, title: str, level: int = 0):
        self.title = title
        self.level = level
        self.indent = "  " * level
    
    def __enter__(self):
        print(f"\n{self.indent}📋 {self.title}")
        print(f"{self.indent}{'=' * (len(self.title) + 3)}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{self.indent}✅ {self.title} 완료\n")


def summarize_response(response: str, max_length: int = 100) -> str:
    """응답을 요약하여 표시"""
    if len(response) <= max_length:
        return response
    
    # 문장 단위로 자르기
    sentences = response.split('. ')
    summary = ""
    for sentence in sentences:
        if len(summary + sentence) > max_length:
            break
        summary += sentence + ". "
    
    return summary.rstrip() + "..." if len(response) > max_length else summary


def print_results_table(results: List[Dict], title: str):
    """결과를 테이블 형태로 표시"""
    print(f"\n📊 {title}")
    print("-" * 100)
    
    # 헤더
    print(f"{'질문':<25} {'카테고리':<12} {'신뢰도':<8} {'시간':<8} {'상태':<10} {'방법':<20}")
    print("-" * 100)
    
    # 데이터 행
    for i, result in enumerate(results, 1):
        if result['success']:
            question = result['test_case']['question'][:23] + ".." if len(result['test_case']['question']) > 25 else result['test_case']['question']
            category = result['test_case']['category'][:10] + ".." if len(result['test_case']['category']) > 12 else result['test_case']['category']
            confidence = f"{result.get('confidence', 0):.2f}"
            time_taken = f"{result.get('processing_time', 0):.2f}s"
            status = "✅ 성공" if not result.get('is_restricted', False) else "⚠️ 제한"
            method = result.get('generation_method', 'unknown')[:18] + ".." if len(result.get('generation_method', 'unknown')) > 20 else result.get('generation_method', 'unknown')
            
            print(f"{question:<25} {category:<12} {confidence:<8} {time_taken:<8} {status:<10} {method:<20}")
        else:
            print(f"질문 {i:<25} {'실패':<12} {'N/A':<8} {'N/A':<8} {'❌ 실패':<10} {'N/A':<20}")


class RealTimeStats:
    """실시간 통계 추적"""
    
    def __init__(self):
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'restricted': 0,
            'total_time': 0,
            'total_confidence': 0
        }
    
    def update(self, result: Dict):
        """통계 업데이트"""
        self.stats['total'] += 1
        if result['success']:
            self.stats['success'] += 1
            if result.get('is_restricted', False):
                self.stats['restricted'] += 1
            self.stats['total_time'] += result.get('processing_time', 0)
            self.stats['total_confidence'] += result.get('confidence', 0)
        else:
            self.stats['failed'] += 1
        
        self._print_stats()
    
    def _print_stats(self):
        """실시간 통계 출력"""
        success_rate = (self.stats['success'] / self.stats['total']) * 100 if self.stats['total'] > 0 else 0
        avg_time = self.stats['total_time'] / self.stats['success'] if self.stats['success'] > 0 else 0
        avg_confidence = self.stats['total_confidence'] / self.stats['success'] if self.stats['success'] > 0 else 0
        
        print(f"\r📈 실시간 통계: 성공률 {success_rate:.1f}% | 평균시간 {avg_time:.2f}s | 신뢰도 {avg_confidence:.2f}", end="")


# 전역 로거 인스턴스
logger = ConsoleLogger(LogLevel.INFO)

logger.success("🚀 최종 종합 답변 품질 테스트")
logger.info("=" * 70)

try:
    from source.utils.config import Config
    from source.services.enhanced_chat_service import EnhancedChatService
    from source.utils.langfuse_monitor import get_langfuse_monitor
    logger.success("✅ 모든 모듈 import 성공")
except ImportError as e:
    logger.error(f"❌ 모듈 import 실패: {e}")
    sys.exit(1)


def generate_comprehensive_test_questions() -> List[Dict[str, Any]]:
    """종합 테스트 질문 생성 (5개 질문) - 지능형 스타일 시스템 테스트용"""
    questions = [
        # 간결한 답변 요청
        {"question": "손해배상 청구 방법을 간단히 알려주세요", "category": "민사법", "expected_type": "civil_law", "priority": "high", "expected_style": "concise"},
        
        # 상세한 답변 요청
        {"question": "계약서 작성 방법을 자세히 구체적으로 설명해주세요", "category": "계약서", "expected_type": "contract", "priority": "high", "expected_style": "detailed"},
        
        # 대화형 답변 요청
        {"question": "부동산 매매 절차를 도와주세요", "category": "부동산", "expected_type": "real_estate", "priority": "high", "expected_style": "interactive"},
        
        # 전문적인 답변 요청
        {"question": "이혼 소송의 법적 근거와 판례를 알려주세요", "category": "가족법", "expected_type": "family_law", "priority": "high", "expected_style": "professional"},
        
        # 친근한 답변 요청
        {"question": "법률 문제로 고민이 많아요. 친근하게 도움을 주세요", "category": "일반법률", "expected_type": "general", "priority": "medium", "expected_style": "friendly"},
    ]
    
    return questions


async def test_comprehensive_answer_quality():
    """개선된 종합 답변 품질 테스트 (LangGraph 통합)"""
    
    try:
        with SectionLogger("🚀 테스트 초기화", 0):
            # 설정 로드
            config = Config()
            logger.success("Config 로드 성공")
            
            # LangGraph 통합 상태 확인
            logger.info("🔍 LangGraph 통합 상태 확인 중...")
            try:
                from source.services.langgraph.integrated_workflow_service import IntegratedWorkflowService
                langgraph_service = IntegratedWorkflowService(config)
                logger.success("✅ LangGraph 통합 서비스 초기화 성공")
                langgraph_available = True
            except Exception as e:
                logger.warning(f"⚠️ LangGraph 통합 서비스 초기화 실패: {e}")
                logger.info("📝 기존 방식으로 테스트 진행")
                langgraph_available = False
            
            # Langfuse 모니터링 상태 확인
            langfuse_monitor = get_langfuse_monitor()
            if langfuse_monitor.is_enabled():
                logger.success("Langfuse 모니터링이 활성화되어 있습니다.")
            else:
                logger.warning("Langfuse 모니터링이 비활성화되어 있습니다.")
                logger.info("환경 변수 LANGFUSE_PUBLIC_KEY와 LANGFUSE_SECRET_KEY를 설정하세요.")
            
            # Enhanced Chat Service 초기화
            chat_service = EnhancedChatService(config)
            logger.success("Enhanced Chat Service 초기화 성공")
            logger.debug(f"Chat service type: {type(chat_service)}")
            logger.debug(f"Chat service has process_message: {hasattr(chat_service, 'process_message')}")
            
            # 테스트 질문 생성
            test_questions = generate_comprehensive_test_questions()
            logger.info(f"📝 총 {len(test_questions)}개의 종합 테스트 질문 생성")
            
            # 우선순위별 분류
            high_priority = [q for q in test_questions if q["priority"] == "high"]
            medium_priority = [q for q in test_questions if q["priority"] == "medium"]
            low_priority = [q for q in test_questions if q["priority"] == "low"]
            
            logger.info(f"📊 우선순위별 질문 수: High({len(high_priority)}), Medium({len(medium_priority)}), Low({len(low_priority)})")
        
        with SectionLogger("🔄 테스트 실행", 0):
            # 테스트 실행
            results = []
            start_time = time.time()
            
            # 진행률 추적기와 실시간 통계 초기화
            progress = ProgressTracker(len(test_questions), "테스트 진행")
            stats = RealTimeStats()
            
            for i, test_case in enumerate(test_questions, 1):
                question = test_case["question"]
                category = test_case["category"]
                expected_type = test_case["expected_type"]
                priority = test_case["priority"]
                expected_style = test_case.get("expected_style", "unknown")
                
                with SectionLogger(f"질문 {i}: {question[:30]}...", 1):
                    logger.info(f"카테고리: {category} | 예상유형: {expected_type} | 우선순위: {priority} | 예상스타일: {expected_style}")
                    
                    # Langfuse 트레이스 생성
                    trace = None
                    if langfuse_monitor.is_enabled():
                        trace = langfuse_monitor.create_trace(
                            name=f"comprehensive_test_question_{i}",
                            user_id=f"comprehensive_test_user_{i}",
                            session_id=f"comprehensive_test_session_{i}"
                        )
                        if trace:
                            logger.debug(f"🔍 Langfuse 트레이스 생성됨: {trace}")
                    
                    try:
                        # 메시지 처리
                        result = await chat_service.process_message(
                            message=question,
                            user_id=f"comprehensive_test_user_{i}",
                            session_id=f"comprehensive_test_session_{i}"
                        )
                        
                        # 결과 분석
                        response = result.get('response', 'N/A')
                        confidence = result.get('confidence', 0.0)
                        processing_time = result.get('processing_time', 0.0)
                        is_restricted = result.get('restricted', False)
                        generation_method = result.get('generation_method', 'unknown')
                        sources = result.get('sources', [])
                        
                        # 간결한 결과 표시
                        logger.success(f"신뢰도: {confidence:.2f} | 시간: {processing_time:.3f}초 | 제한: {is_restricted}")
                        logger.info(f"생성 방법: {generation_method} | 검색 결과: {len(sources)}개")
                        
                        # 응답 요약 표시
                        response_summary = summarize_response(response, 150)
                        logger.info(f"응답 요약: {response_summary}")
                        
                        # Langfuse 로깅
                        if langfuse_monitor.is_enabled() and trace:
                            try:
                                langfuse_monitor.log_generation(
                                    trace_id=trace.id if hasattr(trace, 'id') else str(trace),
                                    name="comprehensive_test_response",
                                    input_data={
                                        "question": question,
                                        "category": category,
                                        "expected_type": expected_type,
                                        "priority": priority,
                                        "expected_style": expected_style
                                    },
                                    output_data={
                                        "response": response,
                                        "confidence": confidence,
                                        "processing_time": processing_time,
                                        "is_restricted": is_restricted,
                                        "generation_method": generation_method,
                                        "sources_count": len(sources)
                                    },
                                    metadata={
                                        "test_case_id": i,
                                        "user_id": f"comprehensive_test_user_{i}",
                                        "session_id": f"comprehensive_test_session_{i}",
                                        "test_type": "comprehensive_quality"
                                    }
                                )
                                logger.debug("🔍 Langfuse 로깅 완료")
                            except Exception as e:
                                logger.warning(f"⚠️ Langfuse 로깅 실패: {e}")
                        
                        # 결과 저장
                        test_result = {
                            'test_case': test_case,
                            'result': result,
                            'success': True,
                            'processing_time': processing_time,
                            'confidence': confidence,
                            'is_restricted': is_restricted,
                            'generation_method': generation_method,
                            'sources_count': len(sources)
                        }
                        results.append(test_result)
                        stats.update(test_result)
                        
                    except Exception as e:
                        logger.error(f"❌ 질문 {i} 처리 실패: {e}")
                        
                        # Langfuse 오류 로깅
                        if langfuse_monitor.is_enabled() and trace:
                            try:
                                langfuse_monitor.log_event(
                                    trace_id=trace.id if hasattr(trace, 'id') else str(trace),
                                    name="comprehensive_test_error",
                                    input_data={
                                        "question": question,
                                        "category": category,
                                        "expected_type": expected_type,
                                        "priority": priority,
                                        "expected_style": expected_style
                                    },
                                    output_data={
                                        "error": str(e),
                                        "error_type": type(e).__name__
                                    },
                                    metadata={
                                        "test_case_id": i,
                                        "user_id": f"comprehensive_test_user_{i}",
                                        "session_id": f"comprehensive_test_session_{i}",
                                        "test_type": "comprehensive_quality",
                                        "success": False
                                    }
                                )
                                logger.debug("🔍 Langfuse 오류 로깅 완료")
                            except Exception as langfuse_error:
                                logger.warning(f"⚠️ Langfuse 오류 로깅 실패: {langfuse_error}")
                        
                        error_result = {
                            'test_case': test_case,
                            'result': None,
                            'success': False,
                            'error': str(e)
                        }
                        results.append(error_result)
                        stats.update(error_result)
                    
                    # 진행률 업데이트
                    progress.update(f"질문 {i} 완료")
            
            total_time = time.time() - start_time
            logger.info(f"총 실행 시간: {total_time:.2f}초")
        
        with SectionLogger("📊 결과 분석", 0):
            # 테스트 결과 요약 테이블
            print_results_table(results, "테스트 결과 요약")
            
            # 기본 통계
            total_tests = len(results)
            successful_tests = sum(1 for r in results if r['success'])
            failed_tests = total_tests - successful_tests
            restricted_tests = sum(1 for r in results if r.get('is_restricted', False))
            
            logger.info(f"총 테스트: {total_tests}")
            logger.success(f"성공한 테스트: {successful_tests}")
            logger.error(f"실패한 테스트: {failed_tests}")
            logger.warning(f"제한된 테스트: {restricted_tests}")
            
            if successful_tests > 0:
                avg_confidence = sum(r.get('confidence', 0) for r in results if r['success']) / successful_tests
                avg_processing_time = sum(r.get('processing_time', 0) for r in results if r['success']) / successful_tests
                
                logger.info(f"평균 신뢰도: {avg_confidence:.2f}")
                logger.info(f"평균 처리 시간: {avg_processing_time:.3f}초")
            
            # LangGraph 통합 분석
            logger.info("🔄 LangGraph 통합 분석")
            langgraph_enabled_count = sum(1 for r in results if r.get('result', {}).get('langgraph_enabled', False))
            langgraph_usage_rate = (langgraph_enabled_count / total_tests) * 100 if total_tests > 0 else 0
            
            # 워크플로우 단계 통계
            workflow_steps_counts = [len(r.get('result', {}).get('workflow_steps', [])) for r in results if r['success']]
            avg_workflow_steps = sum(workflow_steps_counts) / len(workflow_steps_counts) if workflow_steps_counts else 0
            
            logger.info(f"LangGraph 사용: {langgraph_enabled_count}/{total_tests} ({langgraph_usage_rate:.1f}%)")
            logger.info(f"평균 워크플로우 단계: {avg_workflow_steps:.1f}")
            
            # 생성 방법별 분석
            logger.info("🔧 생성 방법별 분석")
            generation_methods = {}
            for result in results:
                if result['success']:
                    method = result.get('generation_method', 'unknown')
                    if method not in generation_methods:
                        generation_methods[method] = {'count': 0, 'total_confidence': 0, 'avg_confidence': 0, 'avg_time': 0}
                    generation_methods[method]['count'] += 1
                    generation_methods[method]['total_confidence'] += result.get('confidence', 0)
                    generation_methods[method]['avg_time'] += result.get('processing_time', 0)
            
            for method, stats in generation_methods.items():
                stats['avg_confidence'] = stats['total_confidence'] / stats['count']
                stats['avg_time'] = stats['avg_time'] / stats['count']
                logger.info(f"{method}: {stats['count']}개, 평균 신뢰도: {stats['avg_confidence']:.2f}, 평균 시간: {stats['avg_time']:.3f}초")
            
            # 우선순위별 분석
            logger.info("📈 우선순위별 분석")
            priority_stats = {}
            for result in results:
                if result['success']:
                    priority = result['test_case']['priority']
                    if priority not in priority_stats:
                        priority_stats[priority] = {'total': 0, 'success': 0, 'avg_conf': 0, 'avg_time': 0}
                    
                    priority_stats[priority]['total'] += 1
                    priority_stats[priority]['success'] += 1
                    priority_stats[priority]['avg_conf'] += result.get('confidence', 0)
                    priority_stats[priority]['avg_time'] += result.get('processing_time', 0)
            
            for priority, stats in priority_stats.items():
                success_rate = (stats['success'] / stats['total']) * 100
                avg_conf = stats['avg_conf'] / stats['success'] if stats['success'] > 0 else 0
                avg_time = stats['avg_time'] / stats['success'] if stats['success'] > 0 else 0
                
                logger.info(f"{priority.upper()}: {stats['success']}/{stats['total']} 성공 ({success_rate:.1f}%), 평균신뢰도 {avg_conf:.2f}, 평균시간 {avg_time:.3f}초")
            
            # 카테고리별 분석
            logger.info("📊 카테고리별 분석")
            categories = {}
            for result in results:
                if result['success']:
                    category = result['test_case']['category']
                    if category not in categories:
                        categories[category] = {'total': 0, 'success': 0, 'restricted': 0, 'avg_time': 0, 'avg_conf': 0}
                    
                    categories[category]['total'] += 1
                    categories[category]['success'] += 1
                    categories[category]['avg_time'] += result.get('processing_time', 0)
                    categories[category]['avg_conf'] += result.get('confidence', 0)
                    if result.get('is_restricted', False):
                        categories[category]['restricted'] += 1
            
            for category, stats in categories.items():
                success_rate = (stats['success'] / stats['total']) * 100
                restriction_rate = (stats['restricted'] / stats['total']) * 100
                avg_time = stats['avg_time'] / stats['success'] if stats['success'] > 0 else 0
                avg_conf = stats['avg_conf'] / stats['success'] if stats['success'] > 0 else 0
                
                logger.info(f"{category}: {stats['success']}/{stats['total']} 성공 ({success_rate:.1f}%), 제한 {restriction_rate:.1f}%, 평균시간 {avg_time:.3f}초, 평균신뢰도 {avg_conf:.2f}")
            
            # 지능형 스타일 시스템 분석
            logger.info("🎨 지능형 스타일 시스템 분석")
            intelligent_results = [r for r in results if r['success'] and 'intelligent_style' in r.get('generation_method', '')]
            fallback_results = [r for r in results if r['success'] and 'fallback' in r.get('generation_method', '')]
            
            if intelligent_results:
                intelligent_avg_conf = sum(r.get('confidence', 0) for r in intelligent_results) / len(intelligent_results)
                intelligent_avg_time = sum(r.get('processing_time', 0) for r in intelligent_results) / len(intelligent_results)
                logger.info(f"지능형 스타일 시스템: {len(intelligent_results)}개, 평균 신뢰도: {intelligent_avg_conf:.2f}, 평균 시간: {intelligent_avg_time:.3f}초")
            
            if fallback_results:
                fallback_avg_conf = sum(r.get('confidence', 0) for r in fallback_results) / len(fallback_results)
                fallback_avg_time = sum(r.get('processing_time', 0) for r in fallback_results) / len(fallback_results)
                logger.info(f"폴백 시스템: {len(fallback_results)}개, 평균 신뢰도: {fallback_avg_conf:.2f}, 평균 시간: {fallback_avg_time:.3f}초")
            
            # 스타일별 분석
            logger.info("🎭 스타일별 분석")
            style_stats = {}
            for result in results:
                if result['success']:
                    expected_style = result['test_case'].get('expected_style', 'unknown')
                    generation_method = result.get('generation_method', 'unknown')
                    
                    if expected_style not in style_stats:
                        style_stats[expected_style] = {'count': 0, 'avg_conf': 0, 'avg_time': 0, 'intelligent_count': 0}
                    
                    style_stats[expected_style]['count'] += 1
                    style_stats[expected_style]['avg_conf'] += result.get('confidence', 0)
                    style_stats[expected_style]['avg_time'] += result.get('processing_time', 0)
                    
                    if 'intelligent_style' in generation_method:
                        style_stats[expected_style]['intelligent_count'] += 1
            
            for style, stats in style_stats.items():
                avg_conf = stats['avg_conf'] / stats['count'] if stats['count'] > 0 else 0
                avg_time = stats['avg_time'] / stats['count'] if stats['count'] > 0 else 0
                intelligent_rate = (stats['intelligent_count'] / stats['count']) * 100 if stats['count'] > 0 else 0
                
                logger.info(f"{style}: {stats['count']}개, 평균 신뢰도: {avg_conf:.2f}, 평균 시간: {avg_time:.3f}초, 지능형 적용률: {intelligent_rate:.1f}%")
            
            # Langfuse 모니터링 결과 분석
            if langfuse_monitor.is_enabled():
                logger.info("🔍 Langfuse 모니터링 결과 분석")
                
                # Langfuse 데이터 플러시
                try:
                    langfuse_monitor.flush()
                    logger.success("✅ Langfuse 데이터 플러시 완료")
                except Exception as e:
                    logger.warning(f"⚠️ Langfuse 데이터 플러시 실패: {e}")
                
                # 모니터링 통계
                langfuse_traces = sum(1 for r in results if r.get('success', False))
                langfuse_errors = sum(1 for r in results if not r.get('success', True))
                
                logger.info(f"Langfuse 트레이스 생성: {langfuse_traces}개")
                logger.info(f"Langfuse 오류 로깅: {langfuse_errors}개")
                logger.info(f"총 모니터링 이벤트: {langfuse_traces + langfuse_errors}개")
                
                if langfuse_traces > 0:
                    logger.info("📊 Langfuse 대시보드에서 상세한 분석을 확인하세요:")
                    logger.info("   - 트레이스 실행 시간 분석")
                    logger.info("   - 응답 품질 메트릭")
                    logger.info("   - 오류 패턴 분석")
                    logger.info("   - 사용자별 성능 통계")
            else:
                logger.warning("⚠️ Langfuse 모니터링이 비활성화되어 있어 상세 분석이 불가능합니다.")
                logger.info("환경 변수를 설정하여 Langfuse 모니터링을 활성화하세요.")
            
            logger.success("✅ 종합 답변 품질 테스트 완료!")
            
            return results
            
    except Exception as e:
        logger.error(f"❌ 종합 답변 품질 테스트 실패: {e}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        return []


if __name__ == "__main__":
    logger.success("🚀 Final Comprehensive Answer Quality Test with Langfuse Monitoring")
    logger.info("=" * 80)
    
    # Langfuse 모니터링 상태 사전 확인
    try:
        langfuse_monitor = get_langfuse_monitor()
        if langfuse_monitor.is_enabled():
            logger.success("✅ Langfuse 모니터링이 활성화되어 있습니다.")
            logger.info("📊 테스트 결과는 Langfuse 대시보드에서 확인할 수 있습니다.")
        else:
            logger.warning("⚠️ Langfuse 모니터링이 비활성화되어 있습니다.")
            logger.info("💡 Langfuse 모니터링을 활성화하려면 환경 변수를 설정하세요:")
            logger.info("   - LANGFUSE_PUBLIC_KEY")
            logger.info("   - LANGFUSE_SECRET_KEY")
    except Exception as e:
        logger.warning(f"⚠️ Langfuse 모니터 초기화 실패: {e}")
    
    logger.info("\n" + "=" * 80)
    
    # 종합 테스트 실행
    results = asyncio.run(test_comprehensive_answer_quality())
    
    logger.success("\n🎉 최종 종합 답변 품질 테스트 완료!")
    logger.info("=" * 80)
    
    # 최종 요약
    if results:
        successful_tests = sum(1 for r in results if r.get('success', False))
        total_tests = len(results)
        logger.info(f"📊 테스트 요약: {successful_tests}/{total_tests} 성공")
        
        try:
            langfuse_monitor = get_langfuse_monitor()
            if langfuse_monitor.is_enabled():
                logger.info("🔍 Langfuse 대시보드에서 상세한 분석 결과를 확인하세요!")
        except:
            pass