# -*- coding: utf-8 -*-
"""
전체 워크플로우 통합 테스트
실제 질문 처리 및 Agentic 모드 검증
"""

import asyncio
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정 (asyncio 호환성을 위해 print 사용)
# logging과 asyncio의 버퍼 분리 문제로 인해 안전한 출력 함수 사용
def safe_print(msg, flush=True):
    """안전한 출력 함수 (로깅 버퍼 이슈 방지)"""
    try:
        print(msg, flush=flush)
    except:
        pass

# 로거는 내부 모듈 로깅용으로만 사용 (출력은 하지 않음)
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)  # 테스트 중 로거 출력 비활성화


# 환경 변수 컨텍스트 매니저
@contextmanager
def env_context(**env_vars):
    """환경 변수 컨텍스트 매니저 (자동 복원)"""
    original = {}
    try:
        for key, value in env_vars.items():
            original[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def test_with_timeout(coro, timeout=120):
    """타임아웃이 있는 테스트 실행"""
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        safe_print(f"❌ 테스트가 {timeout}초 내에 완료되지 않았습니다")
        return False
    except Exception as e:
        safe_print(f"❌ 테스트 실행 중 오류: {e}")
        return False


def validate_workflow_result(result):
    """워크플로우 결과 검증"""
    if not result:
        return False, "결과가 None입니다"
    
    # 필수 필드 확인
    if 'answer' not in result and 'error' not in result:
        return False, "필수 필드 'answer' 또는 'error'가 없습니다"
    
    # 답변 길이 검증 (answer가 있는 경우)
    if 'answer' in result:
        answer = result.get('answer', '')
        if len(answer) < 10:
            return False, f"답변이 너무 짧습니다 ({len(answer)} 자)"
    
    # 신뢰도 검증 (있는 경우)
    if 'confidence' in result:
        confidence = result.get('confidence', 0.0)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            return False, f"신뢰도가 유효하지 않습니다: {confidence}"
    
    return True, "검증 통과"


async def test_basic_workflow_execution():
    """기본 워크플로우 실행 테스트"""
    safe_print("=" * 80)
    safe_print("Test: 기본 워크플로우 실행 테스트")
    safe_print("=" * 80)
    
    with env_context(USE_AGENTIC_MODE="false", TESTING="true"):
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
            from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
            from lawfirm_langgraph.langgraph_core.utils.state_definitions import create_initial_legal_state
            
            # 설정 로드
            config = LangGraphConfig.from_env()
            safe_print(f"Config: use_agentic_mode={config.use_agentic_mode}")
            
            # 워크플로우 서비스 초기화
            workflow_service = LangGraphWorkflowService(config)
            safe_print("✅ WorkflowService 초기화 완료")
            
            # 테스트 질문
            test_query = "계약서 작성 시 주의할 사항은 무엇인가요?"
            session_id = "test_session_001"
            
            safe_print(f"테스트 질문: {test_query}")
            
            # 초기 상태 생성
            initial_state = create_initial_legal_state(test_query, session_id)
            safe_print(f"✅ Initial state 생성: query={initial_state.get('query', 'N/A')}")
            
            # 워크플로우 실행 (타임아웃 포함)
            safe_print("워크플로우 실행 시작... (최대 120초)")
            start_time = time.time()
            
            async def execute_workflow():
                return await workflow_service.process_query(
                    query=test_query,
                    session_id=session_id
                )
            
            result = await test_with_timeout(execute_workflow(), timeout=120)
            elapsed = time.time() - start_time
            
            if result:
                # 결과 검증
                is_valid, message = validate_workflow_result(result)
                
                if is_valid:
                    answer = result.get('answer', '')
                    sources = result.get('sources', [])
                    confidence = result.get('confidence', 0.0)
                    
                    safe_print(f"✅ 워크플로우 실행 성공 ({elapsed:.2f}초)")
                    safe_print(f"   답변 길이: {len(answer)} 자")
                    safe_print(f"   소스 개수: {len(sources)} 개")
                    safe_print(f"   신뢰도: {confidence:.2f}")
                    safe_print("✅ 기본 워크플로우 실행 테스트 성공")
                    return True
                else:
                    safe_print(f"⚠️ 결과 검증 실패: {message}")
                    return False
            else:
                safe_print(f"⚠️ 워크플로우 실행 실패 (경과 시간: {elapsed:.2f}초)")
                return False
                
        except Exception as e:
            safe_print(f"❌ 기본 워크플로우 실행 테스트 실패: {e}")
            safe_print(traceback.format_exc())
            return False


async def test_agentic_mode_execution():
    """Agentic 모드 실행 테스트 (Tool 시스템 확인 및 실제 실행)"""
    safe_print("=" * 80)
    safe_print("Test: Agentic 모드 실행 테스트")
    safe_print("=" * 80)
    
    with env_context(USE_AGENTIC_MODE="true", TESTING="true"):
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
            from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
            
            # 설정 로드
            config = LangGraphConfig.from_env()
            safe_print(f"Config: use_agentic_mode={config.use_agentic_mode}")
            
            if not config.use_agentic_mode:
                safe_print("⚠️ Agentic mode가 활성화되지 않았습니다")
                return False
            
            # 워크플로우 서비스 초기화 (성능 측정 포함)
            init_start = time.time()
            workflow_service = LangGraphWorkflowService(config)
            init_time = time.time() - init_start
            
            safe_print(f"✅ WorkflowService 초기화 완료 (Agentic 모드, {init_time:.2f}초)")
            
            # Tool 시스템 확인
            if hasattr(workflow_service.legal_workflow, 'legal_tools'):
                tools = workflow_service.legal_workflow.legal_tools
                tools_count = len(tools) if tools else 0
                safe_print(f"✅ Tool 시스템 로드: {tools_count}개 Tool")
                
                if tools_count > 0:
                    for tool in tools:
                        safe_print(f"   - {tool.name}")
                    
                    # 실제 Agentic 모드 실행 테스트 (간단한 질의)
                    safe_print("\n[Agentic 모드 실제 실행 테스트]")
                    test_query = "계약서 위반 시 손해배상 책임에 대한 판례를 찾아주세요."
                    safe_print(f"테스트 질의: {test_query}")
                    
                    exec_start = time.time()
                    try:
                        # 타임아웃이 있는 실행
                        result = await test_with_timeout(
                            workflow_service.process_query(
                                query=test_query,
                                session_id="agentic_test_session"
                            ),
                            timeout=180  # Agentic 모드는 더 오래 걸릴 수 있음
                        )
                        exec_time = time.time() - exec_start
                        
                        if result:
                            is_valid, message = validate_workflow_result(result)
                            if is_valid:
                                answer_len = len(result.get('answer', ''))
                                confidence = result.get('confidence', 0.0)
                                safe_print(f"✅ Agentic 모드 실행 성공 ({exec_time:.2f}초)")
                                safe_print(f"   답변 길이: {answer_len} 자")
                                safe_print(f"   신뢰도: {confidence:.2f}")
                                
                                # Tool 사용 여부 확인
                                if 'agentic_tools_used' in result:
                                    tools_used = result.get('agentic_tools_used', [])
                                    safe_print(f"   사용된 Tool: {tools_used}")
                                
                                safe_print("✅ Agentic 모드 실행 테스트 성공 (Tool 시스템 활성 및 실행 검증)")
                                return True
                            else:
                                safe_print(f"⚠️ 결과 검증 실패: {message}")
                                return False
                        else:
                            safe_print(f"⚠️ 실행 결과가 None입니다 (경과 시간: {exec_time:.2f}초)")
                            return False
                    except asyncio.TimeoutError:
                        safe_print(f"❌ Agentic 모드 실행이 180초 내에 완료되지 않았습니다")
                        return False
                    except Exception as e:
                        safe_print(f"❌ Agentic 모드 실행 중 오류: {e}")
                        return False
                else:
                    safe_print("⚠️ Tool 시스템이 비어있습니다")
                    return False
            else:
                safe_print("⚠️ legal_tools 속성을 찾을 수 없습니다")
                return False
        
        except Exception as e:
            safe_print(f"❌ Agentic 모드 실행 테스트 실패: {e}")
            safe_print(traceback.format_exc())
            return False


async def test_component_initialization():
    """각 컴포넌트 초기화 테스트 (성능 측정 포함, 로거 격리)"""
    safe_print("=" * 80)
    safe_print("Test: 컴포넌트 초기화 테스트")
    safe_print("=" * 80)
    
    # 테스트 환경 변수 설정 (로깅 스트림 분리 오류 방지)
    os.environ["TESTING"] = "true"
    
    # 로거 격리: 내부 모듈 로거 레벨 조정 (스트림 분리 오류 방지)
    import logging
    original_levels = {}
    critical_modules = [
        'lawfirm_langgraph.langgraph_core.services.legal_workflow_enhanced',
        'lawfirm_langgraph.langgraph_core.services.workflow_service',
        'source.agents.context_builder',
        'source.agents.answer_generator',
        'source.agents.answer_formatter',
        'source.agents.workflow_routes',
        'lawfirm_langgraph.langgraph_core.utils.workflow_routes',
        'lawfirm_langgraph.langgraph_core.utils.workflow_utils'
    ]
    
    try:
        # 내부 모듈 로거 레벨 임시 조정 (CRITICAL로 설정하여 출력 억제)
        # 스트림 핸들러 제거 및 NullHandler 추가로 멀티스레드 환경에서 안전하게 처리
        for module_name in critical_modules:
            module_logger = logging.getLogger(module_name)
            original_levels[module_name] = module_logger.level
            
            # 기존 StreamHandler 제거 (멀티스레드 환경에서 문제 발생)
            handlers_to_remove = []
            for handler in module_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handlers_to_remove.append(handler)
            for handler in handlers_to_remove:
                module_logger.removeHandler(handler)
            
            # CRITICAL 레벨 설정 및 NullHandler 추가
            module_logger.setLevel(logging.CRITICAL)
            if not module_logger.handlers:
                null_handler = logging.NullHandler()
                module_logger.addHandler(null_handler)
                module_logger.propagate = False  # 전파 중단으로 루트 로거 영향 제거
        
        start_time = time.time()
        
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
        from lawfirm_langgraph.langgraph_core.services.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        from lawfirm_langgraph.langgraph_core.utils.state_definitions import create_initial_legal_state, LegalWorkflowState
        from lawfirm_langgraph.langgraph_core.utils.workflow_constants import WorkflowConstants
        from lawfirm_langgraph.langgraph_core.utils.workflow_routes import WorkflowRoutes
        
        # 설정 로드
        config_start = time.time()
        config = LangGraphConfig.from_env()
        config_time = time.time() - config_start
        safe_print(f"✅ Config 로드 완료 ({config_time:.3f}초)")
        
        # 워크플로우 객체 직접 생성
        workflow_start = time.time()
        workflow = EnhancedLegalQuestionWorkflow(config)
        workflow_time = time.time() - workflow_start
        safe_print(f"✅ EnhancedLegalQuestionWorkflow 초기화 완료 ({workflow_time:.2f}초)")
        
        # 워크플로우 서비스 생성
        service_start = time.time()
        service = LangGraphWorkflowService(config)
        service_time = time.time() - service_start
        safe_print(f"✅ LangGraphWorkflowService 초기화 완료 ({service_time:.2f}초)")
        
        # 상태 생성
        state_start = time.time()
        state = create_initial_legal_state("테스트", "test")
        state_time = time.time() - state_start
        safe_print(f"✅ State 생성 완료: {type(state).__name__} ({state_time:.3f}초)")
        
        # Constants 확인
        constants = WorkflowConstants
        safe_print(f"✅ WorkflowConstants 로드 완료")
        
        # Routes 확인
        routes = WorkflowRoutes()
        safe_print(f"✅ WorkflowRoutes 초기화 완료")
        
        total_time = time.time() - start_time
        safe_print(f"\n✅ 모든 컴포넌트 초기화 성공 (총 {total_time:.2f}초)")
        
        # 성능 기준 확인
        if workflow_time > 10.0:
            safe_print(f"⚠️ 워크플로우 초기화 시간이 길습니다: {workflow_time:.2f}초 (기준: 10초)")
        if service_time > 10.0:
            safe_print(f"⚠️ 서비스 초기화 시간이 길습니다: {service_time:.2f}초 (기준: 10초)")
        
        return True
        
    except Exception as e:
        safe_print(f"❌ 컴포넌트 초기화 테스트 실패: {e}")
        safe_print(traceback.format_exc())
        return False
    finally:
        # 로거 레벨 복원
        for module_name, level in original_levels.items():
            logging.getLogger(module_name).setLevel(level)
        
        # 테스트 환경 변수 제거
        os.environ.pop("TESTING", None)


async def run_all_tests():
    """모든 테스트 실행 (성능 벤치마크 포함)"""
    safe_print("\n" + "=" * 80)
    safe_print("전체 워크플로우 통합 테스트 시작")
    safe_print("=" * 80 + "\n")
    
    test_start_time = time.time()
    results = []
    performance_metrics = {}
    
    # 컴포넌트 초기화 테스트
    test_time = time.time()
    result = await test_component_initialization()
    test_duration = time.time() - test_time
    results.append(("Component Initialization", result))
    performance_metrics["Component Initialization"] = test_duration
    
    # 기본 워크플로우 실행 테스트
    test_time = time.time()
    result = await test_basic_workflow_execution()
    test_duration = time.time() - test_time
    results.append(("Basic Workflow Execution", result))
    performance_metrics["Basic Workflow Execution"] = test_duration
    
    # Agentic 모드 실행 테스트
    test_time = time.time()
    result = await test_agentic_mode_execution()
    test_duration = time.time() - test_time
    results.append(("Agentic Mode Execution", result))
    performance_metrics["Agentic Mode Execution"] = test_duration
    
    total_test_time = time.time() - test_start_time
    
    # 결과 요약
    safe_print("\n" + "=" * 80)
    safe_print("테스트 결과 요약")
    safe_print("=" * 80)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        duration = performance_metrics.get(test_name, 0.0)
        safe_print(f"{status} - {test_name} ({duration:.2f}초)")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    failed = total - passed
    
    safe_print("=" * 80)
    safe_print(f"총 테스트: {total}개 | 통과: {passed}개 | 실패: {failed}개")
    safe_print(f"총 테스트 시간: {total_test_time:.2f}초")
    safe_print("=" * 80)
    
    # 성능 벤치마크 평가
    safe_print("\n[성능 벤치마크 평가]")
    safe_print("-" * 80)
    
    benchmarks = {
        "Component Initialization": 10.0,  # 초기화는 10초 이내
        "Basic Workflow Execution": 120.0,  # 기본 실행은 120초 이내
        "Agentic Mode Execution": 180.0  # Agentic 실행은 180초 이내
    }
    
    all_within_benchmark = True
    for test_name, benchmark_time in benchmarks.items():
        actual_time = performance_metrics.get(test_name, 0.0)
        if actual_time > benchmark_time:
            safe_print(f"⚠️ {test_name}: {actual_time:.2f}초 (기준: {benchmark_time}초) - 기준 초과")
            all_within_benchmark = False
        else:
            safe_print(f"✅ {test_name}: {actual_time:.2f}초 (기준: {benchmark_time}초) - 기준 만족")
    
    if all_within_benchmark:
        safe_print("\n✅ 모든 테스트가 성능 기준을 만족합니다")
    else:
        safe_print("\n⚠️ 일부 테스트가 성능 기준을 초과했습니다")
    
    safe_print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

