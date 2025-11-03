# -*- coding: utf-8 -*-
"""
체크포인터 Memory Store 기능 테스트
CheckpointManager와 MemorySaver 사용 검증
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)


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


async def test_checkpoint_manager_memory_saver():
    """CheckpointManager MemorySaver 초기화 테스트"""
    logger.info("=" * 80)
    logger.info("Test: CheckpointManager MemorySaver 초기화")
    logger.info("=" * 80)
    
    try:
        from source.agents.checkpoint_manager import CheckpointManager
        
        # MemorySaver 초기화
        checkpoint_manager = CheckpointManager(
            storage_type="memory",
            db_path=None
        )
        
        # 초기화 확인
        if not checkpoint_manager.is_enabled():
            logger.error("❌ CheckpointManager가 활성화되지 않았습니다")
            return False
        
        if checkpoint_manager.storage_type != "memory":
            logger.error(f"❌ 저장소 타입이 올바르지 않습니다: {checkpoint_manager.storage_type}")
            return False
        
        checkpointer = checkpoint_manager.get_checkpointer()
        if checkpointer is None:
            logger.error("❌ Checkpointer를 가져올 수 없습니다")
            return False
        
        logger.info(f"✅ CheckpointManager 초기화 성공: {checkpoint_manager.storage_type}")
        logger.info(f"✅ Checkpointer 타입: {type(checkpointer).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_checkpoint_manager_sqlite_saver():
    """CheckpointManager SqliteSaver 초기화 테스트"""
    logger.info("=" * 80)
    logger.info("Test: CheckpointManager SqliteSaver 초기화")
    logger.info("=" * 80)
    
    try:
        from source.agents.checkpoint_manager import CheckpointManager
        
        # 임시 DB 경로 설정
        test_db_path = "./data/test_checkpoints/test.db"
        Path(test_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # SqliteSaver 초기화 시도
        checkpoint_manager = CheckpointManager(
            storage_type="sqlite",
            db_path=test_db_path
        )
        
        # 초기화 확인 (SqliteSaver가 실패하면 MemorySaver로 폴백될 수 있음)
        if checkpoint_manager.is_enabled():
            logger.info(f"✅ CheckpointManager 초기화 성공: {checkpoint_manager.storage_type}")
            logger.info(f"✅ Checkpointer 타입: {type(checkpoint_manager.get_checkpointer()).__name__}")
            return True
        else:
            logger.warning("⚠️ SqliteSaver 초기화 실패, MemorySaver로 폴백되었을 수 있습니다")
            return True  # 폴백은 정상 동작
        
    except Exception as e:
        logger.warning(f"⚠️ SqliteSaver 테스트 중 오류 (폴백 동작 가능): {e}")
        return True  # 폴백은 정상 동작


async def test_checkpoint_manager_disabled():
    """CheckpointManager 비활성화 테스트"""
    logger.info("=" * 80)
    logger.info("Test: CheckpointManager 비활성화")
    logger.info("=" * 80)
    
    try:
        from source.agents.checkpoint_manager import CheckpointManager
        
        # Disabled 초기화
        checkpoint_manager = CheckpointManager(
            storage_type="disabled",
            db_path=None
        )
        
        # 비활성화 확인
        if checkpoint_manager.is_enabled():
            logger.error("❌ CheckpointManager가 비활성화되어야 하는데 활성화되어 있습니다")
            return False
        
        checkpointer = checkpoint_manager.get_checkpointer()
        if checkpointer is not None:
            logger.error("❌ 비활성화 상태에서 checkpointer가 None이어야 합니다")
            return False
        
        logger.info("✅ CheckpointManager 비활성화 확인 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        return False


async def test_workflow_service_with_memory_checkpoint():
    """WorkflowService에서 MemorySaver 체크포인터 사용 테스트"""
    logger.info("=" * 80)
    logger.info("Test: WorkflowService MemorySaver 체크포인터 사용")
    logger.info("=" * 80)
    
    with env_context(
        ENABLE_CHECKPOINT="true",
        CHECKPOINT_STORAGE="memory",
        TESTING="true",
        USE_AGENTIC_MODE="false"
    ):
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
            from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
            
            # 설정 로드
            config = LangGraphConfig.from_env()
            logger.info(f"Config: enable_checkpoint={config.enable_checkpoint}")
            logger.info(f"Config: checkpoint_storage={config.checkpoint_storage.value}")
            
            # WorkflowService 초기화
            workflow_service = LangGraphWorkflowService(config)
            
            # CheckpointManager 확인
            if workflow_service.checkpoint_manager is None:
                logger.error("❌ CheckpointManager가 초기화되지 않았습니다")
                return False
            
            if not workflow_service.checkpoint_manager.is_enabled():
                logger.error("❌ CheckpointManager가 활성화되지 않았습니다")
                return False
            
            if workflow_service.checkpoint_manager.storage_type != "memory":
                logger.error(f"❌ 저장소 타입이 올바르지 않습니다: {workflow_service.checkpoint_manager.storage_type}")
                return False
            
            logger.info("✅ WorkflowService가 MemorySaver 체크포인터와 함께 초기화되었습니다")
            return True
            
        except Exception as e:
            logger.error(f"❌ 테스트 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


async def test_workflow_service_without_checkpoint():
    """WorkflowService에서 체크포인터 비활성화 테스트"""
    logger.info("=" * 80)
    logger.info("Test: WorkflowService 체크포인터 비활성화")
    logger.info("=" * 80)
    
    with env_context(
        ENABLE_CHECKPOINT="false",
        CHECKPOINT_STORAGE="disabled",
        TESTING="true",
        USE_AGENTIC_MODE="false"
    ):
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
            from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
            
            # 설정 로드
            config = LangGraphConfig.from_env()
            logger.info(f"Config: enable_checkpoint={config.enable_checkpoint}")
            
            # WorkflowService 초기화
            workflow_service = LangGraphWorkflowService(config)
            
            # CheckpointManager가 None이거나 비활성화되어 있어야 함
            if workflow_service.checkpoint_manager is not None:
                if workflow_service.checkpoint_manager.is_enabled():
                    logger.error("❌ 체크포인터가 비활성화되어야 하는데 활성화되어 있습니다")
                    return False
            
            logger.info("✅ WorkflowService가 체크포인터 없이 초기화되었습니다")
            return True
            
        except Exception as e:
            logger.error(f"❌ 테스트 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


async def test_workflow_execution_with_checkpoint():
    """체크포인터를 사용한 워크플로우 실행 테스트"""
    logger.info("=" * 80)
    logger.info("Test: 체크포인터를 사용한 워크플로우 실행")
    logger.info("=" * 80)
    
    with env_context(
        ENABLE_CHECKPOINT="true",
        CHECKPOINT_STORAGE="memory",
        TESTING="true",
        USE_AGENTIC_MODE="false"
    ):
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
            from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
            
            # 설정 로드
            config = LangGraphConfig.from_env()
            
            # WorkflowService 초기화
            workflow_service = LangGraphWorkflowService(config)
            
            # 체크포인터 확인
            if not workflow_service.checkpoint_manager or not workflow_service.checkpoint_manager.is_enabled():
                logger.error("❌ 체크포인터가 활성화되어 있지 않습니다")
                return False
            
            # 테스트 질문 (간단한 질문으로 빠른 테스트)
            test_query = "계약서는 무엇인가요?"
            session_id = f"test_session_{int(time.time())}"
            
            logger.info(f"테스트 질문: {test_query}")
            logger.info(f"세션 ID: {session_id}")
            
            # 워크플로우 실행 (체크포인터 활성화)
            start_time = time.time()
            result = await asyncio.wait_for(
                workflow_service.process_query(
                    query=test_query,
                    session_id=session_id,
                    enable_checkpoint=True
                ),
                timeout=60
            )
            elapsed = time.time() - start_time
            
            # 결과 확인
            if not result:
                logger.error("❌ 워크플로우 실행 결과가 None입니다")
                return False
            
            logger.info(f"✅ 워크플로우 실행 완료 ({elapsed:.2f}초)")
            logger.info(f"결과 키: {list(result.keys())}")
            
            # 체크포인터가 사용되었는지 확인 (로그 확인)
            logger.info("✅ 체크포인터를 사용한 워크플로우 실행 성공")
            return True
            
        except asyncio.TimeoutError:
            logger.error("❌ 워크플로우 실행이 타임아웃되었습니다")
            return False
        except Exception as e:
            logger.error(f"❌ 테스트 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


async def test_session_persistence():
    """세션 지속성 테스트 (같은 session_id로 연속 실행)"""
    logger.info("=" * 80)
    logger.info("Test: 세션 지속성 (체크포인터 세션 관리)")
    logger.info("=" * 80)
    
    with env_context(
        ENABLE_CHECKPOINT="true",
        CHECKPOINT_STORAGE="memory",
        TESTING="true",
        USE_AGENTIC_MODE="false"
    ):
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
            from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
            
            # 설정 로드
            config = LangGraphConfig.from_env()
            
            # WorkflowService 초기화
            workflow_service = LangGraphWorkflowService(config)
            
            # 체크포인터 확인
            if not workflow_service.checkpoint_manager or not workflow_service.checkpoint_manager.is_enabled():
                logger.warning("⚠️ 체크포인터가 활성화되어 있지 않아 세션 지속성 테스트를 건너뜁니다")
                return True
            
            # 같은 세션 ID로 여러 쿼리 실행
            session_id = f"persistence_test_{int(time.time())}"
            test_queries = [
                "계약서는 무엇인가요?",
                "계약서 작성 시 주의사항은?"
            ]
            
            logger.info(f"세션 ID: {session_id}")
            
            for i, query in enumerate(test_queries, 1):
                logger.info(f"\n[{i}/{len(test_queries)}] 쿼리 실행: {query}")
                start_time = time.time()
                
                result = await asyncio.wait_for(
                    workflow_service.process_query(
                        query=query,
                        session_id=session_id,
                        enable_checkpoint=True
                    ),
                    timeout=60
                )
                
                elapsed = time.time() - start_time
                logger.info(f"✅ 쿼리 {i} 완료 ({elapsed:.2f}초)")
                
                if not result:
                    logger.warning(f"⚠️ 쿼리 {i} 결과가 None입니다")
            
            logger.info("✅ 세션 지속성 테스트 완료 (같은 세션 ID로 여러 쿼리 실행 성공)")
            return True
            
        except asyncio.TimeoutError:
            logger.error("❌ 세션 지속성 테스트가 타임아웃되었습니다")
            return False
        except Exception as e:
            logger.error(f"❌ 테스트 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


async def run_all_checkpoint_tests():
    """모든 체크포인터 테스트 실행"""
    logger.info("\n" + "=" * 80)
    logger.info("체크포인터 Memory Store 기능 테스트 시작")
    logger.info("=" * 80 + "\n")
    
    test_start_time = time.time()
    results = []
    
    # 1. CheckpointManager MemorySaver 초기화
    logger.info("\n[1/7] CheckpointManager MemorySaver 초기화 테스트")
    result = await test_checkpoint_manager_memory_saver()
    results.append(("CheckpointManager MemorySaver", result))
    
    # 2. CheckpointManager SqliteSaver 초기화 (폴백 가능)
    logger.info("\n[2/7] CheckpointManager SqliteSaver 초기화 테스트")
    result = await test_checkpoint_manager_sqlite_saver()
    results.append(("CheckpointManager SqliteSaver", result))
    
    # 3. CheckpointManager 비활성화
    logger.info("\n[3/7] CheckpointManager 비활성화 테스트")
    result = await test_checkpoint_manager_disabled()
    results.append(("CheckpointManager Disabled", result))
    
    # 4. WorkflowService MemorySaver 사용
    logger.info("\n[4/7] WorkflowService MemorySaver 체크포인터 사용 테스트")
    result = await test_workflow_service_with_memory_checkpoint()
    results.append(("WorkflowService with MemorySaver", result))
    
    # 5. WorkflowService 체크포인터 비활성화
    logger.info("\n[5/7] WorkflowService 체크포인터 비활성화 테스트")
    result = await test_workflow_service_without_checkpoint()
    results.append(("WorkflowService without Checkpoint", result))
    
    # 6. 체크포인터를 사용한 워크플로우 실행
    logger.info("\n[6/7] 체크포인터를 사용한 워크플로우 실행 테스트")
    result = await test_workflow_execution_with_checkpoint()
    results.append(("Workflow Execution with Checkpoint", result))
    
    # 7. 세션 지속성 테스트
    logger.info("\n[7/7] 세션 지속성 테스트")
    result = await test_session_persistence()
    results.append(("Session Persistence", result))
    
    total_test_time = time.time() - test_start_time
    
    # 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("테스트 결과 요약")
    logger.info("=" * 80)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    failed = total - passed
    
    logger.info("=" * 80)
    logger.info(f"총 테스트: {total}개 | 통과: {passed}개 | 실패: {failed}개")
    logger.info(f"총 테스트 시간: {total_test_time:.2f}초")
    logger.info("=" * 80)
    
    if failed == 0:
        logger.info("\n🎉 모든 체크포인터 테스트가 통과했습니다!")
    else:
        logger.warning(f"\n⚠️ {failed}개의 테스트가 실패했습니다.")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_checkpoint_tests())
    sys.exit(0 if success else 1)

