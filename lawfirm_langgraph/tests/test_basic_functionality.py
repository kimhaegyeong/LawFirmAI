# -*- coding: utf-8 -*-
"""
기본 기능 테스트
기존 기능과 Agentic 모드가 정상 동작하는지 확인
"""

import asyncio
import logging
import os
import sys
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


async def test_basic_workflow():
    """기본 워크플로우 테스트 (Agentic 모드 OFF)"""
    logger.info("=" * 80)
    logger.info("Test: 기본 워크플로우 테스트 (Agentic 모드 OFF)")
    logger.info("=" * 80)
    
    try:
        # 설정 로드 (Agentic 모드 OFF)
        original_value = os.environ.get("USE_AGENTIC_MODE")
        os.environ["USE_AGENTIC_MODE"] = "false"
        
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
        from lawfirm_langgraph.langgraph_core.utils.state_definitions import create_initial_legal_state
        
        config = LangGraphConfig.from_env()
        logger.info(f"Config loaded: use_agentic_mode={config.use_agentic_mode}")
        
        # 워크플로우 서비스 초기화
        workflow_service = LangGraphWorkflowService(config)
        logger.info("WorkflowService initialized successfully")
        
        # 초기 상태 생성
        initial_state = create_initial_legal_state("테스트 질문입니다", "test_session")
        logger.info(f"Initial state created: query={initial_state.get('query', 'N/A')}")
        
        logger.info("✅ 기본 워크플로우 테스트 성공")
        
        # 환경 변수 복원
        if original_value:
            os.environ["USE_AGENTIC_MODE"] = original_value
        elif "USE_AGENTIC_MODE" in os.environ:
            del os.environ["USE_AGENTIC_MODE"]
        
        return True
    except Exception as e:
        logger.error(f"❌ 기본 워크플로우 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 환경 변수 복원
        if original_value:
            os.environ["USE_AGENTIC_MODE"] = original_value
        elif "USE_AGENTIC_MODE" in os.environ:
            del os.environ["USE_AGENTIC_MODE"]
        
        return False


async def test_agentic_mode():
    """Agentic 모드 테스트 (설정만 확인)"""
    logger.info("=" * 80)
    logger.info("Test: Agentic 모드 설정 테스트")
    logger.info("=" * 80)
    
    try:
        # 설정 로드 (Agentic 모드 ON)
        original_value = os.environ.get("USE_AGENTIC_MODE")
        os.environ["USE_AGENTIC_MODE"] = "true"
        
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
        
        config = LangGraphConfig.from_env()
        logger.info(f"Config loaded: use_agentic_mode={config.use_agentic_mode}")
        
        if not config.use_agentic_mode:
            logger.warning("⚠️ Agentic mode 설정이 false입니다. 환경 변수 확인 필요")
            return False
        
        # 워크플로우 서비스 초기화 (Tool 시스템 로드 확인)
        workflow_service = LangGraphWorkflowService(config)
        logger.info("WorkflowService initialized with Agentic mode")
        
        # 워크플로우 객체 확인
        if hasattr(workflow_service, 'workflow') and hasattr(workflow_service.workflow, 'legal_tools'):
            tools_count = len(workflow_service.workflow.legal_tools) if workflow_service.workflow.legal_tools else 0
            logger.info(f"Tools loaded: {tools_count} tools available")
        else:
            logger.warning("⚠️ Tool 시스템 정보를 확인할 수 없습니다")
        
        logger.info("✅ Agentic 모드 설정 테스트 성공")
        
        # 환경 변수 복원
        if original_value:
            os.environ["USE_AGENTIC_MODE"] = original_value
        elif "USE_AGENTIC_MODE" in os.environ:
            del os.environ["USE_AGENTIC_MODE"]
        
        return True
    except Exception as e:
        logger.error(f"❌ Agentic 모드 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 환경 변수 복원
        if original_value:
            os.environ["USE_AGENTIC_MODE"] = original_value
        elif "USE_AGENTIC_MODE" in os.environ:
            del os.environ["USE_AGENTIC_MODE"]
        
        return False


async def run_all_tests():
    """모든 테스트 실행"""
    logger.info("\n" + "=" * 80)
    logger.info("기능 테스트 시작")
    logger.info("=" * 80 + "\n")
    
    results = []
    
    # 테스트 실행
    results.append(("Basic Workflow", await test_basic_workflow()))
    results.append(("Agentic Mode", await test_agentic_mode()))
    
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
    logger.info("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

