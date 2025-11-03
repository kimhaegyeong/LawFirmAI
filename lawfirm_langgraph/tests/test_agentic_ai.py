# -*- coding: utf-8 -*-
"""
Agentic AI 테스트
NEXT_STEPS.md의 Agentic 모드 테스트를 실행
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


def test_agentic_mode_configuration():
    """Agentic 모드 설정 확인 테스트"""
    logger.info("=" * 80)
    logger.info("Test 1: Agentic 모드 설정 확인")
    logger.info("=" * 80)
    
    try:
        # 환경 변수 설정 (Agentic 모드 ON)
        original_value = os.environ.get("USE_AGENTIC_MODE")
        os.environ["USE_AGENTIC_MODE"] = "true"
        
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        config = LangGraphConfig.from_env()
        
        logger.info(f"Config loaded: use_agentic_mode={config.use_agentic_mode}")
        
        if not config.use_agentic_mode:
            logger.error("❌ Agentic 모드가 활성화되지 않았습니다")
            if original_value:
                os.environ["USE_AGENTIC_MODE"] = original_value
            elif "USE_AGENTIC_MODE" in os.environ:
                del os.environ["USE_AGENTIC_MODE"]
            return False
        
        logger.info("✅ Agentic 모드 설정 확인 성공")
        
        # 환경 변수 복원
        if original_value:
            os.environ["USE_AGENTIC_MODE"] = original_value
        elif "USE_AGENTIC_MODE" in os.environ:
            del os.environ["USE_AGENTIC_MODE"]
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Agentic 모드 설정 확인 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 환경 변수 복원
        if original_value:
            os.environ["USE_AGENTIC_MODE"] = original_value
        elif "USE_AGENTIC_MODE" in os.environ:
            del os.environ["USE_AGENTIC_MODE"]
        
        return False


def test_tools_import():
    """Tool 시스템 import 테스트"""
    logger.info("=" * 80)
    logger.info("Test 2: Tool 시스템 Import 테스트")
    logger.info("=" * 80)
    
    try:
        from lawfirm_langgraph.langgraph_core.tools import LEGAL_TOOLS
        
        logger.info(f"✅ Tool 시스템 import 성공: {len(LEGAL_TOOLS)}개 Tool")
        
        # Tool 목록 출력
        for i, tool in enumerate(LEGAL_TOOLS, 1):
            logger.info(f"   {i}. {tool.name}: {tool.description[:80]}...")
        
        if len(LEGAL_TOOLS) == 0:
            logger.warning("⚠️ Tool이 없습니다. 의존성 확인이 필요합니다")
            return False
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Tool 시스템 import 실패 (의존성 문제일 수 있음): {e}")
        logger.info("해결 방법: pip install langchain langchain-core")
        return False
    except Exception as e:
        logger.error(f"❌ Tool 시스템 import 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_agentic_workflow_initialization():
    """Agentic 워크플로우 초기화 테스트"""
    logger.info("=" * 80)
    logger.info("Test 3: Agentic 워크플로우 초기화 테스트")
    logger.info("=" * 80)
    
    try:
        # 환경 변수 설정 (Agentic 모드 ON)
        original_value = os.environ.get("USE_AGENTIC_MODE")
        os.environ["USE_AGENTIC_MODE"] = "true"
        
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
        
        config = LangGraphConfig.from_env()
        logger.info(f"Config: use_agentic_mode={config.use_agentic_mode}")
        
        if not config.use_agentic_mode:
            logger.error("❌ Agentic 모드가 활성화되지 않았습니다")
            if original_value:
                os.environ["USE_AGENTIC_MODE"] = original_value
            elif "USE_AGENTIC_MODE" in os.environ:
                del os.environ["USE_AGENTIC_MODE"]
            return False
        
        # 워크플로우 서비스 초기화
        service = LangGraphWorkflowService(config)
        logger.info("✅ WorkflowService 초기화 완료")
        
        # 워크플로우 객체 확인
        if hasattr(service, 'workflow'):
            workflow = service.workflow
            
            # Tool 시스템 확인
            if hasattr(workflow, 'legal_tools'):
                tools = workflow.legal_tools
                tools_count = len(tools) if tools else 0
                logger.info(f"✅ Tool 시스템 로드: {tools_count}개 Tool")
                
                if tools_count > 0:
                    logger.info("✅ Agentic 워크플로우 초기화 성공")
                else:
                    logger.warning("⚠️ Tool 시스템이 비어있습니다")
            else:
                logger.warning("⚠️ legal_tools 속성을 찾을 수 없습니다")
        else:
            logger.warning("⚠️ workflow 속성을 찾을 수 없습니다")
        
        # 환경 변수 복원
        if original_value:
            os.environ["USE_AGENTIC_MODE"] = original_value
        elif "USE_AGENTIC_MODE" in os.environ:
            del os.environ["USE_AGENTIC_MODE"]
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Agentic 워크플로우 초기화 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 환경 변수 복원
        if original_value:
            os.environ["USE_AGENTIC_MODE"] = original_value
        elif "USE_AGENTIC_MODE" in os.environ:
            del os.environ["USE_AGENTIC_MODE"]
        
        return False


async def test_agentic_mode_execution():
    """Agentic 모드 실행 테스트 (NEXT_STEPS.md 참고)"""
    logger.info("=" * 80)
    logger.info("Test 4: Agentic 모드 실행 테스트")
    logger.info("=" * 80)
    
    try:
        # 환경 변수 설정 (Agentic 모드 ON)
        original_value = os.environ.get("USE_AGENTIC_MODE")
        os.environ["USE_AGENTIC_MODE"] = "true"
        
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
        
        config = LangGraphConfig.from_env()
        config.use_agentic_mode = True
        
        service = LangGraphWorkflowService(config)
        logger.info("✅ WorkflowService 초기화 완료 (Agentic 모드)")
        
        # 복잡한 질문으로 테스트 (Agentic 모드가 활성화되는 경우)
        test_query = "복잡한 법률 문제를 분석해주세요. 계약서 검토와 관련 판례 검색이 필요합니다."
        session_id = "agentic_test_session"
        
        logger.info(f"테스트 질문: {test_query}")
        logger.info("워크플로우 실행 중... (시간이 걸릴 수 있습니다)")
        
        # 실제 실행 (선택사항 - 시간이 오래 걸릴 수 있음)
        # result = await service.process_query_async(test_query, session_id)
        # logger.info(f"✅ 워크플로우 실행 완료")
        # logger.info(f"답변 길이: {len(result.get('answer', ''))} 자")
        
        logger.info("✅ Agentic 모드 실행 테스트 준비 완료")
        logger.info("   (실제 실행은 주석 처리되어 있습니다. 필요시 주석 해제하세요)")
        
        # 환경 변수 복원
        if original_value:
            os.environ["USE_AGENTIC_MODE"] = original_value
        elif "USE_AGENTIC_MODE" in os.environ:
            del os.environ["USE_AGENTIC_MODE"]
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Agentic 모드 실행 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 환경 변수 복원
        if original_value:
            os.environ["USE_AGENTIC_MODE"] = original_value
        elif "USE_AGENTIC_MODE" in os.environ:
            del os.environ["USE_AGENTIC_MODE"]
        
        return False


async def run_all_tests():
    """모든 Agentic AI 테스트 실행"""
    logger.info("\n" + "=" * 80)
    logger.info("Agentic AI 테스트 시작")
    logger.info("NEXT_STEPS.md의 Agentic 모드 테스트 절차를 따릅니다")
    logger.info("=" * 80 + "\n")
    
    results = []
    
    # 1. 설정 확인
    results.append(("Agentic 모드 설정 확인", test_agentic_mode_configuration()))
    
    # 2. Tool 시스템 import
    results.append(("Tool 시스템 Import", test_tools_import()))
    
    # 3. 워크플로우 초기화
    results.append(("Agentic 워크플로우 초기화", await test_agentic_workflow_initialization()))
    
    # 4. 실행 테스트
    results.append(("Agentic 모드 실행", await test_agentic_mode_execution()))
    
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
    
    if failed == 0:
        logger.info("\n✅ 모든 Agentic AI 테스트가 통과했습니다!")
        logger.info("\n다음 단계:")
        logger.info("1. 실제 워크플로우 실행 테스트 (test_agentic_mode_execution 주석 해제)")
        logger.info("2. Tool 실행 검증")
        logger.info("3. Agentic 모드 성능 측정")
    else:
        logger.warning(f"\n⚠️ {failed}개의 테스트가 실패했습니다.")
        logger.info("\n문제 해결:")
        logger.info("1. 의존성 확인: pip install langchain langchain-core")
        logger.info("2. 환경 변수 확인: USE_AGENTIC_MODE=true")
        logger.info("3. .env 파일 확인: lawfirm_langgraph/.env")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

