# -*- coding: utf-8 -*-
"""
모든 테스트 실행 스크립트
기본 기능, 전체 워크플로우 테스트를 순차적으로 실행
"""

import asyncio
import logging
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


def run_import_test():
    """Import 테스트 실행 (기본 기능 테스트로 대체)"""
    logger.info("\n" + "=" * 80)
    logger.info("1단계: 기본 Import 검증")
    logger.info("=" * 80)
    
    try:
        # 기본 import 확인
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
        from lawfirm_langgraph.langgraph_core.utils.state_definitions import LegalWorkflowState
        logger.info("✅ 모든 핵심 모듈 import 성공")
        return True
    except ImportError as e:
        logger.error(f"❌ Import 실패: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Import 검증 실패: {e}")
        return False


async def run_basic_functionality_test():
    """기본 기능 테스트 실행"""
    logger.info("\n" + "=" * 80)
    logger.info("2단계: 기본 기능 테스트")
    logger.info("=" * 80)
    
    try:
        import test_basic_functionality
        return await test_basic_functionality.run_all_tests()
    except ImportError as e:
        logger.error(f"❌ 테스트 파일을 찾을 수 없습니다: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 기본 기능 테스트 실행 실패: {e}")
        return False


async def run_full_workflow_test():
    """전체 워크플로우 테스트 실행"""
    logger.info("\n" + "=" * 80)
    logger.info("3단계: 전체 워크플로우 통합 테스트")
    logger.info("=" * 80)
    
    try:
        import test_full_workflow
        return await test_full_workflow.run_all_tests()
    except ImportError as e:
        logger.error(f"❌ 테스트 파일을 찾을 수 없습니다: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 전체 워크플로우 테스트 실행 실패: {e}")
        return False


async def check_dependencies():
    """의존성 확인"""
    logger.info("\n" + "=" * 80)
    logger.info("의존성 확인")
    logger.info("=" * 80)
    
    missing = []
    
    # 필수 의존성 확인
    dependencies = {
        'langchain': 'langchain',
        'langchain_core': 'langchain-core',
        'langgraph': 'langgraph',
        'google.generativeai': 'google-generativeai',
    }
    
    for module, package in dependencies.items():
        try:
            __import__(module)
            logger.info(f"✅ {package} 설치됨")
        except ImportError:
            logger.warning(f"⚠️ {package} 미설치")
            missing.append(package)
    
    if missing:
        logger.error("\n❌ 다음 패키지를 설치해주세요:")
        logger.error(f"pip install {' '.join(missing)}")
        return False
    
    logger.info("✅ 모든 필수 의존성이 설치되어 있습니다")
    return True


async def main():
    """메인 실행 함수"""
    logger.info("\n" + "=" * 80)
    logger.info("LawFirm LangGraph 전체 테스트 스위트")
    logger.info("=" * 80 + "\n")
    
    # 의존성 확인
    if not await check_dependencies():
        logger.error("\n❌ 의존성 확인 실패. 테스트를 중단합니다.")
        return False
    
    results = []
    
    # 1단계: Import 검증
    results.append(("Import Verification", run_import_test()))
    
    # 2단계: 기본 기능 테스트
    results.append(("Basic Functionality Test", await run_basic_functionality_test()))
    
    # 3단계: 전체 워크플로우 테스트
    results.append(("Full Workflow Test", await run_full_workflow_test()))
    
    # 최종 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("최종 테스트 결과 요약")
    logger.info("=" * 80)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    failed = total - passed
    
    logger.info("=" * 80)
    logger.info(f"총 테스트 스위트: {total}개 | 통과: {passed}개 | 실패: {failed}개")
    logger.info("=" * 80)
    
    if failed == 0:
        logger.info("\n🎉 모든 테스트가 통과했습니다!")
    else:
        logger.warning(f"\n⚠️ {failed}개의 테스트 스위트가 실패했습니다.")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

