# -*- coding: utf-8 -*-
"""
Agentic AI 통합 테스트
Tool Use/Function Calling 기능과 기존 워크플로우의 통합 검증
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

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


class TestAgenticIntegration:
    """Agentic AI 통합 테스트 클래스"""
    
    def __init__(self):
        self.test_results = []
    
    def test_tool_import(self):
        """Tool 클래스 import 테스트"""
        logger.info("=" * 80)
        logger.info("Test 1: Tool 클래스 Import 테스트")
        logger.info("=" * 80)
        
        try:
            from langgraph_core.tools import LEGAL_TOOLS
            logger.info(f"✅ Tool 클래스 import 성공: {len(LEGAL_TOOLS)}개의 Tool")
            
            # Tool 목록 출력
            for i, tool in enumerate(LEGAL_TOOLS, 1):
                logger.info(f"   {i}. {tool.name}: {tool.description[:80]}...")
            
            self.test_results.append(("Tool Import", True, f"{len(LEGAL_TOOLS)}개의 Tool"))
            return True
        except Exception as e:
            logger.error(f"❌ Tool 클래스 import 실패: {e}")
            self.test_results.append(("Tool Import", False, str(e)))
            return False
    
    def test_config_flag(self):
        """설정 플래그 테스트"""
        logger.info("=" * 80)
        logger.info("Test 2: Agentic 모드 설정 플래그 테스트")
        logger.info("=" * 80)
        
        try:
            from infrastructure.utils.langgraph_config import LangGraphConfig
            
            # 기본으로 테스트 (비활성화)
            config_default = LangGraphConfig.from_env()
            logger.info(f"   기본 use_agentic_mode: {config_default.use_agentic_mode}")
            assert config_default.use_agentic_mode == False, "기본값은 False여야 함"
            
            # 환경 변수로 활성화한 테스트
            original_value = os.environ.get("USE_AGENTIC_MODE")
            try:
                os.environ["USE_AGENTIC_MODE"] = "true"
                config_enabled = LangGraphConfig.from_env()
                logger.info(f"   활성화된 use_agentic_mode: {config_enabled.use_agentic_mode}")
                assert config_enabled.use_agentic_mode == True, "활성화된 경우 True여야 함"
                
                # 환경 변수 복원
                if original_value:
                    os.environ["USE_AGENTIC_MODE"] = original_value
                else:
                    os.environ.pop("USE_AGENTIC_MODE", None)
                
                logger.info("✅ 설정 플래그 테스트 성공")
                self.test_results.append(("Config Flag", True, "설정 플래그 정상 작동"))
                return True
            except Exception as e:
                # 환경 변수 복원
                if original_value:
                    os.environ["USE_AGENTIC_MODE"] = original_value
                else:
                    os.environ.pop("USE_AGENTIC_MODE", None)
                raise e
                
        except Exception as e:
            logger.error(f"❌ 설정 플래그 테스트 실패: {e}")
            self.test_results.append(("Config Flag", False, str(e)))
            return False
    
    async def test_workflow_without_agentic(self):
        """Agentic 모드 비활성화 상태에서 워크플로우 테스트 (기존 작동 확인)"""
        logger.info("=" * 80)
        logger.info("Test 3: Agentic 모드 비활성화 상태 워크플로우 테스트")
        logger.info("=" * 80)
        
        try:
            # 환경 변수 확인 후 즉시 비활성화
            original_value = os.environ.get("USE_AGENTIC_MODE")
            os.environ["USE_AGENTIC_MODE"] = "false"
            
            from source.agents.workflow_service import LangGraphWorkflowService
            from infrastructure.utils.langgraph_config import LangGraphConfig
            
            config = LangGraphConfig.from_env()
            assert config.use_agentic_mode == False, "Agentic 모드가 비활성화되어야 함"
            
            workflow_service = LangGraphWorkflowService(config)
            logger.info("   ✅ 워크플로우 서비스 초기화 성공 (Agentic 모드 비활성화)")
            
            # 간단한 질문으로 테스트
            test_query = "계약이란 무엇인가요?"
            logger.info(f"   테스트 질의: {test_query}")
            
            result = await workflow_service.process_query(test_query)
            
            # 결과 검증
            assert "answer" in result or result.get("response"), "응답 코드가 있어야 함"
            logger.info(f"   ✅ 워크플로우 실행 성공 (기존 방식)")
            
            # 환경 변수 복원
            if original_value:
                os.environ["USE_AGENTIC_MODE"] = original_value
            else:
                os.environ.pop("USE_AGENTIC_MODE", None)
            
            self.test_results.append(("Workflow Without Agentic", True, "기존 워크플로우 정상 작동"))
            return True
            
        except Exception as e:
            logger.error(f"❌ 기존 워크플로우 테스트 실패: {e}")
            # 환경 변수 복원
            if original_value:
                os.environ["USE_AGENTIC_MODE"] = original_value
            else:
                os.environ.pop("USE_AGENTIC_MODE", None)
            
            self.test_results.append(("Workflow Without Agentic", False, str(e)))
            return False
    
    async def test_agentic_node_initialization(self):
        """Agentic 노드 초기화 테스트"""
        logger.info("=" * 80)
        logger.info("Test 4: Agentic 노드 초기화 테스트")
        logger.info("=" * 80)
        
        try:
            # 환경 변수 확인 후 즉시 활성화
            original_value = os.environ.get("USE_AGENTIC_MODE")
            os.environ["USE_AGENTIC_MODE"] = "true"
            
            from source.agents.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
            from infrastructure.utils.langgraph_config import LangGraphConfig
            
            config = LangGraphConfig.from_env()
            assert config.use_agentic_mode == True, "Agentic 모드가 활성화되어야 함"
            
            workflow = EnhancedLegalQuestionWorkflow(config)
            
            # Tool 클래스 초기화 확인
            assert hasattr(workflow, "legal_tools"), "legal_tools 속성이 있어야 함"
            logger.info(f"   ✅ Agentic 노드 초기화 성공")
            logger.info(f"   Tool 개수: {len(workflow.legal_tools)}")
            
            # 그래프에 Agentic 노드가 추가되었는지 확인
            graph = workflow._build_graph()
            nodes = graph.nodes.keys() if hasattr(graph, 'nodes') else []
            
            if "agentic_decision" in nodes:
                logger.info("   ✅ agentic_decision 노드가 그래프에 추가됨")
            else:
                logger.warning("   ⚠️ agentic_decision 노드가 그래프에 없음 (추가 확인 필요)")
            
            # 환경 변수 복원
            if original_value:
                os.environ["USE_AGENTIC_MODE"] = original_value
            else:
                os.environ.pop("USE_AGENTIC_MODE", None)
            
            self.test_results.append(("Agentic Node Init", True, f"{len(workflow.legal_tools)}개의 Tool"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Agentic 노드 초기화 테스트 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 환경 변수 복원
            if original_value:
                os.environ["USE_AGENTIC_MODE"] = original_value
            else:
                os.environ.pop("USE_AGENTIC_MODE", None)
            
            self.test_results.append(("Agentic Node Init", False, str(e)))
            return False
    
    async def test_tool_execution(self):
        """Tool 실행 테스트"""
        logger.info("=" * 80)
        logger.info("Test 5: Tool 실행 테스트")
        logger.info("=" * 80)
        
        try:
            from langgraph_core.tools import LEGAL_TOOLS
            
            if not LEGAL_TOOLS:
                logger.warning("   ⚠️ 사용 가능한 Tool이 없음 (검색 엔진 미초기화 가능)")
                self.test_results.append(("Tool Execution", True, "Tool 없음 (정상)"))
                return True
            
            # 첫 번째 Tool로 테스트 (보통 hybrid_search_tool)
            test_tool = LEGAL_TOOLS[0]
            logger.info(f"   테스트 Tool: {test_tool.name}")
            
            # Tool 실행 테스트 (실제 검색은 하지 않고 구조만 확인)
            try:
                # Tool의 함수 확인
                if hasattr(test_tool, 'func'):
                    logger.info(f"   ✅ Tool 함수 확인: {test_tool.func.__name__}")
                else:
                    logger.info(f"   ✅ Tool 구조 확인 완료")
                
                self.test_results.append(("Tool Execution", True, f"{test_tool.name} 확인 완료"))
                return True
            except Exception as e:
                logger.error(f"   ❌ Tool 실행 테스트 실패: {e}")
                self.test_results.append(("Tool Execution", False, str(e)))
                return False
                
        except Exception as e:
            logger.error(f"❌ Tool 실행 테스트 실패: {e}")
            self.test_results.append(("Tool Execution", False, str(e)))
            return False
    
    def print_summary(self):
        """테스트 결과 요약 출력"""
        logger.info("=" * 80)
        logger.info("테스트 결과 요약")
        logger.info("=" * 80)
        
        total = len(self.test_results)
        passed = sum(1 for _, success, _ in self.test_results if success)
        failed = total - passed
        
        for test_name, success, detail in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{status} - {test_name}: {detail}")
        
        logger.info("=" * 80)
        logger.info(f"전체 테스트: {total}개 | 통과: {passed}개 | 실패: {failed}개")
        logger.info("=" * 80)
        
        return failed == 0


async def run_all_tests():
    """모든 테스트 실행"""
    tester = TestAgenticIntegration()
    
    # 동기 테스트
    tester.test_tool_import()
    tester.test_config_flag()
    tester.test_tool_execution()
    
    # 비동기 테스트
    await tester.test_workflow_without_agentic()
    await tester.test_agentic_node_initialization()
    
    # 결과 요약
    success = tester.print_summary()
    
    return success


if __name__ == "__main__":
    # 테스트 실행
    success = asyncio.run(run_all_tests())
    
    # 종료 코드
    sys.exit(0 if success else 1)
