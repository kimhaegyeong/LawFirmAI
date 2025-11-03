# -*- coding: utf-8 -*-
"""
Agentic AI 워크플로우 실행 테스트
실제 Agentic 모드에서 워크플로우 실행 및 Tool 사용 검증
"""

import asyncio
import logging
import os
import sys
import time
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


async def test_agentic_mode_enabled():
    """Agentic 모드 활성화된 상태에서 워크플로우 테스트"""
    logger.info("=" * 80)
    logger.info("Agentic 모드 활성화된 상태 워크플로우 테스트")
    logger.info("=" * 80)
    
    # 환경 변수 설정
    original_value = os.environ.get("USE_AGENTIC_MODE")
    os.environ["USE_AGENTIC_MODE"] = "true"
    
    try:
        from source.agents.workflow_service import LangGraphWorkflowService
        from infrastructure.utils.langgraph_config import LangGraphConfig
        
        config = LangGraphConfig.from_env()
        assert config.use_agentic_mode == True, "Agentic 모드가 활성화되어야 함"
        
        logger.info("✅ Agentic 모드 활성화 확인")
        
        workflow_service = LangGraphWorkflowService(config)
        logger.info("✅ 워크플로우 서비스 초기화 완료")
        
        # 복잡한 질문 (Agentic 노드가 사용될 가능성이 있음)
        test_queries = [
            "계약 일반 약관에서 약속금은 어떻게 계산하나요?",
            "이혼 소송 절차를 알려주세요",
            "민법 제923조에 관련된 최근 판례를 찾아주세요"
        ]
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- 테스트 질의 {i}: {query} ---")
            
            try:
                start_time = time.time()
                result = await workflow_service.process_query(query)
                execution_time = time.time() - start_time
                
                logger.info(f"✅ 실행 시간: {execution_time:.2f}초")
                
                # 결과 검증
                if "answer" in result or result.get("response"):
                    answer = result.get("answer") or result.get("response", "")
                    logger.info(f"✅ 응답 생성됨 ({len(answer)}자)")
                    
                    # Agentic Tool 호출 정보 확인
                    if "agentic_tool_calls" in str(result):
                        logger.info("✅ Agentic Tool이 사용된 것으로 보임")
                    else:
                        logger.info("⚠️ Agentic Tool 사용 정보 없음 (기존 프로세일 수도 있음)")
                    
                    # 검색 결과 확인
                    if "sources" in result or "retrieved_docs" in result:
                        sources = result.get("sources", []) or result.get("retrieved_docs", [])
                        logger.info(f"✅ 검색 결과: {len(sources)}개")
                    
                else:
                    logger.warning("⚠️ 응답 코드 없음")
                    
            except Exception as e:
                logger.error(f"❌ 질의 처리 실패: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("\n" + "=" * 80)
        logger.info("테스트 완료")
        logger.info("=" * 80)
        
    finally:
        # 환경 변수 복원
        if original_value:
            os.environ["USE_AGENTIC_MODE"] = original_value
        else:
            os.environ.pop("USE_AGENTIC_MODE", None)


async def test_tool_selection():
    """Tool 선택 작동 테스트"""
    logger.info("=" * 80)
    logger.info("Tool 선택 작동 테스트")
    logger.info("=" * 80)
    
    try:
        from langgraph_core.tools import LEGAL_TOOLS
        
        logger.info(f"사용 가능한 Tool: {len(LEGAL_TOOLS)}개")
        for i, tool in enumerate(LEGAL_TOOLS, 1):
            logger.info(f"  {i}. {tool.name}")
            logger.info(f"     설명: {tool.description[:100]}...")
        
        # 각 Tool의 입력 스키마 확인
        for tool in LEGAL_TOOLS:
            if hasattr(tool, 'args_schema'):
                logger.info(f"\n{tool.name} 입력 스키마:")
                schema = tool.args_schema
                if hasattr(schema, 'schema'):
                    for field_name, field_info in schema.schema().get('properties', {}).items():
                        logger.info(f"  - {field_name}: {field_info.get('description', 'N/A')}")
        
        logger.info("\n✅ Tool 선택 테스트 완료")
        
    except Exception as e:
        logger.error(f"❌ Tool 선택 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_agentic_mode_enabled())
    asyncio.run(test_tool_selection())
