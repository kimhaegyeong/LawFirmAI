# -*- coding: utf-8 -*-
"""
Sources 워크플로우 테스트 스크립트
실제 langgraph 워크플로우를 실행하여 sources 데이터를 테스트합니다.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

# DEBUG_SOURCES 환경변수 설정
os.environ['DEBUG_SOURCES'] = 'true'

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_sources_workflow():
    """sources 워크플로우 테스트"""
    print("\n" + "=" * 80)
    print("Sources 워크플로우 테스트 시작")
    print("=" * 80)
    
    try:
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
        
        # 설정 로드
        logger.info("1️⃣  설정 로드 중...")
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False
        logger.info(f"   ✅ LangGraph 활성화: {config.langgraph_enabled}")
        
        # 서비스 초기화
        logger.info("\n2️⃣  LangGraphWorkflowService 초기화 중...")
        service = LangGraphWorkflowService(config)
        logger.info("   ✅ 서비스 초기화 완료")
        
        # 테스트 질의
        test_query = "전세금 반환 보증에 대해 설명해주세요"
        logger.info(f"\n3️⃣  질의 처리 중: {test_query}")
        
        result = await service.process_query(
            query=test_query,
            session_id="sources_test",
            enable_checkpoint=False
        )
        
        # 결과 분석
        logger.info("\n4️⃣  Sources 데이터 분석:")
        logger.info("=" * 80)
        
        sources = result.get("sources", [])
        sources_detail = result.get("sources_detail", [])
        related_questions = result.get("metadata", {}).get("related_questions", [])
        
        logger.info(f"\n📊 Sources 통계:")
        logger.info(f"   - sources: {len(sources)}개")
        logger.info(f"   - sources_detail: {len(sources_detail)}개")
        logger.info(f"   - related_questions: {len(related_questions)}개")
        
        # sources와 sources_detail 개수 확인
        if len(sources) != len(sources_detail):
            logger.warning(f"\n⚠️  개수 불일치: sources={len(sources)}, sources_detail={len(sources_detail)}")
        else:
            logger.info(f"\n✅ 개수 일치: sources={len(sources)}, sources_detail={len(sources_detail)}")
        
        # sources_detail 상세 분석
        logger.info(f"\n📋 Sources Detail 분석:")
        for idx, detail in enumerate(sources_detail[:10], 1):
            logger.info(f"\n   [{idx}] {detail.get('name', 'N/A')}")
            logger.info(f"       - type: {detail.get('type', 'N/A')}")
            logger.info(f"       - case_name: {detail.get('case_name', 'N/A')}")
            logger.info(f"       - case_number: {detail.get('case_number', 'N/A')}")
            logger.info(f"       - court: {detail.get('court', 'N/A')}")
            logger.info(f"       - url: {detail.get('url', 'N/A')}")
            metadata = detail.get('metadata', {})
            if metadata:
                logger.info(f"       - metadata.court: {metadata.get('court', 'N/A')}")
                logger.info(f"       - metadata.doc_id: {metadata.get('doc_id', 'N/A')}")
                logger.info(f"       - metadata.casenames: {metadata.get('casenames', 'N/A')}")
        
        # related_questions 확인
        if related_questions:
            logger.info(f"\n❓ Related Questions ({len(related_questions)}개):")
            for idx, question in enumerate(related_questions[:5], 1):
                logger.info(f"   {idx}. {question}")
        else:
            logger.warning("\n⚠️  Related Questions가 없습니다!")
        
        # 비어있는 metadata 확인
        empty_metadata_count = 0
        for detail in sources_detail:
            if detail.get("type") == "case_paragraph":
                metadata = detail.get("metadata", {})
                if isinstance(metadata, dict):
                    court = metadata.get("court") or ""
                    doc_id = metadata.get("doc_id") or ""
                    casenames = metadata.get("casenames") or ""
                    if not str(court).strip() and not str(doc_id).strip() and not str(casenames).strip():
                        empty_metadata_count += 1
                        logger.warning(f"   ⚠️ Empty metadata: {detail.get('name')}")
        
        if empty_metadata_count > 0:
            logger.warning(f"\n⚠️  총 {empty_metadata_count}개의 비어있는 metadata 발견")
        else:
            logger.info("\n✅ 모든 metadata가 채워져 있습니다!")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ 테스트 완료!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_sources_workflow())

