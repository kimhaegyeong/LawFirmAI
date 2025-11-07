"""
워크플로우 디버깅 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 디렉토리를 sys.path에 추가
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
if lawfirm_langgraph_path.exists():
    sys.path.insert(0, str(lawfirm_langgraph_path))

import os
import asyncio
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

async def test_workflow():
    """워크플로우 테스트"""
    try:
        from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        
        print("=" * 50)
        print("LangGraph 워크플로우 테스트")
        print("=" * 50)
        
        # 환경 변수 확인
        print("\n[환경 변수 확인]")
        print(f"GOOGLE_API_KEY: {'설정됨' if os.getenv('GOOGLE_API_KEY') else '설정되지 않음'}")
        print(f"GOOGLE_MODEL: {os.getenv('GOOGLE_MODEL', '기본값 사용')}")
        print(f"LANGGRAPH_ENABLED: {os.getenv('LANGGRAPH_ENABLED', 'true')}")
        
        # 설정 로드
        print("\n[설정 로드]")
        config = LangGraphConfig.from_env()
        print(f"LangGraph 활성화: {config.langgraph_enabled}")
        print(f"LLM Provider: {config.llm_provider}")
        print(f"Google Model: {config.google_model}")
        print(f"Google API Key: {'설정됨' if config.google_api_key else '설정되지 않음'}")
        
        # 워크플로우 서비스 초기화
        print("\n[워크플로우 서비스 초기화]")
        workflow_service = LangGraphWorkflowService(config)
        print("✓ 워크플로우 서비스 초기화 완료")
        
        # 테스트 쿼리
        test_query = "계약서 작성 시 주의사항은 무엇인가요?"
        print(f"\n[테스트 쿼리 실행]")
        print(f"쿼리: {test_query}")
        
        result = await workflow_service.process_query(
            query=test_query,
            session_id="test_session_001",
            enable_checkpoint=False
        )
        
        print("\n[결과]")
        print(f"답변: {result.get('answer', 'N/A')[:200]}...")
        print(f"신뢰도: {result.get('confidence', 0.0)}")
        print(f"쿼리 타입: {result.get('query_type', 'N/A')}")
        print(f"처리 시간: {result.get('processing_time', 0.0)}초")
        
        if result.get('errors'):
            print(f"\n[오류]")
            for error in result.get('errors', []):
                print(f"  - {error}")
        
        print("\n✓ 테스트 완료")
        
    except ImportError as e:
        print(f"\n✗ Import 오류: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_workflow())

