# -*- coding: utf-8 -*-
"""
semantic_search_engine 전달 확인 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow


def test_semantic_search_engine_delivery():
    """semantic_search_engine 전달 확인 테스트"""
    print("=" * 80)
    print("🔍 semantic_search_engine 전달 확인 테스트")
    print("=" * 80)
    
    try:
        # Config 로드
        config = LangGraphConfig()
        
        # Workflow 초기화
        print("\n[1] EnhancedLegalQuestionWorkflow 초기화 중...")
        workflow = EnhancedLegalQuestionWorkflow(config)
        
        # semantic_search 확인
        print("\n[2] semantic_search 확인:")
        if hasattr(workflow, 'semantic_search') and workflow.semantic_search:
            print(f"  ✅ workflow.semantic_search: {type(workflow.semantic_search).__name__}")
            print(f"     - Available: {workflow.semantic_search.is_available() if hasattr(workflow.semantic_search, 'is_available') else 'N/A'}")
        else:
            print("  ❌ workflow.semantic_search: None 또는 없음")
        
        # SearchExecutionProcessor 확인
        print("\n[3] SearchExecutionProcessor 확인:")
        if hasattr(workflow, 'search_execution_processor') and workflow.search_execution_processor:
            processor = workflow.search_execution_processor
            print(f"  ✅ search_execution_processor: {type(processor).__name__}")
            
            # semantic_search_engine 확인
            if hasattr(processor, 'semantic_search_engine'):
                engine = processor.semantic_search_engine
                if engine:
                    print(f"  ✅ processor.semantic_search_engine: {type(engine).__name__}")
                else:
                    print("  ⚠️ processor.semantic_search_engine: None")
            else:
                print("  ❌ processor.semantic_search_engine: 속성 없음")
            
            # search_handler 확인
            if hasattr(processor, 'search_handler') and processor.search_handler:
                handler = processor.search_handler
                print(f"  ✅ processor.search_handler: {type(handler).__name__}")
                
                # search_handler의 semantic_search_engine 확인
                if hasattr(handler, 'semantic_search_engine'):
                    handler_engine = handler.semantic_search_engine
                    if handler_engine:
                        print(f"  ✅ handler.semantic_search_engine: {type(handler_engine).__name__}")
                    else:
                        print("  ⚠️ handler.semantic_search_engine: None")
                else:
                    print("  ❌ handler.semantic_search_engine: 속성 없음")
                
                # search_handler의 semantic_search 확인
                if hasattr(handler, 'semantic_search'):
                    handler_search = handler.semantic_search
                    if handler_search:
                        print(f"  ✅ handler.semantic_search: {type(handler_search).__name__}")
                    else:
                        print("  ⚠️ handler.semantic_search: None")
                else:
                    print("  ❌ handler.semantic_search: 속성 없음")
            else:
                print("  ❌ processor.search_handler: None 또는 없음")
        else:
            print("  ❌ search_execution_processor: None 또는 없음")
        
        # 전달 경로 확인
        print("\n[4] 전달 경로 확인:")
        if (hasattr(workflow, 'semantic_search') and workflow.semantic_search and
            hasattr(workflow, 'search_execution_processor') and workflow.search_execution_processor):
            processor = workflow.search_execution_processor
            if hasattr(processor, 'semantic_search_engine'):
                if processor.semantic_search_engine == workflow.semantic_search:
                    print("  ✅ 전달 경로 정상: workflow.semantic_search → processor.semantic_search_engine")
                else:
                    print("  ⚠️ 전달 경로 불일치: workflow.semantic_search ≠ processor.semantic_search_engine")
            else:
                print("  ❌ processor.semantic_search_engine 속성 없음")
        else:
            print("  ❌ 전달 경로 확인 불가: 필수 컴포넌트 없음")
        
        print("\n" + "=" * 80)
        print("✅ 테스트 완료")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_semantic_search_engine_delivery()

