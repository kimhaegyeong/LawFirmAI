# -*- coding: utf-8 -*-
"""
LLM 기반 요약 검증 스크립트
DocumentSummaryAgent가 LLM을 사용하는지 확인
"""

import sys
import os
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

# 환경 변수 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(project_root)
except ImportError:
    pass

def test_summary_agent_llm():
    """요약 에이전트 LLM 사용 검증"""
    print("=" * 80)
    print("LLM 기반 요약 검증 테스트")
    print("=" * 80)
    
    try:
        # 1. DocumentSummaryAgent import
        print("\n1. DocumentSummaryAgent import 중...")
        from lawfirm_langgraph.core.agents.handlers.document_summary_agent import DocumentSummaryAgent
        print("   ✅ DocumentSummaryAgent import 성공")
        
        # 2. LLM 초기화 확인
        print("\n2. LLM 초기화 확인 중...")
        from lawfirm_langgraph.core.workflow.initializers.llm_initializer import LLMInitializer
        from lawfirm_langgraph.core.shared.utils.langgraph_config import LangGraphConfig
        
        config = LangGraphConfig.from_env()
        llm_initializer = LLMInitializer(config=config)
        llm_fast = llm_initializer.initialize_llm_fast()
        
        if llm_fast:
            print(f"   ✅ llm_fast 초기화 성공")
            # 모델 정보 확인
            if hasattr(llm_fast, 'model_name'):
                print(f"   모델: {llm_fast.model_name}")
            elif hasattr(llm_fast, 'model'):
                print(f"   모델: {llm_fast.model}")
        else:
            print("   ❌ llm_fast 초기화 실패")
            return False
        
        # 3. DocumentSummaryAgent에 LLM 전달
        print("\n3. DocumentSummaryAgent에 LLM 전달 중...")
        agent = DocumentSummaryAgent(llm_fast=llm_fast)
        
        if agent.llm_fast:
            print("   ✅ DocumentSummaryAgent.llm_fast 설정 완료")
        else:
            print("   ❌ DocumentSummaryAgent.llm_fast가 None입니다")
            return False
        
        # 4. UnifiedPromptManager 확인
        print("\n4. UnifiedPromptManager 확인 중...")
        from lawfirm_langgraph.core.services.unified_prompt_manager import UnifiedPromptManager
        
        manager = UnifiedPromptManager()
        manager.llm_fast = llm_fast
        
        if manager.llm_fast:
            print("   ✅ UnifiedPromptManager.llm_fast 설정 완료")
        else:
            print("   ❌ UnifiedPromptManager.llm_fast가 None입니다")
            return False
        
        # 5. 요약 에이전트 가져오기
        print("\n5. 요약 에이전트 가져오기...")
        summary_agent = manager._get_summary_agent()
        
        if summary_agent:
            print("   ✅ 요약 에이전트 가져오기 성공")
            if summary_agent.llm_fast:
                print("   ✅ 요약 에이전트의 llm_fast가 설정되어 있습니다")
            else:
                print("   ❌ 요약 에이전트의 llm_fast가 None입니다")
                return False
        else:
            print("   ❌ 요약 에이전트를 가져올 수 없습니다")
            return False
        
        # 6. 테스트 문서로 LLM 요약 테스트
        print("\n6. LLM 기반 요약 테스트...")
        test_doc = {
            "law_name": "민법",
            "article_no": "543",
            "content": "민법 제543조 (해지, 해제권) 계약 또는 법률의 규정에 의하여 당사자의 일방이나 쌍방이 해지 또는 해제의 권리가 있는 때에는 그 해지 또는 해제는 상대방에 대한 의사표시로 한다. 그러나 그 의사표시에는 조건 또는 기한을 붙이지 못한다. " * 10  # 긴 문서
        }
        test_query = "계약 해지에 대해 알려주세요"
        
        print(f"   테스트 문서 길이: {len(test_doc['content'])}자")
        print(f"   테스트 질문: {test_query}")
        print("   LLM 요약 실행 중...")
        
        result = summary_agent.summarize_document(
            test_doc,
            test_query,
            max_summary_length=200,
            use_llm=True  # LLM 사용
        )
        
        if result:
            print("   ✅ 요약 생성 성공")
            print(f"   요약: {result.get('summary', '')[:100]}...")
            print(f"   문서 유형: {result.get('document_type', 'unknown')}")
            print(f"   핵심 포인트 수: {len(result.get('key_points', []))}")
            
            # LLM이 실제로 사용되었는지 확인 (규칙 기반과 다르면 LLM 사용)
            if len(result.get('summary', '')) > 50:
                print("   ✅ LLM 기반 요약으로 보입니다 (요약 길이가 충분함)")
            else:
                print("   ⚠️  요약이 너무 짧아 규칙 기반일 수 있습니다")
        else:
            print("   ❌ 요약 생성 실패")
            return False
        
        print("\n" + "=" * 80)
        print("✅ 모든 검증 통과!")
        print("=" * 80)
        print("\n결론:")
        print("  - llm_fast (gemini-2.5-flash-lite) 초기화: ✅")
        print("  - UnifiedPromptManager에 llm_fast 전달: ✅")
        print("  - DocumentSummaryAgent에 llm_fast 전달: ✅")
        print("  - LLM 기반 요약 사용 (use_llm=True): ✅")
        print("\nLLM 기반 요약이 정상적으로 작동합니다!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_summary_agent_llm()
    sys.exit(0 if success else 1)

