# -*- coding: utf-8 -*-
"""
LangGraph 워크플로우 테스트
LangGraph가 제대로 작동하는지 확인하는 간단한 테스트
"""

import asyncio
import os
import sys
import time

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, '.')
sys.path.insert(0, 'source')
sys.path.insert(0, 'source/services')
sys.path.insert(0, 'source/utils')
sys.path.insert(0, 'source/models')
sys.path.insert(0, 'source/data')

print("🚀 LangGraph 워크플로우 테스트")
print("=" * 50)

def test_langgraph_import():
    """LangGraph import 테스트"""
    print("\n1. LangGraph 기본 모듈 import 테스트")
    try:
        from langgraph.graph import END, StateGraph
        print("✅ LangGraph 기본 모듈 import 성공")
        return True
    except ImportError as e:
        print(f"❌ LangGraph 기본 모듈 import 실패: {e}")
        return False

def test_langchain_import():
    """LangChain import 테스트"""
    print("\n2. LangChain 모듈 import 테스트")
    try:
        from langchain_core.messages import AIMessage, HumanMessage
        print("✅ LangChain Core import 성공")

        try:
            from langchain_community.llms import Ollama
            print("✅ LangChain Community import 성공")
        except ImportError as e:
            print(f"⚠️ LangChain Community import 실패: {e}")

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("✅ LangChain Google GenAI import 성공")
        except ImportError as e:
            print(f"⚠️ LangChain Google GenAI import 실패: {e}")

        return True
    except ImportError as e:
        print(f"❌ LangChain Core import 실패: {e}")
        return False

def test_project_modules():
    """프로젝트 모듈 import 테스트"""
    print("\n3. 프로젝트 모듈 import 테스트")

    # LangGraph 설정 테스트
    try:
        from source.utils.langgraph_config import LangGraphConfig, langgraph_config
        print("✅ LangGraph 설정 모듈 import 성공")
        print(f"   - LangGraph 활성화: {langgraph_config.langgraph_enabled}")
        print(f"   - 체크포인트 저장소: {langgraph_config.checkpoint_storage.value}")
        print(f"   - 최대 반복: {langgraph_config.max_iterations}")
    except ImportError as e:
        print(f"❌ LangGraph 설정 모듈 import 실패: {e}")
        return False

    # 워크플로우 서비스 테스트
    try:
        from source.services.langgraph_workflow.integrated_workflow_service import (
            IntegratedWorkflowService,
        )
        print("✅ 통합 워크플로우 서비스 import 성공")
    except ImportError as e:
        print(f"❌ 통합 워크플로우 서비스 import 실패: {e}")
        return False

    # 향상된 워크플로우 테스트
    try:
        from source.services.langgraph_workflow.legal_workflow_enhanced import (
            EnhancedLegalQuestionWorkflow,
        )
        print("✅ 향상된 법률 워크플로우 import 성공")
    except ImportError as e:
        print(f"❌ 향상된 법률 워크플로우 import 실패: {e}")
        return False

    return True

def test_workflow_initialization():
    """워크플로우 초기화 테스트"""
    print("\n4. 워크플로우 초기화 테스트")

    try:
        from source.services.langgraph_workflow.integrated_workflow_service import (
            IntegratedWorkflowService,
        )
        from source.utils.langgraph_config import langgraph_config

        print("🚀 통합 워크플로우 서비스 초기화 중...")
        workflow_service = IntegratedWorkflowService(langgraph_config)
        print("✅ 통합 워크플로우 서비스 초기화 성공")

        # 그래프 정보 확인
        if hasattr(workflow_service, 'graph_app'):
            print(f"✅ 그래프 앱 생성됨: {workflow_service.graph_app}")
        else:
            print("⚠️ 그래프 앱이 생성되지 않음")

        return workflow_service

    except Exception as e:
        print(f"❌ 워크플로우 초기화 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return None

async def test_workflow_execution(workflow_service):
    """워크플로우 실행 테스트"""
    print("\n5. 워크플로우 실행 테스트")

    if not workflow_service:
        print("❌ 워크플로우 서비스가 없어서 실행 테스트를 건너뜁니다.")
        return False

    try:
        test_query = "퇴직금 계산 방법을 알려주세요."
        print(f"테스트 쿼리: {test_query}")

        print("🚀 워크플로우 실행 중...")
        result = await workflow_service.process_query(
            query=test_query,
            context="노동법 관련 질문",
            session_id="test_session",
            user_id="test_user"
        )

        print("✅ 워크플로우 실행 성공")
        print(f"결과 타입: {type(result)}")
        print(f"결과 키: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

        if isinstance(result, dict) and 'response' in result:
            response = result['response']
            print(f"응답 길이: {len(response)}")
            print(f"응답 미리보기: {response[:200]}...")

        return True

    except Exception as e:
        print(f"❌ 워크플로우 실행 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return False

def main():
    """메인 테스트 함수"""
    print("🔍 LangGraph 워크플로우 종합 테스트 시작")
    print("=" * 60)

    # 1. 기본 모듈 import 테스트
    langgraph_ok = test_langgraph_import()
    langchain_ok = test_langchain_import()

    if not langgraph_ok or not langchain_ok:
        print("\n❌ 기본 모듈 import 실패로 테스트를 중단합니다.")
        print("💡 필요한 패키지를 설치하세요:")
        print("   pip install langgraph langchain-core langchain-community langchain-google-genai")
        return

    # 2. 프로젝트 모듈 import 테스트
    project_ok = test_project_modules()
    if not project_ok:
        print("\n❌ 프로젝트 모듈 import 실패로 테스트를 중단합니다.")
        return

    # 3. 워크플로우 초기화 테스트
    workflow_service = test_workflow_initialization()

    # 4. 워크플로우 실행 테스트
    if workflow_service:
        execution_ok = asyncio.run(test_workflow_execution(workflow_service))

        if execution_ok:
            print("\n🎉 LangGraph 워크플로우 테스트 완료!")
            print("✅ 모든 테스트가 성공했습니다.")
        else:
            print("\n⚠️ 워크플로우 실행 테스트 실패")
    else:
        print("\n⚠️ 워크플로우 초기화 실패")

    print("\n" + "=" * 60)
    print("📊 테스트 요약:")
    print(f"   - LangGraph import: {'✅' if langgraph_ok else '❌'}")
    print(f"   - LangChain import: {'✅' if langchain_ok else '❌'}")
    print(f"   - 프로젝트 모듈: {'✅' if project_ok else '❌'}")
    print(f"   - 워크플로우 초기화: {'✅' if workflow_service else '❌'}")
    print(f"   - 워크플로우 실행: {'✅' if workflow_service and asyncio.run(test_workflow_execution(workflow_service)) else '❌'}")

if __name__ == "__main__":
    main()
