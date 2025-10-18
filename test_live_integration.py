#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Integration Live Test
실제 Ollama와 연동하여 LangGraph 기능 테스트
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 환경 변수 설정
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"

async def test_ollama_connection():
    """Ollama 연결 테스트"""
    print("=== Ollama 연결 테스트 ===")
    
    try:
        from langchain_community.llms import Ollama
        
        # Ollama 클라이언트 생성
        ollama_llm = Ollama(
            model="qwen2.5:7b",
            base_url="http://localhost:11434"
        )
        
        # 간단한 테스트 질문
        test_prompt = "안녕하세요. 간단한 인사말을 해주세요."
        print(f"테스트 질문: {test_prompt}")
        
        # 응답 생성
        response = ollama_llm.invoke(test_prompt)
        print(f"Ollama 응답: {response}")
        
        return True
    except Exception as e:
        print(f"Ollama 연결 실패: {e}")
        return False

async def test_langgraph_workflow():
    """LangGraph 워크플로우 테스트"""
    print("\n=== LangGraph 워크플로우 테스트 ===")
    
    try:
        from source.services.langgraph.workflow_service import LangGraphWorkflowService
        from source.utils.langgraph_config import LangGraphConfig
        
        # 임시 데이터베이스 파일 사용
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        # 설정 생성
        config = LangGraphConfig.from_env()
        config.checkpoint_db_path = db_path
        
        # 워크플로우 서비스 초기화
        service = LangGraphWorkflowService(config)
        print("워크플로우 서비스 초기화 성공")
        
        # 간단한 법률 질문 테스트
        test_query = "계약서 작성 시 주의사항은 무엇인가요?"
        print(f"테스트 질문: {test_query}")
        
        # 워크플로우 실행
        result = await service.process_query(test_query)
        
        print("워크플로우 실행 결과:")
        print(f"  - 답변: {result.get('answer', 'N/A')[:100]}...")
        print(f"  - 신뢰도: {result.get('confidence', 0)}")
        print(f"  - 소스: {len(result.get('sources', []))}개")
        print(f"  - 처리 단계: {len(result.get('processing_steps', []))}개")
        print(f"  - 세션 ID: {result.get('session_id', 'N/A')}")
        
        # 임시 파일 정리
        os.unlink(db_path)
        
        return True
    except Exception as e:
        print(f"워크플로우 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_chat_service_integration():
    """ChatService 통합 테스트"""
    print("\n=== ChatService 통합 테스트 ===")
    
    try:
        from source.services.chat_service import ChatService
        from source.utils.config import Config
        
        # ChatService 초기화
        config = Config()
        chat_service = ChatService(config)
        
        print(f"ChatService 초기화 완료")
        print(f"LangGraph 활성화: {chat_service.use_langgraph}")
        
        if chat_service.use_langgraph:
            print("LangGraph 서비스 사용 가능")
            
            # 간단한 메시지 처리 테스트
            test_message = "이혼 절차에 대해 알려주세요."
            print(f"테스트 메시지: {test_message}")
            
            result = await chat_service.process_message(test_message)
            
            print("ChatService 처리 결과:")
            print(f"  - 응답: {result.get('response', 'N/A')[:100]}...")
            print(f"  - 신뢰도: {result.get('confidence', 0)}")
            print(f"  - 처리 시간: {result.get('processing_time', 0):.2f}초")
        else:
            print("LangGraph 서비스가 비활성화됨")
        
        return True
    except Exception as e:
        print(f"ChatService 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """메인 테스트 함수"""
    print("LangGraph 통합 실시간 테스트")
    print("=" * 50)
    
    tests = [
        ("Ollama 연결", test_ollama_connection),
        ("LangGraph 워크플로우", test_langgraph_workflow),
        ("ChatService 통합", test_chat_service_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n[{passed + 1}/{total}] {test_name} 테스트 중...")
            if await test_func():
                passed += 1
                print(f"[OK] {test_name} 테스트 성공")
            else:
                print(f"[FAIL] {test_name} 테스트 실패")
        except Exception as e:
            print(f"[ERROR] {test_name} 테스트 중 오류: {e}")
    
    print("\n" + "=" * 50)
    print(f"테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("모든 테스트가 성공했습니다!")
        print("LangGraph 통합이 정상적으로 작동합니다.")
    else:
        print("일부 테스트가 실패했습니다.")
        print("Ollama 서버 상태와 모델 설치를 확인해주세요.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
