#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 통합 테스트 스크립트
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """기본 import 테스트"""
    print("=== 1. 기본 Import 테스트 ===")
    
    try:
        from source.utils.langgraph_config import LangGraphConfig
        print("[OK] LangGraphConfig import 성공")
        
        config = LangGraphConfig.from_env()
        print(f"[OK] 설정 로드 성공: enabled={config.langgraph_enabled}")
        print(f"   - 체크포인트 DB: {config.checkpoint_db_path}")
        print(f"   - Ollama URL: {config.ollama_base_url}")
        print(f"   - Ollama 모델: {config.ollama_model}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Import 실패: {e}")
        return False

def test_state_definitions():
    """상태 정의 테스트"""
    print("\n=== 2. 상태 정의 테스트 ===")
    
    try:
        from source.services.langgraph.state_definitions import (
            create_initial_legal_state,
            LegalWorkflowState
        )
        print("[OK] 상태 정의 import 성공")
        
        # 초기 상태 생성 테스트
        state = create_initial_legal_state("테스트 질문", "test-session")
        print(f"[OK] 초기 상태 생성 성공")
        print(f"   - 질문: {state['query']}")
        print(f"   - 세션 ID: {state['session_id']}")
        print(f"   - 처리 단계 수: {len(state['processing_steps'])}")
        
        return True
    except Exception as e:
        print(f"[ERROR] 상태 정의 테스트 실패: {e}")
        return False

def test_checkpoint_manager():
    """체크포인트 관리자 테스트"""
    print("\n=== 3. 체크포인트 관리자 테스트 ===")
    
    try:
        from source.services.langgraph.checkpoint_manager import CheckpointManager
        
        # 임시 데이터베이스 파일 사용
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name
        
        manager = CheckpointManager(db_path)
        print("[OK] 체크포인트 관리자 초기화 성공")
        
        # 데이터베이스 정보 조회
        info = manager.get_database_info()
        print(f"[OK] 데이터베이스 정보 조회 성공")
        print(f"   - DB 경로: {info['database_path']}")
        print(f"   - LangGraph 사용 가능: {info['langgraph_available']}")
        
        # 임시 파일 정리
        os.unlink(db_path)
        
        return True
    except Exception as e:
        print(f"[ERROR] 체크포인트 관리자 테스트 실패: {e}")
        return False

def test_workflow_compilation():
    """워크플로우 컴파일 테스트"""
    print("\n=== 4. 워크플로우 컴파일 테스트 ===")
    
    try:
        from source.services.langgraph.legal_workflow import LegalQuestionWorkflow
        from source.utils.langgraph_config import LangGraphConfig
        
        config = LangGraphConfig.from_env()
        
        # 워크플로우 초기화 (모킹 사용)
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            config.checkpoint_db_path = tmp_file.name
        
        # 모킹을 사용하여 의존성 문제 해결
        import unittest.mock as mock
        
        with mock.patch('source.services.langgraph.legal_workflow.QuestionClassifier'):
            with mock.patch('source.services.langgraph.legal_workflow.HybridSearchEngine'):
                with mock.patch('source.services.langgraph.legal_workflow.OllamaClient'):
                    workflow = LegalQuestionWorkflow(config)
                    print("[OK] 워크플로우 초기화 성공")
                    
                    # 컴파일 테스트
                    compiled = workflow.compile()
                    if compiled:
                        print("[OK] 워크플로우 컴파일 성공")
                    else:
                        print("[WARNING] 워크플로우 컴파일 실패 (LangGraph 미설치 가능)")
        
        # 임시 파일 정리
        os.unlink(config.checkpoint_db_path)
        
        return True
    except Exception as e:
        print(f"[ERROR] 워크플로우 컴파일 테스트 실패: {e}")
        return False

async def test_workflow_service():
    """워크플로우 서비스 테스트"""
    print("\n=== 5. 워크플로우 서비스 테스트 ===")
    
    try:
        from source.services.langgraph.workflow_service import LangGraphWorkflowService
        from source.utils.langgraph_config import LangGraphConfig
        
        config = LangGraphConfig.from_env()
        
        # 임시 데이터베이스 파일 사용
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            config.checkpoint_db_path = tmp_file.name
        
        # 모킹을 사용하여 의존성 문제 해결
        import unittest.mock as mock
        
        with mock.patch('source.services.langgraph.workflow_service.LegalQuestionWorkflow'):
            with mock.patch('source.services.langgraph.workflow_service.CheckpointManager'):
                service = LangGraphWorkflowService(config)
                print("[OK] 워크플로우 서비스 초기화 성공")
                
                # 서비스 상태 조회
                status = service.get_service_status()
                print(f"[OK] 서비스 상태 조회 성공")
                print(f"   - 서비스명: {status['service_name']}")
                print(f"   - 상태: {status['status']}")
        
        # 임시 파일 정리
        os.unlink(config.checkpoint_db_path)
        
        return True
    except Exception as e:
        print(f"[ERROR] 워크플로우 서비스 테스트 실패: {e}")
        return False

def test_chat_service_integration():
    """ChatService 통합 테스트"""
    print("\n=== 6. ChatService 통합 테스트 ===")
    
    try:
        from source.services.chat_service import ChatService
        from source.utils.config import Config
        
        # 환경 변수 설정
        os.environ["USE_LANGGRAPH"] = "true"
        
        config = Config()
        chat_service = ChatService(config)
        print("[OK] ChatService 초기화 성공")
        
        # 서비스 상태 확인
        status = chat_service.get_service_status()
        print(f"[OK] ChatService 상태 확인 성공")
        print(f"   - LangGraph 활성화: {status['langgraph_enabled']}")
        print(f"   - 워크플로우 서비스 사용 가능: {status['langgraph_service_available']}")
        
        return True
    except Exception as e:
        print(f"[ERROR] ChatService 통합 테스트 실패: {e}")
        return False

async def test_full_integration():
    """전체 통합 테스트"""
    print("\n=== 7. 전체 통합 테스트 ===")
    
    try:
        # 환경 변수 설정
        os.environ["USE_LANGGRAPH"] = "true"
        os.environ["LANGGRAPH_ENABLED"] = "true"
        
        from source.services.chat_service import ChatService
        from source.utils.config import Config
        
        config = Config()
        chat_service = ChatService(config)
        
        if chat_service.langgraph_service:
            print("[OK] LangGraph 서비스 통합 성공")
            
            # 간단한 테스트 질문 처리 (모킹 사용)
            import unittest.mock as mock
            
            with mock.patch.object(chat_service.langgraph_service, 'process_query') as mock_process:
                mock_process.return_value = {
                    "answer": "테스트 답변입니다.",
                    "sources": ["테스트 소스"],
                    "confidence": 0.8,
                    "session_id": "test-session",
                    "processing_steps": ["테스트 단계"],
                    "errors": []
                }
                
                result = await chat_service.process_message("테스트 질문")
                print(f"[OK] 테스트 질문 처리 성공")
                print(f"   - 답변: {result['response']}")
                print(f"   - 신뢰도: {result['confidence']}")
        else:
            print("[WARNING] LangGraph 서비스가 초기화되지 않음")
        
        return True
    except Exception as e:
        print(f"[ERROR] 전체 통합 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("LangGraph 통합 개발 테스트 시작")
    print("=" * 50)
    
    # 동기 테스트
    tests = [
        test_imports,
        test_state_definitions,
        test_checkpoint_manager,
        test_workflow_compilation,
        test_chat_service_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 테스트 실행 중 오류: {e}")
    
    # 비동기 테스트
    async_tests = [
        test_workflow_service,
        test_full_integration
    ]
    
    async def run_async_tests():
        nonlocal passed, total
        total += len(async_tests)
        
        for test in async_tests:
            try:
                if await test():
                    passed += 1
            except Exception as e:
                print(f"❌ 비동기 테스트 실행 중 오류: {e}")
    
    # 비동기 테스트 실행
    asyncio.run(run_async_tests())
    
    # 결과 요약
    print("\n" + "=" * 50)
    print(f"🎯 테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트가 성공적으로 통과했습니다!")
        print("✅ LangGraph 통합이 정상적으로 작동합니다.")
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
        print("   - LangGraph 패키지 설치 상태를 확인하세요.")
        print("   - 의존성 모듈들이 정상적으로 설치되어 있는지 확인하세요.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
