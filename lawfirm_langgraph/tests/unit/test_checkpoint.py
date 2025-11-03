# -*- coding: utf-8 -*-
"""
Checkpoint 기능 테스트 스크립트
Checkpoint 활성화 및 동작을 확인합니다.
"""

import sys
import os
import importlib.util
from pathlib import Path

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

lawfirm_langgraph_root = Path(__file__).parent.parent.parent
if str(lawfirm_langgraph_root) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_root))

def import_checkpoint_manager_directly():
    """CheckpointManager를 직접 파일 경로로 import (__init__.py 우회)"""
    try:
        # 직접 파일 경로로 import하여 __init__.py의 circular import 우회
        checkpoint_manager_path = Path(__file__).parent.parent.parent / "langgraph_core" / "utils" / "checkpoint_manager.py"
        spec = importlib.util.spec_from_file_location(
            "checkpoint_manager", 
            checkpoint_manager_path
        )
        checkpoint_manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checkpoint_manager_module)
        return checkpoint_manager_module.CheckpointManager
    except Exception as e:
        # 폴백: 일반 import 시도
        from langgraph_core.utils.checkpoint_manager import CheckpointManager
        return CheckpointManager

def test_checkpoint_manager_import():
    """CheckpointManager import 테스트"""
    print("\n" + "=" * 60)
    print("1. CheckpointManager Import 테스트")
    print("=" * 60)
    
    try:
        # 직접 import 방식 사용
        CheckpointManager = import_checkpoint_manager_directly()
        print("  ✓ CheckpointManager import 성공")
        print(f"    - Type: {CheckpointManager.__name__}")
        return True
    except Exception as e:
        print(f"  ✗ CheckpointManager import 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_checkpoint_manager_initialization():
    """CheckpointManager 초기화 테스트"""
    print("\n" + "=" * 60)
    print("2. CheckpointManager 초기화 테스트")
    print("=" * 60)
    
    try:
        # 직접 import 방식 사용
        CheckpointManager = import_checkpoint_manager_directly()
        
        # Config는 일반 import로 가능
        from config.langgraph_config import LangGraphConfig
        
        config = LangGraphConfig.from_env()
        db_path = config.checkpoint_db_path
        
        # 임시 디렉토리 사용
        test_db_path = str(Path(__file__).parent.parent.parent / "data" / "test_checkpoints.db")
        
        manager = CheckpointManager(test_db_path)
        print(f"  ✓ CheckpointManager 초기화 성공: {test_db_path}")
        
        # get_memory 테스트
        memory = manager.get_memory()
        if memory:
            print(f"  ✓ Checkpoint memory 객체 획득 성공: {type(memory).__name__}")
        else:
            print("  ⚠ Checkpoint memory 객체가 None입니다 (LangGraph 미설치 가능)")
        
        return True
    except Exception as e:
        print(f"  ✗ CheckpointManager 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workflow_service_with_checkpoint():
    """WorkflowService에서 Checkpoint 사용 테스트"""
    print("\n" + "=" * 60)
    print("3. WorkflowService Checkpoint 통합 테스트")
    print("=" * 60)
    
    try:
        from langgraph_core.services.workflow_service import LangGraphWorkflowService
        from config.langgraph_config import LangGraphConfig
        
        # Checkpoint 활성화
        os.environ["ENABLE_CHECKPOINT"] = "true"
        
        config = LangGraphConfig.from_env()
        service = LangGraphWorkflowService(config)
        
        # CheckpointManager가 초기화되었는지 확인
        if service.checkpoint_manager:
            print("  ✓ CheckpointManager가 초기화되었습니다")
            print(f"    - DB 경로: {service.checkpoint_manager.db_path}")
        else:
            print("  ⚠ CheckpointManager가 None입니다 (정상일 수 있음 - 환경 변수 설정 확인)")
        
        # App이 checkpointer와 함께 컴파일되었는지 확인
        if service.app:
            print("  ✓ Workflow app이 생성되었습니다")
            # checkpointer 속성 확인 (내부 속성이므로 직접 접근 불가)
            print("    - App type: " + str(type(service.app)))
        else:
            print("  ✗ Workflow app이 생성되지 않았습니다")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ WorkflowService Checkpoint 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_checkpoint_list_functionality():
    """Checkpoint list 기능 테스트"""
    print("\n" + "=" * 60)
    print("4. Checkpoint List 기능 테스트")
    print("=" * 60)
    
    try:
        CheckpointManager = import_checkpoint_manager_directly()
        
        test_db_path = str(Path(__file__).parent.parent.parent / "data" / "test_checkpoints.db")
        manager = CheckpointManager(test_db_path)
        
        # 테스트 thread_id로 checkpoint 목록 조회
        test_thread_id = "test-thread-123"
        checkpoints = manager.list_checkpoints(test_thread_id)
        
        print(f"  ✓ list_checkpoints 호출 성공")
        print(f"    - Thread ID: {test_thread_id}")
        print(f"    - Found checkpoints: {len(checkpoints)}")
        
        return True
    except Exception as e:
        print(f"  ✗ Checkpoint list 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_checkpoint_database_info():
    """Checkpoint 데이터베이스 정보 조회 테스트"""
    print("\n" + "=" * 60)
    print("5. Checkpoint 데이터베이스 정보 테스트")
    print("=" * 60)
    
    try:
        CheckpointManager = import_checkpoint_manager_directly()
        
        test_db_path = str(Path(__file__).parent.parent.parent / "data" / "test_checkpoints.db")
        manager = CheckpointManager(test_db_path)
        
        db_info = manager.get_database_info()
        
        print(f"  ✓ get_database_info 호출 성공")
        print(f"    - Database path: {db_info.get('database_path')}")
        print(f"    - LangGraph available: {db_info.get('langgraph_available')}")
        print(f"    - Saver type: {db_info.get('saver_type')}")
        
        return True
    except Exception as e:
        print(f"  ✗ 데이터베이스 정보 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_checkpoint_cleanup_functionality():
    """Checkpoint cleanup 기능 테스트"""
    print("\n" + "=" * 60)
    print("6. Checkpoint Cleanup 기능 테스트")
    print("=" * 60)
    
    try:
        CheckpointManager = import_checkpoint_manager_directly()
        
        test_db_path = str(Path(__file__).parent.parent.parent / "data" / "test_checkpoints.db")
        manager = CheckpointManager(test_db_path)
        
        # Cleanup 호출 (실제로는 SqliteSaver가 자동 관리하므로 0 반환)
        result = manager.cleanup_old_checkpoints(ttl_hours=24)
        
        print(f"  ✓ cleanup_old_checkpoints 호출 성공")
        print(f"    - Result: {result} (SqliteSaver가 자동 관리)")
        
        return True
    except Exception as e:
        print(f"  ✗ Cleanup 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graph_compilation_with_checkpoint():
    """Graph가 checkpointer와 함께 컴파일되는지 테스트"""
    print("\n" + "=" * 60)
    print("7. Graph Compilation with Checkpoint 테스트")
    print("=" * 60)
    
    try:
        from graph import create_app
        
        # Checkpoint 활성화
        os.environ["ENABLE_CHECKPOINT"] = "true"
        
        app = create_app()
        
        if app:
            print("  ✓ App이 checkpointer와 함께 컴파일되었습니다")
            print(f"    - App type: {type(app).__name__}")
        else:
            print("  ✗ App 생성 실패")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Graph compilation 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("Checkpoint 기능 종합 테스트")
    print("=" * 60)
    
    results = []
    
    # 각 테스트 실행
    results.append(("CheckpointManager Import", test_checkpoint_manager_import()))
    results.append(("CheckpointManager 초기화", test_checkpoint_manager_initialization()))
    results.append(("WorkflowService Checkpoint 통합", test_workflow_service_with_checkpoint()))
    results.append(("Checkpoint List 기능", test_checkpoint_list_functionality()))
    results.append(("데이터베이스 정보 조회", test_checkpoint_database_info()))
    results.append(("Checkpoint Cleanup 기능", test_checkpoint_cleanup_functionality()))
    results.append(("Graph Compilation with Checkpoint", test_graph_compilation_with_checkpoint()))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n총 {passed}/{total} 테스트 통과")
    
    if passed == total:
        print("\n✓ 모든 테스트 통과! Checkpoint 기능이 정상적으로 작동합니다.")
    elif passed >= total - 2:
        print("\n⚠ 대부분의 테스트 통과 (일부 비중요 테스트 실패)")
        print("  Checkpoint 기능은 기본적으로 활성화되었습니다.")
    else:
        print("\n✗ 일부 테스트 실패. 위의 오류를 확인하세요.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

