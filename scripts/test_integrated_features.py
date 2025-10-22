#!/usr/bin/env python3
"""
통합된 기능들의 테스트 및 검증 스크립트
새로 통합된 품질 개선 및 테스트 기능들을 검증합니다.
"""

import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_unified_rebuild_manager():
    """통합 재구축 매니저 테스트"""
    print("=== 통합 재구축 매니저 테스트 ===\n")
    
    try:
        # 직접 파일에서 import
        sys.path.append(str(Path(__file__).parent / "core"))
        from unified_rebuild_manager import UnifiedRebuildManager, RebuildConfig, RebuildMode
        
        # 간단한 재구축 모드 테스트
        print("1. 간단한 재구축 모드 테스트:")
        config = RebuildConfig(
            mode=RebuildMode.SIMPLE,
            db_path="data/lawfirm.db",
            raw_data_dir="data/raw",
            backup_enabled=False,  # 테스트용으로 백업 비활성화
            log_level="INFO",
            batch_size=10,
            include_articles=False,
            quality_check=False,
            quality_fix_enabled=False
        )
        
        manager = UnifiedRebuildManager(config)
        print("   ✅ UnifiedRebuildManager 초기화 성공")
        
        # 품질 개선 전용 모드 테스트
        print("\n2. 품질 개선 전용 모드 테스트:")
        quality_config = RebuildConfig(
            mode=RebuildMode.QUALITY_FIX,
            db_path="data/lawfirm.db",
            backup_enabled=False,
            log_level="INFO",
            batch_size=10,
            quality_fix_enabled=True
        )
        
        quality_manager = UnifiedRebuildManager(quality_config)
        print("   ✅ 품질 개선 매니저 초기화 성공")
        
        print("\n   ⚠️  실제 실행은 데이터 손실 위험이 있으므로 건너뜀")
        
    except Exception as e:
        print(f"   ❌ UnifiedRebuildManager 테스트 실패: {e}")
        return False
    
    return True

def test_unified_vector_manager():
    """통합 벡터 매니저 테스트"""
    print("\n=== 통합 벡터 매니저 테스트 ===\n")
    
    try:
        # 직접 파일에서 import
        sys.path.append(str(Path(__file__).parent / "core"))
        from unified_vector_manager import UnifiedVectorManager, VectorConfig, EmbeddingModel, BuildMode
        
        # 기본 벡터 설정 테스트
        print("1. 기본 벡터 설정 테스트:")
        config = VectorConfig(
            model=EmbeddingModel.KO_SROBERTA,
            build_mode=BuildMode.FULL,
            db_path="data/lawfirm.db",
            embeddings_dir="data/embeddings",
            batch_size=16,
            chunk_size=100,
            use_gpu=False,  # CPU 모드로 테스트
            memory_optimized=True,
            log_level="INFO"
        )
        
        manager = UnifiedVectorManager(config)
        print("   ✅ UnifiedVectorManager 초기화 성공")
        
        # CPU 최적화 모드 테스트
        print("\n2. CPU 최적화 모드 테스트:")
        cpu_config = VectorConfig(
            model=EmbeddingModel.KO_SROBERTA,
            build_mode=BuildMode.CPU_OPTIMIZED,
            use_gpu=False,
            memory_optimized=True
        )
        
        cpu_manager = UnifiedVectorManager(cpu_config)
        print("   ✅ CPU 최적화 매니저 초기화 성공")
        
        print("\n   ⚠️  실제 벡터 빌드는 시간이 오래 걸리므로 건너뜀")
        
    except Exception as e:
        print(f"   ❌ UnifiedVectorManager 테스트 실패: {e}")
        return False
    
    return True

def test_unified_test_suite():
    """통합 테스트 스위트 테스트"""
    print("\n=== 통합 테스트 스위트 테스트 ===\n")
    
    try:
        # 직접 파일에서 import
        sys.path.append(str(Path(__file__).parent / "testing"))
        from unified_test_suite import UnifiedTestSuite, TestConfig, TestType, ExecutionMode
        
        # 기본 검증 테스트 설정
        print("1. 기본 검증 테스트 설정:")
        config = TestConfig(
            test_type=TestType.VALIDATION,
            execution_mode=ExecutionMode.SEQUENTIAL,
            max_workers=2,
            batch_size=5,
            timeout_seconds=60,
            log_level="INFO",
            enable_chat_service=False,  # 테스트용으로 비활성화
            use_improved_validation=False,
            save_results=False
        )
        
        test_suite = UnifiedTestSuite(config)
        print("   ✅ UnifiedTestSuite 초기화 성공")
        
        # 벡터 임베딩 테스트 설정
        print("\n2. 벡터 임베딩 테스트 설정:")
        vector_config = TestConfig(
            test_type=TestType.VECTOR_EMBEDDING,
            execution_mode=ExecutionMode.SEQUENTIAL,
            max_workers=1,
            batch_size=3,
            timeout_seconds=120,
            save_results=False
        )
        
        vector_suite = UnifiedTestSuite(vector_config)
        print("   ✅ 벡터 임베딩 테스트 스위트 초기화 성공")
        
        # 시맨틱 검색 테스트 설정
        print("\n3. 시맨틱 검색 테스트 설정:")
        semantic_config = TestConfig(
            test_type=TestType.SEMANTIC_SEARCH,
            execution_mode=ExecutionMode.SEQUENTIAL,
            max_workers=1,
            batch_size=3,
            timeout_seconds=120,
            save_results=False
        )
        
        semantic_suite = UnifiedTestSuite(semantic_config)
        print("   ✅ 시맨틱 검색 테스트 스위트 초기화 성공")
        
        # 간단한 테스트 실행
        print("\n4. 간단한 테스트 실행:")
        test_queries = [
            "계약서 작성 방법",
            "법률 상담 문의"
        ]
        
        results = test_suite.run_tests(test_queries)
        print(f"   ✅ 테스트 실행 완료: {results['summary'].get('success_rate', 0):.1f}% 성공률")
        
    except Exception as e:
        print(f"   ❌ UnifiedTestSuite 테스트 실패: {e}")
        return False
    
    return True

def test_base_manager():
    """기본 매니저 테스트"""
    print("\n=== 기본 매니저 테스트 ===\n")
    
    try:
        # 직접 파일에서 import
        sys.path.append(str(Path(__file__).parent / "core"))
        from base_manager import BaseManager, BaseConfig, ScriptConfigManager, ProgressTracker, ErrorHandler, PerformanceMonitor
        
        # 기본 설정 테스트
        print("1. 기본 설정 테스트:")
        config = BaseConfig(
            log_level="INFO",
            log_dir="logs",
            results_dir="results",
            backup_enabled=True,
            timeout_seconds=300
        )
        
        print("   ✅ BaseConfig 생성 성공")
        
        # 설정 관리자 테스트
        print("\n2. 설정 관리자 테스트:")
        config_manager = ScriptConfigManager()
        db_config = config_manager.get_database_config()
        vector_config = config_manager.get_vector_config()
        
        print("   ✅ ScriptConfigManager 초기화 성공")
        print(f"   ✅ 데이터베이스 설정: {db_config.get('path', 'N/A')}")
        print(f"   ✅ 벡터 모델: {vector_config.get('model', 'N/A')}")
        
        # 진행률 추적기 테스트
        print("\n3. 진행률 추적기 테스트:")
        tracker = ProgressTracker(total=10, description="테스트 진행")
        for i in range(10):
            tracker.update()
            time.sleep(0.1)
        tracker.finish()
        print("   ✅ ProgressTracker 테스트 완료")
        
        # 에러 핸들러 테스트
        print("\n4. 에러 핸들러 테스트:")
        import logging
        logger = logging.getLogger("test")
        error_handler = ErrorHandler(logger)
        
        try:
            raise ValueError("테스트 에러")
        except Exception as e:
            error_handler.handle_error(e, "테스트 컨텍스트")
        
        error_summary = error_handler.get_error_summary()
        print(f"   ✅ ErrorHandler 테스트 완료: {error_summary['total_errors']}개 에러 처리")
        
        # 성능 모니터 테스트
        print("\n5. 성능 모니터 테스트:")
        monitor = PerformanceMonitor()
        monitor.start_timer("test_operation")
        time.sleep(0.1)
        duration = monitor.end_timer("test_operation")
        
        summary = monitor.get_summary()
        print(f"   ✅ PerformanceMonitor 테스트 완료: {duration:.3f}초")
        
    except Exception as e:
        print(f"   ❌ BaseManager 테스트 실패: {e}")
        return False
    
    return True

def test_file_structure():
    """파일 구조 테스트"""
    print("\n=== 파일 구조 테스트 ===\n")
    
    try:
        # 핵심 파일들 존재 확인
        core_files = [
            "scripts/core/unified_rebuild_manager.py",
            "scripts/core/unified_vector_manager.py",
            "scripts/core/base_manager.py",
            "scripts/testing/unified_test_suite.py"
        ]
        
        print("1. 핵심 파일 존재 확인:")
        for file_path in core_files:
            if Path(file_path).exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} - 파일 없음")
                return False
        
        # 폴더 구조 확인
        print("\n2. 폴더 구조 확인:")
        folders = [
            "scripts/core",
            "scripts/testing",
            "scripts/analysis",
            "scripts/utilities",
            "scripts/deprecated"
        ]
        
        for folder in folders:
            if Path(folder).exists():
                print(f"   ✅ {folder}")
            else:
                print(f"   ❌ {folder} - 폴더 없음")
                return False
        
        # README 파일 확인
        print("\n3. 문서 파일 확인:")
        readme_files = [
            "scripts/README.md",
            "scripts/deprecated/README.md"
        ]
        
        for readme in readme_files:
            if Path(readme).exists():
                print(f"   ✅ {readme}")
            else:
                print(f"   ❌ {readme} - 파일 없음")
                return False
        
    except Exception as e:
        print(f"   ❌ 파일 구조 테스트 실패: {e}")
        return False
    
    return True

def main():
    """메인 함수"""
    print("🚀 통합된 기능들의 테스트 및 검증 시작\n")
    
    start_time = time.time()
    
    # 테스트 실행
    tests = [
        ("파일 구조", test_file_structure),
        ("기본 매니저", test_base_manager),
        ("통합 재구축 매니저", test_unified_rebuild_manager),
        ("통합 벡터 매니저", test_unified_vector_manager),
        ("통합 테스트 스위트", test_unified_test_suite)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                print(f"✅ {test_name} 테스트 통과")
            else:
                print(f"❌ {test_name} 테스트 실패")
        except Exception as e:
            print(f"❌ {test_name} 테스트 오류: {e}")
            results[test_name] = False
    
    # 결과 요약
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*50}")
    print("📊 테스트 결과 요약")
    print(f"{'='*50}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ 통과" if success else "❌ 실패"
        print(f"{test_name}: {status}")
    
    print(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")
    print(f"소요 시간: {duration:.2f}초")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과! 통합 시스템이 정상적으로 작동합니다.")
    else:
        print(f"\n⚠️  {total-passed}개 테스트 실패. 문제를 확인해주세요.")
    
    # 결과 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"results/integration_test_results_{timestamp}.json"
    
    try:
        Path("results").mkdir(exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'total_tests': total,
                'passed_tests': passed,
                'success_rate': passed/total*100,
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 결과 저장: {result_file}")
        
    except Exception as e:
        print(f"\n⚠️  결과 저장 실패: {e}")

if __name__ == "__main__":
    main()
