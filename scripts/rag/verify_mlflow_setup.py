"""
MLflow 설정 및 통합 검증 스크립트

MLflow 설치, 모듈 import, 기본 기능을 검증합니다.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "utils"))
sys.path.insert(0, str(project_root / "scripts" / "rag"))

def check_mlflow_installation():
    """MLflow 설치 확인"""
    print("=" * 80)
    print("1. MLflow 설치 확인")
    print("=" * 80)
    try:
        import mlflow
        try:
            version = mlflow.__version__
        except AttributeError:
            try:
                import pkg_resources
                version = pkg_resources.get_distribution("mlflow").version
            except:
                version = "설치됨 (버전 확인 불가)"
        print(f"✓ MLflow 설치됨: 버전 {version}")
        return True
    except ImportError:
        print("✗ MLflow가 설치되지 않았습니다.")
        print("  설치 명령: pip install mlflow>=2.8.0")
        return False

def check_module_imports():
    """모듈 import 확인"""
    print("\n" + "=" * 80)
    print("2. 모듈 Import 확인")
    print("=" * 80)
    
    results = {}
    
    try:
        from mlflow_manager import MLflowFAISSManager
        print("✓ mlflow_manager 모듈 import 성공")
        results['mlflow_manager'] = True
    except ImportError as e:
        print(f"✗ mlflow_manager 모듈 import 실패: {e}")
        results['mlflow_manager'] = False
    
    try:
        from build_index import build_and_save_index
        print("✓ build_index 모듈 import 성공")
        results['build_index'] = True
    except ImportError as e:
        print(f"✗ build_index 모듈 import 실패: {e}")
        results['build_index'] = False
    
    try:
        from evaluate import MLflowRAGEvaluator
        print("✓ evaluate 모듈 import 성공")
        results['evaluate'] = True
    except ImportError as e:
        print(f"✗ evaluate 모듈 import 실패: {e}")
        results['evaluate'] = False
    
    try:
        from optimize_search_quality import RAGSearchQualityOptimizer
        print("✓ optimize_search_quality 모듈 import 성공")
        results['optimize_search_quality'] = True
    except ImportError as e:
        print(f"✗ optimize_search_quality 모듈 import 실패: {e}")
        results['optimize_search_quality'] = False
    
    return results

def check_environment_variables():
    """환경 변수 확인"""
    print("\n" + "=" * 80)
    print("3. 환경 변수 확인")
    print("=" * 80)
    
    import os
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        print(f"✓ MLFLOW_TRACKING_URI: {mlflow_uri}")
    else:
        default_uri = "file://./mlflow/mlruns"
        print(f"⚠ MLFLOW_TRACKING_URI가 설정되지 않음 (기본값: {default_uri})")
        print(f"  .env 파일에 MLFLOW_TRACKING_URI={default_uri} 추가 권장")
    
    return mlflow_uri is not None

def check_mlflow_manager_initialization():
    """MLflowFAISSManager 초기화 테스트"""
    print("\n" + "=" * 80)
    print("4. MLflowFAISSManager 초기화 테스트")
    print("=" * 80)
    
    try:
        from mlflow_manager import MLflowFAISSManager
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        tracking_uri = f"file://{temp_dir}/mlruns"
        
        manager = MLflowFAISSManager(
            experiment_name="test_verification",
            tracking_uri=tracking_uri
        )
        
        print(f"✓ MLflowFAISSManager 초기화 성공")
        print(f"  Experiment: {manager.experiment_name}")
        print(f"  Experiment ID: {manager.experiment_id}")
        print(f"  Tracking URI: {manager.tracking_uri}")
        
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
    except Exception as e:
        print(f"✗ MLflowFAISSManager 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_database_connection():
    """데이터베이스 연결 확인"""
    print("\n" + "=" * 80)
    print("5. 데이터베이스 연결 확인")
    print("=" * 80)
    
    db_path = project_root / "data" / "lawfirm_v2.db"
    
    if not db_path.exists():
        print(f"⚠ 데이터베이스 파일을 찾을 수 없음: {db_path}")
        print("  실제 데이터 테스트는 건너뜁니다.")
        return False
    
    print(f"✓ 데이터베이스 파일 존재: {db_path}")
    
    try:
        from embedding_version_manager import EmbeddingVersionManager
        
        evm = EmbeddingVersionManager(str(db_path))
        versions = evm.list_versions()
        
        if versions:
            print(f"✓ 임베딩 버전 {len(versions)}개 발견")
            active_versions = [v for v in versions if v.get('is_active')]
            if active_versions:
                print("\n활성 버전:")
                for v in active_versions[:3]:
                    print(f"  - ID: {v['id']}, Name: {v['version_name']}, "
                          f"Strategy: {v['chunking_strategy']}")
            return True
        else:
            print("⚠ 임베딩 버전이 없습니다.")
            return False
            
    except Exception as e:
        print(f"✗ 데이터베이스 연결 실패: {e}")
        return False

def main():
    """메인 검증 함수"""
    print("\n" + "=" * 80)
    print("MLflow 통합 검증")
    print("=" * 80)
    print()
    
    results = {
        'mlflow_installed': check_mlflow_installation(),
        'modules_imported': check_module_imports(),
        'env_vars_set': check_environment_variables(),
        'manager_initialized': check_mlflow_manager_initialization(),
        'database_connected': check_database_connection()
    }
    
    print("\n" + "=" * 80)
    print("검증 결과 요약")
    print("=" * 80)
    
    all_passed = True
    for check_name, result in results.items():
        if isinstance(result, dict):
            all_passed = all_passed and all(result.values())
            for module, status in result.items():
                status_str = "✓" if status else "✗"
                print(f"{status_str} {module}")
        else:
            status_str = "✓" if result else "✗"
            print(f"{status_str} {check_name}")
            if not result:
                all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 모든 검증 통과!")
        print("\n다음 단계:")
        print("1. 실제 데이터로 인덱스 빌드 테스트")
        print("2. MLflow UI 실행: mlflow ui --backend-store-uri file://./mlflow/mlruns")
    else:
        print("⚠ 일부 검증 실패")
        print("\n해결 방법:")
        if not results['mlflow_installed']:
            print("- MLflow 설치: pip install mlflow>=2.8.0")
        if not results['env_vars_set']:
            print("- .env 파일에 MLFLOW_TRACKING_URI 설정 추가")
    print("=" * 80)

if __name__ == "__main__":
    main()

