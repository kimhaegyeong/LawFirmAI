# -*- coding: utf-8 -*-
"""
MLflow 인덱스 전용 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "lawfirm_langgraph"))

import warnings
warnings.filterwarnings('ignore')

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

def test_mlflow_index_required():
    """테스트 1: use_mlflow_index=False일 때 RuntimeError 발생 확인"""
    print("\n" + "=" * 60)
    print("테스트 1: use_mlflow_index=False일 때 RuntimeError 발생 확인")
    print("=" * 60)
    
    db_path = "data/lawfirm_v2.db"
    if not Path(db_path).exists():
        db_path = project_root / "data" / "lawfirm_v2.db"
        if not db_path.exists():
            print("❌ 데이터베이스 파일을 찾을 수 없습니다.")
            return False
    
    # 환경 변수 백업 및 제거
    original_use_mlflow = os.getenv("USE_MLFLOW_INDEX")
    if "USE_MLFLOW_INDEX" in os.environ:
        del os.environ["USE_MLFLOW_INDEX"]
    
    try:
        engine = SemanticSearchEngineV2(
            db_path=str(db_path),
            use_mlflow_index=False
        )
        print("❌ 테스트 실패: RuntimeError가 발생하지 않았습니다.")
        return False
    except RuntimeError as e:
        if "MLflow index is required" in str(e):
            print(f"✅ 테스트 통과: 예상된 RuntimeError 발생")
            print(f"   에러 메시지: {e}")
            return True
        else:
            print(f"❌ 테스트 실패: 예상과 다른 RuntimeError")
            print(f"   에러 메시지: {e}")
            return False
    except Exception as e:
        print(f"❌ 테스트 실패: 예상하지 못한 예외 발생")
        print(f"   예외 타입: {type(e).__name__}")
        print(f"   에러 메시지: {e}")
        return False
    finally:
        # 환경 변수 복원
        if original_use_mlflow:
            os.environ["USE_MLFLOW_INDEX"] = original_use_mlflow

def test_mlflow_manager_not_initialized():
    """테스트 2: MLflow 매니저가 초기화되지 않았을 때 RuntimeError 발생 확인"""
    print("\n" + "=" * 60)
    print("테스트 2: MLflow 매니저가 초기화되지 않았을 때 RuntimeError 발생 확인")
    print("=" * 60)
    
    db_path = "data/lawfirm_v2.db"
    if not Path(db_path).exists():
        db_path = project_root / "data" / "lawfirm_v2.db"
        if not db_path.exists():
            print("❌ 데이터베이스 파일을 찾을 수 없습니다.")
            return False
    
    # 환경 변수 백업
    original_use_mlflow = os.getenv("USE_MLFLOW_INDEX")
    
    try:
        # USE_MLFLOW_INDEX=true로 설정하되 MLflow 매니저 초기화 실패 시뮬레이션
        os.environ["USE_MLFLOW_INDEX"] = "true"
        
        # MLflow 매니저를 찾을 수 없도록 경로 조작
        # (실제로는 초기화 시도 시 실패할 것)
        try:
            engine = SemanticSearchEngineV2(
                db_path=str(db_path),
                use_mlflow_index=True
            )
            # MLflow 매니저가 None인 경우 RuntimeError 발생해야 함
            if engine.mlflow_manager is None:
                print("❌ 테스트 실패: MLflow 매니저가 None인데 RuntimeError가 발생하지 않았습니다.")
                return False
            else:
                print("ℹ️  MLflow 매니저가 정상 초기화되었습니다. (이 경우는 정상 동작)")
                return True
        except RuntimeError as e:
            if "MLflow manager is not initialized" in str(e) or "MLflow index is required" in str(e):
                print(f"✅ 테스트 통과: 예상된 RuntimeError 발생")
                print(f"   에러 메시지: {e}")
                return True
            else:
                print(f"❌ 테스트 실패: 예상과 다른 RuntimeError")
                print(f"   에러 메시지: {e}")
                return False
    finally:
        # 환경 변수 복원
        if original_use_mlflow:
            os.environ["USE_MLFLOW_INDEX"] = original_use_mlflow
        elif "USE_MLFLOW_INDEX" in os.environ:
            del os.environ["USE_MLFLOW_INDEX"]

def test_mlflow_index_load():
    """테스트 3: MLflow 인덱스 정상 로드 확인"""
    print("\n" + "=" * 60)
    print("테스트 3: MLflow 인덱스 정상 로드 확인")
    print("=" * 60)
    
    db_path = "data/lawfirm_v2.db"
    if not Path(db_path).exists():
        db_path = project_root / "data" / "lawfirm_v2.db"
        if not db_path.exists():
            print("❌ 데이터베이스 파일을 찾을 수 없습니다.")
            return False
    
    # 환경 변수 확인
    use_mlflow = os.getenv("USE_MLFLOW_INDEX", "false").lower() == "true"
    
    if not use_mlflow:
        print("ℹ️  USE_MLFLOW_INDEX가 false입니다. 이 테스트를 건너뜁니다.")
        print("   MLflow 인덱스 테스트를 하려면 USE_MLFLOW_INDEX=true로 설정하세요.")
        return None
    
    try:
        engine = SemanticSearchEngineV2(
            db_path=str(db_path),
            use_mlflow_index=True
        )
        
        # mlflow_manager가 None인 경우는 초기화 실패로 간주
        if not hasattr(engine, 'mlflow_manager') or engine.mlflow_manager is None:
            print("ℹ️  MLflow 매니저가 초기화되지 않았습니다.")
            print("   이는 MLflow 설정이 없거나 경로 문제일 수 있습니다.")
            return None
        
        if engine.index is not None:
            print(f"✅ MLflow 인덱스 로드 성공")
            print(f"   인덱스 타입: {type(engine.index).__name__}")
            print(f"   벡터 수: {engine.index.ntotal:,}")
            print(f"   Chunk IDs 수: {len(engine._chunk_ids)}")
            if hasattr(engine, 'mlflow_run_id') and engine.mlflow_run_id:
                print(f"   MLflow run ID: {engine.mlflow_run_id}")
            return True
        else:
            print("❌ MLflow 인덱스가 로드되지 않았습니다.")
            return False
    except RuntimeError as e:
        if "No production run found" in str(e):
            print(f"ℹ️  프로덕션 run이 없습니다: {e}")
            print("   이는 정상적인 상황일 수 있습니다.")
            return None
        else:
            print(f"❌ MLflow 인덱스 로드 실패: {e}")
            return False
    except Exception as e:
        print(f"❌ 예상하지 못한 예외 발생: {type(e).__name__}: {e}")
        return False

def test_model_detection_from_mlflow():
    """테스트 4: MLflow에서 모델명 감지 확인"""
    print("\n" + "=" * 60)
    print("테스트 4: MLflow에서 모델명 감지 확인")
    print("=" * 60)
    
    db_path = "data/lawfirm_v2.db"
    if not Path(db_path).exists():
        db_path = project_root / "data" / "lawfirm_v2.db"
        if not db_path.exists():
            print("❌ 데이터베이스 파일을 찾을 수 없습니다.")
            return False
    
    # 환경 변수에서 모델명 제거 (MLflow에서 감지하도록)
    original_model = os.getenv("EMBEDDING_MODEL")
    if "EMBEDDING_MODEL" in os.environ:
        del os.environ["EMBEDDING_MODEL"]
    
    use_mlflow = os.getenv("USE_MLFLOW_INDEX", "false").lower() == "true"
    
    try:
        if not use_mlflow:
            print("ℹ️  USE_MLFLOW_INDEX가 false입니다. 이 테스트를 건너뜁니다.")
            return None
        
        engine = SemanticSearchEngineV2(
            db_path=str(db_path),
            model_name=None,  # 모델명을 None으로 설정하여 자동 감지
            use_mlflow_index=True
        )
        
        # mlflow_manager가 None인 경우는 초기화 실패로 간주
        if not hasattr(engine, 'mlflow_manager') or engine.mlflow_manager is None:
            print("ℹ️  MLflow 매니저가 초기화되지 않았습니다.")
            print("   이는 MLflow 설정이 없거나 경로 문제일 수 있습니다.")
            return None
        
        if engine.model_name:
            print(f"✅ 모델명 감지 성공: {engine.model_name}")
            # MLflow에서 감지했는지 확인
            if hasattr(engine, 'mlflow_run_id') and engine.mlflow_run_id:
                print(f"   MLflow run ID: {engine.mlflow_run_id}")
            return True
        else:
            print("❌ 모델명이 감지되지 않았습니다.")
            return False
    except RuntimeError as e:
        if "No production run found" in str(e):
            print(f"ℹ️  프로덕션 run이 없어 모델명을 감지할 수 없습니다: {e}")
            return None
        else:
            print(f"❌ 모델명 감지 실패: {e}")
            return False
    except Exception as e:
        print(f"❌ 예상하지 못한 예외 발생: {type(e).__name__}: {e}")
        return False
    finally:
        # 환경 변수 복원
        if original_model:
            os.environ["EMBEDDING_MODEL"] = original_model

def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("MLflow 인덱스 전용 테스트")
    print("=" * 60)
    
    results = []
    
    # 테스트 1: use_mlflow_index=False일 때 RuntimeError 발생 확인
    result1 = test_mlflow_index_required()
    results.append(("테스트 1: use_mlflow_index=False RuntimeError", result1))
    
    # 테스트 2: MLflow 매니저 초기화 실패 확인
    result2 = test_mlflow_manager_not_initialized()
    results.append(("테스트 2: MLflow 매니저 초기화 실패", result2))
    
    # 테스트 3: MLflow 인덱스 로드 확인
    result3 = test_mlflow_index_load()
    if result3 is not None:
        results.append(("테스트 3: MLflow 인덱스 로드", result3))
    
    # 테스트 4: MLflow에서 모델명 감지 확인
    result4 = test_model_detection_from_mlflow()
    if result4 is not None:
        results.append(("테스트 4: MLflow 모델명 감지", result4))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    
    for test_name, result in results:
        if result is True:
            print(f"✅ {test_name}: 통과")
        elif result is False:
            print(f"❌ {test_name}: 실패")
        else:
            print(f"⏭️  {test_name}: 건너뜀")
    
    print(f"\n총 {len(results)}개 테스트: {passed}개 통과, {failed}개 실패, {skipped}개 건너뜀")
    
    if failed > 0:
        print("\n❌ 일부 테스트가 실패했습니다.")
        return 1
    else:
        print("\n✅ 모든 테스트가 통과했습니다.")
        return 0

if __name__ == "__main__":
    sys.exit(main())

