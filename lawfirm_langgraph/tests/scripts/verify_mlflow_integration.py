# -*- coding: utf-8 -*-
"""MLflow 통합 확인 스크립트"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import mlflow

print("=" * 60)
print("MLflow 통합 확인")
print("=" * 60)

# mlflow/mlruns 확인
mlruns_path = project_root / "mlflow" / "mlruns"
tracking_uri = f"file:///{str(mlruns_path.absolute()).replace(chr(92), '/')}"

print(f"\nTracking URI: {tracking_uri}")
print(f"mlruns 경로: {mlruns_path}")
print(f"존재 여부: {mlruns_path.exists()}")

if mlruns_path.exists():
    mlflow.set_tracking_uri(tracking_uri)
    experiments = mlflow.search_experiments()
    
    print(f"\n✅ 통합 완료: {len(experiments)}개 실험\n")
    
    total_runs = 0
    for exp in experiments:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
        run_count = len(runs) if not runs.empty else 0
        total_runs += run_count
        print(f"  - {exp.name}: {run_count}개 Run")
    
    print(f"\n총 {total_runs}개 Run")
    
    print(f"\n💡 MLflow UI 실행:")
    print(f"   mlflow ui --backend-store-uri mlflow/mlruns")
    print(f"\n   또는:")
    print(f"   cd mlflow")
    print(f"   mlflow ui --backend-store-uri ./mlruns")
else:
    print(f"\n❌ {mlruns_path} 디렉토리가 없습니다")

