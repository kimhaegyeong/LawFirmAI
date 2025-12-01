#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLflow에서 최근 사용된 벡터 임베딩 확인 스크립트
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드
try:
    from dotenv import load_dotenv
    root_env = _PROJECT_ROOT / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=str(root_env), override=True)
    scripts_env = _PROJECT_ROOT / "scripts" / ".env"
    if scripts_env.exists():
        load_dotenv(dotenv_path=str(scripts_env), override=True)
except ImportError:
    pass

try:
    from scripts.rag.mlflow_manager import MLflowFAISSManager
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
    from rag.mlflow_manager import MLflowFAISSManager

import mlflow
from mlflow.tracking import MlflowClient


def format_timestamp(timestamp) -> str:
    """타임스탬프를 읽기 쉬운 형식으로 변환"""
    try:
        # pandas Timestamp인 경우
        if hasattr(timestamp, 'timestamp'):
            dt = datetime.fromtimestamp(timestamp.timestamp())
        # 밀리초 정수인 경우
        elif isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp / 1000)
        # datetime 객체인 경우
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            return str(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return str(timestamp)


def get_version_info(mlflow_manager: MLflowFAISSManager, run_id: str) -> Optional[Dict[str, Any]]:
    """version_info.json 로드"""
    try:
        # 로컬 파일 시스템에서 먼저 시도
        if hasattr(mlflow_manager, 'load_version_info_from_local'):
            version_info = mlflow_manager.load_version_info_from_local(run_id)
            if version_info:
                return version_info
        
        # MLflow에서 다운로드
        version_info = mlflow.artifacts.load_dict(f"runs:/{run_id}/version_info.json")
        return version_info
    except Exception as e:
        return None


def print_run_info(run_data: Dict[str, Any], index: int = None):
    """Run 정보 출력"""
    run_id = run_data.get("run_id", "N/A")
    version = run_data.get("version", "N/A")
    status = run_data.get("status", "N/A")
    start_time = run_data.get("start_time", 0)
    
    prefix = f"[{index}] " if index is not None else ""
    print(f"\n{prefix}{'='*80}")
    print(f"Run ID: {run_id}")
    print(f"Version: {version}")
    print(f"Status: {status}")
    if start_time:
        print(f"Created: {format_timestamp(start_time)}")
    
    params = run_data.get("params", {})
    metrics = run_data.get("metrics", {})
    
    if params:
        print(f"\nParameters:")
        for key, value in params.items():
            if key.startswith("params."):
                print(f"  {key.replace('params.', '')}: {value}")
    
    if metrics:
        print(f"\nMetrics:")
        for key, value in metrics.items():
            if key.startswith("metrics."):
                print(f"  {key.replace('metrics.', '')}: {value}")


def print_embedding_info(version_info: Dict[str, Any]):
    """임베딩 정보 출력"""
    embedding_config = version_info.get("embedding_config", {})
    chunking_config = version_info.get("chunking_config", {})
    
    print(f"\n📊 Embedding Configuration:")
    if embedding_config:
        model = embedding_config.get("model", "N/A")
        dimension = embedding_config.get("dimension", "N/A")
        print(f"  Model: {model}")
        print(f"  Dimension: {dimension}")
    else:
        print("  No embedding config found")
    
    print(f"\n📝 Chunking Configuration:")
    if chunking_config:
        chunk_size = chunking_config.get("chunk_size", "N/A")
        chunk_overlap = chunking_config.get("chunk_overlap", "N/A")
        print(f"  Chunk Size: {chunk_size}")
        print(f"  Chunk Overlap: {chunk_overlap}")
    else:
        print("  No chunking config found")
    
    document_count = version_info.get("document_count", 0)
    total_chunks = version_info.get("total_chunks", 0)
    print(f"\n📈 Statistics:")
    print(f"  Documents: {document_count:,}")
    print(f"  Total Chunks: {total_chunks:,}")


def main():
    """메인 함수"""
    print("=" * 80)
    print("MLflow 벡터 임베딩 조회")
    print("=" * 80)
    
    try:
        # MLflow 매니저 초기화
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "faiss_index_versions")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        
        print(f"\n🔧 Configuration:")
        print(f"  Experiment: {experiment_name}")
        print(f"  Tracking URI: {tracking_uri or 'Default (file://)'}")
        
        mlflow_manager = MLflowFAISSManager(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri
        )
        
        # 프로덕션 run 확인
        print(f"\n{'='*80}")
        print("🔍 프로덕션 Run 확인")
        print("=" * 80)
        production_run_id = mlflow_manager.get_production_run()
        if production_run_id:
            print(f"✅ 프로덕션 Run ID: {production_run_id}")
            
            # 프로덕션 run 상세 정보
            runs = mlflow_manager.list_runs(
                filter_string=f"tags.status='production_ready'",
                max_results=1
            )
            if runs:
                run_data = runs[0]
                print_run_info(run_data)
                
                # version_info 로드
                version_info = get_version_info(mlflow_manager, production_run_id)
                if version_info:
                    print_embedding_info(version_info)
        else:
            print("⚠️  프로덕션 run을 찾을 수 없습니다.")
        
        # 최근 runs 조회
        print(f"\n{'='*80}")
        print("📋 최근 Runs (최대 10개)")
        print("=" * 80)
        
        recent_runs = mlflow_manager.list_runs(max_results=10)
        
        if not recent_runs:
            print("❌ MLflow에 run이 없습니다.")
            return
        
        for idx, run_data in enumerate(recent_runs, 1):
            run_id = run_data.get("run_id")
            version = run_data.get("version", "N/A")
            status = run_data.get("status", "N/A")
            start_time = run_data.get("start_time", 0)
            
            print(f"\n[{idx}] Run ID: {run_id}")
            print(f"    Version: {version}")
            print(f"    Status: {status}")
            if start_time:
                print(f"    Created: {format_timestamp(start_time)}")
            
            # version_info 로드 시도 (에러 무시)
            try:
                version_info = get_version_info(mlflow_manager, run_id)
                if version_info:
                    embedding_config = version_info.get("embedding_config", {})
                    model = embedding_config.get("model", "N/A")
                    dimension = embedding_config.get("dimension", "N/A")
                    total_chunks = version_info.get("total_chunks", 0)
                    
                    print(f"    Model: {model}")
                    print(f"    Dimension: {dimension}")
                    print(f"    Total Chunks: {total_chunks:,}")
                else:
                    print(f"    ⚠️  version_info 없음")
            except Exception as e:
                print(f"    ⚠️  version_info 로드 실패: {str(e)[:50]}")
        
        # 환경 변수에서 지정된 run_id 확인
        env_run_id = os.getenv("MLFLOW_RUN_ID")
        if env_run_id:
            print(f"\n{'='*80}")
            print(f"🔧 환경 변수 MLFLOW_RUN_ID: {env_run_id}")
            print("=" * 80)
            
            try:
                version_info = get_version_info(mlflow_manager, env_run_id)
                if version_info:
                    print_embedding_info(version_info)
                else:
                    print(f"⚠️  Run ID {env_run_id}의 version_info를 찾을 수 없습니다.")
            except Exception as e:
                print(f"❌ Run ID {env_run_id} 조회 실패: {e}")
        
        print(f"\n{'='*80}")
        print("✅ 조회 완료")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

