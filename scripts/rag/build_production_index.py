"""
프로덕션 인덱스 빌드 스크립트

최적 파라미터를 사용하여 프로덕션 버전의 FAISS 인덱스를 빌드하고 MLflow에 저장합니다.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "utils"))
sys.path.insert(0, str(project_root / "scripts" / "rag"))

from build_index import build_and_save_index
from mlflow_manager import MLflowFAISSManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_optimized_params(params_path: str) -> Dict[str, Any]:
    """최적 파라미터 로드"""
    params_file = Path(params_path)
    if not params_file.exists():
        raise FileNotFoundError(f"최적 파라미터 파일을 찾을 수 없습니다: {params_path}")
    
    with open(params_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get("search_parameters", {})


def tag_as_production(run_id: str, mlflow_manager: MLflowFAISSManager):
    """MLflow run을 프로덕션 버전으로 태깅"""
    try:
        import mlflow
        
        mlflow.end_run()
        
        client = mlflow.tracking.MlflowClient()
        client.set_tag(run_id, "environment", "production")
        client.set_tag(run_id, "status", "production_ready")
        client.set_tag(run_id, "optimized", "true")
        client.set_tag(run_id, "production_date", datetime.now().isoformat())
        
        logger.info(f"프로덕션 태그 추가 완료: {run_id}")
    except Exception as e:
        logger.warning(f"프로덕션 태그 추가 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="프로덕션 인덱스 빌드")
    parser.add_argument(
        "--version-name",
        default=None,
        help="버전 이름 (None이면 자동 생성: production-YYYYMMDD-HHMMSS)"
    )
    parser.add_argument(
        "--embedding-version-id",
        type=int,
        required=True,
        help="임베딩 버전 ID"
    )
    parser.add_argument(
        "--chunking-strategy",
        required=True,
        help="청킹 전략"
    )
    parser.add_argument(
        "--db-path",
        default="data/lawfirm_v2.db",
        help="데이터베이스 경로"
    )
    parser.add_argument(
        "--model-name",
        default="jhgan/ko-sroberta-multitask",
        help="임베딩 모델명"
    )
    parser.add_argument(
        "--optimized-params-path",
        default="data/ml_config/optimized_search_params.json",
        help="최적 파라미터 파일 경로"
    )
    parser.add_argument(
        "--tag-production",
        action="store_true",
        default=True,
        help="프로덕션 태그 추가"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("프로덕션 인덱스 빌드 시작")
    logger.info("=" * 80)
    
    try:
        optimized_params = load_optimized_params(args.optimized_params_path)
        logger.info(f"최적 파라미터 로드 완료: {optimized_params}")
        
        if args.version_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            version_name = f"production-{timestamp}"
        else:
            version_name = args.version_name
        
        logger.info(f"버전 이름: {version_name}")
        
        success = build_and_save_index(
            version_name=version_name,
            embedding_version_id=args.embedding_version_id,
            chunking_strategy=args.chunking_strategy,
            db_path=args.db_path,
            use_mlflow=True,
            local_backup=True,
            model_name=args.model_name
        )
        
        if not success:
            logger.error("인덱스 빌드 실패")
            sys.exit(1)
        
        if args.tag_production:
            mlflow_manager = MLflowFAISSManager()
            runs = mlflow_manager.list_runs()
            if runs:
                latest_run = runs[0]
                run_id = latest_run.get("run_id")
                if run_id:
                    tag_as_production(run_id, mlflow_manager)
        
        logger.info("=" * 80)
        logger.info("프로덕션 인덱스 빌드 완료")
        logger.info("=" * 80)
        logger.info(f"버전: {version_name}")
        logger.info(f"최적 파라미터: {args.optimized_params_path}")
        logger.info("다음 단계: 프로덕션 환경에서 이 인덱스를 사용하세요")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

