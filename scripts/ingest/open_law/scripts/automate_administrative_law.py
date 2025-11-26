#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
행정법 법령 수집 및 임베딩 자동화 스크립트
수집 → 검증 → 임베딩 생성까지 전체 프로세스를 자동으로 실행합니다.
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    try:
        from dotenv import load_dotenv
        scripts_env = _PROJECT_ROOT / "scripts" / ".env"
        if scripts_env.exists():
            load_dotenv(dotenv_path=str(scripts_env), override=True)
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
    except ImportError:
        pass

from scripts.ingest.open_law.utils import build_database_url

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/open_law/administrative_law_automation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdministrativeLawAutomation:
    """행정법 법령 수집 및 임베딩 자동화 클래스"""
    
    def __init__(self, oc: str, db_url: str, project_root: Path):
        self.oc = oc
        self.db_url = db_url
        self.project_root = project_root
        self.start_time = datetime.now()
        self.steps_completed = []
        self.steps_failed = []
    
    def run_script(self, script_name: str, args: list, description: str = "") -> bool:
        """스크립트 실행"""
        if description:
            logger.info(f"▶ {description}")
        else:
            logger.info(f"▶ 실행: {script_name}")
        
        script_path = self.project_root / "scripts" / "ingest" / "open_law" / "scripts" / script_name
        if not script_path.exists():
            # embedding 스크립트는 다른 경로
            script_path = self.project_root / "scripts" / "ingest" / "open_law" / "embedding" / script_name
            if not script_path.exists():
                logger.error(f"스크립트를 찾을 수 없습니다: {script_name}")
                return False
        
        cmd = [sys.executable, str(script_path)] + args
        
        logger.debug(f"명령어: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            logger.info(f"✅ {description or script_name} 완료")
            if result.stdout:
                logger.debug(f"출력: {result.stdout[-500:]}")  # 마지막 500자만
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {description or script_name} 실패: {e}")
            if e.stdout:
                logger.error(f"출력: {e.stdout[-1000:]}")
            if e.stderr:
                logger.error(f"에러: {e.stderr[-1000:]}")
            return False
    
    def step1_collect_list(self) -> bool:
        """1단계: 법령 목록 수집"""
        logger.info("=" * 80)
        logger.info("1단계: 행정법 법령 목록 수집")
        logger.info("=" * 80)
        
        success = self.run_script(
            "collect_administrative_statutes.py",
            [
                "--oc", self.oc,
                "--phase", "list",
                "--output", "data/raw/open_law/administrative_statutes_list.json"
            ],
            "법령 목록 수집"
        )
        
        if success:
            self.steps_completed.append("법령 목록 수집")
        else:
            self.steps_failed.append("법령 목록 수집")
        
        return success
    
    def step2_collect_content(self) -> bool:
        """2단계: 법령 본문 및 조문 수집"""
        logger.info("=" * 80)
        logger.info("2단계: 행정법 법령 본문 및 조문 수집")
        logger.info("=" * 80)
        
        input_file = self.project_root / "data" / "raw" / "open_law" / "administrative_statutes_list.json"
        if not input_file.exists():
            logger.error(f"법령 목록 파일이 없습니다: {input_file}")
            self.steps_failed.append("법령 본문 수집 (목록 파일 없음)")
            return False
        
        success = self.run_script(
            "collect_administrative_statutes.py",
            [
                "--oc", self.oc,
                "--phase", "content",
                "--input", str(input_file.relative_to(self.project_root)),
                "--db", self.db_url,
                "--rate-limit", "0.5"
            ],
            "법령 본문 및 조문 수집"
        )
        
        if success:
            self.steps_completed.append("법령 본문 수집")
        else:
            self.steps_failed.append("법령 본문 수집")
        
        return success
    
    def step3_validate_data(self) -> bool:
        """3단계: 데이터 검증"""
        logger.info("=" * 80)
        logger.info("3단계: 데이터 검증")
        logger.info("=" * 80)
        
        validate_script = self.project_root / "scripts" / "ingest" / "open_law" / "scripts" / "validate_data.py"
        if not validate_script.exists():
            logger.warning("검증 스크립트가 없어 건너뜁니다.")
            return True
        
        success = self.run_script(
            "validate_data.py",
            [
                "--domain", "administrative_law",
                "--db", self.db_url
            ],
            "데이터 검증"
        )
        
        if success:
            self.steps_completed.append("데이터 검증")
        else:
            self.steps_failed.append("데이터 검증")
        
        return success
    
    def step4_generate_embeddings(self, method: str = 'pgvector', model: str = None, 
                                  batch_size: int = 100, chunking_strategy: str = 'article') -> bool:
        """4단계: 임베딩 생성"""
        logger.info("=" * 80)
        logger.info("4단계: 행정법 법령 임베딩 생성")
        logger.info("=" * 80)
        
        args = [
            "--db", self.db_url,
            "--method", method,
            "--domain", "administrative_law",
            "--batch-size", str(batch_size),
            "--chunking-strategy", chunking_strategy
        ]
        
        if model:
            args.extend(["--model", model])
        
        if method == 'faiss':
            args.extend([
                "--output-dir", 
                str(self.project_root / "data" / "embeddings" / "open_law_postgresql" / "administrative_law")
            ])
        
        success = self.run_script(
            "generate_statute_embeddings.py",
            args,
            f"임베딩 생성 ({method})"
        )
        
        if success:
            self.steps_completed.append(f"임베딩 생성 ({method})")
        else:
            self.steps_failed.append(f"임베딩 생성 ({method})")
        
        return success
    
    def run_full_pipeline(self, skip_collection: bool = False, skip_validation: bool = False,
                          embedding_method: str = 'pgvector', embedding_model: str = None,
                          batch_size: int = 100, chunking_strategy: str = 'article') -> bool:
        """전체 파이프라인 실행"""
        logger.info("=" * 80)
        logger.info("행정법 법령 수집 및 임베딩 자동화 시작")
        logger.info(f"시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # 1. 법령 목록 수집
        if not skip_collection:
            if not self.step1_collect_list():
                logger.error("법령 목록 수집 실패로 인해 중단합니다.")
                return False
            
            # 2. 법령 본문 수집
            if not self.step2_collect_content():
                logger.error("법령 본문 수집 실패로 인해 중단합니다.")
                return False
        
        # 3. 데이터 검증
        if not skip_validation:
            if not self.step3_validate_data():
                logger.warning("데이터 검증에 문제가 있지만 계속 진행합니다.")
        
        # 4. 임베딩 생성
        if not self.step4_generate_embeddings(
            method=embedding_method,
            model=embedding_model,
            batch_size=batch_size,
            chunking_strategy=chunking_strategy
        ):
            logger.error("임베딩 생성 실패")
            return False
        
        # 완료
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        logger.info("=" * 80)
        logger.info("✅ 전체 자동화 프로세스 완료")
        logger.info(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"소요 시간: {duration}")
        logger.info(f"완료된 단계: {', '.join(self.steps_completed)}")
        if self.steps_failed:
            logger.warning(f"실패한 단계: {', '.join(self.steps_failed)}")
        logger.info("=" * 80)
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='행정법 법령 수집 및 임베딩 자동화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 프로세스 실행 (수집 + 검증 + 임베딩)
  python automate_administrative_law.py --oc YOUR_OC

  # 수집만 실행
  python automate_administrative_law.py --oc YOUR_OC --skip-embedding

  # 임베딩만 실행 (이미 수집된 데이터 사용)
  python automate_administrative_law.py --oc YOUR_OC --skip-collection

  # FAISS 임베딩 생성
  python automate_administrative_law.py --oc YOUR_OC --skip-collection --embedding-method faiss
        """
    )
    parser.add_argument(
        '--oc',
        default=os.getenv('LAW_OPEN_API_OC'),
        help='사용자 이메일 ID (환경변수: LAW_OPEN_API_OC)'
    )
    parser.add_argument(
        '--db',
        default=build_database_url(),
        help='PostgreSQL 데이터베이스 URL (환경변수: DATABASE_URL)'
    )
    parser.add_argument(
        '--skip-collection',
        action='store_true',
        help='수집 단계 건너뛰기 (이미 수집된 데이터 사용)'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='검증 단계 건너뛰기'
    )
    parser.add_argument(
        '--skip-embedding',
        action='store_true',
        help='임베딩 생성 건너뛰기'
    )
    parser.add_argument(
        '--embedding-method',
        choices=['pgvector', 'faiss', 'both'],
        default='pgvector',
        help='임베딩 생성 방법 (기본값: pgvector)'
    )
    parser.add_argument(
        '--embedding-model',
        default='woong0322/ko-legal-sbert-finetuned',
        help='임베딩 모델 이름 (기본값: woong0322/ko-legal-sbert-finetuned)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='배치 크기 (기본값: 100)'
    )
    parser.add_argument(
        '--chunking-strategy',
        default='article',
        help='청킹 전략 (기본값: article)'
    )
    parser.add_argument(
        '--step',
        choices=['collect_list', 'collect_content', 'validate', 'embedding'],
        help='특정 단계만 실행'
    )
    
    args = parser.parse_args()
    
    # 필수 인자 체크
    if not args.oc:
        logger.error("--oc 인자 또는 LAW_OPEN_API_OC 환경변수가 필요합니다.")
        return 1
    
    if not args.db:
        logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
        return 1
    
    # 로그 디렉토리 생성
    Path('logs/open_law').mkdir(parents=True, exist_ok=True)
    Path('data/raw/open_law').mkdir(parents=True, exist_ok=True)
    
    # 자동화 실행기 생성
    project_root = Path(__file__).resolve().parents[4]
    automation = AdministrativeLawAutomation(args.oc, args.db, project_root)
    
    # 특정 단계만 실행
    if args.step:
        if args.step == 'collect_list':
            automation.step1_collect_list()
        elif args.step == 'collect_content':
            automation.step2_collect_content()
        elif args.step == 'validate':
            automation.step3_validate_data()
        elif args.step == 'embedding':
            automation.step4_generate_embeddings(
                method=args.embedding_method,
                model=args.embedding_model,
                batch_size=args.batch_size,
                chunking_strategy=args.chunking_strategy
            )
    else:
        # 전체 파이프라인 실행
        if args.skip_embedding:
            # 수집만 실행
            if not args.skip_collection:
                automation.step1_collect_list()
                automation.step2_collect_content()
            if not args.skip_validation:
                automation.step3_validate_data()
        else:
            # 전체 실행
            automation.run_full_pipeline(
                skip_collection=args.skip_collection,
                skip_validation=args.skip_validation,
                embedding_method=args.embedding_method,
                embedding_model=args.embedding_model,
                batch_size=args.batch_size,
                chunking_strategy=args.chunking_strategy
            )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

