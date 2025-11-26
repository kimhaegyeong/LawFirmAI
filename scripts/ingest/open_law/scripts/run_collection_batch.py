#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open Law API 데이터 수집 배치 실행 스크립트
전체 수집 프로세스를 자동화하여 실행합니다.
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
# scripts/ingest/open_law/scripts/run_collection_batch.py -> 프로젝트 루트
_PROJECT_ROOT = _CURRENT_FILE.parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드 (utils/env_loader.py 사용)
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

# 공통 유틸리티 임포트
from scripts.ingest.open_law.utils import build_database_url

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/open_law/batch_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CollectionBatchRunner:
    """수집 배치 실행기"""
    
    def __init__(self, oc: str, db_url: str, project_root: Path):
        self.oc = oc
        self.db_url = db_url
        self.project_root = project_root
        self.start_time = datetime.now()
    
    def run_script(self, script_name: str, args: list) -> bool:
        """스크립트 실행"""
        script_path = self.project_root / "scripts" / "ingest" / "open_law" / "scripts" / script_name
        cmd = [sys.executable, str(script_path)] + args
        
        logger.info(f"실행: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            logger.info(f"✅ {script_name} 실행 완료")
            if result.stdout:
                logger.debug(f"출력: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {script_name} 실행 실패: {e}")
            if e.stdout:
                logger.error(f"출력: {e.stdout}")
            if e.stderr:
                logger.error(f"에러: {e.stderr}")
            return False
    
    def init_schema(self) -> bool:
        """스키마 초기화"""
        logger.info("=" * 80)
        logger.info("1단계: PostgreSQL 스키마 초기화")
        logger.info("=" * 80)
        
        init_script = self.project_root / "scripts" / "migrations" / "init_open_law_schema.py"
        cmd = [sys.executable, str(init_script), "--db", self.db_url]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            logger.info("✅ 스키마 초기화 완료")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 스키마 초기화 실패: {e}")
            if e.stderr:
                logger.error(f"에러: {e.stderr}")
            return False
    
    def collect_civil_statutes(self) -> bool:
        """민사법 현행법령 수집"""
        logger.info("=" * 80)
        logger.info("2단계: 민사법 현행법령 수집")
        logger.info("=" * 80)
        
        # 목록 수집
        logger.info("2-1. 법령 목록 수집")
        if not self.run_script("collect_civil_statutes.py", [
            "--oc", self.oc,
            "--phase", "list",
            "--output", "data/raw/open_law/civil_statutes_list.json"
        ]):
            return False
        
        # 본문 수집
        logger.info("2-2. 법령 본문 및 조문 수집")
        if not self.run_script("collect_civil_statutes.py", [
            "--oc", self.oc,
            "--phase", "content",
            "--input", "data/raw/open_law/civil_statutes_list.json",
            "--db", self.db_url
        ]):
            return False
        
        return True
    
    def collect_criminal_statutes(self) -> bool:
        """형법 현행법령 수집"""
        logger.info("=" * 80)
        logger.info("3단계: 형법 현행법령 수집")
        logger.info("=" * 80)
        
        # 목록 수집
        logger.info("3-1. 법령 목록 수집")
        if not self.run_script("collect_criminal_statutes.py", [
            "--oc", self.oc,
            "--phase", "list",
            "--output", "data/raw/open_law/criminal_statutes_list.json"
        ]):
            return False
        
        # 본문 수집
        logger.info("3-2. 법령 본문 및 조문 수집")
        if not self.run_script("collect_criminal_statutes.py", [
            "--oc", self.oc,
            "--phase", "content",
            "--input", "data/raw/open_law/criminal_statutes_list.json",
            "--db", self.db_url
        ]):
            return False
        
        return True
    
    def collect_civil_precedents(self) -> bool:
        """민사법 판례 수집"""
        logger.info("=" * 80)
        logger.info("4단계: 민사법 판례 수집")
        logger.info("=" * 80)
        
        # 목록 수집
        logger.info("4-1. 판례 목록 수집")
        if not self.run_script("collect_civil_precedents.py", [
            "--oc", self.oc,
            "--phase", "list",
            "--max-pages", "200",
            "--output", "data/raw/open_law/civil_precedents_list.json"
        ]):
            return False
        
        # 본문 수집
        logger.info("4-2. 판례 본문 수집")
        if not self.run_script("collect_civil_precedents.py", [
            "--oc", self.oc,
            "--phase", "content",
            "--input", "data/raw/open_law/civil_precedents_list.json",
            "--db", self.db_url
        ]):
            return False
        
        return True
    
    def collect_criminal_precedents(self) -> bool:
        """형법 판례 수집"""
        logger.info("=" * 80)
        logger.info("5단계: 형법 판례 수집")
        logger.info("=" * 80)
        
        # 목록 수집
        logger.info("5-1. 판례 목록 수집")
        if not self.run_script("collect_criminal_precedents.py", [
            "--oc", self.oc,
            "--phase", "list",
            "--max-pages", "100",
            "--output", "data/raw/open_law/criminal_precedents_list.json"
        ]):
            return False
        
        # 본문 수집
        logger.info("5-2. 판례 본문 수집")
        if not self.run_script("collect_criminal_precedents.py", [
            "--oc", self.oc,
            "--phase", "content",
            "--input", "data/raw/open_law/criminal_precedents_list.json",
            "--db", self.db_url
        ]):
            return False
        
        return True
    
    def collect_administrative_statutes(self) -> bool:
        """행정법 현행법령 수집"""
        logger.info("=" * 80)
        logger.info("6단계: 행정법 현행법령 수집")
        logger.info("=" * 80)
        
        # 목록 수집
        logger.info("6-1. 법령 목록 수집")
        if not self.run_script("collect_administrative_statutes.py", [
            "--oc", self.oc,
            "--phase", "list",
            "--output", "data/raw/open_law/administrative_statutes_list.json"
        ]):
            return False
        
        # 본문 수집
        logger.info("6-2. 법령 본문 및 조문 수집")
        if not self.run_script("collect_administrative_statutes.py", [
            "--oc", self.oc,
            "--phase", "content",
            "--input", "data/raw/open_law/administrative_statutes_list.json",
            "--db", self.db_url
        ]):
            return False
        
        return True
    
    def run_all(self, skip_schema: bool = False):
        """전체 수집 프로세스 실행"""
        logger.info("=" * 80)
        logger.info("Open Law API 데이터 수집 배치 실행 시작")
        logger.info(f"시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        steps = []
        
        # 1. 스키마 초기화
        if not skip_schema:
            if not self.init_schema():
                logger.error("스키마 초기화 실패로 인해 중단합니다.")
                return False
            steps.append("스키마 초기화")
        
        # 2. 민사법 현행법령
        if not self.collect_civil_statutes():
            logger.error("민사법 현행법령 수집 실패")
            return False
        steps.append("민사법 현행법령 수집")
        
        # 3. 형법 현행법령
        if not self.collect_criminal_statutes():
            logger.error("형법 현행법령 수집 실패")
            return False
        steps.append("형법 현행법령 수집")
        
        # 4. 민사법 판례
        if not self.collect_civil_precedents():
            logger.error("민사법 판례 수집 실패")
            return False
        steps.append("민사법 판례 수집")
        
        # 5. 형법 판례
        if not self.collect_criminal_precedents():
            logger.error("형법 판례 수집 실패")
            return False
        steps.append("형법 판례 수집")
        
        # 6. 행정법 현행법령
        if not self.collect_administrative_statutes():
            logger.error("행정법 현행법령 수집 실패")
            return False
        steps.append("행정법 현행법령 수집")
        
        # 완료
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        logger.info("=" * 80)
        logger.info("✅ 전체 수집 프로세스 완료")
        logger.info(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"소요 시간: {duration}")
        logger.info(f"완료된 단계: {', '.join(steps)}")
        logger.info("=" * 80)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Open Law API 데이터 수집 배치 실행')
    parser.add_argument(
        '--oc',
        default=os.getenv('LAW_OPEN_API_OC'),
        help='사용자 이메일 ID (환경변수: LAW_OPEN_API_OC)'
    )
    parser.add_argument(
        '--db',
        default=build_database_url(),
        help='PostgreSQL 데이터베이스 URL (환경변수: DATABASE_URL 또는 개별 POSTGRES_* 변수)'
    )
    parser.add_argument(
        '--skip-schema',
        action='store_true',
        help='스키마 초기화 건너뛰기'
    )
    parser.add_argument(
        '--step',
        choices=['schema', 'civil_statutes', 'criminal_statutes', 
                 'civil_precedents', 'criminal_precedents', 'administrative_statutes'],
        help='특정 단계만 실행'
    )
    
    args = parser.parse_args()
    
    # 필수 인자 체크
    if not args.oc:
        logger.error("--oc 인자 또는 LAW_OPEN_API_OC 환경변수가 필요합니다.")
        return
    
    if not args.db:
        logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
        return
    
    # 로그 디렉토리 생성
    Path('logs/open_law').mkdir(parents=True, exist_ok=True)
    Path('data/raw/open_law').mkdir(parents=True, exist_ok=True)
    
    # 배치 실행기 생성
    project_root = Path(__file__).resolve().parents[3]
    runner = CollectionBatchRunner(args.oc, args.db, project_root)
    
    # 특정 단계만 실행
    if args.step:
        if args.step == 'schema':
            runner.init_schema()
        elif args.step == 'civil_statutes':
            runner.collect_civil_statutes()
        elif args.step == 'criminal_statutes':
            runner.collect_criminal_statutes()
        elif args.step == 'civil_precedents':
            runner.collect_civil_precedents()
        elif args.step == 'criminal_precedents':
            runner.collect_criminal_precedents()
        elif args.step == 'administrative_statutes':
            runner.collect_administrative_statutes()
    else:
        # 전체 실행
        runner.run_all(skip_schema=args.skip_schema)


if __name__ == '__main__':
    main()

