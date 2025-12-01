#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실행 전 체크 스크립트
데이터 수집을 시작하기 전에 필요한 환경과 설정을 확인합니다.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text, inspect

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
# scripts/ingest/open_law/scripts/preflight_check.py -> 프로젝트 루트
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
        # scripts/.env 파일 우선 로드
        scripts_env = _PROJECT_ROOT / "scripts" / ".env"
        if scripts_env.exists():
            load_dotenv(dotenv_path=str(scripts_env), override=True)
        # 프로젝트 루트 .env 파일 로드
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
    except ImportError:
        pass

# 공통 유틸리티 임포트
try:
    from scripts.ingest.open_law.utils import build_database_url
except ImportError:
    # 직접 구현 (fallback)
    from urllib.parse import quote_plus
    def build_database_url():
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB')
        user = os.getenv('POSTGRES_USER')
        password = os.getenv('POSTGRES_PASSWORD')
        if db and user and password:
            encoded_password = quote_plus(password)
            return f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
        return None

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment_variables():
    """환경 변수 확인"""
    print("=" * 80)
    print("1. 환경 변수 확인")
    print("=" * 80)
    
    issues = []
    
    # DATABASE_URL 확인 (직접 설정 또는 개별 변수로부터 구성)
    db_url = build_database_url()
    if not db_url:
        print("  ❌ DATABASE_URL 환경 변수가 설정되지 않았습니다.")
        print("     DATABASE_URL 또는 (POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD) 설정 필요")
        issues.append("DATABASE_URL 환경 변수 설정 필요")
    else:
        # 비밀번호 마스킹
        masked_url = db_url
        if '@' in masked_url and ':' in masked_url.split('@')[0]:
            parts = masked_url.split('@')
            if len(parts) == 2:
                user_pass = parts[0].split('://')[1] if '://' in parts[0] else parts[0]
                if ':' in user_pass:
                    user = user_pass.split(':')[0]
                    masked_url = masked_url.replace(user_pass, f"{user}:***")
        print(f"  ✅ DATABASE_URL: {masked_url[:70]}...")
    
    # LAW_OPEN_API_OC 확인
    oc = os.getenv('LAW_OPEN_API_OC')
    if not oc:
        print("  ❌ LAW_OPEN_API_OC 환경 변수가 설정되지 않았습니다.")
        issues.append("LAW_OPEN_API_OC 환경 변수 설정 필요")
    else:
        print(f"  ✅ LAW_OPEN_API_OC: {oc}")
    
    print()
    return issues


def check_database_connection(db_url: str):
    """데이터베이스 연결 확인"""
    print("=" * 80)
    print("2. 데이터베이스 연결 확인")
    print("=" * 80)
    
    issues = []
    
    try:
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            echo=False
        )
        
        with engine.connect() as conn:
            # 연결 테스트
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
            print("  ✅ 데이터베이스 연결 성공")
            
            # 스키마 확인
            inspector = inspect(engine)
            required_tables = ['statutes', 'statutes_articles', 'precedents', 'precedent_contents']
            existing_tables = inspector.get_table_names()
            
            print(f"  📋 기존 테이블: {len(existing_tables)}개")
            for table in required_tables:
                if table in existing_tables:
                    print(f"    ✅ {table} 테이블 존재")
                else:
                    print(f"    ⚠️  {table} 테이블 없음 (스키마 초기화 필요)")
                    issues.append(f"{table} 테이블이 없습니다. 스키마 초기화를 실행하세요.")
        
    except Exception as e:
        print(f"  ❌ 데이터베이스 연결 실패: {e}")
        issues.append(f"데이터베이스 연결 실패: {e}")
    
    print()
    return issues


def check_directories():
    """필요한 디렉토리 확인"""
    print("=" * 80)
    print("3. 디렉토리 확인")
    print("=" * 80)
    
    issues = []
    required_dirs = [
        'logs/open_law',
        'data/raw/open_law'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path} 존재")
        else:
            print(f"  📁 {dir_path} 생성 중...")
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {dir_path} 생성 완료")
    
    print()
    return issues


def check_python_packages():
    """필요한 Python 패키지 확인"""
    print("=" * 80)
    print("4. Python 패키지 확인")
    print("=" * 80)
    
    issues = []
    required_packages = {
        'sqlalchemy': 'SQLAlchemy',
        'psycopg2': 'psycopg2-binary',
        'requests': 'requests'
    }
    
    for package, display_name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {display_name} 설치됨")
        except ImportError:
            print(f"  ❌ {display_name} 설치되지 않음")
            issues.append(f"{display_name} 패키지 설치 필요: pip install {display_name}")
    
    print()
    return issues


def check_api_access(oc: str):
    """API 접근 테스트"""
    print("=" * 80)
    print("5. API 접근 테스트")
    print("=" * 80)
    
    issues = []
    
    try:
        from scripts.ingest.open_law.client import OpenLawClient
        
        client = OpenLawClient(oc)
        client.rate_limit_delay = 0.1  # 테스트용 빠른 요청
        
        # 간단한 API 호출 테스트
        response = client.search_statutes(query="민법", page=1, display=1)
        
        if response:
            print("  ✅ API 접근 성공")
        else:
            print("  ⚠️  API 응답이 비어있습니다")
            issues.append("API 응답이 비어있습니다. OC 값이 올바른지 확인하세요.")
    
    except Exception as e:
        print(f"  ❌ API 접근 실패: {e}")
        issues.append(f"API 접근 실패: {e}")
    
    print()
    return issues


def main():
    parser = argparse.ArgumentParser(description='실행 전 체크')
    parser.add_argument(
        '--db',
        default=os.getenv('DATABASE_URL'),
        help='PostgreSQL 데이터베이스 URL (환경변수: DATABASE_URL)'
    )
    parser.add_argument(
        '--oc',
        default=os.getenv('LAW_OPEN_API_OC'),
        help='사용자 이메일 ID (환경변수: LAW_OPEN_API_OC)'
    )
    parser.add_argument(
        '--skip-api-test',
        action='store_true',
        help='API 접근 테스트 건너뛰기'
    )
    
    args = parser.parse_args()
    
    all_issues = []
    
    # 1. 환경 변수 확인
    env_issues = check_environment_variables()
    all_issues.extend(env_issues)
    
    # 2. 디렉토리 확인
    dir_issues = check_directories()
    all_issues.extend(dir_issues)
    
    # 3. Python 패키지 확인
    pkg_issues = check_python_packages()
    all_issues.extend(pkg_issues)
    
    # 4. 데이터베이스 연결 확인
    db_url = args.db or build_database_url()
    if db_url:
        db_issues = check_database_connection(db_url)
        all_issues.extend(db_issues)
    else:
        print("=" * 80)
        print("2. 데이터베이스 연결 확인 (건너뜀)")
        print("=" * 80)
        print("  ⚠️  DATABASE_URL이 설정되지 않아 건너뜁니다.")
        print()
    
    # 5. API 접근 테스트
    if not args.skip_api_test and args.oc:
        api_issues = check_api_access(args.oc)
        all_issues.extend(api_issues)
    else:
        print("=" * 80)
        print("5. API 접근 테스트 (건너뜀)")
        print("=" * 80)
        if not args.oc:
            print("  ⚠️  LAW_OPEN_API_OC가 설정되지 않아 건너뜁니다.")
        else:
            print("  ⚠️  --skip-api-test 옵션으로 인해 건너뜁니다.")
        print()
    
    # 결과 요약
    print("=" * 80)
    print("체크 결과 요약")
    print("=" * 80)
    
    if all_issues:
        print(f"⚠️  {len(all_issues)}개의 문제가 발견되었습니다:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print()
        print("위 문제를 해결한 후 다시 실행하세요.")
        return 1
    else:
        print("✅ 모든 체크를 통과했습니다!")
        print()
        print("다음 명령으로 수집을 시작할 수 있습니다:")
        print()
        print("  python scripts/ingest/open_law/scripts/run_collection_batch.py \\")
        print("      --oc $LAW_OPEN_API_OC \\")
        print("      --db $DATABASE_URL")
        print()
        return 0


if __name__ == '__main__':
    sys.exit(main())

