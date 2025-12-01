# -*- coding: utf-8 -*-
"""
PostgreSQL 스키마 검증 스크립트
PostgreSQL 데이터베이스의 스키마가 올바르게 생성되었는지 검증합니다.
"""

import sys
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
    from lawfirm_langgraph.config.app_config import Config
    from scripts.migrations.utils.schema_validator import validate_schema
except ImportError:
    print("❌ 필수 모듈을 import할 수 없습니다.")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """메인 함수"""
    try:
        # 설정 로드
        config = Config()
        database_url = config.database_url
        
        if not database_url:
            print("❌ DATABASE_URL이 설정되지 않았습니다.")
            sys.exit(1)
        
        print(f"데이터베이스 URL: {database_url[:50]}...")
        
        # DatabaseAdapter 초기화
        adapter = DatabaseAdapter(database_url)
        print(f"데이터베이스 타입: {adapter.db_type}")
        
        if adapter.db_type != 'postgresql':
            print("⚠️  PostgreSQL이 아닙니다. 일부 검증이 건너뛰어집니다.")
        
        # 스키마 검증 (유틸리티 함수 사용)
        success, errors = validate_schema(adapter)
        
        if not success:
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

