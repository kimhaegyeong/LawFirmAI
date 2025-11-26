#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판례 청킹 상태 검증 스크립트
민사법 청킹 완료 여부 확인
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
# scripts/ingest/open_law/verify_chunking_status.py -> parents[3] = 프로젝트 루트
_PROJECT_ROOT = _CURRENT_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드 (utils.env_loader 사용)
try:
    from utils.env_loader import ensure_env_loaded
    # ensure_env_loaded는 프로젝트 루트를 기대함
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    # 폴백: 직접 dotenv 사용
    try:
        from dotenv import load_dotenv
        # scripts/.env 파일 우선 로드
        scripts_env = _PROJECT_ROOT / "scripts" / ".env"
        if scripts_env.exists():
            load_dotenv(dotenv_path=str(scripts_env), override=True)
        # 프로젝트 루트 .env 파일 로드
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=True)
    except ImportError:
        pass

# 데이터베이스 URL 빌드
try:
    from scripts.ingest.open_law.utils import build_database_url
except ImportError:
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


def verify_chunking_status(db_url: str, domain: Optional[str] = None):
    """
    청킹 상태 검증
    
    Args:
        db_url: 데이터베이스 URL
        domain: 도메인 필터 (civil_law, criminal_law 등)
    """
    # 데이터베이스 연결
    engine = create_engine(
        db_url,
        poolclass=QueuePool,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False
    )
    
    with engine.connect() as conn:
        # 1. 전체 통계 (간단 요약)
        total_query = """
            SELECT 
                COUNT(DISTINCT p.id) as total_precedents,
                COUNT(DISTINCT pc.id) as total_contents,
                COUNT(DISTINCT CASE WHEN pch.id IS NOT NULL THEN pc.id END) as chunked_contents,
                COUNT(DISTINCT CASE WHEN pch.id IS NULL THEN pc.id END) as unchunked_contents,
                COUNT(pch.id) as total_chunks
            FROM precedents p
            LEFT JOIN precedent_contents pc ON p.id = pc.precedent_id
            LEFT JOIN precedent_chunks pch ON pc.id = pch.precedent_content_id
        """
        
        if domain:
            total_query += " WHERE p.domain = :domain"
            params = {"domain": domain}
        else:
            params = {}
        
        result = conn.execute(text(total_query), params)
        row = result.fetchone()
        
        total_precedents = row[0] or 0
        total_contents = row[1] or 0
        chunked_contents = row[2] or 0
        unchunked_contents = row[3] or 0
        total_chunks = row[4] or 0
        
        if total_contents > 0:
            chunking_rate = (chunked_contents / total_contents) * 100
        else:
            chunking_rate = 0
        
        # 간단한 요약 출력
        print("=" * 80)
        if domain:
            print(f"📊 {domain} 청킹 진행 상황")
        else:
            print("📊 전체 청킹 진행 상황")
        print("=" * 80)
        print(f"✅ 완료: {chunked_contents:,}개 / {total_contents:,}개 ({chunking_rate:.2f}%)")
        print(f"⏳ 남은 작업: {unchunked_contents:,}개")
        print(f"📦 생성된 청크: {total_chunks:,}개")
        print()
        
        # 2. 도메인별 통계 (간단 요약)
        
        domain_query = """
            SELECT 
                p.domain,
                COUNT(DISTINCT p.id) as precedents,
                COUNT(DISTINCT pc.id) as contents,
                COUNT(DISTINCT CASE WHEN pch.id IS NOT NULL THEN pc.id END) as chunked_contents,
                COUNT(pch.id) as chunks
            FROM precedents p
            LEFT JOIN precedent_contents pc ON p.id = pc.precedent_id
            LEFT JOIN precedent_chunks pch ON pc.id = pch.precedent_content_id
            GROUP BY p.domain
            ORDER BY p.domain
        """
        
        if not domain:
            # 도메인별 통계는 전체 조회 시에만 표시
            result = conn.execute(text(domain_query))
            for row in result:
                domain_name = row[0] or "NULL"
                contents = row[2] or 0
                chunked = row[3] or 0
                chunks = row[4] or 0
                
                rate = (chunked / contents * 100) if contents > 0 else 0
                status = "✅" if contents > 0 and chunked == contents else "⏳"
                
                print(f"{status} {domain_name}: {chunked:,}/{contents:,} ({rate:.1f}%) - {chunks:,}개 청크")
            print()
        
        # 3. 섹션 타입별 청킹 통계 (간단 요약)
        
        section_query = """
            SELECT 
                pc.section_type,
                COUNT(DISTINCT pc.id) as content_count,
                COUNT(pch.id) as chunk_count,
                AVG(pch.chunk_length) as avg_chunk_length,
                MIN(pch.chunk_length) as min_chunk_length,
                MAX(pch.chunk_length) as max_chunk_length,
                AVG(
                    (SELECT COUNT(*) FROM precedent_chunks pch2 
                     WHERE pch2.precedent_content_id = pc.id)
                ) as avg_chunks_per_content
            FROM precedent_contents pc
            LEFT JOIN precedent_chunks pch ON pc.id = pch.precedent_content_id
            JOIN precedents p ON pc.precedent_id = p.id
        """
        
        if domain:
            section_query += " WHERE p.domain = :domain"
            params = {"domain": domain}
        else:
            params = {}
        
        section_query += " GROUP BY pc.section_type ORDER BY pc.section_type"
        
        result = conn.execute(text(section_query), params)
        section_summary = []
        for row in result:
            section_type = row[0] or "NULL"
            content_count = row[1] or 0
            chunk_count = row[2] or 0
            avg_length = row[3] or 0
            avg_chunks = row[6] or 0
            section_summary.append((section_type, content_count, chunk_count, avg_length, avg_chunks))
        
        if section_summary:
            print("📋 섹션 타입별 요약:")
            for section_type, content_count, chunk_count, avg_length, avg_chunks in section_summary:
                print(f"  • {section_type}: {chunk_count:,}개 청크 (평균 {avg_length:.0f}자, 내용당 {avg_chunks:.1f}개)")
        print()
        
        # 4. 청킹되지 않은 데이터 확인 (간단 요약)
        
        unchunked_query = """
            SELECT 
                p.domain,
                pc.section_type,
                COUNT(*) as count
            FROM precedent_contents pc
            JOIN precedents p ON pc.precedent_id = p.id
            WHERE NOT EXISTS (
                SELECT 1 FROM precedent_chunks pch
                WHERE pch.precedent_content_id = pc.id
            )
        """
        
        if domain:
            unchunked_query += " AND p.domain = :domain"
            params = {"domain": domain}
        else:
            params = {}
        
        unchunked_query += " GROUP BY p.domain, pc.section_type ORDER BY p.domain, pc.section_type"
        
        result = conn.execute(text(unchunked_query), params)
        unchunked_rows = result.fetchall()
        
        if unchunked_rows:
            total_unchunked = sum(row[2] or 0 for row in unchunked_rows)
            print(f"⚠️  미완료: {total_unchunked:,}개 (", end="")
            details = []
            for row in unchunked_rows:
                section_type = row[1] or "NULL"
                count = row[2] or 0
                if count > 0:
                    details.append(f"{section_type} {count:,}개")
            print(", ".join(details) + ")")
        else:
            print("✅ 모든 데이터가 청킹되었습니다!")
        print()
        
        # 5. 최근 청킹된 데이터 확인 (간단 요약)
        
        recent_query = """
            SELECT 
                p.domain,
                pc.section_type,
                pch.created_at,
                pch.chunk_length
            FROM precedent_chunks pch
            JOIN precedent_contents pc ON pch.precedent_content_id = pc.id
            JOIN precedents p ON pc.precedent_id = p.id
        """
        
        if domain:
            recent_query += " WHERE p.domain = :domain"
            params = {"domain": domain}
        else:
            params = {}
        
        recent_query += " ORDER BY pch.created_at DESC LIMIT 10"
        
        result = conn.execute(text(recent_query), params)
        recent_rows = result.fetchall()
        if recent_rows:
            latest_time = recent_rows[0][2] if recent_rows else None
            if latest_time:
                print(f"🕐 최근 청킹: {latest_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(latest_time, 'strftime') else latest_time}")
        print()
        
        # 6. 최종 결론
        print("=" * 80)
        
        if domain:
            if unchunked_contents == 0 and total_contents > 0:
                print(f"✅ {domain} 도메인의 청킹이 완료되었습니다!")
                print(f"   - 총 {total_contents:,}개 내용 모두 청킹 완료")
                print(f"   - 총 {total_chunks:,}개 청크 생성")
            elif unchunked_contents > 0:
                print(f"⚠️ {domain} 도메인의 청킹이 완료되지 않았습니다.")
                print(f"   - {unchunked_contents:,}개 내용이 아직 청킹되지 않음")
                print(f"   - 청킹 완료율: {chunking_rate:.2f}%")
            else:
                print(f"ℹ️ {domain} 도메인에 데이터가 없습니다.")
        else:
            if unchunked_contents == 0 and total_contents > 0:
                print("✅ 모든 도메인의 청킹이 완료되었습니다!")
            elif unchunked_contents > 0:
                print(f"⚠️ 일부 데이터가 아직 청킹되지 않았습니다.")
                print(f"   - {unchunked_contents:,}개 내용이 아직 청킹되지 않음")
                print(f"   - 청킹 완료율: {chunking_rate:.2f}%")
            else:
                print("ℹ️ 청킹할 데이터가 없습니다.")
        print()


def main():
    parser = argparse.ArgumentParser(description='판례 청킹 상태 검증')
    parser.add_argument(
        '--db',
        default=None,
        help='PostgreSQL 데이터베이스 URL (기본값: 환경변수에서 자동 로드)'
    )
    parser.add_argument(
        '--domain',
        choices=['civil_law', 'criminal_law'],
        default=None,
        help='도메인 필터 (기본값: 전체)'
    )
    
    args = parser.parse_args()
    
    # 데이터베이스 URL 확인 (우선순위: --db 인자 > build_database_url())
    # build_database_url()은 PostgreSQL URL만 반환해야 함 (SQLite URL 무시)
    db_url = args.db
    if not db_url:
        # build_database_url()이 SQLite URL을 반환하는 경우를 방지하기 위해
        # 직접 PostgreSQL 환경 변수를 확인
        db_url = build_database_url()
        
        # build_database_url()이 SQLite URL을 반환한 경우, None으로 처리
        if db_url and not db_url.startswith('postgresql'):
            print(f"⚠️ build_database_url()이 SQLite URL을 반환했습니다. PostgreSQL 환경 변수를 확인합니다.")
            # PostgreSQL 환경 변수 직접 확인
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            db = os.getenv('POSTGRES_DB')
            user = os.getenv('POSTGRES_USER')
            password = os.getenv('POSTGRES_PASSWORD')
            if db and user and password:
                from urllib.parse import quote_plus
                encoded_password = quote_plus(password)
                db_url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
                print(f"✅ PostgreSQL URL 구성: postgresql://{user}:***@{host}:{port}/{db}")
            else:
                db_url = None
                print(f"❌ PostgreSQL 환경 변수가 설정되지 않았습니다.")
                print(f"   POSTGRES_DB: {db or 'None'}")
                print(f"   POSTGRES_USER: {user or 'None'}")
                print(f"   POSTGRES_PASSWORD: {'설정됨' if password else 'None'}")
    
    if not db_url:
        print("❌ 오류: 데이터베이스 연결 정보가 필요합니다.")
        print()
        print("다음 중 하나의 방법으로 데이터베이스 URL을 제공하세요:")
        print("1. --db 인자 사용:")
        print("   python verify_chunking_status.py --db postgresql://user:pass@host:5432/dbname --domain civil_law")
        print()
        print("2. 환경 변수 설정 (.env 파일 또는 환경 변수):")
        print("   - DATABASE_URL=postgresql://user:pass@host:5432/dbname")
        print("   또는")
        print("   - POSTGRES_HOST=localhost")
        print("   - POSTGRES_PORT=5432")
        print("   - POSTGRES_DB=lawfirm")
        print("   - POSTGRES_USER=lawfirm_user")
        print("   - POSTGRES_PASSWORD=lawfirm_password")
        print()
        print("현재 환경 변수 상태:")
        print(f"  DATABASE_URL: {os.getenv('DATABASE_URL', '설정되지 않음')}")
        print(f"  POSTGRES_HOST: {os.getenv('POSTGRES_HOST', '설정되지 않음')}")
        print(f"  POSTGRES_PORT: {os.getenv('POSTGRES_PORT', '설정되지 않음')}")
        print(f"  POSTGRES_DB: {os.getenv('POSTGRES_DB', '설정되지 않음')}")
        print(f"  POSTGRES_USER: {os.getenv('POSTGRES_USER', '설정되지 않음')}")
        print(f"  POSTGRES_PASSWORD: {'설정됨' if os.getenv('POSTGRES_PASSWORD') else '설정되지 않음'}")
        print()
        print("💡 .env 파일 위치 확인:")
        root_env = _PROJECT_ROOT / ".env"
        scripts_env = _PROJECT_ROOT / "scripts" / ".env"
        print(f"  프로젝트 루트 .env: {root_env} ({'존재' if root_env.exists() else '없음'})")
        print(f"  scripts/.env: {scripts_env} ({'존재' if scripts_env.exists() else '없음'})")
        return
    
    # 디버깅: 데이터베이스 URL 확인
    if db_url:
        print(f"🔍 데이터베이스 URL: {db_url[:50]}..." if len(db_url) > 50 else f"🔍 데이터베이스 URL: {db_url}")
        print()
    
    try:
        verify_chunking_status(
            db_url=db_url,
            domain=args.domain
        )
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

