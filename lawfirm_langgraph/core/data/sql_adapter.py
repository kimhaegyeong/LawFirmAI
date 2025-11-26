# -*- coding: utf-8 -*-
"""
SQL Adapter
SQLite와 PostgreSQL 간 SQL 문법 변환 유틸리티
"""

import re
from typing import Optional

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

logger = get_logger(__name__)


class SQLAdapter:
    """SQL 문법 변환 어댑터"""
    
    @staticmethod
    def convert_parameter_placeholder(sql: str, db_type: str) -> str:
        """
        파라미터 플레이스홀더 변환
        
        Args:
            sql: SQL 쿼리
            db_type: 'sqlite' 또는 'postgresql'
        
        Returns:
            변환된 SQL 쿼리
        """
        if db_type == 'postgresql':
            # SQLite ? → PostgreSQL %s
            # 단, 문자열 내부의 ?는 변환하지 않음
            def replace_placeholder(match):
                return match.group(0).replace('?', '%s')
            
            # 문자열 리터럴을 제외한 ?를 %s로 변환
            parts = []
            in_string = False
            string_char = None
            i = 0
            
            while i < len(sql):
                char = sql[i]
                
                if char in ("'", '"') and (i == 0 or sql[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                    parts.append(char)
                elif char == '?' and not in_string:
                    parts.append('%s')
                else:
                    parts.append(char)
                i += 1
            
            return ''.join(parts)
        elif db_type == 'sqlite':
            # PostgreSQL %s → SQLite ?
            return sql.replace('%s', '?')
        else:
            return sql
    
    @staticmethod
    def convert_table_check_query(table_name: str, db_type: str) -> tuple[str, tuple]:
        """
        테이블 존재 확인 쿼리 변환
        
        Args:
            table_name: 테이블명
            db_type: 'sqlite' 또는 'postgresql'
        
        Returns:
            (쿼리, 파라미터) 튜플
        """
        if db_type == 'postgresql':
            query = (
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname='public' AND tablename=%s"
            )
            params = (table_name,)
        else:  # sqlite
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            params = (table_name,)
        
        return query, params
    
    @staticmethod
    def convert_group_concat(sql: str, db_type: str) -> str:
        """
        GROUP_CONCAT → STRING_AGG 변환
        
        Args:
            sql: SQL 쿼리
            db_type: 'sqlite' 또는 'postgresql'
        
        Returns:
            변환된 SQL 쿼리
        """
        if db_type == 'postgresql':
            # GROUP_CONCAT(column, separator) → STRING_AGG(column, separator)
            pattern = r'GROUP_CONCAT\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
            
            def replace_group_concat(match):
                column = match.group(1).strip()
                separator = match.group(2).strip()
                # 문자열 리터럴 처리
                if separator.startswith("'") and separator.endswith("'"):
                    # 이스케이프 처리
                    sep_value = separator[1:-1]
                    # PostgreSQL의 E'' 문자열 리터럴 사용
                    return f"STRING_AGG({column}, E'{sep_value}')"
                else:
                    return f"STRING_AGG({column}, {separator})"
            
            sql = re.sub(pattern, replace_group_concat, sql, flags=re.IGNORECASE)
        # SQLite는 변환 불필요
        
        return sql
    
    @staticmethod
    def convert_auto_increment(sql: str, db_type: str) -> str:
        """
        AUTOINCREMENT → SERIAL 변환
        
        Args:
            sql: SQL 쿼리
            db_type: 'sqlite' 또는 'postgresql'
        
        Returns:
            변환된 SQL 쿼리
        """
        if db_type == 'postgresql':
            # INTEGER PRIMARY KEY AUTOINCREMENT → SERIAL PRIMARY KEY
            sql = re.sub(
                r'INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT',
                'SERIAL PRIMARY KEY',
                sql,
                flags=re.IGNORECASE
            )
            # INTEGER PRIMARY KEY → SERIAL PRIMARY KEY (AUTOINCREMENT 없이)
            sql = re.sub(
                r'INTEGER\s+PRIMARY\s+KEY(?!\s+AUTOINCREMENT)',
                'SERIAL PRIMARY KEY',
                sql,
                flags=re.IGNORECASE
            )
        # SQLite는 변환 불필요
        
        return sql
    
    @staticmethod
    def convert_blob_type(sql: str, db_type: str) -> str:
        """
        BLOB 타입 변환
        
        Args:
            sql: SQL 쿼리
            db_type: 'sqlite' 또는 'postgresql'
        
        Returns:
            변환된 SQL 쿼리
        """
        if db_type == 'postgresql':
            # BLOB → BYTEA (일반 데이터) 또는 VECTOR (임베딩)
            # VECTOR는 별도 처리 필요하므로 여기서는 BYTEA로 변환
            sql = re.sub(r'\bBLOB\b', 'BYTEA', sql, flags=re.IGNORECASE)
        # SQLite는 변환 불필요
        
        return sql
    
    @staticmethod
    def convert_fts5_to_postgresql_fts(sql: str, db_type: str) -> str:
        """
        FTS5 쿼리를 PostgreSQL Full-Text Search로 변환
        
        Args:
            sql: SQL 쿼리
            db_type: 'sqlite' 또는 'postgresql'
        
        Returns:
            변환된 SQL 쿼리
        """
        if db_type != 'postgresql':
            return sql
        
        # FTS5 가상 테이블명을 기본 테이블명으로 변환
        # statute_articles_fts → statute_articles
        # case_paragraphs_fts → case_paragraphs
        # decision_paragraphs_fts → decision_paragraphs
        # interpretation_paragraphs_fts → interpretation_paragraphs
        fts_table_mapping = {
            'statute_articles_fts': 'statute_articles',
            'case_paragraphs_fts': 'case_paragraphs',
            'decision_paragraphs_fts': 'decision_paragraphs',
            'interpretation_paragraphs_fts': 'interpretation_paragraphs',
        }
        
        for fts_table, base_table in fts_table_mapping.items():
            # FROM 절의 가상 테이블명 변경
            sql = re.sub(
                rf'\b{re.escape(fts_table)}\b',
                base_table,
                sql,
                flags=re.IGNORECASE
            )
        
        # MATCH ? → text_search_vector @@ to_tsquery('korean', %s)
        # 패턴: table_name MATCH ? 또는 table_name MATCH %s
        sql = re.sub(
            r"(\w+)\s+MATCH\s+([?%s])",
            lambda m: f"{m.group(1)}.text_search_vector @@ to_tsquery('korean', {m.group(2)})",
            sql,
            flags=re.IGNORECASE
        )
        
        # bm25(table_name) → ts_rank(to_tsvector('korean', table_name.text), to_tsquery('korean', %s))
        # bm25() 함수는 ORDER BY 절에서만 사용되므로, 별도 처리 필요
        # 실제로는 쿼리 변환 시점에 파라미터를 알 수 없으므로, 
        # legal_data_connector_v2.py에서 직접 처리하는 것이 더 정확함
        
        # rowid → id (PostgreSQL에서는 rowid가 없고 id를 사용)
        sql = re.sub(r'\browid\b', 'id', sql, flags=re.IGNORECASE)
        
        return sql
    
    @staticmethod
    def convert_sql(sql: str, db_type: str) -> str:
        """
        SQL 쿼리 전체 변환
        
        Args:
            sql: SQL 쿼리
            db_type: 'sqlite' 또는 'postgresql'
        
        Returns:
            변환된 SQL 쿼리
        """
        # 순서 중요: 복잡한 변환부터 먼저
        sql = SQLAdapter.convert_group_concat(sql, db_type)
        sql = SQLAdapter.convert_auto_increment(sql, db_type)
        sql = SQLAdapter.convert_blob_type(sql, db_type)
        sql = SQLAdapter.convert_fts5_to_postgresql_fts(sql, db_type)
        sql = SQLAdapter.convert_parameter_placeholder(sql, db_type)
        
        return sql
    
    @staticmethod
    def convert_row_to_dict(row) -> dict:
        """
        Row 객체를 dict로 변환
        
        Args:
            row: sqlite3.Row 또는 psycopg2.extras.RealDictRow
        
        Returns:
            dict 객체
        """
        if hasattr(row, 'keys'):
            # sqlite3.Row 또는 RealDictRow
            return dict(row)
        elif hasattr(row, '__iter__') and not isinstance(row, (str, bytes)):
            # 튜플 또는 리스트
            return dict(row)
        else:
            return row

