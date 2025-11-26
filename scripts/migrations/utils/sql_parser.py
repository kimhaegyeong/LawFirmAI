"""
SQL 파서 유틸리티
함수 정의를 올바르게 파싱하기 위한 유틸리티
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def parse_sql_statements(sql_text: str) -> List[str]:
    """
    SQL 문장을 파싱하되, 함수 정의를 올바르게 처리
    
    Args:
        sql_text: SQL 텍스트
    
    Returns:
        SQL 문장 리스트
    """
    statements = []
    
    # 주석 제거 (-- 로 시작하는 줄)
    lines = sql_text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('--'):
            cleaned_lines.append(line)
    
    sql_text = '\n'.join(cleaned_lines)
    
    # 세미콜론으로 분리하되, 함수 내부는 제외
    # $$ ... $$ LANGUAGE plpgsql; 형식의 함수를 하나의 문장으로 처리
    parts = re.split(r';(?=(?:[^\$]*\$\$[^\$]*\$\$)*[^\$]*$)', sql_text)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # 함수 정의 시작 확인 (CREATE OR REPLACE FUNCTION)
        if re.search(r'CREATE\s+(OR\s+REPLACE\s+)?FUNCTION', part, re.IGNORECASE):
            # $$ 태그 찾기
            dollar_matches = re.findall(r'\$\$[^\$]*\$\$', part)
            if dollar_matches:
                # 함수 정의는 세미콜론까지 포함하여 하나의 문장으로 처리
                statements.append(part + ';')
            else:
                statements.append(part + ';')
        else:
            # 일반 SQL 문장
            statements.append(part + ';')
    
    return statements

