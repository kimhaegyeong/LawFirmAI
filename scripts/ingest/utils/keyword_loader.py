# -*- coding: utf-8 -*-
"""
키워드 로드 유틸리티
키워드 문자열 또는 파일에서 키워드를 로드
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def load_keywords(keywords_str: Optional[str] = None, keyword_file: Optional[str] = None) -> List[str]:
    """
    키워드 로드
    
    Args:
        keywords_str: 검색 키워드 (쉼표 구분)
        keyword_file: 키워드 파일 경로
    
    Returns:
        중복 제거된 키워드 리스트
    """
    keywords = []
    
    if keywords_str:
        keywords.extend([k.strip() for k in keywords_str.split(',') if k.strip()])
    
    if keyword_file:
        file_path = Path(keyword_file)
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                keywords.extend([line.strip() for line in f if line.strip()])
        else:
            logger.warning(f"키워드 파일을 찾을 수 없습니다: {keyword_file}")
    
    return list(set(keywords))  # 중복 제거

