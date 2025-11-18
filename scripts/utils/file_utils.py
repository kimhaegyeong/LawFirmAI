#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 처리 유틸리티

JSON 파일 읽기/쓰기 등 파일 관련 공통 함수
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def load_json_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> Union[Dict[str, Any], List[Any]]:
    """
    JSON 파일을 안전하게 로드
    
    Args:
        file_path: JSON 파일 경로
        encoding: 파일 인코딩 (기본값: utf-8)
    
    Returns:
        Dict 또는 List: JSON 데이터
    
    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        json.JSONDecodeError: JSON 파싱 오류
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {file_path}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        logger.debug(f"JSON 파일 로드 성공: {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류 {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"JSON 파일 로드 실패 {file_path}: {e}")
        raise


def save_json_file(
    data: Union[Dict[str, Any], List[Any]],
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    ensure_ascii: bool = False,
    indent: int = 2
) -> None:
    """
    JSON 파일을 안전하게 저장
    
    Args:
        data: 저장할 데이터 (Dict 또는 List)
        file_path: 저장할 파일 경로
        encoding: 파일 인코딩 (기본값: utf-8)
        ensure_ascii: ASCII만 사용 여부 (기본값: False)
        indent: 들여쓰기 공백 수 (기본값: 2)
    
    Raises:
        OSError: 파일 쓰기 오류
    """
    file_path = Path(file_path)
    
    # 디렉토리 생성
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
        logger.debug(f"JSON 파일 저장 성공: {file_path}")
    except Exception as e:
        logger.error(f"JSON 파일 저장 실패 {file_path}: {e}")
        raise


def load_json_files(directory: Union[str, Path], pattern: str = "*.json") -> List[Dict[str, Any]]:
    """
    디렉토리 내의 모든 JSON 파일 로드
    
    Args:
        directory: 디렉토리 경로
        pattern: 파일 패턴 (기본값: *.json)
    
    Returns:
        List[Dict]: 로드된 JSON 데이터 리스트
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"디렉토리가 존재하지 않습니다: {directory}")
        return []
    
    results = []
    for json_file in directory.glob(pattern):
        try:
            data = load_json_file(json_file)
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
        except Exception as e:
            logger.warning(f"JSON 파일 로드 실패 (건너뜀): {json_file} - {e}")
            continue
    
    return results

