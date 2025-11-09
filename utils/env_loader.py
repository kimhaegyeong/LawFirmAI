"""
중앙 집중식 환경 변수 로더
모든 .env 파일을 일관된 순서로 로드합니다.
"""
import os
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


def load_all_env_files(project_root: Optional[Path] = None) -> List[str]:
    """
    모든 .env 파일을 우선순위에 따라 로드
    
    로딩 순서 (우선순위 낮음 → 높음):
    1. 프로젝트 루트 .env (공통 설정, 최저 우선순위)
    2. lawfirm_langgraph/.env (LangGraph 설정)
    3. api/.env (API 설정, 최고 우선순위)
    
    Args:
        project_root: 프로젝트 루트 경로 (None이면 자동 감지)
    
    Returns:
        List[str]: 로드된 .env 파일 경로 리스트
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv not installed. .env files will not be loaded.")
        logger.warning("Install with: pip install python-dotenv")
        return []
    
    if project_root is None:
        # 현재 파일 기준으로 프로젝트 루트 찾기
        # utils/env_loader.py -> 프로젝트 루트
        project_root = Path(__file__).parent.parent
    
    loaded_files = []
    
    # 1. 프로젝트 루트 .env (공통 설정, 최저 우선순위)
    root_env = project_root / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=str(root_env), override=False)
        loaded_files.append(str(root_env))
        logger.debug(f"✅ Loaded: {root_env}")
    else:
        logger.debug(f"⚠️  Not found: {root_env}")
    
    # 2. lawfirm_langgraph/.env (LangGraph 설정)
    langgraph_env = project_root / "lawfirm_langgraph" / ".env"
    if langgraph_env.exists():
        load_dotenv(dotenv_path=str(langgraph_env), override=False)
        loaded_files.append(str(langgraph_env))
        logger.debug(f"✅ Loaded: {langgraph_env}")
    else:
        logger.debug(f"⚠️  Not found: {langgraph_env}")
    
    # 3. api/.env (API 설정, 최고 우선순위)
    api_env = project_root / "api" / ".env"
    if api_env.exists():
        load_dotenv(dotenv_path=str(api_env), override=True)
        loaded_files.append(str(api_env))
        logger.debug(f"✅ Loaded: {api_env}")
    else:
        logger.debug(f"⚠️  Not found: {api_env}")
    
    if loaded_files:
        logger.info(f"✅ Loaded {len(loaded_files)} .env file(s): {', '.join([Path(f).name for f in loaded_files])}")
    else:
        logger.warning("⚠️  No .env files found. Using environment variables only.")
    
    return loaded_files


def ensure_env_loaded(project_root: Optional[Path] = None) -> None:
    """
    환경 변수가 로드되었는지 확인하고, 로드되지 않았다면 로드
    
    이 함수는 여러 번 호출되어도 안전합니다 (이미 로드된 변수는 override=False로 처리)
    
    Args:
        project_root: 프로젝트 루트 경로 (None이면 자동 감지)
    """
    # 이미 로드되었는지 확인 (간단한 체크)
    # dotenv는 이미 로드된 변수를 override=False로 다시 로드해도 안전
    load_all_env_files(project_root)

