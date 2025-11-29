"""
중앙 집중식 환경 변수 로더
모든 .env 파일을 일관된 순서로 로드합니다.
"""
import inspect
import logging
from pathlib import Path
from typing import Optional, List, Set

logger = logging.getLogger(__name__)

# 환경 변수 로드 상태 추적 (캐싱)
_loaded_project_roots: Set[Path] = set()


def _find_caller_env_file(project_root: Path) -> Optional[Path]:
    """
    호출 스택에서 실행 중인 스크립트의 .env 파일 찾기
    
    Args:
        project_root: 프로젝트 루트 경로
    
    Returns:
        Optional[Path]: 찾은 .env 파일 경로, 없으면 None
    """
    try:
        # 호출 스택 확인
        stack = inspect.stack()
        
        # 호출 스택을 역순으로 탐색하여 실행 중인 스크립트 찾기
        for frame_info in stack:
            frame = frame_info.frame
            filename = frame.f_globals.get('__file__')
            
            if filename and filename != __file__:
                # 실행 중인 스크립트 경로
                script_path = Path(filename).resolve()
                
                # 스크립트 디렉토리부터 상위로 올라가며 .env 파일 검색
                current_dir = script_path.parent
                
                # 프로젝트 루트를 넘어가지 않도록 제한
                while current_dir != project_root.parent and current_dir != current_dir.parent:
                    env_file = current_dir / ".env"
                    if env_file.exists():
                        logger.debug(f"🔍 Found caller .env: {env_file}")
                        return env_file
                    
                    # 프로젝트 루트에 도달하면 중단
                    if current_dir == project_root:
                        break
                    
                    current_dir = current_dir.parent
                
                # 스크립트를 찾았으면 더 이상 탐색하지 않음
                break
    except Exception as e:
        logger.debug(f"⚠️  Failed to find caller .env file: {e}")
    
    return None


def load_all_env_files(project_root: Optional[Path] = None) -> List[str]:
    """
    모든 .env 파일을 우선순위에 따라 로드
    
    로딩 순서 (우선순위 낮음 → 높음):
    1. 현재 실행 중인 스크립트 위치의 .env 파일 (최우선)
    2. 프로젝트 루트 .env (상위 설정)
    3. lawfirm_langgraph/.env (LangGraph 설정, 최종)
    
    주의: 하위 .env 파일이 상위 .env 파일의 환경변수를 덮어씁니다.
    즉, 나중에 로드되는 .env 파일은 override=True로 로드되어 이전 설정을 덮어쓸 수 있습니다.
    
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
    
    # 1. 현재 실행 중인 스크립트 위치의 .env 파일 (최우선)
    caller_env = _find_caller_env_file(project_root)
    if caller_env:
        load_dotenv(dotenv_path=str(caller_env), override=False)
        loaded_files.append(str(caller_env))
        logger.debug(f"✅ Loaded (caller): {caller_env}")
    else:
        logger.debug("⚠️  No caller .env file found")
    
    # 2. 프로젝트 루트 .env (상위 설정)
    # 이전에 로드된 환경변수를 덮어쓸 수 있음 (override=True)
    root_env = project_root / ".env"
    if root_env.exists():
        # 호출자 .env와 동일한 파일이 아닌 경우에만 로드
        if not caller_env or root_env.resolve() != caller_env.resolve():
            load_dotenv(dotenv_path=str(root_env), override=True)
            loaded_files.append(str(root_env))
            logger.debug(f"✅ Loaded (root): {root_env}")
        else:
            logger.debug(f"⏭️  Skipped (same as caller): {root_env}")
    else:
        logger.debug(f"⚠️  Not found: {root_env}")
    
    # 3. lawfirm_langgraph/.env (LangGraph 설정, 최종)
    # 이전에 로드된 환경변수를 덮어쓸 수 있음 (override=True)
    langgraph_env = project_root / "lawfirm_langgraph" / ".env"
    if langgraph_env.exists():
        # 호출자 .env와 동일한 파일이 아닌 경우에만 로드
        if not caller_env or langgraph_env.resolve() != caller_env.resolve():
            load_dotenv(dotenv_path=str(langgraph_env), override=True)
            loaded_files.append(str(langgraph_env))
            logger.debug(f"✅ Loaded (langgraph): {langgraph_env}")
        else:
            logger.debug(f"⏭️  Skipped (same as caller): {langgraph_env}")
    else:
        logger.debug(f"⚠️  Not found: {langgraph_env}")
    
    if loaded_files:
        # 중복 방지: 파일명만 표시 (전체 경로는 DEBUG 레벨에서만)
        file_names = [Path(f).name for f in loaded_files]
        # 중복 제거 (같은 파일명이 여러 번 나타나는 경우)
        unique_file_names = list(dict.fromkeys(file_names))
        if len(unique_file_names) != len(file_names):
            logger.debug(f"Some .env files were skipped (duplicates): {', '.join(file_names)}")
        logger.info(f"✅ Loaded {len(loaded_files)} .env file(s): {', '.join(unique_file_names)}")
    else:
        logger.warning("⚠️  No .env files found. Using environment variables only.")
    
    return loaded_files


def ensure_env_loaded(project_root: Optional[Path] = None) -> None:
    """
    환경 변수가 로드되었는지 확인하고, 로드되지 않았다면 로드
    
    이 함수는 여러 번 호출되어도 안전합니다.
    같은 project_root에 대해서는 한 번만 로드하여 중복 로드를 방지합니다.
    
    Args:
        project_root: 프로젝트 루트 경로 (None이면 자동 감지)
    """
    # project_root 결정
    if project_root is None:
        project_root = Path(__file__).parent.parent
    else:
        project_root = Path(project_root).resolve()
    
    # 이미 로드된 project_root인지 확인
    if project_root in _loaded_project_roots:
        logger.debug(f"⏭️  Environment variables already loaded for: {project_root}")
        return
    
    # 환경 변수 로드
    load_all_env_files(project_root)
    
    # 로드 완료 표시
    _loaded_project_roots.add(project_root)

