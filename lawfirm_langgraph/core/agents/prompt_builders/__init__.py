# -*- coding: utf-8 -*-
"""
Prompt Builders Module
프롬프트 빌더 모듈
"""

import sys
from pathlib import Path

# 상위 디렉토리의 prompt_builders.py 파일에서 import
try:
    # core/agents/prompt_builders.py 파일에서 직접 import
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # prompt_builders.py 모듈을 직접 import
    import importlib.util
    prompt_builders_file = parent_dir / "prompt_builders.py"
    if prompt_builders_file.exists():
        spec = importlib.util.spec_from_file_location("prompt_builders_module", prompt_builders_file)
        prompt_builders_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompt_builders_module)
        QueryBuilder = getattr(prompt_builders_module, "QueryBuilder", None)
    else:
        QueryBuilder = None
except Exception:
    QueryBuilder = None

__all__ = ["QueryBuilder"]
