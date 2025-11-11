# -*- coding: utf-8 -*-
"""
Import 오류 확인 스크립트
"""
import sys
import importlib
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent.parent))

# 확인할 주요 모듈들
modules_to_check = [
    "core.conversation.conversation_flow_tracker",
    "core.conversation.conversation_manager",
    "core.agents.prompt_builders.unified_prompt_manager",
    "core.processing.integration.term_integration_system",
    "core.processing.extractors.ai_keyword_generator",
    "core.classification.analyzers.emotion_intent_analyzer",
    "core.generation.validators.confidence_calculator",
    "core.search.handlers.search_service",
    "core.search.processors.result_merger",
]

print("=== Import 오류 확인 ===")
errors = []
success = []

for module_name in modules_to_check:
    try:
        importlib.import_module(module_name)
        success.append(module_name)
        print(f"✓ {module_name}")
    except ImportError as e:
        errors.append((module_name, str(e)))
        print(f"✗ {module_name}: {e}")
    except Exception as e:
        errors.append((module_name, f"Unexpected error: {e}"))
        print(f"⚠ {module_name}: {e}")

print(f"\n=== 결과 ===")
print(f"성공: {len(success)}개")
print(f"오류: {len(errors)}개")

if errors:
    print("\n=== 오류 상세 ===")
    for module, error in errors:
        print(f"{module}: {error}")

sys.exit(0 if not errors else 1)

