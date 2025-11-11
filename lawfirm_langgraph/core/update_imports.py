# -*- coding: utf-8 -*-
"""
Import 경로 업데이트 스크립트
"""
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Import 경로 매핑 (old -> new)
IMPORT_MAPPINGS = {
    # Conversation
    r"from core\.services\.conversation_flow_tracker": "from core.conversation.conversation_flow_tracker",
    r"from core\.services\.conversation_manager": "from core.conversation.conversation_manager",
    r"from core\.services\.conversation_quality_monitor": "from core.conversation.conversation_quality_monitor",
    r"from core\.services\.multi_turn_handler": "from core.conversation.multi_turn_handler",
    r"from core\.services\.contextual_memory_manager": "from core.conversation.contextual_memory_manager",
    r"from core\.services\.integrated_session_manager": "from core.conversation.integrated_session_manager",
    r"from \.\.services\.conversation_flow_tracker": "from ..conversation.conversation_flow_tracker",
    r"from \.\.services\.conversation_manager": "from ..conversation.conversation_manager",
    
    # Prompt
    r"from core\.services\.unified_prompt_manager": "from core.agents.prompt_builders.unified_prompt_manager",
    r"from core\.services\.prompt_optimizer": "from core.agents.prompt_builders.prompt_optimizer",
    r"from core\.services\.prompt_templates": "from core.agents.prompt_builders.prompt_templates",
    r"from core\.services\.dynamic_prompt_updater": "from core.agents.prompt_builders.dynamic_prompt_updater",
    
    # Processing
    r"from core\.services\.term_integration_system": "from core.processing.integration.term_integration_system",
    r"from core\.services\.legal_basis_integration_service": "from core.processing.integration.legal_basis_integration_service",
    r"from core\.services\.ai_keyword_generator": "from core.processing.extractors.ai_keyword_generator",
    r"from core\.services\.keyword_database_loader": "from core.processing.extractors.keyword_database_loader",
    r"from core\.services\.keyword_coverage_enhancer": "from core.processing.extractors.keyword_coverage_enhancer",
    r"from core\.services\.database_keyword_manager": "from core.processing.extractors.database_keyword_manager",
    r"from core\.services\.hybrid_keyword_manager": "from core.processing.extractors.hybrid_keyword_manager",
    r"from core\.services\.multi_method_term_extractor": "from core.processing.extractors.multi_method_term_extractor",
    r"from core\.services\.legal_term_extractor": "from core.processing.extractors.legal_term_extractor",
    r"from core\.services\.legal_term_expander": "from core.processing.extractors.legal_term_expander",
    r"from core\.services\.legal_term_validator": "from core.processing.extractors.legal_term_validator",
    r"from core\.services\.nlp_term_extraction_pipeline": "from core.processing.extractors.nlp_term_extraction_pipeline",
    r"from core\.services\.legal_text_preprocessor": "from core.processing.processors.legal_text_preprocessor",
    
    # Classification
    r"from core\.services\.domain_specific_extractor": "from core.classification.extractors.domain_specific_extractor",
    r"from core\.services\.emotion_intent_analyzer": "from core.classification.analyzers.emotion_intent_analyzer",
    
    # Generation
    r"from core\.services\.confidence_calculator": "from core.generation.validators.confidence_calculator",
    r"from core\.services\.source_validator": "from core.generation.validators.source_validator",
    r"from core\.services\.unified_source_formatter": "from core.generation.formatters.unified_source_formatter",
    r"from core\.services\.gemini_validation_pipeline": "from core.generation.validators.gemini_validation_pipeline",
    
    # Search
    r"from core\.services\.search_service": "from core.search.handlers.search_service",
    r"from core\.services\.result_merger": "from core.search.processors.result_merger",
    
    # Context
    r"from core\.services\.context_manager": "from core.agents.handlers.context_manager",
    r"from core\.services\.context_compressor": "from core.agents.handlers.context_compressor",
    r"from core\.services\.context_quality_enhancer": "from core.agents.handlers.context_quality_enhancer",
    
    # Models
    r"from core\.services\.optimized_model_manager": "from core.models.managers.optimized_model_manager",
    
    # Data
    r"from core\.services\.sql_router": "from core.data.routers.sql_router",
    
    # Shared
    r"from core\.services\.performance_monitoring": "from core.shared.monitoring.performance_monitoring",
    r"from core\.services\.performance_monitor": "from core.shared.monitoring.performance_monitor",
    r"from core\.services\.performance_evaluator": "from core.shared.monitoring.performance_evaluator",
    r"from core\.services\.feedback_system": "from core.shared.feedback.feedback_system",
    r"from core\.services\.user_profile_manager": "from core.shared.profiles.user_profile_manager",
    r"from core\.services\.ab_test_manager": "from core.shared.testing.ab_test_manager",
}

def update_file_imports(file_path: Path):
    """파일의 import 경로 업데이트"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        updated = False
        
        for old_pattern, new_path in IMPORT_MAPPINGS.items():
            pattern = re.compile(old_pattern)
            if pattern.search(content):
                content = pattern.sub(new_path, content)
                updated = True
        
        if updated and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"⚠ {file_path}: {e}")
        return False

def find_python_files(directory: Path):
    """재귀적으로 Python 파일 찾기"""
    for path in directory.rglob("*.py"):
        if path.name != "__init__.py" and "refactor_services.py" not in str(path) and "update_imports.py" not in str(path):
            yield path

print("=== Import 경로 업데이트 ===")
updated_count = 0

for py_file in find_python_files(BASE_DIR):
    if update_file_imports(py_file):
        print(f"✓ {py_file.relative_to(BASE_DIR)}")
        updated_count += 1

print(f"\n=== 완료 ===")
print(f"업데이트: {updated_count}개 파일")

