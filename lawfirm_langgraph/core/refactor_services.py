# -*- coding: utf-8 -*-
"""
Services 폴더 재구성 스크립트
"""
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
SERVICES_DIR = BASE_DIR / "services"

# 새 폴더 구조 생성
new_dirs = [
    "conversation",
    "shared/cache",
    "shared/clients",
    "shared/monitoring",
    "shared/feedback",
    "shared/profiles",
    "shared/testing",
    "classification/extractors",
    "classification/analyzers",
    "processing/integration",
    "search/handlers",
    "search/processors",
    "agents/prompt_builders",
    "models/managers",
    "data/routers",
]

print("=== 폴더 생성 ===")
for dir_path in new_dirs:
    full_path = BASE_DIR / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    # __init__.py 생성
    init_file = full_path / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# -*- coding: utf-8 -*-\n")
    print(f"✓ {dir_path}")

# 파일 이동 매핑
file_moves = {
    # Conversation
    "conversation_flow_tracker.py": "conversation/conversation_flow_tracker.py",
    "conversation_manager.py": "conversation/conversation_manager.py",
    "conversation_quality_monitor.py": "conversation/conversation_quality_monitor.py",
    "multi_turn_handler.py": "conversation/multi_turn_handler.py",
    "contextual_memory_manager.py": "conversation/contextual_memory_manager.py",
    "integrated_session_manager.py": "conversation/integrated_session_manager.py",
    
    # Cache
    "workflow_cache_manager.py": "shared/cache/workflow_cache_manager.py",
    "keyword_cache.py": "shared/cache/keyword_cache.py",
    "integrated_cache_system.py": "shared/cache/integrated_cache_system.py",
    
    # Clients
    "gemini_client.py": "shared/clients/gemini_client.py",
    "ollama_client.py": "shared/clients/ollama_client.py",
    "langfuse_client.py": "shared/clients/langfuse_client.py",
    
    # Monitoring
    "performance_monitoring.py": "shared/monitoring/performance_monitoring.py",
    "performance_monitor.py": "shared/monitoring/performance_monitor.py",
    "performance_evaluator.py": "shared/monitoring/performance_evaluator.py",
    
    # Feedback
    "feedback_system.py": "shared/feedback/feedback_system.py",
    
    # Profiles
    "user_profile_manager.py": "shared/profiles/user_profile_manager.py",
    
    # Testing
    "ab_test_manager.py": "shared/testing/ab_test_manager.py",
    
    # Classification
    "domain_specific_extractor.py": "classification/extractors/domain_specific_extractor.py",
    "emotion_intent_analyzer.py": "classification/analyzers/emotion_intent_analyzer.py",
    
    # Processing
    "term_integration_system.py": "processing/integration/term_integration_system.py",
    "legal_basis_integration_service.py": "processing/integration/legal_basis_integration_service.py",
    "ai_keyword_generator.py": "processing/extractors/ai_keyword_generator.py",
    "keyword_database_loader.py": "processing/extractors/keyword_database_loader.py",
    "keyword_coverage_enhancer.py": "processing/extractors/keyword_coverage_enhancer.py",
    "database_keyword_manager.py": "processing/extractors/database_keyword_manager.py",
    "hybrid_keyword_manager.py": "processing/extractors/hybrid_keyword_manager.py",
    "multi_method_term_extractor.py": "processing/extractors/multi_method_term_extractor.py",
    "legal_term_extractor.py": "processing/extractors/legal_term_extractor.py",
    "legal_term_expander.py": "processing/extractors/legal_term_expander.py",
    "legal_term_validator.py": "processing/extractors/legal_term_validator.py",
    "nlp_term_extraction_pipeline.py": "processing/extractors/nlp_term_extraction_pipeline.py",
    "legal_text_preprocessor.py": "processing/processors/legal_text_preprocessor.py",
    "document_processor.py": "processing/processors/document_processor.py",
    
    # Search
    "search_service.py": "search/handlers/search_service.py",
    "result_merger.py": "search/processors/result_merger.py",
    
    # Prompt
    "unified_prompt_manager.py": "agents/prompt_builders/unified_prompt_manager.py",
    "prompt_optimizer.py": "agents/prompt_builders/prompt_optimizer.py",
    "prompt_templates.py": "agents/prompt_builders/prompt_templates.py",
    "dynamic_prompt_updater.py": "agents/prompt_builders/dynamic_prompt_updater.py",
    
    # Models
    "optimized_model_manager.py": "models/managers/optimized_model_manager.py",
    
    # Data
    "sql_router.py": "data/routers/sql_router.py",
    
    # Context (agents/handlers에 이미 context_builder.py가 있으므로 확인 필요)
    "context_manager.py": "agents/handlers/context_manager.py",
    "context_compressor.py": "agents/handlers/context_compressor.py",
    "context_quality_enhancer.py": "agents/handlers/context_quality_enhancer.py",
    
    # Generation (중복 확인 필요 - generation 폴더에 이미 있을 수 있음)
    "confidence_calculator.py": "generation/validators/confidence_calculator.py",
    "source_validator.py": "generation/validators/source_validator.py",
    "unified_source_formatter.py": "generation/formatters/unified_source_formatter.py",
    "gemini_validation_pipeline.py": "generation/validators/gemini_validation_pipeline.py",
}

print("\n=== 파일 이동 ===")
moved_count = 0
skipped_count = 0

for src_file, dst_path in file_moves.items():
    src_path = SERVICES_DIR / src_file
    dst_full_path = BASE_DIR / dst_path
    
    if not src_path.exists():
        print(f"⚠ {src_file} - 소스 파일 없음")
        skipped_count += 1
        continue
    
    if dst_full_path.exists():
        print(f"⚠ {src_file} - 대상 파일 이미 존재 (건너뜀)")
        skipped_count += 1
        continue
    
    # 대상 디렉토리 생성
    dst_full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 파일 이동
    shutil.move(str(src_path), str(dst_full_path))
    print(f"✓ {src_file} → {dst_path}")
    moved_count += 1

print(f"\n=== 완료 ===")
print(f"이동: {moved_count}개")
print(f"건너뜀: {skipped_count}개")

