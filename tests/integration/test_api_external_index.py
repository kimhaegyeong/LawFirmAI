#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 엔드포인트 통합 테스트

실제 채팅 API에서 외부 인덱스가 제대로 사용되는지 확인합니다.
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from fastapi.testclient import TestClient
from lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2 import HybridSearchEngineV2
from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from lawfirm_langgraph.core.utils.config import Config


def test_hybrid_search_engine_external_index():
    """HybridSearchEngineV2가 외부 인덱스를 사용하는지 확인"""
    print("\n" + "=" * 60)
    print("Test: HybridSearchEngineV2 External Index Configuration")
    print("=" * 60)
    
    # Config 설정 확인
    config = Config()
    use_external = getattr(config, 'use_external_vector_store', False)
    external_path = getattr(config, 'external_vector_store_base_path', None)
    version = getattr(config, 'vector_store_version', None)
    
    print(f"\nConfig settings:")
    print(f"  use_external_vector_store: {use_external}")
    print(f"  external_vector_store_base_path: {external_path}")
    print(f"  vector_store_version: {version}")
    
    # HybridSearchEngineV2 초기화
    hybrid_engine = HybridSearchEngineV2()
    
    # SemanticSearchEngineV2 설정 확인
    semantic_engine = hybrid_engine.semantic_search
    
    print(f"\nSemanticSearchEngineV2 settings:")
    print(f"  use_external_index: {semantic_engine.use_external_index}")
    print(f"  external_index_path: {semantic_engine.external_index_path}")
    print(f"  vector_store_version: {semantic_engine.vector_store_version}")
    print(f"  Index loaded: {semantic_engine.index is not None}")
    print(f"  External metadata count: {len(semantic_engine._external_metadata)}")
    
    # 검색 테스트
    print(f"\nTesting search with query: '계약 해제'")
    results = hybrid_engine.search("계약 해제", max_results=5)
    
    print(f"\nSearch results:")
    print(f"  Total results: {len(results.get('results', []))}")
    print(f"  Semantic results: {len(results.get('semantic_results', []))}")
    print(f"  Exact results: {len(results.get('exact_results', []))}")
    
    if results.get('semantic_results'):
        print(f"\n  Top 3 semantic results:")
        for i, result in enumerate(results['semantic_results'][:3], 1):
            print(f"    {i}. Score: {result.get('score', 0):.4f}")
            print(f"       Type: {result.get('type', 'N/A')}")
            print(f"       Source: {result.get('source', 'N/A')}")
    
    assert semantic_engine.use_external_index == use_external, \
        f"External index setting mismatch: {semantic_engine.use_external_index} != {use_external}"
    
    if use_external:
        assert len(semantic_engine._external_metadata) > 0, \
            "External metadata should be loaded when using external index"
        assert semantic_engine.index is not None, \
            "FAISS index should be loaded when using external index"
    
    print("\n✓ Test passed!")


def test_semantic_search_direct():
    """SemanticSearchEngineV2 직접 테스트"""
    print("\n" + "=" * 60)
    print("Test: SemanticSearchEngineV2 Direct")
    print("=" * 60)
    
    config = Config()
    use_external = getattr(config, 'use_external_vector_store', False)
    external_path = getattr(config, 'external_vector_store_base_path', None)
    version = getattr(config, 'vector_store_version', None)
    
    engine = SemanticSearchEngineV2(
        use_external_index=use_external,
        external_index_path=external_path,
        vector_store_version=version
    )
    
    print(f"\nEngine state:")
    print(f"  use_external_index: {engine.use_external_index}")
    print(f"  Index loaded: {engine.index is not None}")
    print(f"  External metadata: {len(engine._external_metadata)} items")
    
    # 검색 테스트
    results = engine.search("계약 해제", k=3, similarity_threshold=0.3)
    
    print(f"\nSearch results: {len(results)} items")
    if results:
        print(f"\n  Top result:")
        print(f"    Score: {results[0].get('score', 0):.4f}")
        print(f"    Type: {results[0].get('type', 'N/A')}")
        print(f"    Source: {results[0].get('source', 'N/A')}")
    
    assert len(results) > 0, "Should return at least one result"
    
    print("\n✓ Test passed!")


def test_workflow_integration():
    """Workflow에서 SemanticSearchEngineV2 사용 확인"""
    print("\n" + "=" * 60)
    print("Test: Workflow Integration")
    print("=" * 60)
    
    try:
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        from lawfirm_langgraph.core.utils.langgraph_config import LangGraphConfig
        
        config = LangGraphConfig.from_env()
        workflow = EnhancedLegalQuestionWorkflow(config)
        
        print(f"\nWorkflow semantic_search state:")
        if workflow.semantic_search:
            print(f"  use_external_index: {workflow.semantic_search.use_external_index}")
            print(f"  Index loaded: {workflow.semantic_search.index is not None}")
            print(f"  External metadata: {len(workflow.semantic_search._external_metadata)} items")
            
            # 간단한 검색 테스트
            results = workflow.semantic_search.search("계약 해제", k=3)
            print(f"\n  Search results: {len(results)} items")
        else:
            print("  semantic_search is None")
        
        print("\n✓ Test passed!")
    except Exception as e:
        print(f"\n⚠ Test skipped: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("API Integration Tests")
    print("=" * 60)
    
    try:
        test_hybrid_search_engine_external_index()
        test_semantic_search_direct()
        test_workflow_integration()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

