#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""상세 통합 테스트"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    print("=" * 60)
    print("Integration Test - Detailed")
    print("=" * 60)
    
    print("\n1. Importing SemanticSearchEngineV2...")
    from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
    
    print("2. Initializing engine with DB path (DB-based index)...")
    engine = SemanticSearchEngineV2(
        db_path='data/lawfirm_v2.db',
        use_external_index=False  # DB 기반 인덱스 사용
    )
    
    print("3. Checking engine state...")
    print(f"   Model name: {engine.model_name}")
    print(f"   Index path: {engine.index_path}")
    print(f"   Use external index: {engine.use_external_index}")
    print(f"   Index loaded: {engine.index is not None}")
    print(f"   Embedder loaded: {engine.embedder is not None}")
    print(f"   Chunk IDs count: {len(engine._chunk_ids)}")
    
    if hasattr(engine, 'diagnose'):
        diagnosis = engine.diagnose()
        print(f"\n4. Diagnosis:")
        for key, value in diagnosis.items():
            print(f"   {key}: {value}")
    
    print("\n5. Testing search...")
    print("   Query: '계약 해제'")
    
    try:
        # 낮은 threshold로 테스트
        results = engine.search('계약 해제', k=5, similarity_threshold=0.1)
        print(f"   Found {len(results)} results")
        
        if results:
            print("\n   Top 3 results:")
            for i, result in enumerate(results[:3], 1):
                print(f"\n   {i}. Score: {result.get('score', 0):.4f}")
                print(f"      Type: {result.get('type', 'N/A')}")
                print(f"      Source: {result.get('source', 'N/A')}")
                text = result.get('text', '')[:100]
                print(f"      Text: {text}...")
        else:
            print("   No results found")
            print("\n   Checking database...")
            import sqlite3
            conn = sqlite3.connect('data/lawfirm_v2.db')
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]
            print(f"   Embeddings in DB: {count}")
            cursor = conn.execute("SELECT COUNT(*) FROM text_chunks")
            count = cursor.fetchone()[0]
            print(f"   Text chunks in DB: {count}")
            conn.close()
            
    except Exception as e:
        print(f"   Error during search: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

