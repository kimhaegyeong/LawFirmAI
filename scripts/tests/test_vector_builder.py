#!/usr/bin/env python3
"""
벡터 임베딩 생성 중 발생하는 'int' object is not iterable 에러 디버깅
"""

import json
import sys
from pathlib import Path

# Add source to path
sys.path.append(str(Path(__file__).parent.parent.parent / "source"))

from scripts.build_ml_enhanced_vector_db import MLEnhancedVectorBuilder

def test_vector_builder():
    """벡터 빌더 테스트"""
    input_dir = Path("data/processed/assembly/law/20251013_ml")
    
    # 첫 번째 파일 로드 (law_page 파일만)
    json_files = [f for f in input_dir.rglob("ml_enhanced_law_page_*.json")]
    if not json_files:
        print("No law page JSON files found!")
        return
    
    print(f"Found {len(json_files)} files")
    
    # 벡터 빌더 초기화
    try:
        vector_builder = MLEnhancedVectorBuilder()
        print("[OK] Vector builder initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize vector builder: {e}")
        return
    
    # 처음 2개 파일로 테스트
    test_files = json_files[:2]
    print(f"Testing with {len(test_files)} files")
    
    try:
        # 배치 처리 테스트
        batch_documents = vector_builder._process_batch(test_files)
        print(f"[OK] Batch processing completed: {len(batch_documents)} documents")
        
        # 벡터 인덱스에 추가 테스트
        if batch_documents:
            success = vector_builder._add_documents_to_index(batch_documents)
            if success:
                print("[OK] Documents added to vector index successfully")
            else:
                print("[ERROR] Failed to add documents to vector index")
        
    except Exception as e:
        print(f"[ERROR] Error during vector builder processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_builder()

