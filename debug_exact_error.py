#!/usr/bin/env python3
"""
벡터 임베딩 생성 중 발생하는 'int' object is not iterable 에러 디버깅
"""

import json
import sys
import os
from pathlib import Path

# Windows 콘솔에서 UTF-8 인코딩 설정
if os.name == 'nt':  # Windows
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except AttributeError:
        # 이미 UTF-8로 설정된 경우 무시
        pass

# Add source to path
sys.path.append(str(Path(__file__).parent / "source"))

from scripts.build_ml_enhanced_vector_db import MLEnhancedVectorBuilder

def debug_exact_error():
    """정확한 에러 위치 찾기"""
    input_dir = Path("data/processed/assembly/law/20251013_ml")
    
    # 첫 번째 파일 로드 (law_page 파일만)
    json_files = [f for f in input_dir.rglob("ml_enhanced_law_page_*.json")]
    if not json_files:
        print("No law page JSON files found!")
        return
    
    first_file = json_files[0]
    print(f"Testing with file: {first_file}")
    
    # 벡터 빌더 초기화
    try:
        vector_builder = MLEnhancedVectorBuilder()
        print("[OK] Vector builder initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize vector builder: {e}")
        return
    
    # 단일 파일 처리 테스트
    try:
        documents = vector_builder._process_single_file(first_file)
        print(f"[OK] Single file processing completed: {len(documents)} documents")
        
    except Exception as e:
        print(f"[ERROR] Error during single file processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_exact_error()
