# -*- coding: utf-8 -*-
"""
외부 인덱스 메타데이터 확인 스크립트
"""

import sys
import os
from pathlib import Path
import json

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 외부 인덱스 메타데이터 경로
metadata_path = project_root / "data" / "embeddings" / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_faiss_index.json"

if not metadata_path.exists():
    print(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
    sys.exit(1)

print("="*80)
print("외부 인덱스 메타데이터 구조 확인")
print("="*80)

with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata_content = json.load(f)

print(f"메타데이터 타입: {type(metadata_content)}")

if isinstance(metadata_content, dict):
    print(f"메타데이터 키: {list(metadata_content.keys())}")
    
    if 'documents' in metadata_content:
        documents = metadata_content['documents']
        print(f"\n'documents' 키 사용: {len(documents)}개 문서")
        if documents:
            print(f"\n첫 번째 문서 샘플:")
            print(f"  타입: {type(documents[0])}")
            if isinstance(documents[0], dict):
                print(f"  키: {list(documents[0].keys())}")
                print(f"  내용: {json.dumps(documents[0], ensure_ascii=False, indent=2)[:500]}")
    
    elif 'document_metadata' in metadata_content and 'document_texts' in metadata_content:
        metadata_list = metadata_content['document_metadata']
        texts_list = metadata_content['document_texts']
        print(f"\n'document_metadata'와 'document_texts' 키 사용")
        print(f"  document_metadata: {len(metadata_list)}개")
        print(f"  document_texts: {len(texts_list)}개")
        
        if metadata_list:
            print(f"\n첫 번째 메타데이터 샘플:")
            print(f"  타입: {type(metadata_list[0])}")
            if isinstance(metadata_list[0], dict):
                print(f"  키: {list(metadata_list[0].keys())}")
                print(f"  내용: {json.dumps(metadata_list[0], ensure_ascii=False, indent=2)[:500]}")
                
                # chunk_id 관련 키 확인
                chunk_id_keys = [k for k in metadata_list[0].keys() if 'chunk' in k.lower() or 'id' in k.lower()]
                if chunk_id_keys:
                    print(f"\n  chunk_id 관련 키: {chunk_id_keys}")
                    for key in chunk_id_keys:
                        print(f"    {key}: {metadata_list[0].get(key)}")
        
        if texts_list:
            print(f"\n첫 번째 텍스트 샘플:")
            print(f"  길이: {len(texts_list[0])}자")
            print(f"  내용: {texts_list[0][:200]}...")
    
    # 샘플 몇 개 더 확인
    if 'document_metadata' in metadata_content:
        print(f"\n" + "="*80)
        print("메타데이터 샘플 (처음 5개)")
        print("="*80)
        for i, meta in enumerate(metadata_content['document_metadata'][:5], 1):
            print(f"\n  샘플 {i}:")
            if isinstance(meta, dict):
                for key, value in meta.items():
                    if key in ['chunk_id', 'id', 'document_id', 'source_id', 'text_chunk_id']:
                        print(f"    {key}: {value}")
                    elif isinstance(value, (str, int, float)) and len(str(value)) < 100:
                        print(f"    {key}: {value}")

elif isinstance(metadata_content, list):
    print(f"\n리스트 타입: {len(metadata_content)}개 항목")
    if metadata_content:
        print(f"\n첫 번째 항목 샘플:")
        print(f"  타입: {type(metadata_content[0])}")
        if isinstance(metadata_content[0], dict):
            print(f"  키: {list(metadata_content[0].keys())}")
            print(f"  내용: {json.dumps(metadata_content[0], ensure_ascii=False, indent=2)[:500]}")

