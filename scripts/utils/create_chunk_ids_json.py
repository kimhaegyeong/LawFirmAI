#!/usr/bin/env python3
"""
chunk_ids.json 파일 생성 스크립트
FAISS 인덱스와 함께 사용할 chunk_ids 매핑 파일을 생성합니다.
"""
import json
import sqlite3
import sys
from pathlib import Path

def create_chunk_ids_json(db_path: str, version_id: int, output_path: str):
    """버전 ID에 해당하는 chunk_ids를 JSON 파일로 저장"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute(
        "SELECT DISTINCT chunk_id FROM embeddings WHERE version_id = ? ORDER BY chunk_id",
        (version_id,)
    )
    chunk_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_ids, f, indent=2)
    
    print(f"Created {output_file} with {len(chunk_ids)} chunk_ids")
    if chunk_ids:
        print(f"Range: {min(chunk_ids)} - {max(chunk_ids)}")
    
    return len(chunk_ids)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python create_chunk_ids_json.py <db_path> <version_id> <output_path>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    version_id = int(sys.argv[2])
    output_path = sys.argv[3]
    
    create_chunk_ids_json(db_path, version_id, output_path)

