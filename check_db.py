#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite 데이터베이스 상태 확인 스크립트
"""

import sqlite3
import os
from pathlib import Path

def check_database():
    """데이터베이스 상태 확인"""
    db_path = "data/lawfirm.db"
    
    if not os.path.exists(db_path):
        print(f"❌ 데이터베이스 파일이 존재하지 않습니다: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테이블 목록 조회
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print("📋 테이블 목록:")
        for table in tables:
            print(f"  - {table[0]}")
        
        print("\n📊 데이터 통계:")
        
        # documents 테이블 확인
        try:
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            print(f"  - Documents: {doc_count:,}개")
            
            # 문서 타입별 통계
            cursor.execute("SELECT document_type, COUNT(*) FROM documents GROUP BY document_type")
            doc_types = cursor.fetchall()
            print("    문서 타입별:")
            for doc_type, count in doc_types:
                print(f"      - {doc_type}: {count:,}개")
                
        except sqlite3.OperationalError as e:
            print(f"  - Documents 테이블 오류: {e}")
        
        # 메타데이터 테이블들 확인
        metadata_tables = [
            'law_metadata', 'precedent_metadata', 'constitutional_metadata',
            'interpretation_metadata', 'administrative_rule_metadata', 'local_ordinance_metadata'
        ]
        
        for table in metadata_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  - {table}: {count:,}개")
            except sqlite3.OperationalError:
                print(f"  - {table}: 테이블 없음")
        
        # 샘플 데이터 확인
        print("\n🔍 샘플 데이터:")
        try:
            cursor.execute("SELECT id, document_type, title, LENGTH(content) as content_length FROM documents LIMIT 5")
            samples = cursor.fetchall()
            for sample in samples:
                print(f"  - ID: {sample[0]}, Type: {sample[1]}, Title: {sample[2][:50]}..., Length: {sample[3]:,}자")
        except sqlite3.OperationalError as e:
            print(f"  샘플 데이터 조회 오류: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ 데이터베이스 접근 오류: {e}")

def check_embeddings():
    """임베딩 데이터 확인"""
    print("\n🔍 임베딩 데이터 확인:")
    
    embedding_files = [
        "data/embeddings/legal_vector_index.faiss",
        "data/embeddings/legal_vector_index.json",
        "data/embeddings/metadata.json"
    ]
    
    for file_path in embedding_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✅ {file_path}: {file_size:,} bytes")
        else:
            print(f"  ❌ {file_path}: 파일 없음")
    
    # JSON 메타데이터 내용 확인
    metadata_file = "data/embeddings/legal_vector_index.json"
    if os.path.exists(metadata_file):
        try:
            import json
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"\n📈 임베딩 메타데이터:")
            print(f"  - 모델명: {metadata.get('model_name', 'N/A')}")
            print(f"  - 차원: {metadata.get('dimension', 'N/A')}")
            print(f"  - 문서 수: {metadata.get('document_count', 'N/A'):,}개")
            print(f"  - 생성일: {metadata.get('created_at', 'N/A')}")
            
        except Exception as e:
            print(f"  메타데이터 읽기 오류: {e}")

if __name__ == "__main__":
    print("🔍 LawFirmAI 데이터베이스 및 임베딩 상태 확인")
    print("=" * 50)
    
    check_database()
    check_embeddings()
    
    print("\n✅ 확인 완료")
