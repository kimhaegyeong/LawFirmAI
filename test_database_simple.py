#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 데이터베이스 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_database_schema():
    """데이터베이스 스키마 테스트"""
    print("데이터베이스 스키마 테스트")
    print("=" * 40)
    
    try:
        import sqlite3
        
        db_path = project_root / "data" / "lawfirm.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # 주요 테이블 확인
        expected_tables = [
            'assembly_laws',
            'assembly_articles', 
            'precedent_cases',
            'precedent_sections',
            'precedent_parties',
            'processed_files'
        ]
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        print(f"기존 테이블 수: {len(existing_tables)}")
        
        missing_tables = []
        for table in expected_tables:
            if table in existing_tables:
                print(f"테이블 '{table}': 존재")
            else:
                print(f"테이블 '{table}': 없음")
                missing_tables.append(table)
        
        conn.close()
        
        if missing_tables:
            print(f"누락된 테이블: {missing_tables}")
            return False
        else:
            print("모든 주요 테이블이 존재합니다")
            return True
            
    except Exception as e:
        print(f"데이터베이스 스키마 테스트 실패: {e}")
        return False

def test_database_data():
    """데이터베이스 데이터 테스트"""
    print("\n데이터베이스 데이터 테스트")
    print("=" * 40)
    
    try:
        import sqlite3
        
        db_path = project_root / "data" / "lawfirm.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # 법률 데이터 확인
        cursor.execute("SELECT COUNT(*) FROM assembly_laws")
        law_count = cursor.fetchone()[0]
        print(f"법률 문서 수: {law_count:,}")
        
        cursor.execute("SELECT COUNT(*) FROM assembly_articles")
        article_count = cursor.fetchone()[0]
        print(f"법률 조문 수: {article_count:,}")
        
        # 판례 데이터 확인
        cursor.execute("SELECT COUNT(*) FROM precedent_cases")
        precedent_count = cursor.fetchone()[0]
        print(f"판례 사건 수: {precedent_count:,}")
        
        cursor.execute("SELECT COUNT(*) FROM precedent_sections")
        section_count = cursor.fetchone()[0]
        print(f"판례 섹션 수: {section_count:,}")
        
        # 처리된 파일 확인
        cursor.execute("SELECT COUNT(*) FROM processed_files")
        processed_count = cursor.fetchone()[0]
        print(f"처리된 파일 수: {processed_count:,}")
        
        conn.close()
        
        if law_count > 0 and article_count > 0:
            print("법률 데이터가 정상적으로 존재합니다")
            return True
        else:
            print("법률 데이터가 부족합니다")
            return False
            
    except Exception as e:
        print(f"데이터베이스 데이터 테스트 실패: {e}")
        return False

def test_vector_embeddings():
    """벡터 임베딩 테스트"""
    print("\n벡터 임베딩 테스트")
    print("=" * 40)
    
    try:
        import json
        
        embeddings_dir = project_root / "data" / "embeddings"
        
        # 법률 벡터 테스트
        law_vector_file = embeddings_dir / "ml_enhanced_ko_sroberta" / "ml_enhanced_faiss_index.json"
        if law_vector_file.exists():
            with open(law_vector_file, 'r', encoding='utf-8') as f:
                law_data = json.load(f)
            
            print(f"법률 벡터 파일 크기: {law_vector_file.stat().st_size:,} bytes")
            print(f"법률 벡터 수: {len(law_data.get('vectors', []))}")
        else:
            print("법률 벡터 파일이 없습니다")
            return False
        
        # 판례 벡터 테스트
        precedent_vector_file = embeddings_dir / "ml_enhanced_ko_sroberta_precedents" / "ml_enhanced_ko_sroberta_precedents.json"
        if precedent_vector_file.exists():
            with open(precedent_vector_file, 'r', encoding='utf-8') as f:
                precedent_data = json.load(f)
            
            print(f"판례 벡터 파일 크기: {precedent_vector_file.stat().st_size:,} bytes")
            print(f"판례 벡터 수: {len(precedent_data.get('vectors', []))}")
        else:
            print("판례 벡터 파일이 없습니다")
            return False
        
        return True
        
    except Exception as e:
        print(f"벡터 임베딩 테스트 실패: {e}")
        return False

def test_search_functionality():
    """검색 기능 테스트"""
    print("\n검색 기능 테스트")
    print("=" * 40)
    
    try:
        import sqlite3
        
        db_path = project_root / "data" / "lawfirm.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # 기본 검색 테스트
        search_query = "손해배상"
        cursor.execute("""
            SELECT law_name, article_number, content 
            FROM assembly_articles 
            WHERE content LIKE ? 
            LIMIT 5
        """, (f"%{search_query}%",))
        
        results = cursor.fetchall()
        print(f"'{search_query}' 검색 결과: {len(results)}개")
        
        for i, (law_name, article_number, content) in enumerate(results, 1):
            print(f"  {i}. {law_name} {article_number}")
            print(f"     내용: {content[:100]}...")
        
        conn.close()
        
        if len(results) > 0:
            print("검색 기능이 정상적으로 작동합니다")
            return True
        else:
            print("검색 결과가 없습니다")
            return False
            
    except Exception as e:
        print(f"검색 기능 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("LawFirmAI 데이터베이스 테스트 시작")
    print("=" * 60)
    
    tests = [
        ("데이터베이스 스키마", test_database_schema),
        ("데이터베이스 데이터", test_database_data),
        ("벡터 임베딩", test_vector_embeddings),
        ("검색 기능", test_search_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} 테스트 중 오류 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "통과" if result else "실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n전체 결과: {passed}/{len(results)} 테스트 통과")
    
    if passed == len(results):
        print("모든 데이터베이스 테스트가 성공적으로 완료되었습니다!")
    else:
        print("일부 테스트가 실패했습니다. 로그를 확인하세요.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
