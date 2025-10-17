#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 Gradio 테스트 스크립트 (유니코드 문제 해결)
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_gradio_import():
    """Gradio 모듈 임포트 테스트"""
    print("Gradio 모듈 임포트 테스트")
    print("=" * 40)
    
    try:
        import gradio as gr
        print(f"Gradio 버전: {gr.__version__}")
        return True
    except ImportError as e:
        print(f"Gradio 임포트 실패: {e}")
        return False

def test_gradio_interface_creation():
    """Gradio 인터페이스 생성 테스트"""
    print("\nGradio 인터페이스 생성 테스트")
    print("=" * 40)
    
    try:
        import gradio as gr
        
        def simple_chat(message):
            return f"테스트 응답: {message}"
        
        # 간단한 인터페이스 생성
        interface = gr.Interface(
            fn=simple_chat,
            inputs="text",
            outputs="text",
            title="LawFirmAI 테스트",
            description="간단한 테스트 인터페이스"
        )
        
        print("Gradio 인터페이스 생성 성공")
        return True
        
    except Exception as e:
        print(f"Gradio 인터페이스 생성 실패: {e}")
        return False

def test_database_connection():
    """데이터베이스 연결 테스트"""
    print("\n데이터베이스 연결 테스트")
    print("=" * 40)
    
    try:
        import sqlite3
        
        # 데이터베이스 파일 확인
        db_path = project_root / "data" / "lawfirm.db"
        if not db_path.exists():
            print(f"데이터베이스 파일이 없습니다: {db_path}")
            return False
        
        # 데이터베이스 연결 테스트
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # 테이블 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"데이터베이스 연결 성공")
        print(f"테이블 수: {len(tables)}")
        
        # 주요 테이블 데이터 확인
        cursor.execute("SELECT COUNT(*) FROM assembly_articles")
        article_count = cursor.fetchone()[0]
        print(f"법률 조문 수: {article_count:,}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"데이터베이스 연결 실패: {e}")
        return False

def test_vector_embeddings():
    """벡터 임베딩 파일 테스트"""
    print("\n벡터 임베딩 파일 테스트")
    print("=" * 40)
    
    try:
        embeddings_dir = project_root / "data" / "embeddings"
        
        # 법률 벡터 파일 확인
        law_vector_dir = embeddings_dir / "ml_enhanced_ko_sroberta"
        if law_vector_dir.exists():
            faiss_file = law_vector_dir / "ml_enhanced_faiss_index.faiss"
            json_file = law_vector_dir / "ml_enhanced_faiss_index.json"
            
            if faiss_file.exists() and json_file.exists():
                print(f"법률 벡터 파일 존재")
                print(f"FAISS 파일 크기: {faiss_file.stat().st_size:,} bytes")
                print(f"JSON 파일 크기: {json_file.stat().st_size:,} bytes")
            else:
                print(f"법률 벡터 파일이 없습니다")
                return False
        else:
            print(f"법률 벡터 디렉토리가 없습니다")
            return False
        
        # 판례 벡터 파일 확인
        precedent_vector_dir = embeddings_dir / "ml_enhanced_ko_sroberta_precedents"
        if precedent_vector_dir.exists():
            faiss_file = precedent_vector_dir / "ml_enhanced_ko_sroberta_precedents.faiss"
            json_file = precedent_vector_dir / "ml_enhanced_ko_sroberta_precedents.json"
            
            if faiss_file.exists() and json_file.exists():
                print(f"판례 벡터 파일 존재")
                print(f"FAISS 파일 크기: {faiss_file.stat().st_size:,} bytes")
                print(f"JSON 파일 크기: {json_file.stat().st_size:,} bytes")
            else:
                print(f"판례 벡터 파일이 없습니다")
                return False
        else:
            print(f"판례 벡터 디렉토리가 없습니다")
            return False
        
        return True
        
    except Exception as e:
        print(f"벡터 임베딩 파일 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("LawFirmAI Gradio 테스트 시작")
    print("=" * 60)
    
    tests = [
        ("Gradio 모듈 임포트", test_gradio_import),
        ("Gradio 인터페이스 생성", test_gradio_interface_creation),
        ("데이터베이스 연결", test_database_connection),
        ("벡터 임베딩 파일", test_vector_embeddings)
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
        print("모든 테스트가 성공적으로 완료되었습니다!")
        print("\nGradio 앱을 실행하려면:")
        print("python app.py")
    else:
        print("일부 테스트가 실패했습니다. 로그를 확인하세요.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()