# -*- coding: utf-8 -*-
"""
Vector Store Data Checker
벡터 스토어 데이터 확인 및 샘플 데이터 추가
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, '.')

def check_and_add_sample_data():
    """벡터 스토어 데이터 확인 및 샘플 데이터 추가"""
    try:
        from source.data.vector_store import LegalVectorStore
        from source.data.database import DatabaseManager
        
        print("🔍 벡터 스토어 데이터 확인 중...")
        
        # 벡터 스토어 초기화
        vector_store = LegalVectorStore()
        
        # 데이터베이스 초기화
        db_manager = DatabaseManager()
        
        # 벡터 스토어 통계 확인
        stats = vector_store.get_stats()
        print(f"벡터 스토어 통계: {stats}")
        
        # 데이터베이스에서 샘플 조문 가져오기
        sample_articles = db_manager.search_assembly_documents("민법", limit=5)
        print(f"데이터베이스에서 {len(sample_articles)}개의 샘플 조문 발견")
        
        if sample_articles:
            # 샘플 데이터를 벡터 스토어에 추가
            print("📝 샘플 데이터를 벡터 스토어에 추가 중...")
            
            documents = []
            metadatas = []
            
            for article in sample_articles:
                content = f"법령명: {article['law_name']}\n조문번호: 제{article['article_number']}조\n내용: {article['content']}"
                documents.append(content)
                
                metadata = {
                    'law_name': article['law_name'],
                    'article_number': article['article_number'],
                    'article_title': article.get('article_title', ''),
                    'law_id': article['law_id'],
                    'article_type': article.get('article_type', 'main'),
                    'is_supplementary': article.get('is_supplementary', False),
                    'parsing_quality_score': article.get('parsing_quality_score', 0.0)
                }
                metadatas.append(metadata)
            
            # 벡터 스토어에 추가
            success = vector_store.add_documents(documents, metadatas)
            
            if success:
                print("✅ 샘플 데이터 추가 성공")
                
                # 다시 통계 확인
                new_stats = vector_store.get_stats()
                print(f"업데이트된 벡터 스토어 통계: {new_stats}")
            else:
                print("❌ 샘플 데이터 추가 실패")
        else:
            print("❌ 데이터베이스에 샘플 데이터가 없습니다")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    print("🚀 Vector Store Data Checker")
    print("=" * 50)
    
    check_and_add_sample_data()
    
    print("\n🎉 데이터 확인 완료!")



