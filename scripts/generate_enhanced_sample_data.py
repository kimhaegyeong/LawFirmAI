# -*- coding: utf-8 -*-
"""
Enhanced Sample Data Generator
향상된 샘플 데이터 생성기
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, '.')

def generate_enhanced_sample_data():
    """향상된 샘플 데이터 생성"""
    try:
        from source.data.vector_store import LegalVectorStore
        from source.data.database import DatabaseManager
        
        print("🔍 향상된 샘플 데이터 생성 중...")
        
        # 벡터 스토어 초기화
        vector_store = LegalVectorStore()
        
        # 데이터베이스 초기화
        db_manager = DatabaseManager()
        
        # 다양한 법령에서 샘플 데이터 수집
        law_names = ["민법", "형법", "상법", "행정법", "민사소송법", "형사소송법"]
        
        all_documents = []
        all_metadatas = []
        
        for law_name in law_names:
            print(f"📚 {law_name} 데이터 수집 중...")
            
            # 데이터베이스에서 해당 법령의 조문들 가져오기
            articles = db_manager.search_assembly_documents(law_name, limit=20)
            
            for article in articles:
                content = f"법령명: {article['law_name']}\n조문번호: 제{article['article_number']}조\n내용: {article['content']}"
                all_documents.append(content)
                
                metadata = {
                    'law_name': article['law_name'],
                    'article_number': article['article_number'],
                    'article_title': article.get('article_title', ''),
                    'law_id': article['law_id'],
                    'article_type': article.get('article_type', 'main'),
                    'is_supplementary': article.get('is_supplementary', False),
                    'parsing_quality_score': article.get('parsing_quality_score', 0.0)
                }
                all_metadatas.append(metadata)
        
        print(f"📝 총 {len(all_documents)}개의 문서 수집 완료")
        
        if all_documents:
            # 벡터 스토어에 추가
            print("📝 벡터 스토어에 데이터 추가 중...")
            success = vector_store.add_documents(all_documents, all_metadatas)
            
            if success:
                print("✅ 샘플 데이터 추가 성공")
                
                # 통계 확인
                stats = vector_store.get_stats()
                print(f"업데이트된 벡터 스토어 통계: {stats}")
                
                # 테스트 검색 실행
                print("\n🔍 테스트 검색 실행...")
                test_queries = [
                    "민법 제750조",
                    "형법 제250조",
                    "상법 제1조",
                    "불법행위",
                    "손해배상"
                ]
                
                for query in test_queries:
                    results = vector_store.search(query, top_k=3)
                    print(f"'{query}' 검색 결과: {len(results)}개")
                    if results:
                        for i, result in enumerate(results[:2]):
                            metadata = result.get('metadata', {})
                            print(f"  {i+1}. {metadata.get('law_name', 'N/A')} 제{metadata.get('article_number', 'N/A')}조 (유사도: {result.get('similarity', 0.0):.3f})")
                
            else:
                print("❌ 샘플 데이터 추가 실패")
        else:
            print("❌ 수집된 데이터가 없습니다")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    print("🚀 Enhanced Sample Data Generator")
    print("=" * 50)
    
    generate_enhanced_sample_data()
    
    print("\n🎉 샘플 데이터 생성 완료!")



