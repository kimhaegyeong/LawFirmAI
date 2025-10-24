#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
현행법령 조문별 벡터 임베딩 생성 스크립트
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore
from source.utils.logger import setup_logging

class ArticleVectorEmbedder:
    """조문별 벡터 임베딩 생성 클래스"""
    
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger("article_vector_embedder")
        self.db_manager = DatabaseManager()
        self.vector_store = LegalVectorStore()
        
        # 통계 정보
        self.stats = {
            'total_articles': 0,
            'processed_articles': 0,
            'embedding_errors': []
        }
    
    def create_article_embeddings(self) -> Dict[str, Any]:
        """조문별 벡터 임베딩 생성"""
        self.logger.info("조문별 벡터 임베딩 생성 시작")
        
        try:
            # 1. 조문 데이터 조회
            articles = self._get_articles_data()
            self.stats['total_articles'] = len(articles)
            
            if not articles:
                self.logger.warning("처리할 조문 데이터가 없습니다.")
                return self.stats
            
            self.logger.info(f"총 {len(articles)}개 조문 발견")
            
            # 2. 배치별로 벡터 임베딩 생성 (배치 5부터 시작)
            batch_size = 1000  # 원래 배치 크기로 복원
            start_batch = 16  # 배치 16부터 시작
            start_index = (start_batch - 1) * batch_size
            total_batches = (len(articles) + batch_size - 1) // batch_size
            
            self.logger.info(f"배치 {start_batch}부터 시작 (인덱스 {start_index}부터, 배치 크기: {batch_size})")
            
            for i in range(start_index, len(articles), batch_size):
                batch_articles = articles[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                self.logger.info(f"배치 {batch_num}/{total_batches} 처리 중... ({len(batch_articles)}개 조문)")
                
                # 배치별 문서와 메타데이터 준비
                documents, metadatas = self._prepare_embedding_data(batch_articles)
                
                # 벡터 스토어에 추가
                success = self.vector_store.add_documents(documents, metadatas)
                
                if success:
                    self.stats['processed_articles'] += len(documents)
                    self.logger.info(f"✅ 배치 {batch_num} 완료: {len(documents)}개 조문")
                else:
                    self.logger.error(f"❌ 배치 {batch_num} 실패")
                    break
            
            self.logger.info(f"벡터 임베딩 생성 완료: 총 {self.stats['processed_articles']}개")
            
            # 4. 통계 출력
            self._print_statistics()
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"벡터 임베딩 생성 실패: {e}")
            raise
    
    def _get_articles_data(self) -> List[Dict[str, Any]]:
        """조문 데이터 조회"""
        query = """
            SELECT ca.*, cl.ministry_name, cl.effective_date
            FROM current_laws_articles ca
            JOIN current_laws cl ON ca.law_id = cl.law_id
            ORDER BY ca.law_name_korean, ca.article_number, ca.paragraph_number, ca.sub_paragraph_number
        """
        
        try:
            return self.db_manager.execute_query(query)
        except Exception as e:
            self.logger.error(f"조문 데이터 조회 실패: {e}")
            return []
    
    def _prepare_embedding_data(self, articles: List[Dict[str, Any]]) -> tuple[List[str], List[Dict[str, Any]]]:
        """임베딩용 문서와 메타데이터 준비"""
        documents = []
        metadatas = []
        
        for article in articles:
            try:
                # 조문별 문서 생성
                document = self._create_article_document(article)
                metadata = self._create_article_metadata(article)
                
                documents.append(document)
                metadatas.append(metadata)
                
            except Exception as e:
                error_msg = f"조문 {article.get('article_id', 'Unknown')} 처리 실패: {e}"
                self.logger.error(error_msg)
                self.stats['embedding_errors'].append(error_msg)
        
        return documents, metadatas
    
    def _create_article_document(self, article: Dict[str, Any]) -> str:
        """조문 문서 생성"""
        parts = []
        
        # 기본 정보
        parts.append(f"법령명: {article['law_name_korean']}")
        parts.append(f"조문번호: 제{article['article_number']}조")
        
        if article.get('article_title'):
            parts.append(f"제목: {article['article_title']}")
        
        # 조문 내용
        parts.append(f"내용: {article['article_content']}")
        
        # 항 내용
        if article.get('paragraph_content'):
            parts.append(f"항: {article['paragraph_content']}")
        
        # 호 내용
        if article.get('sub_paragraph_content'):
            parts.append(f"호: {article['sub_paragraph_content']}")
        
        # 소관부처 정보
        if article.get('ministry_name'):
            parts.append(f"소관부처: {article['ministry_name']}")
        
        return "\n".join(parts)
    
    def _create_article_metadata(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """조문 메타데이터 생성"""
        return {
            'law_id': article['law_id'],
            'law_name': article['law_name_korean'],
            'article_number': str(article['article_number']),
            'article_id': article['article_id'],
            'article_title': article.get('article_title', ''),
            'paragraph_number': str(article.get('paragraph_number', '')),
            'sub_paragraph_number': article.get('sub_paragraph_number', ''),
            'source_system': 'current_laws',
            'document_type': 'current_law_article',
            'quality_score': article.get('quality_score', 0.9),
            'ministry_name': article.get('ministry_name', ''),
            'effective_date': article.get('effective_date', ''),
            'parsing_method': article.get('parsing_method', 'batch_parser'),
            'is_supplementary': article.get('is_supplementary', False)
        }
    
    def _print_statistics(self):
        """통계 정보 출력"""
        print("\n" + "="*60)
        print("📊 조문별 벡터 임베딩 생성 통계")
        print("="*60)
        print(f"총 조문 수: {self.stats['total_articles']:,}개")
        print(f"처리된 조문: {self.stats['processed_articles']:,}개")
        print(f"처리 실패: {len(self.stats['embedding_errors'])}개")
        
        if self.stats['embedding_errors']:
            print("\n⚠️ 처리 실패 목록:")
            for error in self.stats['embedding_errors'][:5]:  # 최대 5개만 표시
                print(f"  - {error}")
        
        # 벡터 스토어 통계
        try:
            vector_stats = self.vector_store.get_stats()
            print(f"\n📈 벡터 스토어 통계:")
            print(f"  총 문서 수: {vector_stats.get('documents_count', 0):,}개")
            print(f"  벡터 차원: {vector_stats.get('vector_dimension', 0)}")
            print(f"  인덱스 크기: {vector_stats.get('index_size_mb', 0):.2f}MB")
        except Exception as e:
            print(f"벡터 스토어 통계 조회 실패: {e}")


def main():
    """메인 실행 함수"""
    embedder = ArticleVectorEmbedder()
    stats = embedder.create_article_embeddings()
    
    print(f"\n🎉 조문별 벡터 임베딩 생성 완료!")
    print(f"총 {stats['processed_articles']:,}개 조문의 벡터 임베딩이 생성되었습니다.")


if __name__ == "__main__":
    main()
