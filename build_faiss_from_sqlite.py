#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite 데이터를 FAISS 벡터 인덱스로 변환하는 스크립트
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from source.data.vector_store import LegalVectorStore
from source.data.database import DatabaseManager
from source.data.data_processor import LegalDataProcessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/build_faiss_from_sqlite.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SQLiteToFAISSBuilder:
    """SQLite 데이터를 FAISS로 변환하는 클래스"""
    
    def __init__(self):
        """초기화"""
        self.db_manager = DatabaseManager()
        self.vector_store = LegalVectorStore()
        self.processor = LegalDataProcessor()
        
        logger.info("SQLiteToFAISSBuilder 초기화 완료")
    
    def load_documents_from_sqlite(self) -> List[Dict[str, Any]]:
        """SQLite에서 문서 데이터 로드"""
        logger.info("SQLite에서 문서 데이터 로드 시작...")
        
        try:
            # documents 테이블에서 모든 문서 조회
            query = """
                SELECT d.id, d.document_type, d.title, d.content, d.source_url,
                       lm.law_name, lm.article_number, lm.promulgation_date, lm.enforcement_date, lm.department,
                       pm.case_number, pm.court_name, pm.decision_date, pm.case_type
                FROM documents d
                LEFT JOIN law_metadata lm ON d.id = lm.document_id
                LEFT JOIN precedent_metadata pm ON d.id = pm.document_id
                ORDER BY d.document_type, d.id
            """
            
            documents = self.db_manager.execute_query(query)
            logger.info(f"SQLite에서 {len(documents)}개 문서 로드 완료")
            
            return documents
            
        except Exception as e:
            logger.error(f"SQLite에서 문서 로드 실패: {e}")
            return []
    
    def process_documents_for_embedding(self, documents: List[Dict[str, Any]]) -> tuple:
        """문서를 임베딩용으로 전처리"""
        logger.info("문서 전처리 시작...")
        
        texts = []
        metadatas = []
        
        for doc in documents:
            try:
                # 텍스트 정리
                content = doc.get('content', '')
                if not content:
                    logger.warning(f"빈 내용의 문서 건너뛰기: {doc.get('id')}")
                    continue
                
                # 텍스트 전처리
                cleaned_text = self.processor.clean_text(content)
                if not cleaned_text:
                    logger.warning(f"전처리 후 빈 텍스트 문서 건너뛰기: {doc.get('id')}")
                    continue
                
                # 제목과 내용 결합
                title = doc.get('title', '')
                full_text = f"{title}\n\n{cleaned_text}"
                
                texts.append(full_text)
                
                # 메타데이터 구성
                metadata = {
                    'document_id': doc.get('id'),
                    'document_type': doc.get('document_type'),
                    'title': title,
                    'law_name': doc.get('law_name', ''),
                    'case_number': doc.get('case_number', ''),
                    'court_name': doc.get('court_name', ''),
                    'article_number': doc.get('article_number', ''),
                    'promulgation_date': doc.get('promulgation_date', ''),
                    'decision_date': doc.get('decision_date', ''),
                    'department': doc.get('department', ''),
                    'case_type': doc.get('case_type', '')
                }
                
                metadatas.append(metadata)
                
            except Exception as e:
                logger.error(f"문서 전처리 실패 ({doc.get('id')}): {e}")
                continue
        
        logger.info(f"전처리 완료: {len(texts)}개 문서")
        return texts, metadatas
    
    def build_faiss_index(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> bool:
        """FAISS 인덱스 구축"""
        logger.info("FAISS 인덱스 구축 시작...")
        
        try:
            # 벡터 스토어에 문서 추가
            success = self.vector_store.add_documents(texts, metadatas)
            
            if success:
                logger.info(f"FAISS 인덱스 구축 완료: {len(texts)}개 문서")
                
                # 인덱스 저장
                save_success = self.vector_store.save_index("data/embeddings/legal_vector_index")
                
                if save_success:
                    logger.info("FAISS 인덱스 저장 완료")
                    
                    # 통계 정보 출력
                    stats = self.vector_store.get_stats()
                    logger.info(f"벡터 스토어 통계: {stats}")
                    
                    return True
                else:
                    logger.error("FAISS 인덱스 저장 실패")
                    return False
            else:
                logger.error("FAISS 인덱스 구축 실패")
                return False
                
        except Exception as e:
            logger.error(f"FAISS 인덱스 구축 중 오류: {e}")
            return False
    
    def run(self) -> bool:
        """전체 프로세스 실행"""
        logger.info("SQLite → FAISS 변환 프로세스 시작")
        
        try:
            # 1. SQLite에서 문서 로드
            documents = self.load_documents_from_sqlite()
            if not documents:
                logger.error("로드할 문서가 없습니다")
                return False
            
            # 2. 문서 전처리
            texts, metadatas = self.process_documents_for_embedding(documents)
            if not texts:
                logger.error("전처리된 문서가 없습니다")
                return False
            
            # 3. FAISS 인덱스 구축
            success = self.build_faiss_index(texts, metadatas)
            
            if success:
                logger.info("✅ SQLite → FAISS 변환 완료!")
                return True
            else:
                logger.error("❌ SQLite → FAISS 변환 실패")
                return False
                
        except Exception as e:
            logger.error(f"전체 프로세스 실행 중 오류: {e}")
            return False

if __name__ == "__main__":
    print("🔄 SQLite → FAISS 벡터 인덱스 구축 시작")
    print("=" * 50)
    
    builder = SQLiteToFAISSBuilder()
    success = builder.run()
    
    if success:
        print("\n✅ 벡터 인덱스 구축 완료!")
        print("이제 하이브리드 검색을 사용할 수 있습니다.")
    else:
        print("\n❌ 벡터 인덱스 구축 실패")
        print("로그를 확인해주세요.")
