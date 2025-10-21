#!/usr/bin/env python3
"""
메모리 효율적인 벡터 임베딩 생성 스크립트
진행상황을 실시간으로 확인할 수 있고 메모리 사용량을 최적화합니다.
"""

import sys
import os
import sqlite3
import logging
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Generator
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/efficient_vector_build.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EfficientVectorBuilder:
    """메모리 효율적인 벡터 임베딩 빌더"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        self.db_path = db_path
        self.model = None
        self.index = None
        self.dimension = 768
        self.batch_size = 32  # 작은 배치 크기로 메모리 절약
        self.chunk_size = 1000  # 청크 크기
        
        # 임베딩 저장 경로
        self.embeddings_dir = Path("data/embeddings")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_model(self):
        """모델 초기화 (메모리 효율적)"""
        try:
            logger.info("Model initialization...")
            
            # GPU 사용 가능하면 사용, 아니면 CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Device: {device}")
            
            # 모델 로드
            self.model = SentenceTransformer('jhgan/ko-sroberta-multitask', device=device)
            
            # 메모리 절약을 위한 설정
            if device == 'cpu':
                self.model.half()  # Float16으로 변환하여 메모리 절약
            
            logger.info("Model initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    def get_document_count(self) -> int:
        """전체 문서 수 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 법률 문서 수
            cursor.execute("SELECT COUNT(*) FROM assembly_laws")
            law_count = cursor.fetchone()[0]
            
            # 조문 수
            cursor.execute("SELECT COUNT(*) FROM assembly_articles")
            article_count = cursor.fetchone()[0]
            
            conn.close()
            
            total_count = law_count + article_count
            logger.info(f"Total documents: {total_count} (Laws: {law_count}, Articles: {article_count})")
            return total_count
            
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def load_documents_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        """배치 단위로 문서 로드"""
        documents = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 법률 문서 로드
            cursor.execute("""
                SELECT 
                    law_id as id,
                    law_name as title,
                    full_text as content,
                    'law' as type,
                    law_type as category,
                    keywords,
                    summary
                FROM assembly_laws 
                WHERE full_text IS NOT NULL AND LENGTH(full_text) > 50
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            laws = cursor.fetchall()
            for law in laws:
                documents.append({
                    'id': f"law_{law['id']}",
                    'title': law['title'],
                    'content': law['content'],
                    'type': 'law',
                    'category': law['category'],
                    'keywords': law['keywords'],
                    'summary': law['summary']
                })
            
            # 조문 문서 로드 (법률 문서가 부족한 경우)
            remaining_limit = limit - len(documents)
            if remaining_limit > 0:
                cursor.execute("""
                    SELECT 
                        a.article_number as id,
                        a.article_title as title,
                        a.article_content as content,
                        'article' as type,
                        l.law_name as category,
                        '' as keywords,
                        '' as summary
                    FROM assembly_articles a
                    JOIN assembly_laws l ON a.law_id = l.law_id
                    WHERE a.article_content IS NOT NULL AND LENGTH(a.article_content) > 20
                    LIMIT ? OFFSET ?
                """, (remaining_limit, max(0, offset - len(documents))))
                
                articles = cursor.fetchall()
                for article in articles:
                    documents.append({
                        'id': f"article_{article['id']}",
                        'title': f"{article['category']} 제{article['id']}조",
                        'content': article['content'],
                        'type': 'article',
                        'category': article['category'],
                        'keywords': '',
                        'summary': ''
                    })
            
            conn.close()
            return documents
            
        except Exception as e:
            logger.error(f"문서 로드 실패: {e}")
            return []
    
    def create_embeddings_batch(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """배치 단위로 임베딩 생성"""
        try:
            # 텍스트 준비
            texts = []
            for doc in documents:
                # 제목과 내용을 결합하여 임베딩 생성
                text = f"{doc['title']}\n{doc['content']}"
                texts.append(text)
            
            # 임베딩 생성
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return np.array([])
    
    def save_embeddings_batch(self, embeddings: np.ndarray, documents: List[Dict[str, Any]], batch_idx: int):
        """배치 단위로 임베딩 저장"""
        try:
            # 임베딩을 파일로 저장
            embeddings_file = self.embeddings_dir / f"embeddings_batch_{batch_idx:04d}.npy"
            metadata_file = self.embeddings_dir / f"metadata_batch_{batch_idx:04d}.json"
            
            # 임베딩 저장
            np.save(embeddings_file, embeddings)
            
            # 메타데이터 저장
            metadata = []
            for i, doc in enumerate(documents):
                metadata.append({
                    'id': doc['id'],
                    'title': doc['title'],
                    'type': doc['type'],
                    'category': doc['category'],
                    'embedding_index': i
                })
            
            import json
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"배치 {batch_idx} 저장 완료: {len(embeddings)}개 임베딩")
            
        except Exception as e:
            logger.error(f"임베딩 저장 실패: {e}")
    
    def build_faiss_index(self) -> bool:
        """FAISS 인덱스 구축"""
        try:
            logger.info("FAISS 인덱스 구축 시작...")
            
            # 모든 배치 파일 로드
            batch_files = list(self.embeddings_dir.glob("embeddings_batch_*.npy"))
            batch_files.sort()
            
            if not batch_files:
                logger.error("임베딩 파일을 찾을 수 없습니다.")
                return False
            
            logger.info(f"총 {len(batch_files)}개 배치 파일 발견")
            
            # 첫 번째 배치로 인덱스 초기화
            first_embeddings = np.load(batch_files[0])
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (코사인 유사도)
            
            # 모든 임베딩을 인덱스에 추가
            total_embeddings = 0
            for i, batch_file in enumerate(batch_files):
                logger.info(f"배치 {i+1}/{len(batch_files)} 처리 중...")
                
                embeddings = np.load(batch_file)
                
                # 정규화 (코사인 유사도를 위해)
                faiss.normalize_L2(embeddings)
                
                # 인덱스에 추가
                self.index.add(embeddings)
                total_embeddings += len(embeddings)
                
                # 메모리 정리
                del embeddings
                gc.collect()
            
            logger.info(f"FAISS 인덱스 구축 완료: {total_embeddings}개 벡터")
            
            # 인덱스 저장
            index_path = self.embeddings_dir / "legal_vector_index.faiss"
            faiss.write_index(self.index, str(index_path))
            logger.info(f"인덱스 저장 완료: {index_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"FAISS 인덱스 구축 실패: {e}")
            return False
    
    def build_vector_embeddings(self) -> bool:
        """벡터 임베딩 구축 메인 함수"""
        try:
            logger.info("=" * 60)
            logger.info("Memory Efficient Vector Embedding Build Started")
            logger.info("=" * 60)
            
            # 1. 모델 초기화
            if not self.initialize_model():
                return False
            
            # 2. 전체 문서 수 확인
            total_docs = self.get_document_count()
            if total_docs == 0:
                logger.error("처리할 문서가 없습니다.")
                return False
            
            # 3. 배치 단위로 처리
            total_batches = (total_docs + self.chunk_size - 1) // self.chunk_size
            logger.info(f"Total batches to process: {total_batches}")
            
            processed_docs = 0
            
            for batch_idx in range(total_batches):
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches}...")
                
                # 문서 로드
                offset = batch_idx * self.chunk_size
                documents = self.load_documents_batch(offset, self.chunk_size)
                
                if not documents:
                    logger.warning(f"배치 {batch_idx + 1}에서 문서를 로드할 수 없습니다.")
                    continue
                
                logger.info(f"  - Loaded documents: {len(documents)}")
                
                # 임베딩 생성
                logger.info("  - Creating embeddings...")
                embeddings = self.create_embeddings_batch(documents)
                
                if len(embeddings) == 0:
                    logger.warning(f"배치 {batch_idx + 1}에서 임베딩을 생성할 수 없습니다.")
                    continue
                
                # 임베딩 저장
                logger.info("  - Saving embeddings...")
                self.save_embeddings_batch(embeddings, documents, batch_idx)
                
                processed_docs += len(documents)
                
                # 진행률 출력
                progress = (batch_idx + 1) / total_batches * 100
                logger.info(f"  - Progress: {progress:.1f}% ({processed_docs}/{total_docs} documents)")
                
                # 메모리 정리
                del documents, embeddings
                gc.collect()
            
            # 4. FAISS 인덱스 구축
            logger.info("\nBuilding FAISS index...")
            if not self.build_faiss_index():
                return False
            
            # 5. 통계 정보 저장
            stats = {
                'total_documents': processed_docs,
                'total_batches': total_batches,
                'embedding_dimension': self.dimension,
                'model_name': 'jhgan/ko-sroberta-multitask',
                'created_at': datetime.now().isoformat(),
                'index_type': 'flat',
                'similarity_metric': 'cosine'
            }
            
            stats_file = self.embeddings_dir / "vector_stats.json"
            import json
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            logger.info("=" * 60)
            logger.info("Vector embedding build completed!")
            logger.info(f"Total processed documents: {processed_docs}")
            logger.info(f"Embedding dimension: {self.dimension}")
            logger.info(f"Model: jhgan/ko-sroberta-multitask")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Vector embedding build failed: {e}")
            return False
    
    def test_vector_search(self, query: str = "계약서 검토", top_k: int = 5) -> bool:
        """벡터 검색 테스트"""
        try:
            logger.info(f"\n벡터 검색 테스트: '{query}'")
            
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # 검색 수행
            scores, indices = self.index.search(query_embedding, top_k)
            
            logger.info("검색 결과:")
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                logger.info(f"  {i+1}. 점수: {score:.4f}, 인덱스: {idx}")
            
            return True
            
        except Exception as e:
            logger.error(f"벡터 검색 테스트 실패: {e}")
            return False

def main():
    """메인 함수"""
    # 로그 디렉토리 생성
    Path("logs").mkdir(exist_ok=True)
    
    # 벡터 빌더 초기화
    builder = EfficientVectorBuilder()
    
    # 벡터 임베딩 구축
    success = builder.build_vector_embeddings()
    
    if success:
        # 테스트 수행
        builder.test_vector_search()
        logger.info("벡터 임베딩 구축 및 테스트 완료!")
    else:
        logger.error("벡터 임베딩 구축 실패!")

if __name__ == "__main__":
    main()
