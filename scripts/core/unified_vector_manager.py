#!/usr/bin/env python3
"""
통합 벡터 임베딩 매니저
다양한 임베딩 모델과 빌드 전략을 지원하는 통합 시스템
"""

import sys
import os
from pathlib import Path
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Generator
import numpy as np
import faiss
import torch
import gc
from enum import Enum
from dataclasses import dataclass
import json
import pickle

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence_transformers not available. Install with: pip install sentence-transformers")
    SentenceTransformer = None


class EmbeddingModel(Enum):
    """지원하는 임베딩 모델"""
    KO_SROBERTA = "jhgan/ko-sroberta-multitask"
    KO_BERT = "jhgan/ko-sbert-nli"
    MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    KOREAN_BERT = "klue/bert-base"


class BuildMode(Enum):
    """빌드 모드"""
    FULL = "full"           # 전체 재구축
    INCREMENTAL = "incremental"  # 증분 빌드
    RESUMABLE = "resumable"  # 재시작 가능한 빌드
    CPU_OPTIMIZED = "cpu_optimized"  # CPU 최적화


@dataclass
class VectorConfig:
    """벡터 임베딩 설정"""
    model: EmbeddingModel = EmbeddingModel.KO_SROBERTA
    build_mode: BuildMode = BuildMode.FULL
    db_path: str = "data/lawfirm.db"
    embeddings_dir: str = "data/embeddings"
    batch_size: int = 32
    chunk_size: int = 1000
    dimension: int = 768
    use_gpu: bool = True
    memory_optimized: bool = True
    log_level: str = "INFO"


class UnifiedVectorManager:
    """통합 벡터 임베딩 매니저"""
    
    def __init__(self, config: VectorConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.model = None
        self.index = None
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        
        # 임베딩 저장 경로
        self.embeddings_dir = Path(self.config.embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(f"vector_manager_{self.config.build_mode.value}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # 핸들러가 이미 있으면 제거
        if logger.handlers:
            logger.handlers.clear()
            
        # 파일 핸들러
        log_file = f"logs/unified_vector_{self.config.build_mode.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def build_vector_index(self) -> Dict[str, Any]:
        """벡터 인덱스 구축"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting vector index build in {self.config.build_mode.value} mode")
        
        results = {
            'mode': self.config.build_mode.value,
            'model': self.config.model.value,
            'start_time': self.start_time.isoformat(),
            'phases': {},
            'errors': [],
            'summary': {}
        }
        
        try:
            # Phase 1: 모델 초기화
            results['phases']['model_init'] = self._initialize_model()
            
            # Phase 2: 문서 로드 및 전처리
            results['phases']['data_preparation'] = self._prepare_data()
            
            # Phase 3: 임베딩 생성
            results['phases']['embedding_generation'] = self._generate_embeddings()
            
            # Phase 4: 인덱스 구축
            results['phases']['index_building'] = self._build_index()
            
            # Phase 5: 저장 및 검증
            results['phases']['save_and_validate'] = self._save_and_validate()
            
            # 완료
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = duration
            results['summary'] = {
                'processed_count': self.processed_count,
                'error_count': self.error_count,
                'success_rate': (self.processed_count / (self.processed_count + self.error_count)) * 100 if (self.processed_count + self.error_count) > 0 else 0,
                'total_vectors': self.processed_count,
                'index_size_mb': self._get_index_size()
            }
            
            self.logger.info(f"Vector index build completed successfully in {duration:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Vector index build failed: {e}", exc_info=True)
            results['errors'].append(str(e))
            return results
    
    def _initialize_model(self) -> Dict[str, Any]:
        """모델 초기화"""
        self.logger.info("Phase 1: Model initialization")
        
        phase_results = {
            'model_loaded': False,
            'device': 'cpu',
            'errors': []
        }
        
        try:
            if SentenceTransformer is None:
                raise ImportError("sentence_transformers not available")
            
            # GPU 사용 가능 여부 확인
            device = 'cuda' if self.config.use_gpu and torch.cuda.is_available() else 'cpu'
            phase_results['device'] = device
            
            self.logger.info(f"Loading model: {self.config.model.value} on {device}")
            
            # 모델 로드
            self.model = SentenceTransformer(self.config.model.value, device=device)
            
            # 메모리 최적화 설정
            if self.config.memory_optimized and device == 'cpu':
                self.model.half()  # Float16으로 변환하여 메모리 절약
            
            phase_results['model_loaded'] = True
            self.logger.info("Model initialization completed successfully")
            
        except Exception as e:
            error_msg = f"Model initialization failed: {e}"
            self.logger.error(error_msg)
            phase_results['errors'].append(error_msg)
            
        return phase_results
    
    def _prepare_data(self) -> Dict[str, Any]:
        """데이터 준비"""
        self.logger.info("Phase 2: Data preparation")
        
        phase_results = {
            'total_documents': 0,
            'document_types': {},
            'errors': []
        }
        
        try:
            # 전체 문서 수 조회
            total_docs = self._get_document_count()
            phase_results['total_documents'] = total_docs
            
            # 문서 타입별 수 조회
            doc_types = self._get_document_types()
            phase_results['document_types'] = doc_types
            
            self.logger.info(f"Data preparation completed: {total_docs} documents")
            
        except Exception as e:
            error_msg = f"Data preparation failed: {e}"
            self.logger.error(error_msg)
            phase_results['errors'].append(error_msg)
            
        return phase_results
    
    def _generate_embeddings(self) -> Dict[str, Any]:
        """임베딩 생성"""
        self.logger.info("Phase 3: Embedding generation")
        
        phase_results = {
            'embeddings_generated': 0,
            'batches_processed': 0,
            'errors': []
        }
        
        try:
            if self.config.build_mode == BuildMode.INCREMENTAL:
                phase_results = self._generate_incremental_embeddings()
            elif self.config.build_mode == BuildMode.RESUMABLE:
                phase_results = self._generate_resumable_embeddings()
            else:
                phase_results = self._generate_full_embeddings()
            
            self.logger.info(f"Embedding generation completed: {phase_results['embeddings_generated']} embeddings")
            
        except Exception as e:
            error_msg = f"Embedding generation failed: {e}"
            self.logger.error(error_msg)
            phase_results['errors'].append(error_msg)
            
        return phase_results
    
    def _generate_full_embeddings(self) -> Dict[str, Any]:
        """전체 임베딩 생성"""
        results = {
            'embeddings_generated': 0,
            'batches_processed': 0,
            'errors': []
        }
        
        all_embeddings = []
        all_metadata = []
        
        try:
            # 배치 단위로 문서 처리
            offset = 0
            batch_count = 0
            
            while True:
                documents = self._load_documents_batch(offset, self.config.chunk_size)
                if not documents:
                    break
                
                batch_count += 1
                self.logger.info(f"Processing batch {batch_count}: {len(documents)} documents")
                
                # 텍스트 추출
                texts = [doc['text'] for doc in documents if doc['text']]
                
                if texts:
                    # 임베딩 생성
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.config.batch_size,
                        show_progress_bar=True,
                        convert_to_numpy=True
                    )
                    
                    all_embeddings.append(embeddings)
                    
                    # 메타데이터 저장
                    for i, doc in enumerate(documents):
                        if i < len(embeddings):
                            metadata = {
                                'doc_id': doc['id'],
                                'doc_type': doc['type'],
                                'doc_title': doc.get('title', ''),
                                'embedding_index': len(all_metadata)
                            }
                            all_metadata.append(metadata)
                    
                    results['embeddings_generated'] += len(embeddings)
                    self.processed_count += len(embeddings)
                
                results['batches_processed'] = batch_count
                offset += self.config.chunk_size
                
                # 메모리 정리
                if self.config.memory_optimized:
                    gc.collect()
            
            # 임베딩 저장
            if all_embeddings:
                self._save_embeddings(all_embeddings, all_metadata)
            
        except Exception as e:
            error_msg = f"Full embedding generation failed: {e}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
            
        return results
    
    def _generate_incremental_embeddings(self) -> Dict[str, Any]:
        """증분 임베딩 생성"""
        results = {
            'embeddings_generated': 0,
            'batches_processed': 0,
            'errors': []
        }
        
        try:
            # 기존 임베딩 로드
            existing_embeddings, existing_metadata = self._load_existing_embeddings()
            
            # 새로운 문서만 처리
            new_documents = self._get_new_documents(existing_metadata)
            
            if new_documents:
                self.logger.info(f"Processing {len(new_documents)} new documents")
                
                # 새로운 임베딩 생성
                texts = [doc['text'] for doc in new_documents if doc['text']]
                
                if texts:
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.config.batch_size,
                        show_progress_bar=True,
                        convert_to_numpy=True
                    )
                    
                    # 기존 임베딩과 결합
                    if existing_embeddings is not None:
                        all_embeddings = np.vstack([existing_embeddings, embeddings])
                    else:
                        all_embeddings = embeddings
                    
                    # 메타데이터 업데이트
                    for i, doc in enumerate(new_documents):
                        if i < len(embeddings):
                            metadata = {
                                'doc_id': doc['id'],
                                'doc_type': doc['type'],
                                'doc_title': doc.get('title', ''),
                                'embedding_index': len(existing_metadata) + i
                            }
                            existing_metadata.append(metadata)
                    
                    # 저장
                    self._save_embeddings([all_embeddings], existing_metadata)
                    results['embeddings_generated'] = len(embeddings)
                    self.processed_count += len(embeddings)
            
        except Exception as e:
            error_msg = f"Incremental embedding generation failed: {e}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
            
        return results
    
    def _generate_resumable_embeddings(self) -> Dict[str, Any]:
        """재시작 가능한 임베딩 생성"""
        results = {
            'embeddings_generated': 0,
            'batches_processed': 0,
            'errors': []
        }
        
        try:
            # 체크포인트 로드
            checkpoint = self._load_checkpoint()
            
            if checkpoint:
                self.logger.info(f"Resuming from checkpoint: {checkpoint['last_processed_id']}")
                start_offset = checkpoint['offset']
            else:
                start_offset = 0
            
            # 배치 단위로 문서 처리
            offset = start_offset
            batch_count = checkpoint.get('batch_count', 0) if checkpoint else 0
            
            while True:
                documents = self._load_documents_batch(offset, self.config.chunk_size)
                if not documents:
                    break
                
                batch_count += 1
                self.logger.info(f"Processing batch {batch_count}: {len(documents)} documents")
                
                # 텍스트 추출
                texts = [doc['text'] for doc in documents if doc['text']]
                
                if texts:
                    # 임베딩 생성
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.config.batch_size,
                        show_progress_bar=True,
                        convert_to_numpy=True
                    )
                    
                    # 임베딩 저장 (배치별로)
                    batch_metadata = []
                    for i, doc in enumerate(documents):
                        if i < len(embeddings):
                            metadata = {
                                'doc_id': doc['id'],
                                'doc_type': doc['type'],
                                'doc_title': doc.get('title', ''),
                                'embedding_index': offset + i
                            }
                            batch_metadata.append(metadata)
                    
                    self._save_batch_embeddings(embeddings, batch_metadata, batch_count)
                    results['embeddings_generated'] += len(embeddings)
                    self.processed_count += len(embeddings)
                
                results['batches_processed'] = batch_count
                offset += self.config.chunk_size
                
                # 체크포인트 저장
                self._save_checkpoint({
                    'offset': offset,
                    'batch_count': batch_count,
                    'last_processed_id': documents[-1]['id'] if documents else None
                })
                
                # 메모리 정리
                if self.config.memory_optimized:
                    gc.collect()
            
        except Exception as e:
            error_msg = f"Resumable embedding generation failed: {e}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
            
        return results
    
    def _build_index(self) -> Dict[str, Any]:
        """인덱스 구축"""
        self.logger.info("Phase 4: Index building")
        
        phase_results = {
            'index_built': False,
            'index_type': 'faiss',
            'errors': []
        }
        
        try:
            # 임베딩 로드
            embeddings, metadata = self._load_embeddings()
            
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("No embeddings found")
            
            # FAISS 인덱스 생성
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
            
            # 정규화 (cosine similarity를 위해)
            faiss.normalize_L2(embeddings)
            
            # 인덱스에 벡터 추가
            self.index.add(embeddings)
            
            phase_results['index_built'] = True
            phase_results['index_type'] = 'faiss'
            phase_results['vector_count'] = len(embeddings)
            phase_results['dimension'] = dimension
            
            self.logger.info(f"Index built successfully: {len(embeddings)} vectors, dimension {dimension}")
            
        except Exception as e:
            error_msg = f"Index building failed: {e}"
            self.logger.error(error_msg)
            phase_results['errors'].append(error_msg)
            
        return phase_results
    
    def _save_and_validate(self) -> Dict[str, Any]:
        """저장 및 검증"""
        self.logger.info("Phase 5: Save and validate")
        
        phase_results = {
            'index_saved': False,
            'metadata_saved': False,
            'validation_passed': False,
            'errors': []
        }
        
        try:
            # 인덱스 저장
            index_path = self.embeddings_dir / f"faiss_index_{self.config.model.value.replace('/', '_')}.index"
            faiss.write_index(self.index, str(index_path))
            phase_results['index_saved'] = True
            phase_results['index_path'] = str(index_path)
            
            # 메타데이터 저장
            metadata_path = self.embeddings_dir / f"metadata_{self.config.model.value.replace('/', '_')}.json"
            embeddings, metadata = self._load_embeddings()
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            phase_results['metadata_saved'] = True
            phase_results['metadata_path'] = str(metadata_path)
            
            # 검증
            validation_result = self._validate_index()
            phase_results['validation_passed'] = validation_result['passed']
            phase_results['validation_details'] = validation_result
            
            self.logger.info("Save and validation completed successfully")
            
        except Exception as e:
            error_msg = f"Save and validation failed: {e}"
            self.logger.error(error_msg)
            phase_results['errors'].append(error_msg)
            
        return phase_results
    
    def _get_document_count(self) -> int:
        """전체 문서 수 조회"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                # 법률 문서 수
                cursor.execute("SELECT COUNT(*) FROM laws")
                law_count = cursor.fetchone()[0]
                
                # 조문 수
                cursor.execute("SELECT COUNT(*) FROM articles")
                article_count = cursor.fetchone()[0]
                
                total_count = law_count + article_count
                return total_count
                
        except Exception as e:
            self.logger.error(f"Failed to get document count: {e}")
            return 0
    
    def _get_document_types(self) -> Dict[str, int]:
        """문서 타입별 수 조회"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                types = {}
                
                # 법률 문서
                cursor.execute("SELECT COUNT(*) FROM laws")
                types['laws'] = cursor.fetchone()[0]
                
                # 조문
                cursor.execute("SELECT COUNT(*) FROM articles")
                types['articles'] = cursor.fetchone()[0]
                
                return types
                
        except Exception as e:
            self.logger.error(f"Failed to get document types: {e}")
            return {}
    
    def _load_documents_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        """배치 단위로 문서 로드"""
        documents = []
        
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                # 법률 문서 로드
                cursor.execute("""
                    SELECT law_id, law_name, clean_text, law_type, law_field
                    FROM laws
                    WHERE clean_text IS NOT NULL AND clean_text != ''
                    ORDER BY law_id
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                laws = cursor.fetchall()
                for law in laws:
                    documents.append({
                        'id': law[0],
                        'type': 'law',
                        'title': law[1],
                        'text': law[2],
                        'law_type': law[3],
                        'law_field': law[4]
                    })
                
                # 조문 로드
                cursor.execute("""
                    SELECT article_id, law_id, article_title, article_content
                    FROM articles
                    WHERE article_content IS NOT NULL AND article_content != ''
                    ORDER BY article_id
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                articles = cursor.fetchall()
                for article in articles:
                    documents.append({
                        'id': article[0],
                        'type': 'article',
                        'title': article[2],
                        'text': article[3],
                        'law_id': article[1]
                    })
                
        except Exception as e:
            self.logger.error(f"Failed to load documents batch: {e}")
            
        return documents
    
    def _save_embeddings(self, embeddings: List[np.ndarray], metadata: List[Dict[str, Any]]):
        """임베딩 저장"""
        try:
            # 임베딩 결합
            if len(embeddings) > 1:
                combined_embeddings = np.vstack(embeddings)
            else:
                combined_embeddings = embeddings[0]
            
            # 임베딩 저장
            embeddings_path = self.embeddings_dir / f"embeddings_{self.config.model.value.replace('/', '_')}.npy"
            np.save(embeddings_path, combined_embeddings)
            
            # 메타데이터 저장
            metadata_path = self.embeddings_dir / f"metadata_{self.config.model.value.replace('/', '_')}.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            self.logger.info(f"Embeddings saved: {embeddings_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save embeddings: {e}")
    
    def _load_embeddings(self) -> tuple:
        """임베딩 로드"""
        try:
            embeddings_path = self.embeddings_dir / f"embeddings_{self.config.model.value.replace('/', '_')}.npy"
            metadata_path = self.embeddings_dir / f"metadata_{self.config.model.value.replace('/', '_')}.pkl"
            
            if not embeddings_path.exists() or not metadata_path.exists():
                return None, []
            
            embeddings = np.load(embeddings_path)
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            return embeddings, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            return None, []
    
    def _load_existing_embeddings(self) -> tuple:
        """기존 임베딩 로드"""
        return self._load_embeddings()
    
    def _get_new_documents(self, existing_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """새로운 문서 조회"""
        try:
            existing_ids = {meta['doc_id'] for meta in existing_metadata}
            
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                # 새로운 법률 문서
                cursor.execute("""
                    SELECT law_id, law_name, clean_text, law_type, law_field
                    FROM laws
                    WHERE law_id NOT IN ({}) AND clean_text IS NOT NULL AND clean_text != ''
                """.format(','.join(['?' for _ in existing_ids])), list(existing_ids))
                
                new_documents = []
                laws = cursor.fetchall()
                for law in laws:
                    new_documents.append({
                        'id': law[0],
                        'type': 'law',
                        'title': law[1],
                        'text': law[2],
                        'law_type': law[3],
                        'law_field': law[4]
                    })
                
                return new_documents
                
        except Exception as e:
            self.logger.error(f"Failed to get new documents: {e}")
            return []
    
    def _save_batch_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], batch_num: int):
        """배치별 임베딩 저장"""
        try:
            batch_path = self.embeddings_dir / f"batch_{batch_num:04d}.npy"
            metadata_path = self.embeddings_dir / f"batch_{batch_num:04d}_metadata.pkl"
            
            np.save(batch_path, embeddings)
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
        except Exception as e:
            self.logger.error(f"Failed to save batch embeddings: {e}")
    
    def _save_checkpoint(self, checkpoint: Dict[str, Any]):
        """체크포인트 저장"""
        try:
            checkpoint_path = self.embeddings_dir / "checkpoint.json"
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """체크포인트 로드"""
        try:
            checkpoint_path = self.embeddings_dir / "checkpoint.json"
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _validate_index(self) -> Dict[str, Any]:
        """인덱스 검증"""
        try:
            if self.index is None:
                return {'passed': False, 'error': 'Index not built'}
            
            # 기본 검증
            vector_count = self.index.ntotal
            dimension = self.index.d
            
            # 샘플 검색 테스트
            test_vector = np.random.random((1, dimension)).astype('float32')
            faiss.normalize_L2(test_vector)
            
            distances, indices = self.index.search(test_vector, min(10, vector_count))
            
            validation_result = {
                'passed': True,
                'vector_count': vector_count,
                'dimension': dimension,
                'test_search_passed': len(indices[0]) > 0
            }
            
            return validation_result
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _get_index_size(self) -> float:
        """인덱스 크기 (MB)"""
        try:
            if self.index is None:
                return 0.0
            
            # 대략적인 크기 계산
            vector_count = self.index.ntotal
            dimension = self.index.d
            size_bytes = vector_count * dimension * 4  # float32
            return size_bytes / (1024 * 1024)  # MB
            
        except Exception:
            return 0.0


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Vector Embedding Manager')
    parser.add_argument('--model', choices=['ko-sroberta', 'ko-bert', 'multilingual', 'korean-bert'], 
                       default='ko-sroberta', help='Embedding model')
    parser.add_argument('--mode', choices=['full', 'incremental', 'resumable', 'cpu_optimized'], 
                       default='full', help='Build mode')
    parser.add_argument('--db-path', default='data/lawfirm.db', help='Database path')
    parser.add_argument('--embeddings-dir', default='data/embeddings', help='Embeddings directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    # 모델 매핑
    model_mapping = {
        'ko-sroberta': EmbeddingModel.KO_SROBERTA,
        'ko-bert': EmbeddingModel.KO_BERT,
        'multilingual': EmbeddingModel.MULTILINGUAL,
        'korean-bert': EmbeddingModel.KOREAN_BERT
    }
    
    # 모드 매핑
    mode_mapping = {
        'full': BuildMode.FULL,
        'incremental': BuildMode.INCREMENTAL,
        'resumable': BuildMode.RESUMABLE,
        'cpu_optimized': BuildMode.CPU_OPTIMIZED
    }
    
    # 설정 생성
    config = VectorConfig(
        model=model_mapping[args.model],
        build_mode=mode_mapping[args.mode],
        db_path=args.db_path,
        embeddings_dir=args.embeddings_dir,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        use_gpu=not args.no_gpu,
        log_level=args.log_level
    )
    
    # 매니저 생성 및 실행
    manager = UnifiedVectorManager(config)
    results = manager.build_vector_index()
    
    # 결과 출력
    print(f"\n=== Vector Build Results ===")
    print(f"Model: {results['model']}")
    print(f"Mode: {results['mode']}")
    print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
    print(f"Vectors: {results['summary'].get('total_vectors', 0)}")
    print(f"Index Size: {results['summary'].get('index_size_mb', 0):.2f} MB")
    print(f"Success Rate: {results['summary'].get('success_rate', 0):.1f}%")
    
    if results['errors']:
        print(f"\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
