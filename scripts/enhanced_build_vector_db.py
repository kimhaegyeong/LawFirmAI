#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 벡터DB 구축 스크립트 (전처리된 데이터 → 벡터DB)

전처리된 데이터를 사용하여 벡터DB를 구축합니다.
하이브리드 검색을 위해 SQLite와 FAISS를 모두 구축합니다.

사용법:
    python scripts/enhanced_build_vector_db.py --mode build
    python scripts/enhanced_build_vector_db.py --mode laws
    python scripts/enhanced_build_vector_db.py --mode precedents
"""

import os
import sys
import json
import argparse
import logging
import gc
import psutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 필요한 라이브러리 import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Please install faiss-cpu or faiss-gpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available. Please install sentence-transformers")

from source.data.database import DatabaseManager
from source.data.enhanced_data_processor import EnhancedLegalDataProcessor

logger = logging.getLogger(__name__)


class EnhancedVectorDBBuilder:
    """향상된 벡터DB 구축 클래스"""
    
    def __init__(self, processed_data_dir: str = "./data/processed"):
        """초기화"""
        self.processed_data_dir = Path(processed_data_dir)
        self.embeddings_dir = Path("./data/embeddings")
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # 데이터베이스 관리자 초기화
        self.db_manager = DatabaseManager()
        
        # Sentence-BERT 모델 초기화
        self.model = None
        self.model_name = "jhgan/ko-sroberta-multitask"
        self.dimension = 768
        
        # FAISS 인덱스
        self.faiss_index = None
        self.document_metadata = []
        
        # 구축 통계
        self.build_stats = {
            'start_time': datetime.now().isoformat(),
            'laws_processed': 0,
            'precedents_processed': 0,
            'constitutional_processed': 0,
            'interpretations_processed': 0,
            'total_documents': 0,
            'total_chunks': 0,
            'embeddings_generated': 0,
            'faiss_index_size': 0,
            'sqlite_records': 0,
            'memory_usage': [],
            'errors': []
        }
        
        # 로깅 설정
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/enhanced_vector_db_build_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
    
    def get_memory_usage(self) -> float:
        """현재 메모리 사용률 반환 (GB)"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024 * 1024)
    
    def check_memory_limit(self) -> bool:
        """메모리 사용량 체크"""
        current_memory = self.get_memory_usage()
        self.build_stats['memory_usage'].append({
            'timestamp': datetime.now().isoformat(),
            'memory_gb': current_memory
        })
        
        if current_memory > 12:  # 12GB 이상 사용 시 경고
            logger.warning(f"메모리 사용량이 높습니다: {current_memory:.2f}GB")
            return False
        return True
    
    def force_garbage_collection(self):
        """강제 가비지 컬렉션 실행"""
        gc.collect()
        logger.debug("가비지 컬렉션 실행 완료")
    
    def load_sentence_transformer(self):
        """Sentence-BERT 모델 로드"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers not available")
        
        logger.info(f"Sentence-BERT 모델 로딩 중: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Sentence-BERT 모델 로딩 완료")
        except Exception as e:
            logger.error(f"Sentence-BERT 모델 로딩 실패: {e}")
            raise
    
    def create_faiss_index(self, dimension: int = 768):
        """FAISS 인덱스 생성"""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available")
        
        logger.info(f"FAISS 인덱스 생성 중 (차원: {dimension})")
        try:
            # FlatL2 인덱스 생성 (정확한 검색을 위해)
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.dimension = dimension
            logger.info("FAISS 인덱스 생성 완료")
        except Exception as e:
            logger.error(f"FAISS 인덱스 생성 실패: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """텍스트 임베딩 생성"""
        if not self.model:
            self.load_sentence_transformer()
        
        logger.info(f"{len(texts)}개 텍스트의 임베딩 생성 중...")
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            logger.info(f"임베딩 생성 완료: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    def process_documents_from_processed_data(self, data_type: str) -> List[Dict[str, Any]]:
        """전처리된 데이터에서 문서 처리"""
        type_dir = self.processed_data_dir / data_type
        
        if not type_dir.exists():
            logger.warning(f"{data_type} 디렉토리가 존재하지 않습니다: {type_dir}")
            return []
        
        json_files = list(type_dir.glob("*.json"))
        all_documents = []
        
        logger.info(f"{data_type} 데이터 처리 시작: {len(json_files)}개 파일")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 단일 문서인 경우
                if isinstance(data, dict):
                    if data.get('chunks'):
                        # 청크가 있는 경우 각 청크를 별도 문서로 처리
                        for chunk in data['chunks']:
                            document = {
                                'id': f"{data['id']}_{chunk['id']}",
                                'original_id': data['id'],
                                'chunk_id': chunk['id'],
                                'text': chunk['text'],
                                'metadata': {
                                    'data_type': data_type,
                                    'original_document': data.get('law_name', data.get('case_name', data.get('title', ''))),
                                    'chunk_start': chunk.get('start_pos', 0),
                                    'chunk_end': chunk.get('end_pos', 0),
                                    'chunk_length': chunk.get('length', 0),
                                    'word_count': chunk.get('word_count', 0),
                                    'entities': chunk.get('entities', {}),
                                    'processed_at': data.get('processed_at', ''),
                                    'is_valid': data.get('is_valid', False)
                                }
                            }
                            all_documents.append(document)
                    else:
                        # 청크가 없는 경우 전체 내용을 하나의 문서로 처리
                        content = data.get('cleaned_content', '') or data.get('content', '')
                        if content:
                            document = {
                                'id': data['id'],
                                'original_id': data['id'],
                                'chunk_id': 'full',
                                'text': content,
                                'metadata': {
                                    'data_type': data_type,
                                    'original_document': data.get('law_name', data.get('case_name', data.get('title', ''))),
                                    'chunk_start': 0,
                                    'chunk_end': len(content),
                                    'chunk_length': len(content),
                                    'word_count': len(content.split()),
                                    'entities': data.get('entities', {}),
                                    'processed_at': data.get('processed_at', ''),
                                    'is_valid': data.get('is_valid', False)
                                }
                            }
                            all_documents.append(document)
                
                # 메모리 체크
                if not self.check_memory_limit():
                    self.force_garbage_collection()
                
            except Exception as e:
                logger.error(f"파일 처리 실패 {json_file}: {e}")
                self.build_stats['errors'].append(f"File processing error {json_file}: {str(e)}")
        
        logger.info(f"{data_type} 데이터 처리 완료: {len(all_documents)}개 문서")
        return all_documents
    
    def build_vector_database(self, data_types: List[str] = None):
        """벡터 데이터베이스 구축"""
        if data_types is None:
            data_types = ['laws', 'precedents', 'constitutional_decisions', 'legal_interpretations']
        
        logger.info("벡터 데이터베이스 구축 시작")
        start_time = datetime.now()
        
        try:
            # Sentence-BERT 모델 로드
            self.load_sentence_transformer()
            
            # FAISS 인덱스 생성
            self.create_faiss_index(self.dimension)
            
            # 모든 문서 수집
            all_documents = []
            for data_type in data_types:
                documents = self.process_documents_from_processed_data(data_type)
                all_documents.extend(documents)
                
                # 통계 업데이트
                if data_type == 'laws':
                    self.build_stats['laws_processed'] = len(documents)
                elif data_type == 'precedents':
                    self.build_stats['precedents_processed'] = len(documents)
                elif data_type == 'constitutional_decisions':
                    self.build_stats['constitutional_processed'] = len(documents)
                elif data_type == 'legal_interpretations':
                    self.build_stats['interpretations_processed'] = len(documents)
            
            if not all_documents:
                logger.error("처리할 문서가 없습니다.")
                return False
            
            # 텍스트 추출
            texts = [doc['text'] for doc in all_documents]
            logger.info(f"총 {len(texts)}개 문서의 임베딩 생성 시작")
            
            # 배치 단위로 임베딩 생성 (메모리 절약)
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.generate_embeddings(batch_texts)
                all_embeddings.append(batch_embeddings)
                
                # 메모리 체크
                if not self.check_memory_limit():
                    self.force_garbage_collection()
                
                logger.info(f"배치 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} 완료")
            
            # 모든 임베딩 결합
            embeddings = np.vstack(all_embeddings)
            logger.info(f"전체 임베딩 생성 완료: {embeddings.shape}")
            
            # FAISS 인덱스에 임베딩 추가
            logger.info("FAISS 인덱스에 임베딩 추가 중...")
            self.faiss_index.add(embeddings.astype('float32'))
            
            # 메타데이터 저장
            self.document_metadata = all_documents
            
            # 통계 업데이트
            self.build_stats['total_documents'] = len(all_documents)
            self.build_stats['total_chunks'] = len(all_documents)
            self.build_stats['embeddings_generated'] = len(embeddings)
            self.build_stats['faiss_index_size'] = self.faiss_index.ntotal
            
            # FAISS 인덱스 저장
            self.save_faiss_index()
            
            # SQLite 데이터베이스에 저장
            self.save_to_sqlite(all_documents)
            
            # 임베딩 저장
            self.save_embeddings(embeddings, all_documents)
            
            # 구축 완료
            duration = datetime.now() - start_time
            self.build_stats['end_time'] = datetime.now().isoformat()
            self.build_stats['duration_seconds'] = duration.total_seconds()
            
            logger.info(f"벡터 데이터베이스 구축 완료 (소요시간: {duration})")
            self.print_build_statistics()
            
            return True
            
        except Exception as e:
            logger.error(f"벡터 데이터베이스 구축 실패: {e}")
            self.build_stats['errors'].append(f"Build error: {str(e)}")
            return False
    
    def save_faiss_index(self):
        """FAISS 인덱스 저장"""
        index_path = self.embeddings_dir / "faiss_index.bin"
        logger.info(f"FAISS 인덱스 저장 중: {index_path}")
        
        try:
            faiss.write_index(self.faiss_index, str(index_path))
            logger.info("FAISS 인덱스 저장 완료")
        except Exception as e:
            logger.error(f"FAISS 인덱스 저장 실패: {e}")
            raise
    
    def save_embeddings(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """임베딩과 메타데이터 저장"""
        embeddings_path = self.embeddings_dir / "embeddings.npy"
        metadata_path = self.embeddings_dir / "metadata.json"
        
        logger.info(f"임베딩 저장 중: {embeddings_path}")
        try:
            np.save(embeddings_path, embeddings)
            logger.info("임베딩 저장 완료")
        except Exception as e:
            logger.error(f"임베딩 저장 실패: {e}")
            raise
        
        logger.info(f"메타데이터 저장 중: {metadata_path}")
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            logger.info("메타데이터 저장 완료")
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {e}")
            raise
    
    def save_to_sqlite(self, documents: List[Dict[str, Any]]):
        """SQLite 데이터베이스에 저장"""
        logger.info("SQLite 데이터베이스에 저장 중...")
        
        try:
            # 데이터베이스 테이블 생성 (이미 초기화 시 생성됨)
            pass
            
            # 문서 저장
            for doc in documents:
                doc_data = {
                    'id': doc['id'],
                    'document_type': doc['metadata']['data_type'],
                    'title': doc['metadata']['original_document'],
                    'content': doc['text'],
                    'source_url': None
                }
                
                # 메타데이터 분리
                metadata = doc['metadata']
                law_meta = None
                prec_meta = None
                const_meta = None
                interp_meta = None
                
                if doc['metadata']['data_type'] == 'laws':
                    law_meta = {
                        'law_name': metadata['original_document'],
                        'article_number': None,
                        'promulgation_date': None,
                        'enforcement_date': None,
                        'department': None
                    }
                elif doc['metadata']['data_type'] == 'precedents':
                    prec_meta = {
                        'case_number': doc['original_id'],
                        'court_name': None,
                        'decision_date': None,
                        'case_type': None
                    }
                elif doc['metadata']['data_type'] == 'constitutional_decisions':
                    const_meta = {
                        'case_number': doc['original_id'],
                        'decision_date': None,
                        'case_type': None
                    }
                elif doc['metadata']['data_type'] == 'legal_interpretations':
                    interp_meta = {
                        'interpretation_date': None,
                        'department': None
                    }
                
                self.db_manager.add_document(
                    doc_data=doc_data,
                    law_meta=law_meta,
                    prec_meta=prec_meta,
                    const_meta=const_meta,
                    interp_meta=interp_meta
                )
            
            self.build_stats['sqlite_records'] = len(documents)
            logger.info(f"SQLite 데이터베이스 저장 완료: {len(documents)}개 레코드")
            
        except Exception as e:
            logger.error(f"SQLite 데이터베이스 저장 실패: {e}")
            raise
    
    def print_build_statistics(self):
        """구축 통계 출력"""
        print("\n=== 벡터DB 구축 통계 ===")
        print(f"법령 문서: {self.build_stats['laws_processed']}개")
        print(f"판례 문서: {self.build_stats['precedents_processed']}개")
        print(f"헌재결정례: {self.build_stats['constitutional_processed']}개")
        print(f"법령해석례: {self.build_stats['interpretations_processed']}개")
        print(f"총 문서 수: {self.build_stats['total_documents']}개")
        print(f"총 청크 수: {self.build_stats['total_chunks']}개")
        print(f"생성된 임베딩: {self.build_stats['embeddings_generated']}개")
        print(f"FAISS 인덱스 크기: {self.build_stats['faiss_index_size']}개")
        print(f"SQLite 레코드: {self.build_stats['sqlite_records']}개")
        
        if self.build_stats['memory_usage']:
            max_memory = max(usage['memory_gb'] for usage in self.build_stats['memory_usage'])
            print(f"최대 메모리 사용량: {max_memory:.2f}GB")
        
        if self.build_stats['errors']:
            print(f"오류 수: {len(self.build_stats['errors'])}개")
    
    def save_build_report(self):
        """구축 보고서 저장"""
        report_path = self.embeddings_dir / "vector_db_build_report.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.build_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"구축 보고서 저장 완료: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="향상된 벡터DB 구축 스크립트")
    parser.add_argument("--mode", 
                       choices=["build", "laws", "precedents", "constitutional", "interpretations", "multiple"],
                       default="build",
                       help="구축 모드")
    parser.add_argument("--processed_dir", 
                       default="./data/processed",
                       help="전처리된 데이터 디렉토리")
    parser.add_argument("--types", 
                       nargs='+',
                       choices=["laws", "precedents", "constitutional_decisions", "legal_interpretations"],
                       help="구축할 데이터 타입 (multiple 모드에서 사용)")
    
    args = parser.parse_args()
    
    # 로그 디렉토리 생성
    Path("logs").mkdir(exist_ok=True)
    
    # 벡터DB 구축기 초기화
    builder = EnhancedVectorDBBuilder(args.processed_dir)
    
    try:
        if args.mode == "build":
            # 전체 구축
            success = builder.build_vector_database()
        elif args.mode == "multiple":
            if not args.types:
                print("multiple 모드에서는 --types 옵션을 지정해야 합니다.")
                return
            success = builder.build_vector_database(args.types)
        else:
            # 특정 데이터 타입만 구축
            data_type_map = {
                "laws": "laws",
                "precedents": "precedents", 
                "constitutional": "constitutional_decisions",
                "interpretations": "legal_interpretations"
            }
            data_type = data_type_map[args.mode]
            success = builder.build_vector_database([data_type])
        
        if success:
            print("벡터DB 구축이 성공적으로 완료되었습니다!")
            builder.save_build_report()
        else:
            print("벡터DB 구축이 실패했습니다.")
            builder.save_build_report()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"벡터DB 구축 중 오류 발생: {e}")
        builder.save_build_report()
        sys.exit(1)


if __name__ == "__main__":
    main()
