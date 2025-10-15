#!/usr/bin/env python3
"""
ML 강화 벡터 임베딩 생성기 (CPU 최적화 버전)
"""

import json
import logging
import sys
import os
import signal
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import gc

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from source.data.vector_store import LegalVectorStore

# Windows 콘솔에서 UTF-8 인코딩 설정 (개선된 버전)
if os.name == 'nt':  # Windows
    try:
        # 환경변수 설정
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # 콘솔 인코딩 설정 (안전한 방법)
        if hasattr(sys.stdout, 'buffer'):
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        else:
            # 이미 설정된 경우 무시
            pass
    except Exception as e:
        # 인코딩 설정 실패 시 기본 설정 유지
        print(f"Warning: Could not set UTF-8 encoding: {e}")

# 로깅 설정 (안전한 방법)
def setup_safe_logging():
    """안전한 로깅 설정"""
    try:
        # 기존 핸들러 제거
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 새로운 핸들러 설정
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # 포맷터 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # 루트 로거에 핸들러 추가
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
        
    except Exception as e:
        print(f"Warning: Could not setup logging: {e}")

# 안전한 로깅 설정 적용
setup_safe_logging()

logger = logging.getLogger(__name__)


class CheckpointManager:
    """체크포인트 관리 클래스"""
    
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_data = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """체크포인트 로드"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return {
            'completed_chunks': [],
            'total_chunks': 0,
            'start_time': None,
            'last_update': None
        }
    
    def save_checkpoint(self, completed_chunks: List[int], total_chunks: int):
        """체크포인트 저장"""
        try:
            checkpoint_data = {
                'completed_chunks': completed_chunks,
                'total_chunks': total_chunks,
                'start_time': self.checkpoint_data.get('start_time', time.time()),
                'last_update': time.time()
            }
            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            self.checkpoint_data = checkpoint_data
            logger.info(f"Checkpoint saved: {len(completed_chunks)}/{total_chunks} chunks completed")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def get_remaining_chunks(self, total_chunks: int) -> List[int]:
        """남은 청크 목록 반환"""
        completed = set(self.checkpoint_data.get('completed_chunks', []))
        return [i for i in range(total_chunks) if i not in completed]
    
    def is_resume_needed(self) -> bool:
        """재시작이 필요한지 확인"""
        return len(self.checkpoint_data.get('completed_chunks', [])) > 0
    
    def get_progress_info(self) -> Dict[str, Any]:
        """진행 상황 정보 반환"""
        completed = len(self.checkpoint_data.get('completed_chunks', []))
        total = self.checkpoint_data.get('total_chunks', 0)
        start_time = self.checkpoint_data.get('start_time')
        
        progress_info = {
            'completed_chunks': completed,
            'total_chunks': total,
            'progress_percentage': (completed / max(total, 1)) * 100 if total > 0 else 0
        }
        
        if start_time:
            elapsed_time = time.time() - start_time
            progress_info['elapsed_time'] = elapsed_time
            if completed > 0:
                avg_time_per_chunk = elapsed_time / completed
                remaining_chunks = total - completed
                progress_info['estimated_remaining_time'] = avg_time_per_chunk * remaining_chunks
        
        return progress_info


class CPUOptimizedVectorBuilder:
    """CPU 사용량 최적화된 벡터 빌더 (ko-sroberta-multitask 지원)"""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask", 
                 batch_size: int = 20, chunk_size: int = 200):
        """
        CPU 최적화된 벡터 빌더 초기화
        
        Args:
            model_name: 사용할 임베딩 모델명 (ko-sroberta-multitask 지원)
            batch_size: 파일 배치 크기 (작게 설정)
            chunk_size: 문서 청크 크기 (작게 설정)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        
        # Graceful shutdown 설정
        self.shutdown_requested = False
        self._setup_signal_handlers()
        
        # ko-sroberta-multitask 모델의 임베딩 차원 (768)
        embedding_dimension = 768 if "ko-sroberta-multitask" in model_name.lower() else 1024
        
        # 벡터 스토어 초기화
        self.vector_store = LegalVectorStore(
            model_name=model_name,
            dimension=embedding_dimension,
            index_type="flat"
        )
        
        # 통계 초기화
        self.stats = {
            'total_files_processed': 0,
            'total_laws_processed': 0,
            'total_articles_processed': 0,
            'main_articles_processed': 0,
            'supplementary_articles_processed': 0,
            'total_chunks_created': 0,
            'total_documents_created': 0,
            'errors': []
        }
        
        logger.info(f"CPUOptimizedVectorBuilder initialized with model: {model_name}")
        logger.info(f"Batch size: {batch_size}, Chunk size: {chunk_size}")
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정 (Graceful shutdown)"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
            self.shutdown_requested = True
        
        # Windows와 Unix 모두 지원
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def build_embeddings(self, input_dir: str, output_dir: str, resume: bool = True) -> Dict[str, Any]:
        """
        CPU 최적화된 벡터 임베딩 생성 (체크포인트 지원)
        
        Args:
            input_dir: 입력 디렉토리
            output_dir: 출력 디렉토리
            resume: 중단된 작업 재시작 여부
            
        Returns:
            처리 결과 통계
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 체크포인트 관리자 초기화
        checkpoint_file = output_path / "embedding_checkpoint.json"
        checkpoint_manager = CheckpointManager(str(checkpoint_file))
        
        logger.info(f"Starting CPU-optimized vector embedding generation from: {input_path}")
        
        # 재시작 확인
        if resume and checkpoint_manager.is_resume_needed():
            progress_info = checkpoint_manager.get_progress_info()
            logger.info(f"Resuming from checkpoint: {progress_info['completed_chunks']}/{progress_info['total_chunks']} chunks completed")
            logger.info(f"Progress: {progress_info['progress_percentage']:.1f}%")
            if 'estimated_remaining_time' in progress_info:
                remaining_hours = progress_info['estimated_remaining_time'] / 3600
                logger.info(f"Estimated remaining time: {remaining_hours:.1f} hours")
        
        # JSON 파일 찾기
        json_files = list(input_path.rglob("ml_enhanced_*.json"))
        logger.info(f"Found {len(json_files)} ML-enhanced files to process")
        
        if not json_files:
            logger.warning("No ML-enhanced files found!")
            return self.stats
        
        # 순차 처리로 CPU 사용량 최소화
        all_documents = []
        
        # 파일들을 작은 배치로 나누어 순차 처리
        batches = [json_files[i:i + self.batch_size] for i in range(0, len(json_files), self.batch_size)]
        logger.info(f"Processing {len(batches)} batches with {self.batch_size} files each")
        
        for batch_idx, batch_files in enumerate(tqdm(batches, desc="Processing batches", unit="batch")):
            try:
                batch_documents = self._process_batch_sequential(batch_files, batch_idx)
                all_documents.extend(batch_documents)
                
                # 메모리 정리
                gc.collect()
                
                # 진행 상황 로깅 (안전한 방법)
                if (batch_idx + 1) % 5 == 0:
                    try:
                        logger.info(f"Processed {batch_idx + 1}/{len(batches)} batches")
                    except Exception:
                        # 로깅 실패 시 print 사용
                        print(f"Processed {batch_idx + 1}/{len(batches)} batches")
                
            except Exception as e:
                error_msg = f"Error processing batch {batch_idx}: {e}"
                try:
                    logger.error(error_msg)
                except Exception:
                    print(f"ERROR: {error_msg}")
                self.stats['errors'].append(f"Batch {batch_idx} error: {e}")
        
        # 벡터 임베딩 생성 및 저장 (체크포인트 지원)
        logger.info(f"Creating embeddings for {len(all_documents)} documents...")
        
        total_chunks = (len(all_documents) + self.chunk_size - 1) // self.chunk_size
        
        # 재시작 시 남은 청크만 처리
        if resume and checkpoint_manager.is_resume_needed():
            remaining_chunks = checkpoint_manager.get_remaining_chunks(total_chunks)
            logger.info(f"Processing {len(remaining_chunks)} remaining chunks out of {total_chunks}")
            chunk_indices = remaining_chunks
        else:
            chunk_indices = list(range(total_chunks))
        
        completed_chunks = checkpoint_manager.checkpoint_data.get('completed_chunks', [])
        
        for chunk_idx in tqdm(chunk_indices, desc="Creating embeddings"):
            # Graceful shutdown 확인
            if self.shutdown_requested:
                logger.info("Graceful shutdown requested. Saving checkpoint and exiting...")
                checkpoint_manager.save_checkpoint(completed_chunks, total_chunks)
                logger.info("Checkpoint saved. You can resume later with --resume flag.")
                return self.stats
            
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(all_documents))
            chunk = all_documents[start_idx:end_idx]
            
            texts = [doc['text'] for doc in chunk]
            metadatas = [doc['metadata'] for doc in chunk]
            
            try:
                self.vector_store.add_documents(texts, metadatas)
                completed_chunks.append(chunk_idx)
                
                # 체크포인트 저장 (매 10개 청크마다)
                if len(completed_chunks) % 10 == 0:
                    checkpoint_manager.save_checkpoint(completed_chunks, total_chunks)
                
                # 메모리 정리
                del texts, metadatas
                gc.collect()
                
            except Exception as e:
                error_msg = f"Error creating embeddings for chunk {chunk_idx}: {e}"
                try:
                    logger.error(error_msg)
                except Exception:
                    print(f"ERROR: {error_msg}")
                self.stats['errors'].append(f"Embedding chunk error: {e}")
        
        # 인덱스 저장
        index_path = output_path / "ml_enhanced_faiss_index"
        self.vector_store.save_index(str(index_path))
        
        # 통계 저장
        stats_path = output_path / "ml_enhanced_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        # 완료 시 체크포인트 정리
        if len(completed_chunks) == total_chunks:
            logger.info("All chunks completed. Cleaning up checkpoint file...")
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info("Checkpoint file removed.")
        
        logger.info("CPU-optimized vector embedding generation completed!")
        logger.info(f"Total documents processed: {self.stats['total_documents_created']}")
        logger.info(f"Total articles processed: {self.stats['total_articles_processed']}")
        logger.info(f"Errors: {len(self.stats['errors'])}")
        
        return self.stats
    
    def _process_batch_sequential(self, batch_files: List[Path], batch_idx: int) -> List[Dict[str, Any]]:
        """배치 파일들을 순차 처리"""
        batch_documents = []
        
        for file_path in batch_files:
            try:
                documents = self._process_single_file(file_path)
                batch_documents.extend(documents)
                self.stats['total_files_processed'] += 1
                self.stats['total_documents_created'] += len(documents)
                
            except Exception as e:
                error_msg = f"Error processing file {file_path}: {e}"
                try:
                    logger.error(error_msg)
                except Exception:
                    print(f"ERROR: {error_msg}")
                self.stats['errors'].append(error_msg)
        
        return batch_documents
    
    def _process_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """단일 파일 처리 (최적화 버전)"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            # 파일 구조 확인
            if isinstance(file_data, dict) and 'laws' in file_data:
                laws = file_data['laws']
            elif isinstance(file_data, list):
                laws = file_data
            else:
                laws = [file_data]
            
            for law_data in laws:
                try:
                    # 법률 메타데이터 추출
                    law_metadata = self._extract_law_metadata(law_data)
                    
                    # 본칙 조문 처리
                    articles = law_data.get('articles', [])
                    if not isinstance(articles, list):
                        articles = []
                    
                    # 부칙 조문 처리
                    supplementary_articles = law_data.get('supplementary_articles', [])
                    if not isinstance(supplementary_articles, list):
                        supplementary_articles = []
                    
                    # 모든 조문을 하나의 리스트로 합치기
                    all_articles = articles + supplementary_articles
                    
                    # 문서 생성
                    article_documents = self._create_article_documents_batch(
                        all_articles, law_metadata
                    )
                    documents.extend(article_documents)
                    
                    # 통계 업데이트
                    self.stats['total_laws_processed'] += 1
                    self.stats['total_articles_processed'] += len(all_articles)
                    
                    main_articles = [a for a in all_articles if not a.get('is_supplementary', False)]
                    supp_articles = [a for a in all_articles if a.get('is_supplementary', False)]
                    
                    self.stats['main_articles_processed'] += len(main_articles)
                    self.stats['supplementary_articles_processed'] += len(supp_articles)
                    
                except Exception as e:
                    error_msg = f"Error processing law {law_data.get('law_name', 'Unknown')}: {e}"
                    try:
                        logger.error(error_msg)
                        logger.error(f"Law data keys: {list(law_data.keys())}")
                        if 'articles' in law_data and law_data['articles']:
                            first_article = law_data['articles'][0]
                            logger.error(f"First article keys: {list(first_article.keys())}")
                            logger.error(f"First article sub_articles type: {type(first_article.get('sub_articles'))}")
                            logger.error(f"First article references type: {type(first_article.get('references'))}")
                    except Exception:
                        print(f"ERROR: {error_msg}")
                    continue
        
        except Exception as e:
            error_msg = f"Error reading file {file_path}: {e}"
            try:
                logger.error(error_msg)
            except Exception:
                print(f"ERROR: {error_msg}")
            raise
        
        return documents
    
    def _create_article_documents_batch(self, articles: List[Dict[str, Any]], 
                                      law_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """조문들을 배치로 문서 변환 (최적화 버전)"""
        documents = []
        
        for article in articles:
            try:
                # 조문 메타데이터 생성
                article_metadata = {
                    **law_metadata,
                    'article_number': article.get('article_number', ''),
                    'article_title': article.get('article_title', ''),
                    'article_type': 'main' if not article.get('is_supplementary', False) else 'supplementary',
                    'is_supplementary': article.get('is_supplementary', False),
                    'ml_confidence_score': article.get('ml_confidence_score'),
                    'parsing_method': article.get('parsing_method', 'ml_enhanced'),
                    'word_count': article.get('word_count', 0),
                    'char_count': article.get('char_count', 0),
                    'sub_articles_count': len(article.get('sub_articles', [])) if isinstance(article.get('sub_articles'), list) else 0,
                    'references_count': len(article.get('references', [])) if isinstance(article.get('references'), list) else 0
                }
                
                # 문서 ID 생성
                document_id = f"{law_metadata['law_id']}_article_{article_metadata['article_number']}"
                article_metadata['document_id'] = document_id
                
                # 텍스트 구성 (최적화)
                text_parts = []
                
                # 조문 번호와 제목
                if article_metadata['article_number']:
                    if article_metadata['article_title']:
                        text_parts.append(f"{article_metadata['article_number']}({article_metadata['article_title']})")
                    else:
                        text_parts.append(article_metadata['article_number'])
                
                # 조문 내용
                article_content = article.get('article_content', '')
                if article_content:
                    text_parts.append(article_content)
                
                # 하위 조문들 (안전한 처리)
                sub_articles = article.get('sub_articles', [])
                if isinstance(sub_articles, list):
                    for sub_article in sub_articles:
                        if isinstance(sub_article, dict):
                            sub_content = sub_article.get('content', '')
                            if sub_content:
                                text_parts.append(sub_content)
                elif isinstance(sub_articles, (int, float)):
                    # sub_articles가 숫자인 경우 무시
                    pass
                
                # 최종 텍스트
                full_text = ' '.join(text_parts)
                
                if full_text.strip():
                    document = {
                        'id': document_id,
                        'text': full_text,
                        'metadata': article_metadata,
                        'chunks': [{
                            'id': f"{document_id}_chunk_0",
                            'text': full_text,
                            'start_pos': 0,
                            'end_pos': len(full_text),
                            'entities': article.get('references', []) if isinstance(article.get('references'), list) else []
                        }]
                    }
                    documents.append(document)
                
            except Exception as e:
                error_msg = f"Error processing article {article.get('article_number', 'Unknown')}: {e}"
                try:
                    logger.error(error_msg)
                    logger.error(f"Article keys: {list(article.keys())}")
                    logger.error(f"Article sub_articles: {article.get('sub_articles')}")
                    logger.error(f"Article references: {article.get('references')}")
                except Exception:
                    print(f"ERROR: {error_msg}")
                continue
        
        return documents
    
    def _extract_law_metadata(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """법률 메타데이터 추출"""
        return {
            'law_id': law_data.get('law_id') or f"ml_enhanced_{law_data.get('law_name', 'unknown').replace(' ', '_')}",
            'law_name': law_data.get('law_name', ''),
            'law_type': law_data.get('law_type', ''),
            'category': law_data.get('category', ''),
            'promulgation_number': law_data.get('promulgation_number', ''),
            'promulgation_date': law_data.get('promulgation_date', ''),
            'enforcement_date': law_data.get('enforcement_date', ''),
            'amendment_type': law_data.get('amendment_type', ''),
            'ministry': law_data.get('ministry', ''),
            'ml_enhanced': law_data.get('ml_enhanced', True),
            'parsing_quality_score': law_data.get('data_quality', {}).get('parsing_quality_score', 0.0) if isinstance(law_data.get('data_quality'), dict) else 0.0,
            'processing_version': law_data.get('processing_version', 'ml_enhanced_v1.0'),
            'control_characters_removed': True
        }


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML 강화 벡터 임베딩 생성기 (CPU 최적화 버전)")
    parser.add_argument("--input", required=True, help="입력 디렉토리")
    parser.add_argument("--output", required=True, help="출력 디렉토리")
    parser.add_argument("--batch-size", type=int, default=20, help="배치 크기 (기본값: 20)")
    parser.add_argument("--chunk-size", type=int, default=200, help="청크 크기 (기본값: 200)")
    parser.add_argument("--log-level", default="INFO", help="로그 레벨")
    parser.add_argument("--resume", action="store_true", help="중단된 작업 재시작")
    parser.add_argument("--no-resume", action="store_true", help="체크포인트 무시하고 처음부터 시작")
    
    args = parser.parse_args()
    
    # 로깅 설정 (안전한 방법)
    try:
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True  # 기존 설정 강제 재설정
        )
    except Exception as e:
        print(f"Warning: Could not setup logging: {e}")
        # 기본 print 사용
        print("Using basic print for output")
    
    # 벡터 빌더 초기화 및 실행 (ko-sroberta-multitask 사용)
    builder = CPUOptimizedVectorBuilder(
        model_name="jhgan/ko-sroberta-multitask",
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )
    
    stats = builder.build_embeddings(args.input, args.output, resume=not args.no_resume)
    
    print(f"\n=== 처리 완료 ===")
    print(f"총 파일 수: {stats['total_files_processed']}")
    print(f"총 법률 수: {stats['total_laws_processed']}")
    print(f"총 조문 수: {stats['total_articles_processed']}")
    print(f"총 문서 수: {stats['total_documents_created']}")
    print(f"에러 수: {len(stats['errors'])}")


if __name__ == "__main__":
    main()
