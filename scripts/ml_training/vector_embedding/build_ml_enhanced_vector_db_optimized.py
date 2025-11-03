#!/usr/bin/env python3
"""
ML 강화 벡터 임베딩 생성기 (성능 최적화 버전)
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data.vector_store import LegalVectorStore

# Windows 콘솔에서 UTF-8 인코딩 설정
if os.name == 'nt':  # Windows
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except AttributeError:
        # 이미 UTF-8로 설정된 경우 무시
        pass

logger = logging.getLogger(__name__)


class OptimizedMLEnhancedVectorBuilder:
    """성능 최적화된 ML 강화 벡터 빌더"""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask", 
                 batch_size: int = 200, max_workers: int = None):
        """
        최적화된 벡터 빌더 초기화
        
        Args:
            model_name: 사용할 Sentence-BERT 모델명
            batch_size: 배치 크기 (증가)
            max_workers: 최대 워커 수
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        
        # 벡터 스토어 초기화
        self.vector_store = LegalVectorStore(
            model_name=model_name,
            dimension=768,
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
        
        logger.info(f"OptimizedMLEnhancedVectorBuilder initialized with model: {model_name}")
        logger.info(f"Batch size: {batch_size}, Max workers: {self.max_workers}")
    
    def build_embeddings(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        ML 강화 벡터 임베딩 생성 (최적화 버전)
        
        Args:
            input_dir: 입력 디렉토리
            output_dir: 출력 디렉토리
            
        Returns:
            처리 결과 통계
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting optimized ML-enhanced vector embedding generation from: {input_path}")
        
        # JSON 파일 찾기
        json_files = list(input_path.rglob("ml_enhanced_*.json"))
        logger.info(f"Found {len(json_files)} ML-enhanced files to process")
        
        if not json_files:
            logger.warning("No ML-enhanced files found!")
            return self.stats
        
        # 파일들을 배치로 나누기
        batches = [json_files[i:i + self.batch_size] for i in range(0, len(json_files), self.batch_size)]
        logger.info(f"Processing {len(batches)} batches with {self.batch_size} files each")
        
        # 병렬 처리로 배치 처리
        all_documents = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 배치별로 병렬 처리
            batch_futures = []
            for batch_idx, batch_files in enumerate(batches):
                future = executor.submit(self._process_batch_parallel, batch_files, batch_idx)
                batch_futures.append(future)
            
            # 진행 상황 표시
            for future in tqdm(batch_futures, desc="Processing batches", unit="batch"):
                try:
                    batch_documents, batch_stats = future.result()
                    all_documents.extend(batch_documents)
                    
                    # 통계 업데이트
                    for key in self.stats:
                        if key in batch_stats:
                            self.stats[key] += batch_stats[key]
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    self.stats['errors'].append(f"Batch error: {e}")
        
        # 벡터 임베딩 생성 및 저장
        logger.info(f"Creating embeddings for {len(all_documents)} documents...")
        
        # 문서를 청크로 나누어 처리 (메모리 효율성)
        chunk_size = 1000
        for i in tqdm(range(0, len(all_documents), chunk_size), desc="Creating embeddings"):
            chunk = all_documents[i:i + chunk_size]
            texts = [doc['text'] for doc in chunk]
            metadatas = [doc['metadata'] for doc in chunk]
            self.vector_store.add_documents(texts, metadatas)
        
        # 인덱스 저장
        index_path = output_path / "ml_enhanced_faiss_index"
        self.vector_store.save_index(str(index_path))
        
        # 통계 저장
        stats_path = output_path / "ml_enhanced_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        logger.info("ML-enhanced vector embedding generation completed!")
        logger.info(f"Total documents processed: {self.stats['total_documents_created']}")
        logger.info(f"Total articles processed: {self.stats['total_articles_processed']}")
        logger.info(f"Errors: {len(self.stats['errors'])}")
        
        return self.stats
    
    def _process_batch_parallel(self, batch_files: List[Path], batch_idx: int) -> tuple:
        """배치 파일들을 병렬 처리"""
        batch_documents = []
        batch_stats = {
            'total_files_processed': 0,
            'total_laws_processed': 0,
            'total_articles_processed': 0,
            'main_articles_processed': 0,
            'supplementary_articles_processed': 0,
            'total_chunks_created': 0,
            'total_documents_created': 0,
            'errors': []
        }
        
        for file_path in batch_files:
            try:
                documents = self._process_single_file(file_path)
                batch_documents.extend(documents)
                batch_stats['total_files_processed'] += 1
                batch_stats['total_documents_created'] += len(documents)
                
            except Exception as e:
                error_msg = f"Error processing file {file_path}: {e}"
                logger.error(error_msg)
                batch_stats['errors'].append(error_msg)
        
        return batch_documents, batch_stats
    
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
                    
                    # 문서 생성 (배치 처리)
                    article_documents = self._create_article_documents_batch(
                        all_articles, law_metadata
                    )
                    documents.extend(article_documents)
                    
                except Exception as e:
                    error_msg = f"Error processing law {law_data.get('law_name', 'Unknown')}: {e}"
                    logger.error(error_msg)
                    continue
        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
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
                logger.error(f"Error processing article {article.get('article_number', 'Unknown')}: {e}")
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
    
    parser = argparse.ArgumentParser(description="ML 강화 벡터 임베딩 생성기 (최적화 버전)")
    parser.add_argument("--input", required=True, help="입력 디렉토리")
    parser.add_argument("--output", required=True, help="출력 디렉토리")
    parser.add_argument("--batch-size", type=int, default=200, help="배치 크기 (기본값: 200)")
    parser.add_argument("--max-workers", type=int, default=None, help="최대 워커 수")
    parser.add_argument("--log-level", default="INFO", help="로그 레벨")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 벡터 빌더 초기화 및 실행
    builder = OptimizedMLEnhancedVectorBuilder(
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
    
    stats = builder.build_embeddings(args.input, args.output)
    
    print(f"\n=== 처리 완료 ===")
    print(f"총 파일 수: {stats['total_files_processed']}")
    print(f"총 법률 수: {stats['total_laws_processed']}")
    print(f"총 조문 수: {stats['total_articles_processed']}")
    print(f"총 문서 수: {stats['total_documents_created']}")
    print(f"에러 수: {len(stats['errors'])}")


if __name__ == "__main__":
    main()
