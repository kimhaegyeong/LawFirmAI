#!/usr/bin/env python3
"""
중단??복구 기능???�는 ML 강화 벡터 ?�베???�성�?
"""

import gc
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# ?�로?�트 루트�?Python 경로??추�?
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from source.data.vector_store import LegalVectorStore

# Windows 콘솔?�서 UTF-8 ?�코???�정
if os.name == 'nt':  # Windows
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except AttributeError:
        # ?��? UTF-8�??�정??경우 무시
        pass

logger = logging.getLogger(__name__)


class ResumableVectorBuilder:
    """중단??복구 기능???�는 벡터 빌더"""

    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask",
                 batch_size: int = 20, chunk_size: int = 200,
                 checkpoint_interval: int = 100):
        """
        중단??복구 가?�한 벡터 빌더 초기??

        Args:
            model_name: ?�용???�베??모델�?
            batch_size: ?�일 배치 ?�기
            chunk_size: 문서 �?�� ?�기
            checkpoint_interval: 체크?�인???�??간격 (문서 ??
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.checkpoint_interval = checkpoint_interval

        # Sentence-BERT 모델???�베??차원 (768)
        embedding_dimension = 768

        # 벡터 ?�토??초기??
        self.vector_store = LegalVectorStore(
            model_name=model_name,
            dimension=embedding_dimension,
            index_type="flat"
        )

        # ?�계 초기??
        self.stats = {
            'total_files_processed': 0,
            'total_laws_processed': 0,
            'total_articles_processed': 0,
            'main_articles_processed': 0,
            'supplementary_articles_processed': 0,
            'total_chunks_created': 0,
            'total_documents_created': 0,
            'errors': [],
            'start_time': datetime.now().isoformat(),
            'last_checkpoint': None,
            'processed_files': set(),
            'processed_documents': 0
        }

        logger.info(f"ResumableVectorBuilder initialized with model: {model_name}")
        logger.info(f"Batch size: {batch_size}, Chunk size: {chunk_size}")
        logger.info(f"Checkpoint interval: {checkpoint_interval}")

    def build_embeddings(self, input_dir: str, output_dir: str, resume: bool = True) -> Dict[str, Any]:
        """
        중단??복구 가?�한 벡터 ?�베???�성

        Args:
            input_dir: ?�력 ?�렉?�리
            output_dir: 출력 ?�렉?�리
            resume: ?�전 ?�업 ?�어??진행?��? ?��?

        Returns:
            처리 결과 ?�계
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 체크?�인???�일 경로
        checkpoint_file = output_path / "checkpoint.json"
        progress_file = output_path / "progress.pkl"

        # ?�전 ?�업 복구
        if resume and checkpoint_file.exists():
            self._load_checkpoint(checkpoint_file, progress_file)
            logger.info(f"Resumed from checkpoint: {self.stats['last_checkpoint']}")
            logger.info(f"Already processed: {self.stats['total_documents_created']} documents")

        logger.info(f"Starting resumable vector embedding generation from: {input_path}")

        # JSON ?�일 찾기
        json_files = list(input_path.rglob("ml_enhanced_*.json"))
        logger.info(f"Found {len(json_files)} ML-enhanced files to process")

        if not json_files:
            logger.warning("No ML-enhanced files found!")
            return self.stats

        # ?��? 처리???�일???�외
        remaining_files = [f for f in json_files if str(f) not in self.stats['processed_files']]
        logger.info(f"Remaining files to process: {len(remaining_files)}")

        if not remaining_files:
            logger.info("All files already processed!")
            return self.stats

        # ?�일?�을 ?��? 배치�??�누???�차 처리
        batches = [remaining_files[i:i + self.batch_size] for i in range(0, len(remaining_files), self.batch_size)]
        logger.info(f"Processing {len(batches)} batches with {self.batch_size} files each")

        try:
            for batch_idx, batch_files in enumerate(tqdm(batches, desc="Processing batches", unit="batch")):
                try:
                    batch_documents = self._process_batch_sequential(batch_files, batch_idx)

                    # 문서?�을 ?��? �?���??�누??처리
                    for i in range(0, len(batch_documents), self.chunk_size):
                        chunk = batch_documents[i:i + self.chunk_size]
                        texts = [doc['text'] for doc in chunk]
                        metadatas = [doc['metadata'] for doc in chunk]

                        try:
                            self.vector_store.add_documents(texts, metadatas)
                            self.stats['total_documents_created'] += len(chunk)

                            # 메모�??�리
                            del texts, metadatas
                            gc.collect()

                            # 체크?�인???�??
                            if self.stats['total_documents_created'] % self.checkpoint_interval == 0:
                                self._save_checkpoint(checkpoint_file, progress_file)
                                logger.info(f"Checkpoint saved: {self.stats['total_documents_created']} documents processed")

                        except Exception as e:
                            logger.error(f"Error creating embeddings for chunk {i//self.chunk_size}: {e}")
                            self.stats['errors'].append(f"Embedding chunk error: {e}")

                    # 배치 ?�료 ??체크?�인???�??
                    self._save_checkpoint(checkpoint_file, progress_file)

                    # 메모�??�리
                    gc.collect()

                    # 진행 ?�황 로깅
                    if (batch_idx + 1) % 5 == 0:
                        logger.info(f"Processed {batch_idx + 1}/{len(batches)} batches")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    self.stats['errors'].append(f"Batch {batch_idx} error: {e}")
                    # ?�러가 발생?�도 ?�음 배치 계속 처리
                    continue

        except KeyboardInterrupt:
            logger.info("Process interrupted by user. Saving checkpoint...")
            self._save_checkpoint(checkpoint_file, progress_file)
            logger.info("Checkpoint saved. You can resume later with --resume flag.")
            return self.stats

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self._save_checkpoint(checkpoint_file, progress_file)
            raise

        # 최종 ?�덱???�??
        index_path = output_path / "ml_enhanced_faiss_index"
        self.vector_store.save_index(str(index_path))

        # 최종 ?�계 ?�??
        self.stats['end_time'] = datetime.now().isoformat()
        stats_path = output_path / "ml_enhanced_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        # 체크?�인???�일 ?�리
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if progress_file.exists():
            progress_file.unlink()

        logger.info("Resumable vector embedding generation completed!")
        logger.info(f"Total documents processed: {self.stats['total_documents_created']}")
        logger.info(f"Total articles processed: {self.stats['total_articles_processed']}")
        logger.info(f"Errors: {len(self.stats['errors'])}")

        return self.stats

    def _process_batch_sequential(self, batch_files: List[Path], batch_idx: int) -> List[Dict[str, Any]]:
        """배치 ?�일?�을 ?�차 처리"""
        batch_documents = []

        for file_path in batch_files:
            try:
                # ?�일???��? 처리?�었?��? ?�인
                if str(file_path) in self.stats['processed_files']:
                    continue

                documents = self._process_single_file(file_path)
                batch_documents.extend(documents)
                self.stats['total_files_processed'] += 1
                self.stats['processed_files'].add(str(file_path))

            except Exception as e:
                error_msg = f"Error processing file {file_path}: {e}"
                logger.error(error_msg)
                self.stats['errors'].append(error_msg)

        return batch_documents

    def _process_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """?�일 ?�일 처리"""
        documents = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)

            # ?�일 구조 ?�인
            if isinstance(file_data, dict) and 'laws' in file_data:
                laws = file_data['laws']
            elif isinstance(file_data, list):
                laws = file_data
            else:
                laws = [file_data]

            for law_data in laws:
                try:
                    # 법률 메�??�이??추출
                    law_metadata = self._extract_law_metadata(law_data)

                    # 본칙 조문 처리
                    articles = law_data.get('articles', [])
                    if not isinstance(articles, list):
                        articles = []

                    # 부�?조문 처리
                    supplementary_articles = law_data.get('supplementary_articles', [])
                    if not isinstance(supplementary_articles, list):
                        supplementary_articles = []

                    # 모든 조문???�나??리스?�로 ?�치�?
                    all_articles = articles + supplementary_articles

                    # 문서 ?�성
                    article_documents = self._create_article_documents_batch(
                        all_articles, law_metadata
                    )
                    documents.extend(article_documents)

                    # ?�계 ?�데?�트
                    self.stats['total_laws_processed'] += 1
                    self.stats['total_articles_processed'] += len(all_articles)

                    main_articles = [a for a in all_articles if not a.get('is_supplementary', False)]
                    supp_articles = [a for a in all_articles if a.get('is_supplementary', False)]

                    self.stats['main_articles_processed'] += len(main_articles)
                    self.stats['supplementary_articles_processed'] += len(supp_articles)

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
        """조문?�을 배치�?문서 변??""
        documents = []

        for article in articles:
            try:
                # 조문 메�??�이???�성
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

                # 문서 ID ?�성
                document_id = f"{law_metadata['law_id']}_article_{article_metadata['article_number']}"
                article_metadata['document_id'] = document_id

                # ?�스??구성
                text_parts = []

                # 조문 번호?� ?�목
                if article_metadata['article_number']:
                    if article_metadata['article_title']:
                        text_parts.append(f"{article_metadata['article_number']}({article_metadata['article_title']})")
                    else:
                        text_parts.append(article_metadata['article_number'])

                # 조문 ?�용
                article_content = article.get('article_content', '')
                if article_content:
                    text_parts.append(article_content)

                # ?�위 조문??
                sub_articles = article.get('sub_articles', [])
                if isinstance(sub_articles, list):
                    for sub_article in sub_articles:
                        if isinstance(sub_article, dict):
                            sub_content = sub_article.get('content', '')
                            if sub_content:
                                text_parts.append(sub_content)

                # 최종 ?�스??
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
        """법률 메�??�이??추출"""
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

    def _save_checkpoint(self, checkpoint_file: Path, progress_file: Path):
        """체크?�인???�??""
        try:
            self.stats['last_checkpoint'] = datetime.now().isoformat()

            # set??list�?변?�하??JSON 직렬??가?�하�?만들�?
            checkpoint_stats = self.stats.copy()
            checkpoint_stats['processed_files'] = list(checkpoint_stats['processed_files'])

            # ?�계 ?�보 ?�??
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_stats, f, ensure_ascii=False, indent=2)

            # 벡터 ?�토???�태 ?�??
            vector_state = {
                'document_count': len(self.vector_store.document_metadata),
                'index_trained': self.vector_store.index.is_trained if self.vector_store.index else False
            }

            with open(progress_file, 'wb') as f:
                pickle.dump(vector_state, f)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, checkpoint_file: Path, progress_file: Path):
        """체크?�인??로드"""
        try:
            # ?�계 ?�보 로드
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                saved_stats = json.load(f)

            # ?�계 ?�보 복원
            self.stats.update(saved_stats)

            # processed_files�?set?�로 변??
            if isinstance(self.stats['processed_files'], list):
                self.stats['processed_files'] = set(self.stats['processed_files'])

            # 벡터 ?�토???�태 ?�인
            if progress_file.exists():
                with open(progress_file, 'rb') as f:
                    vector_state = pickle.load(f)
                logger.info(f"Vector store state: {vector_state['document_count']} documents, trained: {vector_state['index_trained']}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            # 체크?�인??로드 ?�패 ??처음부???�작
            self.stats['processed_files'] = set()


def main():
    """메인 ?�수"""
    import argparse

    parser = argparse.ArgumentParser(description="중단??복구 기능???�는 ML 강화 벡터 ?�베???�성�?)
    parser.add_argument("--input", required=True, help="?�력 ?�렉?�리")
    parser.add_argument("--output", required=True, help="출력 ?�렉?�리")
    parser.add_argument("--batch-size", type=int, default=20, help="배치 ?�기 (기본�? 20)")
    parser.add_argument("--chunk-size", type=int, default=200, help="�?�� ?�기 (기본�? 200)")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="체크?�인???�??간격 (기본�? 100)")
    parser.add_argument("--resume", action="store_true", help="?�전 ?�업 ?�어??진행")
    parser.add_argument("--log-level", default="INFO", help="로그 ?�벨")

    args = parser.parse_args()

    # 로깅 ?�정
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 벡터 빌더 초기??�??�행
    builder = ResumableVectorBuilder(
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        checkpoint_interval=args.checkpoint_interval
    )

    stats = builder.build_embeddings(args.input, args.output, resume=args.resume)

    print(f"\n=== 처리 ?�료 ===")
    print(f"�??�일 ?? {stats['total_files_processed']}")
    print(f"�?법률 ?? {stats['total_laws_processed']}")
    print(f"�?조문 ?? {stats['total_articles_processed']}")
    print(f"�?문서 ?? {stats['total_documents_created']}")
    print(f"?�러 ?? {len(stats['errors'])}")

    if stats.get('start_time') and stats.get('end_time'):
        start_time = datetime.fromisoformat(stats['start_time'])
        end_time = datetime.fromisoformat(stats['end_time'])
        duration = end_time - start_time
        print(f"�??�요 ?�간: {duration}")


if __name__ == "__main__":
    main()
