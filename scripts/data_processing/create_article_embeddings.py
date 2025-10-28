#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
현행법령 조문별 벡터 임베딩 생성 스크립트
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import psutil

# PyTorch 관련 import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore


def setup_local_logging():
    """로컬 로깅 설정 (setup_logging 대신 사용)"""
    # 환경 변수로 외부 라이브러리 로깅만 차단
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # 외부 라이브러리 로깅만 차단
    external_loggers = [
        "faiss", "sentence_transformers", "transformers", "torch",
        "numpy", "scipy", "sklearn", "matplotlib", "pandas",
        "requests", "urllib3", "httpx", "aiohttp", "uvicorn", "fastapi",
        "tqdm", "tokenizers", "datasets", "accelerate", "bitsandbytes"
    ]

    for logger_name in external_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False

    # 경고 메시지 필터
    import warnings
    warnings.filterwarnings('ignore')

    # 우리 앱의 로깅은 정상적으로 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

class ArticleVectorEmbedder:
    """조문별 벡터 임베딩 생성 클래스"""

    def __init__(self, start_batch: int = 1, batch_size: int = 1000, use_gpu: bool = False, max_batches: int = None):
        setup_local_logging()
        self.logger = logging.getLogger("article_vector_embedder")
        self.db_manager = DatabaseManager()

        # GPU 사용 여부 결정 (CUDA 및 ROCm 지원)
        device = "cpu"

        if use_gpu and TORCH_AVAILABLE:
            # CUDA (NVIDIA GPU) 체크
            if torch.cuda.is_available():
                device = "cuda"
                self.logger.info("NVIDIA CUDA GPU detected. Using CUDA.")
            # ROCm (AMD GPU) 체크
            elif hasattr(torch, 'hip') and torch.hip.is_available():
                device = "cuda"  # ROCm은 CUDA API를 사용
                self.logger.info("AMD ROCm GPU detected. Using ROCm.")
            else:
                self.logger.warning("GPU requested but neither CUDA nor ROCm available. Using CPU.")
                self.logger.info("Note: For AMD GPUs, ROCm must be installed in WSL.")

        self.logger.info(f"Using device: {device}")

        # 벡터 스토어 초기화
        self.vector_store = LegalVectorStore(device=device)

        # 배치 설정
        self.start_batch = start_batch
        self.batch_size = batch_size
        self.max_batches = max_batches  # 처리할 최대 배치 수

        # 통계 정보
        self.stats = {
            'total_articles': 0,
            'processed_articles': 0,
            'embedding_errors': []
        }

    def create_article_embeddings(self) -> Dict[str, Any]:
        """메모리 최적화된 벡터 임베딩 생성"""
        self.logger.info("조문별 벡터 임베딩 생성 시작 (메모리 최적화)")

        try:
            # 전체 데이터를 메모리에 로드하지 않고 총 개수만 조회
            total_count = self._get_total_articles_count()
            self.stats['total_articles'] = total_count

            if total_count == 0:
                self.logger.warning("처리할 조문 데이터가 없습니다.")
                return self.stats

            self.logger.info(f"총 {total_count}개 조문 발견")

            # 초기 메모리 상태 로깅
            initial_memory = self._get_memory_info()
            self.logger.info(f"초기 메모리 사용량: {initial_memory['process_memory_mb']:.2f}MB")

            # 배치별로 스트리밍 처리
            start_index = (self.start_batch - 1) * self.batch_size
            total_batches = (total_count + self.batch_size - 1) // self.batch_size

            # 최대 배치 수 설정
            if self.max_batches:
                end_batch = min(self.start_batch + self.max_batches - 1, total_batches)
                self.logger.info(f"배치 {self.start_batch}부터 {end_batch}까지 처리 ({self.max_batches}개 배치)")
            else:
                end_batch = total_batches
                self.logger.info(f"배치 {self.start_batch}부터 {end_batch}까지 처리 (전체)")

            self.logger.info(f"배치 크기: {self.batch_size}, 시작 인덱스: {start_index}")

            for batch_num in range(self.start_batch, end_batch + 1):
                current_start_index = (batch_num - 1) * self.batch_size

                # 배치별로 데이터 조회 (메모리 절약)
                batch_articles = self._get_articles_data_streaming(current_start_index, self.batch_size)

                if not batch_articles:
                    self.logger.info(f"배치 {batch_num}: 데이터 없음, 처리 종료")
                    break

                self.logger.info(f"배치 {batch_num}/{total_batches} 처리 중... ({len(batch_articles)}개 조문)")

                # 배치별 문서와 메타데이터 준비
                documents, metadatas = self._prepare_embedding_data(batch_articles)

                # 벡터 스토어에 추가
                success = self.vector_store.add_documents(documents, metadatas)

                if success:
                    self.stats['processed_articles'] += len(documents)
                    self.logger.info(f"✅ 배치 {batch_num} 완료: {len(documents)}개 조문")

                    # 메모리 정리 및 모니터링
                    del batch_articles, documents, metadatas
                    gc.collect()
                    self._monitor_and_cleanup_memory()

                    # 진행률 및 메모리 상태 로깅
                    total_processing_batches = end_batch - self.start_batch + 1
                    progress_percent = (batch_num - self.start_batch + 1) / total_processing_batches * 100
                    current_memory = self._get_memory_info()
                    self.logger.info(f"진행률: {progress_percent:.1f}% ({batch_num - self.start_batch + 1}/{total_processing_batches}), 현재 메모리: {current_memory['process_memory_mb']:.2f}MB")

                else:
                    self.logger.error(f"❌ 배치 {batch_num} 실패")
                    break

            # 최종 메모리 상태 로깅
            final_memory = self._get_memory_info()
            memory_saved = initial_memory['process_memory_mb'] - final_memory['process_memory_mb']
            self.logger.info(f"벡터 임베딩 생성 완료: 총 {self.stats['processed_articles']}개")
            self.logger.info(f"최종 메모리 사용량: {final_memory['process_memory_mb']:.2f}MB")
            if memory_saved > 0:
                self.logger.info(f"메모리 절약: {memory_saved:.2f}MB")

            # 통계 출력
            self._print_statistics()

            return self.stats

        except Exception as e:
            self.logger.error(f"벡터 임베딩 생성 실패: {e}")
            raise

    def _get_total_articles_count(self) -> int:
        """전체 조문 개수 조회 (메모리 절약)"""
        query = "SELECT COUNT(*) as total FROM current_laws_articles"

        try:
            result = self.db_manager.execute_query(query)
            return result[0]['total'] if result else 0
        except Exception as e:
            self.logger.error(f"조문 개수 조회 실패: {e}")
            return 0

    def _get_articles_data_streaming(self, start_index: int, batch_size: int) -> List[Dict[str, Any]]:
        """스트리밍 방식으로 조문 데이터 조회 (메모리 절약)"""
        query = """
            SELECT ca.*, cl.ministry_name, cl.effective_date
            FROM current_laws_articles ca
            JOIN current_laws cl ON ca.law_id = cl.law_id
            ORDER BY ca.law_name_korean, ca.article_number, ca.paragraph_number, ca.sub_paragraph_number
            LIMIT ? OFFSET ?
        """

        try:
            return self.db_manager.execute_query(query, (batch_size, start_index))
        except Exception as e:
            self.logger.error(f"조문 데이터 조회 실패: {e}")
            return []

    def _get_articles_data(self) -> List[Dict[str, Any]]:
        """조문 데이터 조회 (레거시 메서드 - 호환성 유지)"""
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

    def _monitor_and_cleanup_memory(self):
        """메모리 사용량 모니터링 및 자동 정리"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)

            # 메모리 사용량이 4GB를 초과하면 정리
            if memory_mb > 4000:
                self.logger.warning(f"메모리 사용량 높음: {memory_mb:.2f}MB, 정리 시작...")

                # 가비지 컬렉션 강제 실행
                collected = gc.collect()
                self.logger.info(f"가비지 컬렉션으로 {collected}개 객체 정리")

                # 메모리 사용량 재확인
                memory_after = process.memory_info().rss / (1024**2)
                self.logger.info(f"정리 후 메모리 사용량: {memory_after:.2f}MB")

                # 메모리 절약량 계산
                saved_mb = memory_mb - memory_after
                if saved_mb > 0:
                    self.logger.info(f"메모리 절약: {saved_mb:.2f}MB")

        except Exception as e:
            self.logger.error(f"메모리 모니터링 실패: {e}")

    def _get_memory_info(self) -> Dict[str, float]:
        """현재 메모리 사용량 정보 반환"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)

            return {
                'process_memory_mb': memory_mb,
                'available_memory_gb': available_memory_gb,
                'memory_percent': psutil.virtual_memory().percent
            }
        except Exception as e:
            self.logger.error(f"메모리 정보 조회 실패: {e}")
            return {'process_memory_mb': 0, 'available_memory_gb': 0, 'memory_percent': 0}

    def _create_article_document(self, article: Dict[str, Any]) -> str:
        """스마트 자르기 방식의 조문 문서 생성"""
        content_parts = [
            f"법령명: {article['law_name_korean']}",
            f"조문번호: 제{article['article_number']}조"
        ]

        # 제목이 있으면 우선 포함
        if article.get('article_title'):
            content_parts.append(f"제목: {article['article_title']}")

        # 내용을 스마트하게 자르기
        content = article['article_content']
        max_content_length = 400  # 전체 길이의 80% 할당

        if len(content) > max_content_length:
            # 문장 단위로 자르기
            truncated_content = self._smart_truncate_text(content, max_content_length)
            content_parts.append(f"내용: {truncated_content}")
        else:
            content_parts.append(f"내용: {content}")

        # 항 내용 (스마트 자르기 적용)
        if article.get('paragraph_content'):
            para_content = article['paragraph_content']
            if len(para_content) > 100:
                para_content = self._smart_truncate_text(para_content, 100)
            content_parts.append(f"항: {para_content}")

        # 호 내용 (선택적, 길이 제한)
        if article.get('sub_paragraph_content'):
            sub_para_content = article['sub_paragraph_content']
            if len(sub_para_content) > 80:
                sub_para_content = self._smart_truncate_text(sub_para_content, 80)
            content_parts.append(f"호: {sub_para_content}")

        # 소관부처 정보 (선택적)
        if article.get('ministry_name'):
            content_parts.append(f"소관부처: {article['ministry_name']}")

        return "\n".join(content_parts)

    def _smart_truncate_text(self, text: str, max_length: int) -> str:
        """스마트 텍스트 자르기 (문장 단위 보존)"""
        if len(text) <= max_length:
            return text

        # 한국어 문장 구분자들
        sentence_endings = ['。', '.', '!', '?', ';', ':', '다.', '니다.', '요.', '어요.', '아요.']

        # 문장 단위로 분할
        sentences = []
        current_sentence = ""

        for char in text:
            current_sentence += char
            if char in sentence_endings:
                sentences.append(current_sentence.strip())
                current_sentence = ""

        # 마지막 문장이 있으면 추가
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        # 문장 단위로 자르기
        truncated_text = ""
        for sentence in sentences:
            if len(truncated_text + sentence) <= max_length:
                truncated_text += sentence
            else:
                break

        # 결과가 너무 짧으면 단어 단위로 자르기
        if len(truncated_text) < max_length * 0.5:  # 50% 미만이면
            truncated_text = text[:max_length]
            # 단어 경계에서 자르기
            last_space = truncated_text.rfind(' ')
            if last_space > max_length * 0.7:  # 70% 이상이면
                truncated_text = truncated_text[:last_space]

        return truncated_text

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
        print("📊 조문별 벡터 임베딩 생성 통계 (메모리 최적화)")
        print("="*60)
        print(f"총 조문 수: {self.stats['total_articles']:,}개")
        print(f"처리된 조문: {self.stats['processed_articles']:,}개")
        print(f"처리 실패: {len(self.stats['embedding_errors'])}개")

        # 메모리 통계
        memory_info = self._get_memory_info()
        print(f"\n🧠 메모리 사용량:")
        print(f"  프로세스 메모리: {memory_info['process_memory_mb']:.2f}MB")
        print(f"  시스템 사용 가능 메모리: {memory_info['available_memory_gb']:.2f}GB")
        print(f"  시스템 메모리 사용률: {memory_info['memory_percent']:.1f}%")

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
    parser = argparse.ArgumentParser(description='현행법령 조문별 벡터 임베딩 생성')
    parser.add_argument('--start-batch', type=int, default=1,
                       help='시작할 배치 번호 (기본값: 1)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='배치 크기 (기본값: 1000)')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='처리할 최대 배치 수 (기본값: None, 전체 처리)')
    parser.add_argument('--use-gpu', action='store_true', default=False,
                       help='GPU 사용 (기본값: False, AMD GPU는 ROCm 지원 필요)')
    parser.add_argument('--use-cpu', action='store_true', default=False,
                       help='CPU 강제 사용 (AMD GPU 사용자 권장)')

    args = parser.parse_args()

    # GPU 사용 여부 결정
    use_gpu = args.use_gpu and not args.use_cpu

    print(f"🚀 조문별 벡터 임베딩 생성 시작")
    print(f"   시작 배치: {args.start_batch}")
    print(f"   배치 크기: {args.batch_size}")
    print(f"   최대 배치 수: {args.max_batches if args.max_batches else '전체'}")
    print(f"   GPU 사용: {'Yes' if use_gpu else 'No'}")
    print("-" * 50)

    embedder = ArticleVectorEmbedder(
        start_batch=args.start_batch,
        batch_size=args.batch_size,
        use_gpu=use_gpu,
        max_batches=args.max_batches
    )
    stats = embedder.create_article_embeddings()

    print(f"\n🎉 조문별 벡터 임베딩 생성 완료!")
    print(f"총 {stats['processed_articles']:,}개 조문의 벡터 임베딩이 생성되었습니다.")


if __name__ == "__main__":
    main()
