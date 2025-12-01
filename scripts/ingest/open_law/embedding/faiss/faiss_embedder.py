#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS 임베딩 생성기
PostgreSQL 데이터를 읽어 FAISS 인덱스 생성
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    try:
        from dotenv import load_dotenv
        scripts_env = _PROJECT_ROOT / "scripts" / ".env"
        if scripts_env.exists():
            load_dotenv(dotenv_path=str(scripts_env), override=True)
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
    except ImportError:
        pass

# 공통 모듈 임포트
try:
    from scripts.ingest.open_law.embedding.data_loader import PostgreSQLDataLoader
    from scripts.ingest.open_law.embedding.base_embedder import BaseEmbedder
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
    from ingest.open_law.embedding.data_loader import PostgreSQLDataLoader
    from ingest.open_law.embedding.base_embedder import BaseEmbedder

# 데이터베이스 URL 빌드
try:
    from scripts.ingest.open_law.utils import build_database_url
except ImportError:
    from urllib.parse import quote_plus
    def build_database_url():
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB')
        user = os.getenv('POSTGRES_USER')
        password = os.getenv('POSTGRES_PASSWORD')
        if db and user and password:
            encoded_password = quote_plus(password)
            return f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
        return None

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaissEmbedder:
    """FAISS 임베딩 생성기"""
    
    def __init__(
        self,
        db_url: str,
        output_path: Path,
        model_name: str = "jhgan/ko-sroberta-multitask",
        use_mlflow: bool = True,
        mlflow_experiment_name: str = "faiss_index_versions"
    ):
        """
        FAISS 임베딩 생성기 초기화
        
        Args:
            db_url: PostgreSQL 데이터베이스 URL
            output_path: FAISS 인덱스 저장 경로
            model_name: 임베딩 모델 이름
            use_mlflow: MLflow 사용 여부
            mlflow_experiment_name: MLflow 실험 이름
        """
        self.db_url = db_url
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow
        self.model_name = model_name
        
        try:
            from lawfirm_langgraph.core.utils.logger import get_logger
            self.logger = get_logger(__name__)
        except ImportError:
            self.logger = logging.getLogger(__name__)
        
        # 데이터 로더 및 임베딩 생성기 초기화
        self.data_loader = PostgreSQLDataLoader(db_url)
        self.embedder = BaseEmbedder(model_name)
        self.dimension = self.embedder.get_dimension()
        
        # MLflow 관리자 초기화
        self.mlflow_manager = None
        if self.use_mlflow:
            try:
                # 프로젝트 루트 기준으로 임포트
                sys.path.insert(0, str(_PROJECT_ROOT))
                from scripts.rag.mlflow_manager import MLflowFAISSManager
                self.mlflow_manager = MLflowFAISSManager(
                    experiment_name=mlflow_experiment_name
                )
                self.logger.info("MLflow 통합 활성화됨")
            except ImportError as e:
                self.logger.warning(f"MLflow를 사용할 수 없습니다: {e}. MLflow 없이 진행합니다.")
                self.use_mlflow = False
            except Exception as e:
                self.logger.warning(f"MLflow 초기화 실패: {e}. MLflow 없이 진행합니다.")
                self.use_mlflow = False
        
        # 임베딩 및 메타데이터 저장소
        self.embeddings = []
        self.chunk_ids = []
        self.metadata_list = []
        self.domain = None  # 나중에 설정됨
    
    def generate_embeddings(
        self,
        data_type: str,  # 'precedents' or 'statutes'
        batch_size: int = 100,
        limit: Optional[int] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        임베딩 생성
        
        Args:
            data_type: 데이터 타입 ('precedents' or 'statutes')
            batch_size: 배치 크기
            limit: 최대 처리 개수
            domain: 도메인 필터
        
        Returns:
            처리 결과 통계
        """
        self.logger.info(f"{data_type} 임베딩 생성 시작")
        self.domain = domain  # domain 저장 (MLflow 저장 시 사용)
        
        stats = {
            "total_processed": 0,
            "total_embedded": 0,
            "total_failed": 0,
            "errors": []
        }
        
        offset = 0
        
        try:
            while True:
                # 데이터 로드
                if data_type == "precedents":
                    items = self.data_loader.load_precedent_chunks(
                        domain=domain,
                        limit=batch_size,
                        offset=offset
                    )
                elif data_type == "statutes":
                    items = self.data_loader.load_statute_articles(
                        domain=domain,
                        limit=batch_size,
                        offset=offset
                    )
                else:
                    raise ValueError(f"Unknown data_type: {data_type}")
                
                if not items:
                    break
                
                # 텍스트 추출
                if data_type == "precedents":
                    texts = [item["chunk_content"] for item in items]
                    chunk_ids = [item["id"] for item in items]
                    metadata = [{
                        "id": item["id"],
                        "precedent_content_id": item["precedent_content_id"],
                        "chunk_index": item["chunk_index"],
                        "section_type": item["section_type"],
                        "case_name": item["case_name"],
                        "case_number": item["case_number"],
                        "decision_date": item["decision_date"],
                        "court_name": item["court_name"],
                        "domain": item["domain"]
                    } for item in items]
                else:  # statutes
                    texts = [item["article_content"] for item in items]
                    chunk_ids = [item["id"] for item in items]
                    metadata = [{
                        "id": item["id"],
                        "statute_id": item["statute_id"],
                        "article_no": item["article_no"],
                        "article_title": item["article_title"],
                        "law_name_kr": item["law_name_kr"],
                        "law_abbrv": item["law_abbrv"],
                        "domain": item["domain"]
                    } for item in items]
                
                # 임베딩 생성
                try:
                    embeddings = self.embedder.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress=False
                    )
                    
                    # 메모리에 저장
                    self.embeddings.append(embeddings)
                    self.chunk_ids.extend(chunk_ids)
                    self.metadata_list.extend(metadata)
                    
                    stats["total_embedded"] += len(items)
                    stats["total_processed"] += len(items)
                    
                    # 메모리 최적화: 중간 변수 삭제
                    del embeddings
                    del texts
                    del chunk_ids
                    del metadata
                    del items
                    gc.collect()
                    
                    self.logger.info(
                        f"진행 상황: {stats['total_processed']}개 처리, "
                        f"{stats['total_embedded']}개 임베딩 생성"
                    )
                    
                    offset += batch_size
                    
                    if limit and stats["total_processed"] >= limit:
                        break
                
                except Exception as e:
                    self.logger.error(f"배치 처리 실패: {e}")
                    stats["total_failed"] += len(items)
                    stats["errors"].append(f"배치 offset={offset}: {e}")
                    
                    # 메모리 최적화: 에러 발생 시에도 메모리 정리
                    if 'embeddings' in locals():
                        del embeddings
                    if 'texts' in locals():
                        del texts
                    if 'chunk_ids' in locals():
                        del chunk_ids
                    if 'metadata' in locals():
                        del metadata
                    if 'items' in locals():
                        del items
                    gc.collect()
                    
                    offset += batch_size
                    continue
            
            # 모든 임베딩을 하나의 배열로 결합
            if self.embeddings:
                all_embeddings = np.vstack(self.embeddings).astype(np.float32)
                self.logger.info(
                    f"임베딩 배열 생성 완료: {all_embeddings.shape}"
                )
                # 메모리 최적화: 개별 임베딩 리스트 삭제
                del self.embeddings
                gc.collect()
            else:
                all_embeddings = np.array([])
            
            # 통계 저장
            stats["total_embeddings"] = len(self.chunk_ids)
            stats["dimension"] = self.dimension
            
            self.logger.info(
                f"{data_type} 임베딩 생성 완료: "
                f"{stats['total_embedded']}개 생성, "
                f"{stats['total_failed']}개 실패"
            )
            
            return stats
        
        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    def save_embeddings(
        self,
        data_type: str
    ) -> bool:
        """
        임베딩 및 메타데이터 저장
        
        Args:
            data_type: 데이터 타입
        
        Returns:
            성공 여부
        """
        try:
            # 임베딩 배열 결합
            if not self.embeddings:
                self.logger.warning("저장할 임베딩이 없습니다.")
                return False
            
            all_embeddings = np.vstack(self.embeddings).astype(np.float32)
            
            # 파일 경로
            index_file = self.output_path / f"{data_type}_faiss_index.faiss"
            chunk_ids_file = self.output_path / f"{data_type}_chunk_ids.json"
            metadata_file = self.output_path / f"{data_type}_metadata.json"
            stats_file = self.output_path / f"{data_type}_stats.json"
            
            # FAISS 인덱스 저장 (IndexIVFFlat 사용 - 빠른 검색을 위한 IVF 구조)
            # Inner Product (코사인 유사도용, 정규화된 벡터 필요)
            # 벡터 정규화 (numpy 사용)
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 0으로 나누기 방지
            all_embeddings = all_embeddings / norms
            
            # IndexIVFFlat 생성 - faiss_indexer.py 사용
            try:
                from scripts.ingest.open_law.embedding.faiss.faiss_indexer import FaissIndexer
            except ImportError:
                sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
                from ingest.open_law.embedding.faiss.faiss_indexer import FaissIndexer
            
            indexer = FaissIndexer(dimension=self.dimension)
            n_samples = all_embeddings.shape[0]
            
            self.logger.info(f"IndexIVFFlat 생성 중: 벡터 수={n_samples}")
            index = indexer.build_index(
                embeddings=all_embeddings,
                index_type="ivfflat",
                nlist=None  # 자동 계산
            )
            
            # nprobe 설정
            if hasattr(index, 'nlist'):
                nlist = index.nlist
                index.nprobe = min(max(nlist // 10, 1), 100)
            else:
                nlist = None
            
            faiss.write_index(index, str(index_file))
            self.logger.info(
                f"FAISS 인덱스 저장: {index_file} "
                f"({index.ntotal}개 벡터, nlist={nlist if nlist else 'N/A'}, "
                f"nprobe={index.nprobe if hasattr(index, 'nprobe') else 'N/A'}, "
                f"is_trained={index.is_trained if hasattr(index, 'is_trained') else 'N/A'})"
            )
            
            # 메모리 최적화: 인덱스 생성 후 임베딩 배열 삭제
            del all_embeddings
            del index
            gc.collect()
            
            # chunk_ids 저장
            with open(chunk_ids_file, 'w', encoding='utf-8') as f:
                json.dump(self.chunk_ids, f, ensure_ascii=False, indent=2)
            self.logger.info(f"chunk_ids 저장: {chunk_ids_file}")
            
            # metadata 저장
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_list, f, ensure_ascii=False, indent=2)
            self.logger.info(f"metadata 저장: {metadata_file}")
            
            # 통계 저장
            index_type = type(index).__name__
            stats = {
                "data_type": data_type,
                "total_embeddings": len(self.chunk_ids),
                "dimension": self.dimension,
                "index_type": index_type,
                "ntotal": index.ntotal,
                "nlist": index.nlist if hasattr(index, 'nlist') else None,
                "nprobe": index.nprobe if hasattr(index, 'nprobe') else None,
                "is_trained": index.is_trained if hasattr(index, 'is_trained') else True
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            self.logger.info(f"통계 저장: {stats_file}")
            
            # MLflow에 저장
            if self.use_mlflow and self.mlflow_manager:
                try:
                    from datetime import datetime
                    
                    # 버전 이름 생성
                    domain_suffix = f"_{self.domain}" if self.domain else ""
                    version_name = f"{data_type}{domain_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # MLflow run 생성
                    self.logger.info(f"MLflow run 생성 중: {version_name}")
                    run_id = self.mlflow_manager.create_run(
                        version_name=version_name,
                        embedding_version_id=0,  # 임베딩 버전 ID (필요시 파라미터로 받기)
                        chunking_strategy="article",  # 기본값 (필요시 파라미터로 받기)
                        chunking_config={"strategy": "article"},
                        embedding_config={
                            "model_name": self.model_name,
                            "dimension": self.dimension,
                            "index_type": "IndexIVFFlat"
                        },
                        document_count=0,
                        total_chunks=len(self.chunk_ids),
                        status="active"
                    )
                    
                    # ID 매핑 생성 (인덱스 위치 -> chunk_id)
                    id_mapping = {i: chunk_id for i, chunk_id in enumerate(self.chunk_ids)}
                    
                    # MLflow에 인덱스 저장
                    self.logger.info(f"MLflow에 인덱스 저장 중: run_id={run_id}")
                    mlflow_success = self.mlflow_manager.save_index(
                        run_id=run_id,
                        index=index,
                        id_mapping=id_mapping,
                        metadata=self.metadata_list,
                        index_path=index_file
                    )
                    
                    if mlflow_success:
                        self.logger.info(f"✅ MLflow에 인덱스 저장 완료: run_id={run_id}")
                    else:
                        self.logger.warning("⚠️ MLflow 저장 실패 (로컬 파일은 저장됨)")
                        
                except Exception as e:
                    self.logger.error(f"MLflow 저장 중 오류 발생: {e}", exc_info=True)
                    # MLflow 저장 실패해도 로컬 저장은 성공했으므로 계속 진행
            
            return True
        
        except Exception as e:
            self.logger.error(f"임베딩 저장 실패: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='FAISS 임베딩 생성')
    parser.add_argument(
        '--db',
        default=build_database_url() or os.getenv('DATABASE_URL'),
        help='PostgreSQL 데이터베이스 URL'
    )
    parser.add_argument(
        '--data-type',
        choices=['precedents', 'statutes'],
        required=True,
        help='임베딩 생성할 데이터 타입'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/embeddings/open_law_postgresql'),
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='배치 크기'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='최대 처리 개수'
    )
    parser.add_argument(
        '--domain',
        choices=['civil_law', 'criminal_law', 'administrative_law'],
        default=None,
        help='도메인 필터'
    )
    parser.add_argument(
        '--model',
        default='jhgan/ko-sroberta-multitask',
        help='임베딩 모델 이름'
    )
    
    args = parser.parse_args()
    
    if not args.db:
        logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
        return
    
    try:
        embedder = FaissEmbedder(
            args.db,
            args.output_dir,
            model_name=args.model
        )
        
        # 임베딩 생성
        results = embedder.generate_embeddings(
            data_type=args.data_type,
            batch_size=args.batch_size,
            limit=args.limit,
            domain=args.domain
        )
        logger.info(f"임베딩 생성 완료: {results}")
        
        # 저장
        success = embedder.save_embeddings(args.data_type)
        if success:
            logger.info("임베딩 저장 완료")
        else:
            logger.error("임베딩 저장 실패")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"스크립트 실행 실패: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

