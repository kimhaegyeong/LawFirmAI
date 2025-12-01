#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
문서 간 클러스터링 기반 근사 Ground Truth 생성

벡터 스토어의 모든 문서를 클러스터링하여, 같은 클러스터 내 문서들을
서로 관련 문서로 간주하는 Ground Truth 데이터셋을 생성합니다.
"""

import logging
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import argparse
import numpy as np
import time
import gc
from tqdm import tqdm

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Please install scikit-learn")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("hdbscan not available. Install for HDBSCAN clustering")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.error("FAISS not available. Please install faiss-cpu or faiss-gpu")

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.data.vector_store import LegalVectorStore
from scripts.ml_training.vector_embedding.version_manager import VectorStoreVersionManager

# Windows 호환 로깅 설정 (출력 버퍼 문제 해결)
class SafeStreamHandler(logging.StreamHandler):
    """안전한 스트림 핸들러 (버퍼 분리 문제 해결)"""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            if hasattr(stream, 'write'):
                try:
                    stream.write(msg + self.terminator)
                    self.flush()
                except (ValueError, OSError):
                    # 버퍼가 분리된 경우 무시
                    pass
        except Exception:
            self.handleError(record)

if sys.platform == "win32":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[SafeStreamHandler(sys.stdout)],
        force=True
    )
    logging.raiseExceptions = False
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[SafeStreamHandler(sys.stdout)],
        force=True
    )

# 루트 로거의 모든 핸들러를 안전한 핸들러로 교체
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        root_logger.removeHandler(handler)
        root_logger.addHandler(SafeStreamHandler(sys.stdout))

logger = logging.getLogger(__name__)

# tqdm과 로깅의 충돌 방지를 위한 래퍼
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, tqdm_instance):
        super().__init__()
        self.tqdm = tqdm_instance
    
    def emit(self, record):
        try:
            msg = self.format(record)
            if self.tqdm:
                self.tqdm.write(msg, file=sys.stdout)
            else:
                print(msg, file=sys.stdout)
        except Exception:
            self.handleError(record)


class ProgressMonitor:
    """진행 상황 모니터링 클래스"""
    
    def __init__(self):
        self.start_time = None
        self.stage_start_time = None
        self.stage_name = None
        self.stage_progress = 0
        self.stage_total = 0
        self.stages = []
        self.current_stage_index = 0
    
    def start(self):
        """전체 프로세스 시작"""
        self.start_time = time.time()
        logger.info("=" * 80)
        logger.info("Ground Truth 생성 프로세스 시작")
        logger.info("=" * 80)
    
    def start_stage(self, stage_name: str, total: int = 0):
        """단계 시작"""
        if self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            self.stages.append({
                "name": self.stage_name,
                "elapsed": elapsed,
                "progress": self.stage_progress,
                "total": self.stage_total
            })
        
        self.stage_name = stage_name
        self.stage_start_time = time.time()
        self.stage_progress = 0
        self.stage_total = total
        self.current_stage_index = len(self.stages)
        
        logger.info("")
        logger.info(f"[단계 {self.current_stage_index + 1}] {stage_name} 시작...")
        if total > 0:
            logger.info(f"  총 작업량: {total:,}개")
    
    def update_stage_progress(self, progress: int, total: Optional[int] = None):
        """단계 진행 상황 업데이트"""
        self.stage_progress = progress
        if total is not None:
            self.stage_total = total
        
        if self.stage_total > 0 and self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            progress_pct = (progress / self.stage_total * 100) if self.stage_total > 0 else 0
            
            if progress > 0:
                avg_time_per_item = elapsed / progress
                remaining_items = self.stage_total - progress
                remaining_time = avg_time_per_item * remaining_items
                
                total_elapsed = time.time() - self.start_time if self.start_time else 0
                estimated_total_time = total_elapsed + remaining_time
                
                if progress % max(1, self.stage_total // 20) == 0 or progress == self.stage_total:
                    logger.info(
                        f"  진행률: {progress:,}/{self.stage_total:,} ({progress_pct:.1f}%) | "
                        f"경과: {self._format_time(elapsed)} | "
                        f"예상 남은 시간: {self._format_time(remaining_time)}"
                    )
    
    def end_stage(self):
        """단계 종료"""
        if self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            self.stages.append({
                "name": self.stage_name,
                "elapsed": elapsed,
                "progress": self.stage_progress,
                "total": self.stage_total
            })
            
            logger.info(f"[단계 {self.current_stage_index + 1}] {self.stage_name} 완료 ({self._format_time(elapsed)})")
            self._print_overall_progress()
    
    def _print_overall_progress(self):
        """전체 진행 상황 출력"""
        if not self.start_time:
            return
        
        total_elapsed = time.time() - self.start_time
        completed_stages = len(self.stages)
        
        if completed_stages > 0:
            avg_time_per_stage = total_elapsed / completed_stages
            remaining_stages = max(0, 5 - completed_stages)
            estimated_remaining = avg_time_per_stage * remaining_stages
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining)
            
            logger.info("")
            logger.info("-" * 80)
            logger.info("전체 진행 상황:")
            logger.info(f"  완료된 단계: {completed_stages}/5")
            logger.info(f"  총 경과 시간: {self._format_time(total_elapsed)}")
            logger.info(f"  예상 남은 시간: {self._format_time(estimated_remaining)}")
            logger.info(f"  예상 완료 시간: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("-" * 80)
    
    def finish(self):
        """전체 프로세스 종료"""
        if self.stage_start_time:
            self.end_stage()
        
        if self.start_time:
            total_elapsed = time.time() - self.start_time
            logger.info("")
            logger.info("=" * 80)
            logger.info("Ground Truth 생성 프로세스 완료")
            logger.info("=" * 80)
            logger.info(f"총 소요 시간: {self._format_time(total_elapsed)}")
            logger.info("")
            logger.info("단계별 소요 시간:")
            for i, stage in enumerate(self.stages, 1):
                logger.info(f"  [{i}] {stage['name']}: {self._format_time(stage['elapsed'])}")
            logger.info("=" * 80)
    
    def _format_time(self, seconds: float) -> str:
        """시간 포맷팅"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}분 {secs}초"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}시간 {minutes}분 {secs}초"


class CheckpointManager:
    """체크포인트 관리 클래스"""
    
    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        체크포인트 매니저 초기화
        
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리 (None이면 기본 경로 사용)
        """
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path("data/evaluation/checkpoints")
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "ground_truth_checkpoint.json"
        self.embeddings_file = self.checkpoint_dir / "embeddings.npy"
        self.labels_file = self.checkpoint_dir / "labels.npy"
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
    
    def save_checkpoint(self, stage: str, data: Dict[str, Any], embeddings: Optional[np.ndarray] = None, 
                       labels: Optional[np.ndarray] = None) -> bool:
        """
        체크포인트 저장
        
        Args:
            stage: 현재 단계 이름
            data: 저장할 데이터
            embeddings: 임베딩 배열 (선택)
            labels: 클러스터 레이블 배열 (선택)
        
        Returns:
            bool: 저장 성공 여부
        """
        try:
            checkpoint_data = {
                "stage": stage,
                "saved_at": datetime.now().isoformat(),
                "data": data
            }
            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            if embeddings is not None:
                np.save(self.embeddings_file, embeddings)
            
            if labels is not None:
                np.save(self.labels_file, labels)
            
            logger.info(f"체크포인트 저장 완료: {stage} ({self.checkpoint_file})")
            return True
            
        except Exception as e:
            logger.error(f"체크포인트 저장 실패: {e}")
            return False
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        체크포인트 로드
        
        Returns:
            체크포인트 데이터 또는 None
        """
        try:
            if not self.checkpoint_file.exists():
                return None
            
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            stage = checkpoint_data.get("stage")
            data = checkpoint_data.get("data", {})
            
            result = {
                "stage": stage,
                "data": data,
                "saved_at": checkpoint_data.get("saved_at")
            }
            
            if self.embeddings_file.exists():
                result["embeddings"] = np.load(self.embeddings_file)
            
            if self.labels_file.exists():
                result["labels"] = np.load(self.labels_file)
            
            logger.info(f"체크포인트 로드 완료: {stage} (저장 시각: {result['saved_at']})")
            return result
            
        except Exception as e:
            logger.error(f"체크포인트 로드 실패: {e}")
            return None
    
    def clear_checkpoint(self) -> bool:
        """
        체크포인트 삭제
        
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.embeddings_file.exists():
                self.embeddings_file.unlink()
            if self.labels_file.exists():
                self.labels_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            
            logger.info("체크포인트 삭제 완료")
            return True
            
        except Exception as e:
            logger.error(f"체크포인트 삭제 실패: {e}")
            return False
    
    def has_checkpoint(self) -> bool:
        """체크포인트 존재 여부 확인"""
        return self.checkpoint_file.exists()


class ClusteringGroundTruthGenerator:
    """클러스터링 기반 Ground Truth 생성 클래스"""
    
    def __init__(self, vector_store_path: str, model_name: str = "jhgan/ko-sroberta-multitask",
                 checkpoint_dir: Optional[str] = None):
        """
        초기화
        
        Args:
            vector_store_path: 벡터 스토어 경로
            model_name: 임베딩 모델명
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.vector_store_path = Path(vector_store_path)
        self.model_name = model_name
        
        self.vector_store = LegalVectorStore(
            model_name=model_name,
            dimension=768,
            index_type="flat"
        )
        
        self.progress_monitor = ProgressMonitor()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        logger.info(f"Loading vector store from: {self.vector_store_path}")
        
        index_file = self.vector_store_path
        if self.vector_store_path.is_dir():
            if (self.vector_store_path / "ml_enhanced_faiss_index.faiss").exists():
                index_file = self.vector_store_path / "ml_enhanced_faiss_index"
            elif (self.vector_store_path / "index.faiss").exists():
                index_file = self.vector_store_path / "index"
            elif (self.vector_store_path / "faiss_index_jhgan_ko-sroberta-multitask.index").exists():
                index_file = self.vector_store_path / "faiss_index_jhgan_ko-sroberta-multitask"
            else:
                raise ValueError(f"Could not find FAISS index file in {self.vector_store_path}")
        
        faiss_file = str(index_file) + '.faiss'
        if not Path(faiss_file).exists():
            faiss_file = str(index_file)
        
        metadata_file = index_file.parent / f"{index_file.name}.json"
        if not metadata_file.exists():
            metadata_file = index_file.parent / f"{index_file.name.replace('_index', '_metadata')}.json"
        
        if not metadata_file.exists():
            chunk_ids_file = index_file.parent / f"{index_file.name}.chunk_ids.json"
            if chunk_ids_file.exists():
                logger.warning(f"Metadata file not found, but chunk_ids file exists. Will try to load from database.")
                metadata_file = None
            else:
                raise ValueError(f"Could not find metadata file for {index_file}")
        
        logger.info(f"Loading FAISS index from: {faiss_file}")
        
        self.index = faiss.read_index(faiss_file)
        
        if metadata_file and metadata_file.exists():
            logger.info(f"Loading metadata from: {metadata_file}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if 'document_texts' in metadata:
                self.document_texts = metadata.get('document_texts', [])
            elif 'texts' in metadata:
                self.document_texts = metadata.get('texts', [])
            else:
                self.document_texts = []
            
            if 'document_metadata' in metadata:
                self.document_metadata = metadata.get('document_metadata', [])
            else:
                self.document_metadata = []
        else:
            logger.warning("Metadata file not found. Trying to load from database...")
            try:
                import sqlite3
                import os
                
                db_path = os.getenv("DATABASE_PATH", "data/lawfirm_v2.db")
                if not os.path.isabs(db_path):
                    db_path = str(project_root / db_path)
                
                if not Path(db_path).exists():
                    raise ValueError(f"Database file not found: {db_path}")
                
                chunk_ids_file = index_file.parent / f"{index_file.name}.chunk_ids.json"
                if chunk_ids_file.exists():
                    with open(chunk_ids_file, 'r', encoding='utf-8') as f:
                        chunk_ids = json.load(f)
                    
                    logger.info(f"Loading {len(chunk_ids)} chunks from database: {db_path}")
                    
                    conn = sqlite3.connect(db_path)
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    self.document_texts = []
                    self.document_metadata = []
                    
                    batch_size = 1000
                    logger.info(f"Loading {len(chunk_ids)} chunks from database in batches of {batch_size}...")
                    
                    total_batches = (len(chunk_ids) + batch_size - 1) // batch_size
                    for i in tqdm(range(0, len(chunk_ids), batch_size), desc="Loading chunks from database"):
                        batch_ids = chunk_ids[i:i+batch_size]
                        placeholders = ','.join(['?'] * len(batch_ids))
                        query = f"SELECT id, text, source_type, source_id, chunk_index FROM text_chunks WHERE id IN ({placeholders})"
                        cursor.execute(query, batch_ids)
                        rows = cursor.fetchall()
                        
                        chunk_dict = {row['id']: row for row in rows}
                        
                        for chunk_id in batch_ids:
                            if chunk_id in chunk_dict:
                                row = chunk_dict[chunk_id]
                                self.document_texts.append(row['text'])
                                self.document_metadata.append({
                                    'chunk_id': chunk_id,
                                    'source_type': row['source_type'],
                                    'source_id': row['source_id'],
                                    'chunk_index': row['chunk_index']
                                })
                            else:
                                self.document_texts.append("")
                                self.document_metadata.append({'chunk_id': chunk_id})
                        
                        self.progress_monitor.update_stage_progress(i // batch_size + 1, total_batches)
                    
                    conn.close()
                else:
                    raise ValueError("Could not find chunk_ids file or metadata file")
            except Exception as e:
                logger.error(f"Failed to load from database: {e}")
                raise ValueError(f"Could not load metadata: {e}")
        
        logger.info(f"Loaded {len(self.document_texts)} texts and {len(self.document_metadata)} metadata entries")
        
        logger.info(f"Loaded {len(self.document_texts)} documents")
        logger.info(f"Index size: {self.index.ntotal}")
    
    def extract_all_embeddings(self, use_regeneration: bool = True, 
                               checkpoint_data: Optional[Dict[str, Any]] = None,
                               sample_size: Optional[int] = None) -> np.ndarray:
        """FAISS 인덱스에서 모든 임베딩 추출 또는 재생성"""
        if checkpoint_data and "embeddings" in checkpoint_data:
            logger.info("체크포인트에서 임베딩 로드...")
            embeddings = checkpoint_data["embeddings"]
            logger.info(f"로드된 임베딩: {len(embeddings)}개, 차원: {embeddings.shape[1]}")
            return embeddings
        
        n_total = self.index.ntotal
        dimension = self.index.d
        
        if n_total == 0:
            raise ValueError("Vector store is empty")
        
        if use_regeneration:
            texts_to_process = self.document_texts
            
            if sample_size and sample_size < len(self.document_texts):
                logger.info(f"샘플링: {sample_size}개 문서만 임베딩 생성 (전체 {len(self.document_texts)}개 중)")
                indices = np.random.choice(len(self.document_texts), sample_size, replace=False)
                texts_to_process = [self.document_texts[i] for i in indices]
                self.progress_monitor.start_stage("임베딩 재생성 (샘플링)", len(texts_to_process))
            else:
                self.progress_monitor.start_stage("임베딩 재생성", len(texts_to_process))
            
            logger.info(f"Regenerating embeddings from {len(texts_to_process)} document texts (faster than extracting from index)...")
            
            # 배치 단위로 임베딩 생성 및 중간 체크포인트 저장
            batch_size = 64
            total_batches = (len(texts_to_process) + batch_size - 1) // batch_size
            embeddings_list = []
            checkpoint_interval = max(1, total_batches // 5)  # 5번에 한 번씩 체크포인트 저장
            
            try:
                for i in range(0, len(texts_to_process), batch_size):
                    batch_texts = texts_to_process[i:i+batch_size]
                    batch_embeddings = self.vector_store.generate_embeddings(batch_texts, batch_size=batch_size)
                    embeddings_list.append(batch_embeddings)
                    
                    batch_num = i // batch_size + 1
                    self.progress_monitor.update_stage_progress(min(i + batch_size, len(texts_to_process)), len(texts_to_process))
                    
                    # 중간 체크포인트 저장
                    if batch_num % checkpoint_interval == 0 or batch_num == total_batches:
                        partial_embeddings = np.vstack(embeddings_list) if len(embeddings_list) > 1 else embeddings_list[0]
                        self.checkpoint_manager.save_checkpoint(
                            "embeddings_extracting",
                            {
                                "use_regeneration": use_regeneration,
                                "n_total": n_total,
                                "dimension": dimension,
                                "progress": batch_num,
                                "total_batches": total_batches,
                                "completed": batch_num == total_batches
                            },
                            embeddings=partial_embeddings
                        )
                        del partial_embeddings
                        gc.collect()
                    
                    del batch_texts, batch_embeddings
                    if batch_num % 10 == 0:
                        gc.collect()
                
                embeddings = np.vstack(embeddings_list) if len(embeddings_list) > 1 else embeddings_list[0]
                del embeddings_list
            except Exception as e:
                logger.error(f"임베딩 생성 중 오류 발생: {e}")
                if embeddings_list:
                    logger.info("부분 임베딩 저장 중...")
                    partial_embeddings = np.vstack(embeddings_list) if len(embeddings_list) > 1 else embeddings_list[0]
                    self.checkpoint_manager.save_checkpoint(
                        "embeddings_extracting",
                        {
                            "use_regeneration": use_regeneration,
                            "n_total": n_total,
                            "dimension": dimension,
                            "error": str(e),
                            "partial": True
                        },
                        embeddings=partial_embeddings
                    )
                raise
            
            del texts_to_process
            gc.collect()
            
            self.progress_monitor.end_stage()
            logger.info(f"Generated {len(embeddings)} embeddings of dimension {dimension}")
            
            self.checkpoint_manager.save_checkpoint(
                "embeddings_extracted",
                {"use_regeneration": use_regeneration, "n_total": n_total, "dimension": dimension},
                embeddings=embeddings
            )
            return embeddings
        
        self.progress_monitor.start_stage("FAISS 인덱스에서 임베딩 추출", n_total)
        logger.info("Extracting all embeddings from FAISS index...")
        embeddings = np.zeros((n_total, dimension), dtype=np.float32)
        
        try:
            batch_size = 1000
            for i in tqdm(range(0, n_total, batch_size), desc="Extracting embeddings"):
                end_idx = min(i + batch_size, n_total)
                batch_indices = list(range(i, end_idx))
                
                for idx in batch_indices:
                    embedding = self.index.reconstruct(idx)
                    if isinstance(embedding, np.ndarray):
                        embeddings[idx] = embedding.astype(np.float32)
                    else:
                        embeddings[idx] = np.array(embedding, dtype=np.float32)
                
                del batch_indices
                if i % (batch_size * 10) == 0:
                    gc.collect()
                
                self.progress_monitor.update_stage_progress(end_idx, n_total)
        except Exception as e:
            logger.warning(f"Failed to extract embeddings directly from index: {e}")
            logger.info("Falling back to regenerating embeddings from document texts...")
            self.progress_monitor.start_stage("임베딩 재생성 (Fallback)", len(self.document_texts))
            embeddings = self.vector_store.generate_embeddings(self.document_texts, batch_size=64)
            self.progress_monitor.end_stage()
        
        self.progress_monitor.end_stage()
        logger.info(f"Extracted {n_total} embeddings of dimension {dimension}")
        
        self.checkpoint_manager.save_checkpoint(
            "embeddings_extracted",
            {"use_regeneration": use_regeneration, "n_total": n_total, "dimension": dimension},
            embeddings=embeddings
        )
        return embeddings
    
    def find_optimal_clusters(self, embeddings: np.ndarray, max_k: int = 50, sample_size: Optional[int] = None) -> int:
        """Elbow method를 사용하여 최적 클러스터 수 찾기 (샘플링된 데이터 사용)"""
        sample_embeddings = embeddings
        if sample_size and sample_size < len(embeddings):
            logger.info(f"Sampling {sample_size} embeddings from {len(embeddings)} for Elbow method...")
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
        
        inertias = []
        max_k = min(max_k, len(sample_embeddings) // 10, 20)
        k_range = range(2, max_k + 1)
        
        self.progress_monitor.start_stage("Elbow Method (최적 클러스터 수 찾기)", len(k_range))
        logger.info(f"Testing {len(k_range)} cluster numbers (2 to {max_k}) with n_init=3...")
        
        for idx, k in enumerate(tqdm(k_range, desc="Testing cluster numbers"), 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
            kmeans.fit(sample_embeddings)
            inertias.append(kmeans.inertia_)
            self.progress_monitor.update_stage_progress(idx, len(k_range))
        
        if len(inertias) < 2:
            self.progress_monitor.end_stage()
            return 10
        
        diffs = np.diff(inertias)
        diff_diffs = np.diff(diffs)
        optimal_k = k_range[np.argmax(diff_diffs) + 1] if len(diff_diffs) > 0 else k_range[len(k_range) // 2]
        
        self.progress_monitor.end_stage()
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        self.checkpoint_manager.save_checkpoint(
            "optimal_clusters_found",
            {"optimal_k": int(optimal_k), "max_k": max_k, "sample_size": sample_size}
        )
        
        return optimal_k
    
    def cluster_documents(self, embeddings: np.ndarray, algorithm: str = "kmeans", 
                         n_clusters: Optional[int] = None, min_cluster_size: int = 3,
                         sample_size: Optional[int] = None, calculate_silhouette: bool = False) -> np.ndarray:
        """
        문서 클러스터링
        
        Args:
            embeddings: 문서 임베딩 배열
            algorithm: 클러스터링 알고리즘 ("kmeans" 또는 "hdbscan")
            n_clusters: 클러스터 수 (K-means용, None이면 자동 결정)
            min_cluster_size: 최소 클러스터 크기 (HDBSCAN용)
            sample_size: Elbow method용 샘플 크기
            calculate_silhouette: Silhouette score 계산 여부
        
        Returns:
            클러스터 레이블 배열
        """
        self.progress_monitor.start_stage(f"{algorithm.upper()} 클러스터링", 1)
        logger.info(f"Clustering documents using {algorithm}...")
        
        if algorithm == "kmeans":
            if n_clusters is None or n_clusters == "auto":
                elbow_sample_size = sample_size or min(10000, len(embeddings) // 5)
                n_clusters = self.find_optimal_clusters(embeddings, max_k=20, sample_size=elbow_sample_size)
            
            logger.info(f"Running K-means with {n_clusters} clusters (n_init=3, max_iter=100)...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=100)
            labels = kmeans.fit_predict(embeddings)
            self.progress_monitor.update_stage_progress(1, 1)
            
            if calculate_silhouette:
                sample_size_sil = min(5000, len(embeddings))
                if sample_size_sil < len(embeddings):
                    logger.info(f"Calculating silhouette score on {sample_size_sil} samples...")
                    indices = np.random.choice(len(embeddings), sample_size_sil, replace=False)
                    sample_embeddings = embeddings[indices]
                    sample_labels = labels[indices]
                    score = silhouette_score(sample_embeddings, sample_labels)
                else:
                    score = silhouette_score(embeddings, labels)
                logger.info(f"K-means clustering completed. Silhouette score: {score:.3f}")
            else:
                logger.info(f"K-means clustering completed.")
        
        elif algorithm == "hdbscan":
            if not HDBSCAN_AVAILABLE:
                raise ImportError("hdbscan is required for HDBSCAN clustering")
            
            logger.info(f"Running HDBSCAN with min_cluster_size={min_cluster_size}...")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels = clusterer.fit_predict(embeddings)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            logger.info(f"HDBSCAN clustering completed. {n_clusters} clusters, {n_noise} noise points")
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        self.progress_monitor.end_stage()
        
        self.checkpoint_manager.save_checkpoint(
            "clustering_completed",
            {
                "algorithm": algorithm,
                "n_clusters": int(n_clusters) if n_clusters else None,
                "min_cluster_size": min_cluster_size,
                "n_labels": len(labels),
                "unique_clusters": len(set(labels)) - (1 if -1 in labels else 0)
            },
            labels=labels
        )
        
        return labels
    
    def generate_ground_truth(self, labels: np.ndarray, 
                            similarity_threshold: float = 0.7,
                            min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """
        클러스터링 결과로부터 Ground Truth 생성
        
        Args:
            labels: 클러스터 레이블 배열
            similarity_threshold: 클러스터 내 문서 간 최소 유사도 (사용하지 않음, 향후 확장용)
            min_cluster_size: 최소 클러스터 크기
        
        Returns:
            Ground Truth 데이터셋
        """
        cluster_dict = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(idx)
        
        valid_clusters = {k: v for k, v in cluster_dict.items() if len(v) >= min_cluster_size}
        total_entries = sum(len(doc_indices) for doc_indices in valid_clusters.values())
        
        del cluster_dict
        gc.collect()
        
        self.progress_monitor.start_stage("Ground Truth 생성", total_entries)
        logger.info("Generating ground truth from clusters...")
        
        ground_truth = []
        processed = 0
        batch_size = 1000
        
        for cluster_id, doc_indices in tqdm(valid_clusters.items(), desc="Processing clusters"):
            for query_idx in doc_indices:
                relevant_indices = [idx for idx in doc_indices if idx != query_idx]
                
                if not relevant_indices:
                    continue
                
                doc_id = f"doc_{query_idx}"
                if query_idx < len(self.document_metadata):
                    metadata = self.document_metadata[query_idx].copy()
                    doc_id = metadata.get('chunk_id', doc_id)
                else:
                    metadata = {}
                
                ground_truth.append({
                    "query_doc_id": doc_id,
                    "query_idx": int(query_idx),
                    "query_text": self.document_texts[query_idx] if query_idx < len(self.document_texts) else "",
                    "relevant_doc_ids": [f"doc_{idx}" if idx >= len(self.document_metadata) or 'chunk_id' not in self.document_metadata[idx] 
                                        else self.document_metadata[idx].get('chunk_id', f"doc_{idx}") 
                                        for idx in relevant_indices],
                    "relevant_indices": [int(idx) for idx in relevant_indices],
                    "cluster_id": int(cluster_id),
                    "cluster_size": len(doc_indices),
                    "metadata": metadata
                })
                
                processed += 1
                
                if processed % batch_size == 0:
                    gc.collect()
                
                if processed % max(1, total_entries // 20) == 0 or processed == total_entries:
                    self.progress_monitor.update_stage_progress(processed, total_entries)
        
        del valid_clusters
        gc.collect()
        
        self.progress_monitor.end_stage()
        logger.info(f"Generated {len(ground_truth)} ground truth entries")
        return ground_truth
    
    def run(self, algorithm: str = "kmeans", n_clusters: Optional[int] = None,
           min_cluster_size: int = 3, similarity_threshold: float = 0.7,
           sample_size: Optional[int] = None, use_regeneration: bool = True,
           calculate_silhouette: bool = False, resume: bool = True) -> List[Dict[str, Any]]:
        """
        전체 프로세스 실행
        
        Args:
            algorithm: 클러스터링 알고리즘
            n_clusters: 클러스터 수 (K-means용)
            min_cluster_size: 최소 클러스터 크기
            similarity_threshold: 유사도 임계값
            sample_size: 샘플 크기 (None이면 전체 데이터)
            use_regeneration: 임베딩 재생성 사용 여부
            calculate_silhouette: Silhouette score 계산 여부
            resume: 체크포인트에서 재개할지 여부
        
        Returns:
            Ground Truth 데이터셋
        """
        checkpoint_data = None
        if resume and self.checkpoint_manager.has_checkpoint():
            logger.info("=" * 80)
            logger.info("체크포인트 발견 - 이전 작업 복구 중...")
            logger.info("=" * 80)
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            
            if checkpoint_data:
                logger.info(f"복구된 단계: {checkpoint_data['stage']}")
                logger.info(f"저장 시각: {checkpoint_data.get('saved_at', 'N/A')}")
                logger.info("=" * 80)
        
        self.progress_monitor.start()
        
        if checkpoint_data and checkpoint_data["stage"] == "clustering_completed":
            logger.info("체크포인트에서 클러스터링 결과 로드...")
            labels = checkpoint_data["labels"]
            logger.info(f"로드된 레이블: {len(labels)}개")
        else:
            if checkpoint_data and checkpoint_data["stage"] in ["embeddings_extracted", "embeddings_extracting"]:
                checkpoint_stage = checkpoint_data["stage"]
                checkpoint_info = checkpoint_data.get("data", {})
                
                if checkpoint_stage == "embeddings_extracted":
                    logger.info("체크포인트에서 임베딩 로드 - 클러스터링부터 재개...")
                    embeddings = checkpoint_data["embeddings"]
                elif checkpoint_stage == "embeddings_extracting":
                    if checkpoint_info.get("completed", False):
                        logger.info("체크포인트에서 완료된 임베딩 로드...")
                        embeddings = checkpoint_data["embeddings"]
                    else:
                        logger.info(f"체크포인트에서 부분 임베딩 로드 (진행률: {checkpoint_info.get('progress', 0)}/{checkpoint_info.get('total_batches', 0)})")
                        logger.info("임베딩 생성 재개...")
                        embeddings = self.extract_all_embeddings(
                            use_regeneration=use_regeneration,
                            checkpoint_data=checkpoint_data,
                            sample_size=sample_size
                        )
                else:
                    embeddings = self.extract_all_embeddings(
                        use_regeneration=use_regeneration,
                        checkpoint_data=checkpoint_data,
                        sample_size=sample_size
                    )
            else:
                embeddings = self.extract_all_embeddings(
                    use_regeneration=use_regeneration,
                    checkpoint_data=checkpoint_data,
                    sample_size=sample_size
                )
            
            sample_embeddings = embeddings
            sample_indices = None
            
            if sample_size and sample_size < len(embeddings):
                logger.info(f"추가 샘플링: {sample_size}개 임베딩만 클러스터링 (전체 {len(embeddings)}개 중)")
                indices = np.random.choice(len(embeddings), sample_size, replace=False)
                sample_embeddings = embeddings[indices]
                sample_indices = indices
            else:
                sample_indices = np.arange(len(embeddings))
            
            if checkpoint_data and checkpoint_data["stage"] == "optimal_clusters_found":
                logger.info("체크포인트에서 최적 클러스터 수 로드...")
                checkpoint_n_clusters = checkpoint_data["data"].get("optimal_k")
                if checkpoint_n_clusters:
                    n_clusters = checkpoint_n_clusters
                    logger.info(f"복구된 최적 클러스터 수: {n_clusters}")
            
            labels = self.cluster_documents(sample_embeddings, algorithm, n_clusters, min_cluster_size,
                                           sample_size=sample_size, calculate_silhouette=calculate_silhouette)
            
            if sample_size and sample_size < len(embeddings) and sample_indices is not None:
                full_labels = np.full(len(embeddings), -1, dtype=int)
                for i, label in zip(sample_indices, labels):
                    full_labels[i] = label
                labels = full_labels
                
                del sample_embeddings, sample_indices
                gc.collect()
            
            del embeddings
            gc.collect()
        
        ground_truth = self.generate_ground_truth(labels, similarity_threshold, min_cluster_size)
        
        del labels
        gc.collect()
        
        self.checkpoint_manager.save_checkpoint(
            "ground_truth_completed",
            {
                "total_entries": len(ground_truth),
                "similarity_threshold": similarity_threshold,
                "min_cluster_size": min_cluster_size
            }
        )
        
        self.progress_monitor.finish()
        
        logger.info("=" * 80)
        logger.info("작업 완료 - 체크포인트 정리 중...")
        logger.info("=" * 80)
        self.checkpoint_manager.clear_checkpoint()
        logger.info("체크포인트 정리 완료")
        
        gc.collect()
        
        return ground_truth


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="클러스터링 기반 Ground Truth 생성")
    parser.add_argument(
        "--vector-store-path",
        type=str,
        default="data/embeddings/ml_enhanced_ko_sroberta",
        help="벡터 스토어 경로"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/evaluation/rag_ground_truth_clustering.json",
        help="출력 파일 경로"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="kmeans",
        choices=["kmeans", "hdbscan"],
        help="클러스터링 알고리즘"
    )
    parser.add_argument(
        "--n-clusters",
        type=str,
        default=None,
        help="클러스터 수 (K-means용, 'auto' 또는 None이면 자동 결정, 정수 입력 시 해당 값 사용)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="최소 클러스터 크기"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="클러스터 내 문서 간 최소 유사도 (향후 확장용)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="jhgan/ko-sroberta-multitask",
        help="임베딩 모델명"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="샘플 크기 (None이면 전체 데이터 사용)"
    )
    parser.add_argument(
        "--use-regeneration",
        action="store_true",
        default=True,
        help="FAISS 인덱스에서 추출 대신 문서 텍스트에서 임베딩 재생성 (기본값: True, 더 빠름)"
    )
    parser.add_argument(
        "--no-regeneration",
        action="store_false",
        dest="use_regeneration",
        help="FAISS 인덱스에서 직접 임베딩 추출 (느림)"
    )
    parser.add_argument(
        "--calculate-silhouette",
        action="store_true",
        help="Silhouette score 계산 (느림, 선택사항)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="체크포인트 저장 디렉토리 (기본값: data/evaluation/checkpoints)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="체크포인트에서 재개하지 않음 (처음부터 시작)"
    )
    
    args = parser.parse_args()
    
    n_clusters = None
    if args.n_clusters and args.n_clusters.lower() != 'auto':
        try:
            n_clusters = int(args.n_clusters)
        except ValueError:
            logger.warning(f"Invalid n_clusters value: {args.n_clusters}, using auto")
            n_clusters = None
    
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn is required. Please install: pip install scikit-learn")
        sys.exit(1)
    
    if args.algorithm == "hdbscan" and not HDBSCAN_AVAILABLE:
        logger.error("hdbscan is required for HDBSCAN clustering. Please install: pip install hdbscan")
        sys.exit(1)
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        generator = ClusteringGroundTruthGenerator(
            vector_store_path=args.vector_store_path,
            model_name=args.model_name,
            checkpoint_dir=args.checkpoint_dir
        )
        
        ground_truth = generator.run(
            algorithm=args.algorithm,
            n_clusters=n_clusters,
            min_cluster_size=args.min_cluster_size,
            similarity_threshold=args.similarity_threshold,
            sample_size=args.sample_size,
            use_regeneration=args.use_regeneration,
            calculate_silhouette=args.calculate_silhouette,
            resume=not args.no_resume
        )
        
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "vector_store_path": str(args.vector_store_path),
                "model_name": args.model_name,
                "algorithm": args.algorithm,
                "n_clusters": n_clusters if n_clusters else "auto",
                "min_cluster_size": args.min_cluster_size,
                "similarity_threshold": args.similarity_threshold,
                "total_documents": len(generator.document_texts),
                "total_ground_truth_entries": len(ground_truth),
                "sample_size": args.sample_size,
                "use_regeneration": args.use_regeneration,
                "calculate_silhouette": args.calculate_silhouette
            },
            "ground_truth": ground_truth
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Ground truth saved to: {output_path}")
        logger.info(f"Total entries: {len(ground_truth)}")
        
    except Exception as e:
        logger.error(f"Failed to generate ground truth: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

