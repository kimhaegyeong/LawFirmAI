#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
문서 기반 Pseudo-Query 생성

LLM을 사용하여 벡터 스토어의 각 문서에 대한 질문을 생성하고,
원본 문서를 Ground Truth로 사용하는 데이터셋을 생성합니다.
"""

import logging
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse
from tqdm import tqdm
import time
import gc

try:
    from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        from langchain.schema import HumanMessage, SystemMessage
    LANCHAIN_AVAILABLE = True
except ImportError as e:
    LANCHAIN_AVAILABLE = False
    logging.warning(f"LangChain not available: {e}. Please install langchain")

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.data.vector_store import LegalVectorStore

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

root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        root_logger.removeHandler(handler)
        root_logger.addHandler(SafeStreamHandler(sys.stdout))

logger = logging.getLogger(__name__)


class ProgressMonitor:
    """진행 상황 모니터링 클래스"""
    
    def __init__(self):
        self.start_time = None
        self.stage_start_time = None
        self.stage_name = None
        self.stage_progress = 0
        self.stage_total = 0
    
    def start(self):
        """전체 프로세스 시작"""
        self.start_time = time.time()
        logger.info("=" * 80)
        logger.info("Pseudo-Query 생성 프로세스 시작")
        logger.info("=" * 80)
    
    def start_stage(self, stage_name: str, total: int = 0):
        """단계 시작"""
        if self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            logger.info(f"이전 단계 완료 ({self._format_time(elapsed)})")
        
        self.stage_name = stage_name
        self.stage_start_time = time.time()
        self.stage_progress = 0
        self.stage_total = total
        
        logger.info("")
        logger.info(f"[단계] {stage_name} 시작...")
        if total > 0:
            logger.info(f"  총 작업량: {total:,}개")
    
    def update_progress(self, progress: int, total: Optional[int] = None):
        """진행 상황 업데이트"""
        self.stage_progress = progress
        if total is not None:
            self.stage_total = total
        
        if self.stage_total > 0 and self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            progress_pct = (progress / self.stage_total * 100) if self.stage_total > 0 else 0
            
            if progress > 0 and progress % max(1, self.stage_total // 20) == 0:
                avg_time_per_item = elapsed / progress
                remaining_items = self.stage_total - progress
                remaining_time = avg_time_per_item * remaining_items
                
                logger.info(
                    f"  진행률: {progress:,}/{self.stage_total:,} ({progress_pct:.1f}%) | "
                    f"경과: {self._format_time(elapsed)} | "
                    f"예상 남은 시간: {self._format_time(remaining_time)}"
                )
    
    def end_stage(self):
        """단계 종료"""
        if self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            logger.info(f"[단계] {self.stage_name} 완료 ({self._format_time(elapsed)})")
    
    def finish(self):
        """전체 프로세스 종료"""
        if self.stage_start_time:
            self.end_stage()
        
        if self.start_time:
            total_elapsed = time.time() - self.start_time
            logger.info("")
            logger.info("=" * 80)
            logger.info("Pseudo-Query 생성 프로세스 완료")
            logger.info("=" * 80)
            logger.info(f"총 소요 시간: {self._format_time(total_elapsed)}")
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
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path("data/evaluation/checkpoints")
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "pseudo_query_checkpoint.json"
    
    def save_checkpoint(self, stage: str, data: Dict[str, Any], ground_truth: Optional[List[Dict[str, Any]]] = None) -> bool:
        """체크포인트 저장"""
        try:
            checkpoint_data = {
                "stage": stage,
                "saved_at": datetime.now().isoformat(),
                "data": data
            }
            
            if ground_truth:
                checkpoint_data["ground_truth"] = ground_truth
            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"체크포인트 저장 완료: {stage}")
            return True
        except Exception as e:
            logger.error(f"체크포인트 저장 실패: {e}")
            return False
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """체크포인트 로드"""
        try:
            if not self.checkpoint_file.exists():
                return None
            
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"체크포인트 로드 완료: {checkpoint_data.get('stage')}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"체크포인트 로드 실패: {e}")
            return None
    
    def clear_checkpoint(self) -> bool:
        """체크포인트 삭제"""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            logger.info("체크포인트 삭제 완료")
            return True
        except Exception as e:
            logger.error(f"체크포인트 삭제 실패: {e}")
            return False
    
    def has_checkpoint(self) -> bool:
        """체크포인트 존재 여부 확인"""
        return self.checkpoint_file.exists()


class PseudoQueryGenerator:
    """Pseudo-Query 생성 클래스"""
    
    def __init__(self, vector_store_path: str, model_name: str = "jhgan/ko-sroberta-multitask",
                 llm_model: str = "gpt-4", llm_provider: str = "openai",
                 checkpoint_dir: Optional[str] = None):
        """
        초기화
        
        Args:
            vector_store_path: 벡터 스토어 경로
            model_name: 임베딩 모델명
            llm_model: LLM 모델명
            llm_provider: LLM 제공자 ("openai", "anthropic", "google")
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.vector_store_path = Path(vector_store_path)
        self.model_name = model_name
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        
        self.progress_monitor = ProgressMonitor()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        self.vector_store = LegalVectorStore(
            model_name=model_name,
            dimension=768,
            index_type="flat"
        )
        
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
        
        if not self.vector_store.load_index(str(index_file)):
            logger.warning("Failed to load via load_index, trying direct load...")
            import faiss
            self.vector_store.index = faiss.read_index(faiss_file)
            self.vector_store._index_loaded = True
        
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
                    
                    conn = sqlite3.connect(db_path)
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    self.document_texts = []
                    self.document_metadata = []
                    
                    batch_size = 1000
                    logger.info(f"Loading {len(chunk_ids)} chunks from database in batches of {batch_size}...")
                    
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
                    conn.close()
                else:
                    raise ValueError("Could not find chunk_ids file or metadata file")
            except Exception as e:
                logger.error(f"Failed to load from database: {e}")
                raise ValueError(f"Could not load metadata: {e}")
        
        logger.info(f"Loaded {len(self.document_texts)} texts and {len(self.document_metadata)} metadata entries")
        logger.info(f"Loaded {len(self.document_texts)} documents")
        
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """LLM 초기화 (비용 최적화)"""
        if not LANCHAIN_AVAILABLE:
            logger.warning("LangChain not available. Pseudo-query generation will be skipped.")
            return None
        
        try:
            if self.llm_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not found. Pseudo-query generation will be skipped.")
                    return None
                return ChatOpenAI(
                    model_name=self.llm_model,
                    temperature=0.7,
                    max_tokens=200
                )
            
            elif self.llm_provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.warning("ANTHROPIC_API_KEY not found. Pseudo-query generation will be skipped.")
                    return None
                return ChatAnthropic(
                    model=self.llm_model,
                    temperature=0.7,
                    max_tokens=200
                )
            
            elif self.llm_provider == "google":
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    logger.warning("GOOGLE_API_KEY not found. Pseudo-query generation will be skipped.")
                    return None
                # 비용 절감: 저비용 모델 사용, 출력 토큰 제한
                model_name = self.llm_model
                if model_name == "gpt-4" or model_name.startswith("gemini-1.5") or model_name.startswith("gemini-2.0"):
                    model_name = "gemini-2.5-flash-lite"  # 더 저렴한 모델로 변경
                    logger.info(f"비용 절감을 위해 모델을 {model_name}로 변경합니다.")
                
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.7,
                    max_output_tokens=300,  # 출력 토큰 제한 (3개 질문 생성용)
                    google_api_key=api_key
                )
            
            else:
                logger.warning(f"Unknown LLM provider: {self.llm_provider}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None
    
    def _generate_queries_for_document(self, document_text: str, num_queries: int = 3,
                                      max_text_length: int = 2000) -> List[str]:
        """
        단일 문서에 대한 질문 생성 (비용 최적화)
        
        Args:
            document_text: 문서 텍스트
            num_queries: 생성할 질문 수
            max_text_length: 프롬프트에 포함할 최대 텍스트 길이 (비용 절감)
        
        Returns:
            생성된 질문 리스트
        """
        if not self.llm:
            return []
        
        if not document_text or len(document_text.strip()) < 10:
            return []
        
        # 비용 절감: 텍스트 길이 제한
        text_to_use = document_text[:max_text_length]
        if len(document_text) > max_text_length:
            text_to_use = text_to_use.rsplit('.', 1)[0] + '.'  # 문장 단위로 자르기
        
        # 비용 절감: 간결한 프롬프트
        prompt = f"""법률 문서에서 질문 {num_queries}개 생성:

{text_to_use}

형식: 질문1: [내용]
질문2: [내용]"""
        
        try:
            # 비용 절감: SystemMessage 제거 (토큰 절약)
            messages = [HumanMessage(content=prompt)]
            
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            queries = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line.startswith('질문') or ':' in line):
                    query = line.split(':', 1)[-1].strip()
                    if query and len(query) > 5:
                        queries.append(query)
            
            if len(queries) < num_queries:
                logger.debug(f"Generated only {len(queries)} queries, expected {num_queries}")
            
            return queries[:num_queries]
        
        except Exception as e:
            logger.error(f"Failed to generate queries: {e}")
            return []
    
    def generate_ground_truth(self, queries_per_doc: int = 3, 
                            batch_size: int = 10,
                            max_documents: Optional[int] = None,
                            resume: bool = True,
                            max_text_length: int = 2000,
                            min_text_length: int = 50,
                            max_text_length_filter: int = 5000) -> List[Dict[str, Any]]:
        """
        Ground Truth 데이터셋 생성
        
        Args:
            queries_per_doc: 문서당 생성할 질문 수
            batch_size: 배치 크기 (체크포인트 저장 주기)
            max_documents: 최대 처리 문서 수 (None이면 전체)
            resume: 체크포인트에서 재개할지 여부
        
        Returns:
            Ground Truth 데이터셋
        """
        if not self.llm:
            logger.error("LLM not initialized. Cannot generate pseudo-queries.")
            return []
        
        checkpoint_data = None
        if resume and self.checkpoint_manager.has_checkpoint():
            logger.info("=" * 80)
            logger.info("체크포인트 발견 - 이전 작업 복구 중...")
            logger.info("=" * 80)
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            
            if checkpoint_data:
                logger.info(f"복구된 단계: {checkpoint_data.get('stage', 'N/A')}")
                logger.info(f"저장 시각: {checkpoint_data.get('saved_at', 'N/A')}")
                logger.info("=" * 80)
        
        self.progress_monitor.start()
        self.progress_monitor.start_stage("Pseudo-Query 생성", max_documents or len(self.document_texts))
        
        documents_to_process = self.document_texts
        document_indices = list(range(len(self.document_texts)))
        start_idx = 0
        
        if max_documents and max_documents < len(self.document_texts):
            logger.info(f"샘플링: {max_documents}개 문서만 처리 (전체 {len(self.document_texts)}개 중)")
            import random
            indices = random.sample(range(len(self.document_texts)), max_documents)
            documents_to_process = [self.document_texts[i] for i in indices]
            document_indices = indices
        elif max_documents:
            documents_to_process = documents_to_process[:max_documents]
            document_indices = list(range(max_documents))
        
        if checkpoint_data and checkpoint_data.get("stage") == "query_generation":
            checkpoint_info = checkpoint_data.get("data", {})
            start_idx = checkpoint_info.get("processed_count", 0)
            ground_truth = checkpoint_data.get("ground_truth", [])
            logger.info(f"체크포인트에서 복구: {start_idx}개 문서 처리 완료, {len(ground_truth)}개 Ground Truth 생성됨")
        else:
            ground_truth = []
        
        logger.info("Generating pseudo-queries for documents...")
        logger.info(f"비용 절감 설정: 텍스트 길이 제한={max_text_length}자, 질문 수={queries_per_doc}개/문서")
        
        # 문서 필터링 (비용 절감)
        filtered_documents = []
        filtered_indices = []
        skipped_count = 0
        
        for local_idx, doc_text in enumerate(documents_to_process):
            text_len = len(doc_text.strip())
            if text_len < min_text_length:
                skipped_count += 1
                continue
            if text_len > max_text_length_filter:
                skipped_count += 1
                continue
            filtered_documents.append(doc_text)
            original_idx = document_indices[local_idx] if len(document_indices) != len(self.document_texts) else local_idx
            filtered_indices.append(original_idx)
        
        if skipped_count > 0:
            logger.info(f"문서 필터링: {skipped_count}개 문서 제외 (길이 기준: {min_text_length}-{max_text_length_filter}자)")
        
        logger.info(f"처리할 문서 수: {len(filtered_documents)}개")
        
        gc_interval = max(1, batch_size * 5)
        checkpoint_interval = batch_size
        
        for local_idx in tqdm(range(start_idx, len(filtered_documents)), desc="Generating queries", initial=start_idx):
            doc_text = filtered_documents[local_idx]
            original_idx = filtered_indices[local_idx]
            
            if original_idx >= len(self.document_metadata):
                metadata = {}
                doc_id = f"doc_{original_idx}"
            else:
                metadata = self.document_metadata[original_idx].copy()
                doc_id = metadata.get('chunk_id', f"doc_{original_idx}")
            
            queries = self._generate_queries_for_document(doc_text, queries_per_doc, max_text_length)
            
            if not queries:
                continue
            
            for query in queries:
                ground_truth.append({
                    "query": query,
                    "ground_truth_doc_id": doc_id,
                    "ground_truth_idx": original_idx,
                    "ground_truth_text": doc_text,
                    "metadata": metadata
                })
            
            processed_count = local_idx + 1
            
            if processed_count % gc_interval == 0:
                gc.collect()
            
            if processed_count % checkpoint_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    "query_generation",
                    {
                        "processed_count": processed_count,
                        "total_documents": len(documents_to_process),
                        "queries_per_doc": queries_per_doc
                    },
                    ground_truth=ground_truth
                )
            
            self.progress_monitor.update_progress(processed_count, len(filtered_documents))
        
        self.progress_monitor.end_stage()
        logger.info(f"Generated {len(ground_truth)} ground truth entries")
        
        del filtered_documents, filtered_indices, documents_to_process
        gc.collect()
        
        return ground_truth
    
    def run(self, queries_per_doc: int = 3, batch_size: int = 10,
           max_documents: Optional[int] = None, resume: bool = True,
           max_text_length: int = 2000, min_text_length: int = 50,
           max_text_length_filter: int = 5000) -> List[Dict[str, Any]]:
        """
        전체 프로세스 실행
        
        Args:
            queries_per_doc: 문서당 생성할 질문 수
            batch_size: 배치 크기 (체크포인트 저장 주기)
            max_documents: 최대 처리 문서 수
            resume: 체크포인트에서 재개할지 여부
            max_text_length: 프롬프트에 포함할 최대 텍스트 길이 (비용 절감)
            min_text_length: 처리할 최소 텍스트 길이
            max_text_length_filter: 처리할 최대 텍스트 길이
        
        Returns:
            Ground Truth 데이터셋
        """
        ground_truth = self.generate_ground_truth(
            queries_per_doc=queries_per_doc,
            batch_size=batch_size,
            max_documents=max_documents,
            resume=resume,
            max_text_length=max_text_length,
            min_text_length=min_text_length,
            max_text_length_filter=max_text_length_filter
        )
        
        self.checkpoint_manager.save_checkpoint(
            "query_generation_completed",
            {
                "total_entries": len(ground_truth),
                "queries_per_doc": queries_per_doc
            },
            ground_truth=ground_truth
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
    parser = argparse.ArgumentParser(description="Pseudo-Query 기반 Ground Truth 생성")
    parser.add_argument(
        "--vector-store-path",
        type=str,
        default="data/embeddings/ml_enhanced_ko_sroberta",
        help="벡터 스토어 경로"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/evaluation/rag_ground_truth_pseudo_queries.json",
        help="출력 파일 경로"
    )
    parser.add_argument(
        "--queries-per-doc",
        type=int,
        default=3,
        help="문서당 생성할 질문 수 (기본값: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="배치 크기"
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=500,
        help="최대 처리 문서 수 (비용 절감을 위해 기본값: 500, None이면 전체)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="LLM 모델명 (기본값: gemini-2.5-flash-lite)"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="google",
        choices=["openai", "anthropic", "google"],
        help="LLM 제공자 (비용 절감을 위해 기본값: google)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="jhgan/ko-sroberta-multitask",
        help="임베딩 모델명"
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
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=2000,
        help="프롬프트에 포함할 최대 텍스트 길이 (기본값: 2000자)"
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=50,
        help="처리할 최소 텍스트 길이 (너무 짧은 문서 제외, 기본값: 50자)"
    )
    parser.add_argument(
        "--max-text-length-filter",
        type=int,
        default=5000,
        help="처리할 최대 텍스트 길이 (너무 긴 문서 제외, 기본값: 5000자)"
    )
    
    args = parser.parse_args()
    
    if not LANCHAIN_AVAILABLE:
        logger.error("LangChain is required. Please install: pip install langchain langchain-community")
        sys.exit(1)
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        generator = PseudoQueryGenerator(
            vector_store_path=args.vector_store_path,
            model_name=args.model_name,
            llm_model=args.llm_model,
            llm_provider=args.llm_provider,
            checkpoint_dir=args.checkpoint_dir
        )
        
        if not generator.llm:
            logger.error("LLM initialization failed. Please check your API keys.")
            sys.exit(1)
        
        ground_truth = generator.run(
            queries_per_doc=args.queries_per_doc,
            batch_size=args.batch_size,
            max_documents=args.max_documents,
            resume=not args.no_resume,
            max_text_length=args.max_text_length,
            min_text_length=args.min_text_length,
            max_text_length_filter=args.max_text_length_filter
        )
        
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "vector_store_path": str(args.vector_store_path),
                "model_name": args.model_name,
                "llm_model": args.llm_model,
                "llm_provider": args.llm_provider,
                "queries_per_doc": args.queries_per_doc,
                "total_documents": len(generator.document_texts),
                "total_ground_truth_entries": len(ground_truth)
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

