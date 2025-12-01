"""
재임베딩 완료 대기 및 FAISS 인덱스 빌드 스크립트

재임베딩이 완료될 때까지 대기한 후 FAISS 인덱스를 빌드합니다.
"""
import sys
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.monitoring.monitor_re_embedding_progress import monitor_progress
from scripts.build_faiss_index_for_dynamic_chunking import build_faiss_index
from scripts.utils.embedding_version_manager import EmbeddingVersionManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_re_embedding_complete(db_path: str, version_id: int, threshold: float = 0.99):
    """
    재임베딩 완료 여부 확인
    
    Args:
        db_path: 데이터베이스 경로
        version_id: 버전 ID
        threshold: 완료 임계값 (0.99 = 99% 완료 시 완료로 간주)
    
    Returns:
        bool: 완료 여부
    """
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # 전체 문서 수 조회 (기존 버전 기준)
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT source_type || '_' || source_id) as total
        FROM text_chunks
        WHERE embedding_version_id = 1
    """)
    total_docs = cursor.fetchone()[0]
    
    # 재임베딩된 문서 수 조회
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT source_type || '_' || source_id) as processed
        FROM text_chunks
        WHERE embedding_version_id = ?
    """, (version_id,))
    processed_docs = cursor.fetchone()[0]
    
    conn.close()
    
    if total_docs == 0:
        return False
    
    progress = processed_docs / total_docs
    is_complete = progress >= threshold
    
    logger.info(f"재임베딩 진행률: {processed_docs}/{total_docs} ({progress*100:.2f}%)")
    
    return is_complete


def wait_for_re_embedding_complete(
    db_path: str,
    version_id: int,
    check_interval: int = 60,
    timeout: int = 14400  # 4시간
):
    """
    재임베딩 완료까지 대기
    
    Args:
        db_path: 데이터베이스 경로
        version_id: 버전 ID
        check_interval: 확인 간격 (초)
        timeout: 타임아웃 (초)
    
    Returns:
        bool: 완료 여부
    """
    logger.info("=" * 80)
    logger.info("재임베딩 완료 대기")
    logger.info("=" * 80)
    logger.info(f"확인 간격: {check_interval}초")
    logger.info(f"타임아웃: {timeout}초 ({timeout/3600:.1f}시간)")
    logger.info("")
    
    start_time = time.time()
    last_progress = 0
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > timeout:
            logger.warning(f"타임아웃: {timeout}초 경과")
            return False
        
        # 진행 상황 확인
        is_complete = check_re_embedding_complete(db_path, version_id)
        current_progress = check_re_embedding_complete(db_path, version_id, threshold=0.0)
        
        if is_complete:
            logger.info("✓ 재임베딩 완료!")
            return True
        
        # 진행 상황 출력
        if current_progress != last_progress:
            logger.info(f"진행 중... ({elapsed/60:.1f}분 경과)")
            monitor_progress(db_path, version_id)
            last_progress = current_progress
        
        # 대기
        time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(description="재임베딩 완료 대기 및 FAISS 인덱스 빌드")
    parser.add_argument("--db", default="data/lawfirm_v2.db", help="데이터베이스 경로")
    parser.add_argument("--version-id", type=int, required=True, help="임베딩 버전 ID")
    parser.add_argument("--vector-store", default="data/vector_store", help="벡터 스토어 경로")
    parser.add_argument("--check-interval", type=int, default=60, help="확인 간격 (초)")
    parser.add_argument("--timeout", type=int, default=14400, help="타임아웃 (초)")
    parser.add_argument("--skip-wait", action="store_true", help="대기 건너뛰고 바로 빌드")
    
    args = parser.parse_args()
    
    # 재임베딩 완료 대기
    if not args.skip_wait:
        logger.info("재임베딩 완료를 기다리는 중...")
        is_complete = wait_for_re_embedding_complete(
            args.db,
            args.version_id,
            args.check_interval,
            args.timeout
        )
        
        if not is_complete:
            logger.warning("재임베딩이 완료되지 않았습니다. 계속 진행할까요? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                logger.info("작업 취소")
                return
    else:
        logger.info("대기 건너뛰기: 바로 FAISS 인덱스 빌드 진행")
    
    # FAISS 인덱스 빌드
    logger.info("\n" + "=" * 80)
    logger.info("FAISS 인덱스 빌드 시작")
    logger.info("=" * 80)
    
    success = build_faiss_index(
        args.db,
        args.version_id,
        args.vector_store
    )
    
    if success:
        logger.info("\n✓ 모든 작업 완료!")
    else:
        logger.error("\n✗ FAISS 인덱스 빌드 실패")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

