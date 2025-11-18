"""
재임베딩 모니터링 공통 유틸리티 모듈

재임베딩 모니터링 스크립트들에서 공통으로 사용하는 함수들을 제공합니다.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.embedding_version_manager import EmbeddingVersionManager


class ReEmbeddingMonitor:
    """재임베딩 모니터링 공통 클래스"""
    
    def __init__(self, db_path: str, version_id: int):
        """
        초기화
        
        Args:
            db_path: 데이터베이스 경로
            version_id: 버전 ID
        """
        self.db_path = db_path
        self.version_id = version_id
        self.version_manager = EmbeddingVersionManager(db_path)
        self.version_info = None
        self._load_version_info()
    
    def _load_version_info(self) -> bool:
        """버전 정보 로드"""
        self.version_info = self.version_manager.get_version_statistics(self.version_id)
        return self.version_info is not None
    
    def get_version_name(self) -> Optional[str]:
        """버전 이름 반환"""
        return self.version_info.get('version_name') if self.version_info else None
    
    def get_total_documents(self) -> int:
        """전체 문서 수 조회"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute("""
            SELECT COUNT(DISTINCT source_type || '_' || source_id) as total
            FROM text_chunks
            WHERE embedding_version_id = 1
        """)
        total = cursor.fetchone()[0]
        conn.close()
        return total
    
    def get_processed_documents(self) -> int:
        """재임베딩된 문서 수 조회"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute("""
            SELECT COUNT(DISTINCT source_type || '_' || source_id) as processed
            FROM text_chunks
            WHERE embedding_version_id = ?
        """, (self.version_id,))
        processed = cursor.fetchone()[0]
        conn.close()
        return processed
    
    def get_progress(self) -> Tuple[float, int, int]:
        """
        진행률 계산
        
        Returns:
            (progress, processed, total): 진행률(0-1), 처리된 문서 수, 전체 문서 수
        """
        total = self.get_total_documents()
        processed = self.get_processed_documents()
        progress = processed / total if total > 0 else 0.0
        return progress, processed, total
    
    def get_documents_by_type(self) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
        """
        문서 타입별 통계 조회
        
        Returns:
            (total_by_type, processed_by_type): 전체 문서 타입별 수, 처리된 문서 타입별 수
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # 전체 문서 수
        cursor = conn.execute("""
            SELECT 
                source_type,
                COUNT(DISTINCT source_id) as total_documents
            FROM (
                SELECT DISTINCT source_type, source_id 
                FROM text_chunks 
                WHERE embedding_version_id = 1
            )
            GROUP BY source_type
        """)
        
        total_by_type = {}
        for row in cursor.fetchall():
            total_by_type[row['source_type']] = row['total_documents']
        
        # 재임베딩된 문서 수
        cursor = conn.execute("""
            SELECT 
                source_type,
                COUNT(DISTINCT source_id) as processed_documents,
                COUNT(*) as chunk_count
            FROM text_chunks
            WHERE embedding_version_id = ?
            GROUP BY source_type
        """, (self.version_id,))
        
        processed_by_type = {}
        for row in cursor.fetchall():
            processed_by_type[row['source_type']] = {
                'documents': row['processed_documents'],
                'chunks': row['chunk_count']
            }
        
        conn.close()
        return total_by_type, processed_by_type
    
    def get_version_created_at(self) -> Optional[datetime]:
        """버전 생성 시간 조회"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute("""
            SELECT created_at
            FROM embedding_versions
            WHERE id = ?
        """, (self.version_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            try:
                return datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S')
            except:
                return None
        return None
    
    def calculate_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """
        성능 메트릭 계산
        
        Returns:
            성능 메트릭 딕셔너리 또는 None
        """
        created_at = self.get_version_created_at()
        if not created_at:
            return None
        
        progress, processed, total = self.get_progress()
        elapsed = datetime.now() - created_at
        elapsed_seconds = elapsed.total_seconds()
        
        if processed == 0:
            return None
        
        avg_time_per_doc = elapsed_seconds / processed
        docs_per_hour = processed / (elapsed_seconds / 3600) if elapsed_seconds > 0 else 0
        remaining_docs = total - processed
        remaining_seconds = avg_time_per_doc * remaining_docs
        remaining_time = timedelta(seconds=int(remaining_seconds))
        
        return {
            'created_at': created_at,
            'elapsed': elapsed,
            'elapsed_seconds': elapsed_seconds,
            'processed': processed,
            'total': total,
            'remaining': remaining_docs,
            'progress': progress,
            'avg_time_per_doc': avg_time_per_doc,
            'docs_per_hour': docs_per_hour,
            'remaining_time': remaining_time,
            'estimated_completion': datetime.now() + remaining_time
        }
    
    def format_progress_bar(self, progress: float, length: int = 50) -> str:
        """
        진행 바 문자열 생성
        
        Args:
            progress: 진행률 (0-1)
            length: 진행 바 길이
        
        Returns:
            진행 바 문자열
        """
        filled = int(length * progress)
        bar = "█" * filled + "░" * (length - filled)
        return f"[{bar}] {progress*100:.2f}%"
    
    def format_time_delta(self, delta: timedelta) -> str:
        """
        시간 델타를 읽기 쉬운 형식으로 변환
        
        Args:
            delta: 시간 델타
        
        Returns:
            포맷된 시간 문자열
        """
        total_seconds = int(delta.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}초"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return f"{minutes}분"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            if minutes > 0:
                return f"{hours}시간 {minutes}분"
            return f"{hours}시간"
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        모니터링 가능 여부 확인
        
        Returns:
            (is_valid, error_message): 유효 여부, 오류 메시지
        """
        if not Path(self.db_path).exists():
            return False, f"데이터베이스 파일을 찾을 수 없습니다: {self.db_path}"
        
        if not self.version_info:
            return False, f"버전 ID {self.version_id}를 찾을 수 없습니다."
        
        return True, None

