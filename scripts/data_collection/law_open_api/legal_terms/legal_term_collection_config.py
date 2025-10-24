"""
법률 용어 수집 설정 파일

이 파일은 법률 용어 수집 시스템의 설정을 관리합니다.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

class LegalTermCollectionConfig:
    """법률 용어 수집 설정 클래스"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "scripts/data_collection/law_open_api/legal_terms/legal_term_collection_config.yaml"
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        default_config = {
            # API 설정
            "api": {
                "base_url": "http://www.law.go.kr/DRF",
                "oc_id": "test",  # 사용자 이메일 ID
                "display_count": 100,  # 한 번에 가져올 결과 수
                "max_retries": 3,  # 최대 재시도 횟수
                "retry_delays": [180, 300, 600],  # 재시도 대기 시간 (초)
                "request_timeout": 30,  # 요청 타임아웃 (초)
                "rate_limit_delay": 1.0  # 요청 간 대기 시간 (초)
            },
            
            # 수집 설정
            "collection": {
                "start_page": 1,
                "end_page": None,  # None이면 무제한
                "query": "",  # 검색 쿼리
                "gana": "",  # 가나다 검색
                "list_batch_size": 5,  # 목록 수집 배치 크기 (더 자주 저장)
                "detail_batch_size": 20,  # 상세 수집 배치 크기 (더 자주 저장)
                "vector_batch_size": 50,  # 벡터화 배치 크기 (더 자주 저장)
                "auto_resume": True,  # 자동 재개 여부
                "save_raw_data": True,  # 원본 데이터 저장 여부
                "save_batch_files": True,  # 배치 파일 저장 여부
                "progress_save_interval": 5,  # 진행 상황 저장 간격 (페이지)
                "detail_progress_save_interval": 10,  # 상세 수집 진행 상황 저장 간격 (개수)
                
                # 상세 조회 설정
                "collect_details": True,  # 상세 조회 여부
                "detail_collection_delay": 1.0,  # 상세 조회 간 대기 시간 (초)
                "max_detail_retries": 3,  # 상세 조회 최대 재시도 횟수
                "detail_timeout": 30,  # 상세 조회 타임아웃 (초)
                "skip_existing_details": True,  # 기존 상세 데이터 건너뛰기
                "save_every_page": True  # 매 페이지마다 저장 (누락 방지)
            },
            
            # 데이터베이스 설정 (기존 프로젝트와 통합)
            "database": {
                "db_path": "D:/project/LawFirmAI/LawFirmAI/data/lawfirm.db",  # 기존 프로젝트 데이터베이스 사용
                "backup_enabled": True,
                "backup_interval": 1000,  # 백업 간격 (수집된 항목 수)
                "connection_timeout": 30
            },
            
            # 벡터스토어 설정 (기존 프로젝트와 통합)
            "vector_store": {
                "model_name": "jhgan/ko-sroberta-multitask",
                "vector_dimension": 768,
                "index_type": "flat",  # flat, ivf, hnsw
                "similarity_metric": "l2",  # l2, cosine, ip
                "embeddings_dir": "D:/project/LawFirmAI/LawFirmAI/data/embeddings",
                "index_file": "legal_vector_index.faiss",  # 기존 인덱스 파일 사용
                "metadata_file": "legal_vector_index.json"  # 기존 메타데이터 파일 사용
            },
            
            # 로깅 설정
            "logging": {
                "level": "INFO",
                "log_file": "D:/project/LawFirmAI/LawFirmAI/logs/legal_term_collection.log",
                "max_file_size": "10MB",
                "backup_count": 5,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            
            # 모니터링 설정
            "monitoring": {
                "enabled": True,
                "progress_update_interval": 10,  # 진행 상황 업데이트 간격 (초)
                "status_file": "D:/project/LawFirmAI/LawFirmAI/data/legal_term_collection_status.json",
                "progress_file": "D:/project/LawFirmAI/LawFirmAI/data/legal_term_collection_progress.json",
                "metrics_file": "D:/project/LawFirmAI/LawFirmAI/data/legal_term_collection_metrics.json"
            },
            
            # 파일 저장 설정
            "file_storage": {
                "raw_data_dir": "D:/project/LawFirmAI/LawFirmAI/data/raw/law_open_api/legal_terms",
                "processed_data_dir": "D:/project/LawFirmAI/LawFirmAI/data/processed/legal_terms",
                "backup_dir": "D:/project/LawFirmAI/LawFirmAI/data/backups/legal_terms",
                
                # 상세 조회 데이터 저장 경로
                "detail_responses_dir": "D:/project/LawFirmAI/LawFirmAI/data/raw/law_open_api/legal_terms/responses/term_details",
                "detail_batches_dir": "D:/project/LawFirmAI/LawFirmAI/data/raw/law_open_api/legal_terms/batches",
                "list_responses_dir": "D:/project/LawFirmAI/LawFirmAI/data/raw/law_open_api/legal_terms/responses/term_lists",
                
                # 파일 명명 규칙
                "detail_response_prefix": "term_",
                "detail_response_suffix": "_detail.json",
                "detail_batch_prefix": "legal_term_details_batch_",
                "list_batch_prefix": "legal_term_list_batch_",
                
                # 저장 옵션
                "save_raw_responses": True,
                "save_batch_files": True,
                "save_daily_folders": True,
                "file_format": "json",
                "max_file_size_mb": 50
            },
            
            # 성능 설정
            "performance": {
                "max_concurrent_requests": 5,
                "memory_limit_mb": 2048,
                "cache_size": 1000,
                "enable_compression": True,
                "chunk_size": 1024
            }
        }
        
        # 환경 변수에서 설정 오버라이드
        env_overrides = {
            "api.oc_id": os.getenv("LEGAL_API_OC_ID"),
            "api.display_count": os.getenv("LEGAL_API_DISPLAY_COUNT"),
            "api.max_retries": os.getenv("LEGAL_API_MAX_RETRIES"),
            "collection.start_page": os.getenv("COLLECTION_START_PAGE"),
            "collection.end_page": os.getenv("COLLECTION_END_PAGE"),
            "collection.query": os.getenv("COLLECTION_QUERY"),
            "collection.gana": os.getenv("COLLECTION_GANA"),
            "vector_store.model_name": os.getenv("SENTENCE_BERT_MODEL"),
            "database.db_path": os.getenv("DATABASE_PATH"),
            "logging.level": os.getenv("LOG_LEVEL")
        }
        
        # 환경 변수 적용
        for key, value in env_overrides.items():
            if value is not None:
                self._set_nested_value(default_config, key, value)
        
        return default_config
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """중첩된 딕셔너리에 값 설정"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # 타입 변환
        if key.startswith("api.display_count") or key.startswith("api.max_retries"):
            current[keys[-1]] = int(value)
        elif key.startswith("collection.start_page") or key.startswith("collection.end_page"):
            current[keys[-1]] = int(value) if value else None
        else:
            current[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 조회"""
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """설정 값 설정"""
        self._set_nested_value(self._config, key, value)
    
    def get_api_config(self) -> Dict[str, Any]:
        """API 설정 조회"""
        return self.get("api", {})
    
    def get_collection_config(self) -> Dict[str, Any]:
        """수집 설정 조회"""
        return self.get("collection", {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """데이터베이스 설정 조회"""
        return self.get("database", {})
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """벡터스토어 설정 조회"""
        return self.get("vector_store", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """로깅 설정 조회"""
        return self.get("logging", {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """모니터링 설정 조회"""
        return self.get("monitoring", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """성능 설정 조회"""
        return self.get("performance", {})
    
    def get_all_config(self) -> Dict[str, Any]:
        """전체 설정 반환"""
        return self._config.copy()
    
    def update_config(self, new_config: Dict[str, Any]):
        """설정 업데이트"""
        self._update_nested_dict(self._config, new_config)
    
    def _update_nested_dict(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """중첩된 딕셔너리 업데이트"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate_config(self) -> bool:
        """설정 유효성 검증"""
        try:
            # 필수 설정 확인
            required_keys = [
                "api.base_url",
                "api.oc_id",
                "database.db_path",
                "vector_store.model_name"
            ]
            
            for key in required_keys:
                if self.get(key) is None:
                    print(f"필수 설정이 누락되었습니다: {key}")
                    return False
            
            # 값 범위 검증
            if self.get("api.display_count", 0) <= 0 or self.get("api.display_count", 0) > 100:
                print("display_count는 1-100 범위여야 합니다.")
                return False
            
            if self.get("api.max_retries", 0) < 0:
                print("max_retries는 0 이상이어야 합니다.")
                return False
            
            if self.get("collection.start_page", 0) < 1:
                print("start_page는 1 이상이어야 합니다.")
                return False
            
            return True
            
        except Exception as e:
            print(f"설정 검증 중 오류: {e}")
            return False
    
    def save_config(self, file_path: Optional[str] = None):
        """설정 파일 저장"""
        import yaml
        
        save_path = file_path or self.config_file
        
        try:
            # 디렉토리 생성
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # YAML 파일로 저장
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            
            print(f"설정이 저장되었습니다: {save_path}")
            
        except Exception as e:
            print(f"설정 저장 실패: {e}")
    
    def print_config(self):
        """설정 출력"""
        import json
        print(json.dumps(self._config, ensure_ascii=False, indent=2))


# 설정 인스턴스 생성
config = LegalTermCollectionConfig()

# 편의 함수들
def get_config() -> LegalTermCollectionConfig:
    """설정 인스턴스 반환"""
    return config

def get_api_config() -> Dict[str, Any]:
    """API 설정 반환"""
    return config.get_api_config()

def get_collection_config() -> Dict[str, Any]:
    """수집 설정 반환"""
    return config.get_collection_config()

def get_database_config() -> Dict[str, Any]:
    """데이터베이스 설정 반환"""
    return config.get_database_config()

def get_vector_store_config() -> Dict[str, Any]:
    """벡터스토어 설정 반환"""
    return config.get_vector_store_config()

def get_logging_config() -> Dict[str, Any]:
    """로깅 설정 반환"""
    return config.get_logging_config()

def get_monitoring_config() -> Dict[str, Any]:
    """모니터링 설정 반환"""
    return config.get_monitoring_config()

def get_performance_config() -> Dict[str, Any]:
    """성능 설정 반환"""
    return config.get_performance_config()


if __name__ == "__main__":
    # 설정 검증 및 출력
    if config.validate_config():
        print("설정이 유효합니다.")
        config.print_config()
    else:
        print("설정에 오류가 있습니다.")
