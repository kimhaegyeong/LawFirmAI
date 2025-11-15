"""
청킹 설정 관리

질문 유형별 청킹 설정 및 전략 선택 로직
"""
from typing import Dict, Any, Optional
from enum import Enum
import os
import json
import yaml
from pathlib import Path


class QueryType(Enum):
    """질문 유형"""
    LAW_INQUIRY = "law_inquiry"
    PRECEDENT_SEARCH = "precedent_search"
    LEGAL_ADVICE = "legal_advice"
    PROCEDURE_GUIDE = "procedure_guide"
    TERM_EXPLANATION = "term_explanation"
    GENERAL = "general"


class ChunkingConfig:
    """청킹 설정 클래스"""
    
    # 기본 청킹 설정
    DEFAULT_CONFIG = {
        "min_chars": 400,
        "max_chars": 1800,
        "overlap_ratio": 0.25
    }
    
    # 질문 유형별 최적 청킹 사이즈 설정
    QUERY_TYPE_CHUNK_CONFIG = {
        QueryType.LAW_INQUIRY.value: {
            "min_chars": 300,
            "max_chars": 1000,
            "overlap_ratio": 0.15,
            "description": "법령 조문 질문: 작은 청크로 정확한 매칭"
        },
        QueryType.PRECEDENT_SEARCH.value: {
            "min_chars": 500,
            "max_chars": 2000,
            "overlap_ratio": 0.3,
            "description": "판례 검색: 큰 청크로 문맥 보존"
        },
        QueryType.LEGAL_ADVICE.value: {
            "min_chars": 400,
            "max_chars": 1500,
            "overlap_ratio": 0.25,
            "description": "법률 상담: 균형잡힌 청크"
        },
        QueryType.PROCEDURE_GUIDE.value: {
            "min_chars": 400,
            "max_chars": 1500,
            "overlap_ratio": 0.25,
            "description": "절차 안내: 균형잡힌 청크"
        },
        QueryType.TERM_EXPLANATION.value: {
            "min_chars": 300,
            "max_chars": 1200,
            "overlap_ratio": 0.2,
            "description": "용어 설명: 작은 청크로 정확한 매칭"
        },
        QueryType.GENERAL.value: DEFAULT_CONFIG
    }
    
    # 하이브리드 청킹 크기 카테고리 설정
    HYBRID_SIZE_CATEGORIES = {
        "small": {
            "min_chars": 400,
            "max_chars": 800,
            "overlap_ratio": 0.2
        },
        "medium": {
            "min_chars": 800,
            "max_chars": 1500,
            "overlap_ratio": 0.25
        },
        "large": {
            "min_chars": 1500,
            "max_chars": 2500,
            "overlap_ratio": 0.3
        }
    }
    
    # 문서 타입별 기본 청킹 설정
    DOCUMENT_TYPE_CONFIG = {
        "statute_article": {
            "min_chars": 300,
            "max_chars": 1000,
            "overlap_ratio": 0.15
        },
        "case_paragraph": {
            "min_chars": 500,
            "max_chars": 2000,
            "overlap_ratio": 0.3
        },
        "decision_paragraph": {
            "min_chars": 500,
            "max_chars": 2000,
            "overlap_ratio": 0.3
        },
        "interpretation_paragraph": {
            "min_chars": 500,
            "max_chars": 2000,
            "overlap_ratio": 0.3
        }
    }
    
    @classmethod
    def get_config_for_query_type(cls, query_type: Optional[str]) -> Dict[str, Any]:
        """
        질문 유형에 따른 청킹 설정 반환
        
        Args:
            query_type: 질문 유형 문자열
        
        Returns:
            청킹 설정 딕셔너리
        """
        if query_type and query_type in cls.QUERY_TYPE_CHUNK_CONFIG:
            return cls.QUERY_TYPE_CHUNK_CONFIG[query_type]
        return cls.DEFAULT_CONFIG
    
    @classmethod
    def get_config_for_document_type(cls, document_type: str) -> Dict[str, Any]:
        """
        문서 타입에 따른 청킹 설정 반환
        
        Args:
            document_type: 문서 타입 (statute_article, case_paragraph, etc.)
        
        Returns:
            청킹 설정 딕셔너리
        """
        return cls.DOCUMENT_TYPE_CONFIG.get(document_type, cls.DEFAULT_CONFIG)
    
    @classmethod
    def get_hybrid_config(cls, size_category: str) -> Dict[str, Any]:
        """
        하이브리드 청킹 크기 카테고리 설정 반환
        
        Args:
            size_category: 'small', 'medium', 'large'
        
        Returns:
            청킹 설정 딕셔너리
        """
        return cls.HYBRID_SIZE_CATEGORIES.get(size_category, cls.DEFAULT_CONFIG)
    
    @classmethod
    def load_from_file(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        설정 파일에서 청킹 설정 로드
        
        Args:
            config_path: 설정 파일 경로 (None이면 기본 경로 사용)
        
        Returns:
            설정 딕셔너리
        """
        if config_path is None:
            # 기본 설정 파일 경로
            base_dir = Path(__file__).parent.parent.parent.parent
            config_path = base_dir / "config" / "chunking_config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                    return yaml.safe_load(f) or {}
                elif config_path.suffix == '.json':
                    return json.load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load chunking config from {config_path}: {e}")
            return {}
        
        return {}


def get_chunking_config(
    query_type: Optional[str] = None,
    document_type: Optional[str] = None,
    use_config_file: bool = False
) -> Dict[str, Any]:
    """
    청킹 설정 가져오기
    
    Args:
        query_type: 질문 유형 (동적 청킹 시 사용)
        document_type: 문서 타입
        use_config_file: 설정 파일 사용 여부
    
    Returns:
        청킹 설정 딕셔너리
    """
    config = ChunkingConfig()
    
    # 설정 파일에서 로드
    if use_config_file:
        file_config = config.load_from_file()
        if file_config:
            # 설정 파일의 내용을 우선 사용
            strategies_config = file_config.get("chunking_strategies", {})
            if query_type and "dynamic" in strategies_config:
                query_configs = strategies_config["dynamic"].get("query_type_configs", {})
                if query_type in query_configs:
                    return query_configs[query_type]
    
    # 질문 유형별 설정
    if query_type:
        return config.get_config_for_query_type(query_type)
    
    # 문서 타입별 설정
    if document_type:
        return config.get_config_for_document_type(document_type)
    
    # 기본 설정
    return config.DEFAULT_CONFIG

