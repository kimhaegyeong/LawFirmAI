"""
청킹 전략 모듈

Strategy Pattern을 사용한 확장 가능한 청킹 시스템
"""
from .base_chunker import ChunkingStrategy, ChunkResult
from .factory import ChunkingFactory
from .config import ChunkingConfig, get_chunking_config

__all__ = [
    "ChunkingStrategy",
    "ChunkResult",
    "ChunkingFactory",
    "ChunkingConfig",
    "get_chunking_config"
]

