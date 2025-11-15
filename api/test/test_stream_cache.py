"""
스트리밍 캐시 단위 테스트
"""
import pytest
import time
import sys
from pathlib import Path
from unittest.mock import patch

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.routers.chat import StreamCache, get_stream_cache
from api.config import get_api_config


class TestStreamCache:
    """StreamCache 클래스 단위 테스트"""
    
    def test_cache_initialization(self):
        """캐시 초기화 테스트"""
        cache = StreamCache(max_size=10, ttl_seconds=60)
        assert cache.max_size == 10
        assert cache.ttl_seconds == 60
        assert len(cache.cache) == 0
    
    def test_cache_set_and_get(self):
        """캐시 저장 및 조회 테스트"""
        cache = StreamCache(max_size=10, ttl_seconds=60)
        
        message = "테스트 질문"
        content = "테스트 답변"
        metadata = {"sources": ["소스1", "소스2"]}
        session_id = "test_session_1"
        
        # 캐시 저장
        cache.set(message, content, metadata, session_id)
        
        # 캐시 조회
        result = cache.get(message, session_id)
        
        assert result is not None
        assert result["content"] == content
        assert result["metadata"] == metadata
    
    def test_cache_key_generation(self):
        """캐시 키 생성 테스트 (메시지만 기준)"""
        cache = StreamCache()
        
        # 같은 메시지는 세션과 무관하게 같은 키
        key1 = cache._generate_key("테스트", "session1")
        key2 = cache._generate_key("테스트", "session1")
        assert key1 == key2
        
        # 다른 세션이어도 같은 메시지면 같은 키
        key3 = cache._generate_key("테스트", "session2")
        assert key1 == key3
        
        # 다른 메시지는 다른 키
        key4 = cache._generate_key("다른 테스트", "session1")
        assert key1 != key4
    
    def test_cache_ttl_expiration(self):
        """TTL 만료 테스트"""
        cache = StreamCache(max_size=10, ttl_seconds=1)  # 1초 TTL
        
        message = "TTL 테스트"
        content = "TTL 답변"
        session_id = "test_session"
        
        # 캐시 저장
        cache.set(message, content, {}, session_id)
        
        # 즉시 조회 (캐시 히트)
        result = cache.get(message, session_id)
        assert result is not None
        
        # 1.5초 대기
        time.sleep(1.5)
        
        # TTL 만료 후 조회 (캐시 미스)
        result = cache.get(message, session_id)
        assert result is None
    
    def test_cache_lru_eviction(self):
        """LRU 캐시 제거 테스트"""
        cache = StreamCache(max_size=3, ttl_seconds=3600)
        
        # 캐시 크기만큼 저장
        for i in range(3):
            cache.set(f"메시지{i}", f"답변{i}", {}, f"session{i}")
        
        assert len(cache.cache) == 3
        
        # 캐시 크기 초과 저장 (가장 오래된 항목 제거)
        cache.set("메시지3", "답변3", {}, "session3")
        
        assert len(cache.cache) == 3
        
        # 첫 번째 메시지는 제거됨
        result = cache.get("메시지0", "session0")
        assert result is None
        
        # 나머지는 유지됨
        result = cache.get("메시지1", "session1")
        assert result is not None
    
    def test_cache_lru_reordering(self):
        """LRU 재정렬 테스트"""
        cache = StreamCache(max_size=3, ttl_seconds=3600)
        
        # 캐시에 항목 저장
        cache.set("메시지0", "답변0", {}, "session0")
        cache.set("메시지1", "답변1", {}, "session1")
        cache.set("메시지2", "답변2", {}, "session2")
        
        # 첫 번째 항목 조회 (맨 뒤로 이동)
        cache.get("메시지0", "session0")
        
        # 새 항목 추가 (가장 오래된 항목인 "메시지1" 제거)
        cache.set("메시지3", "답변3", {}, "session3")
        
        # "메시지0"은 유지 (최근 조회됨)
        result = cache.get("메시지0", "session0")
        assert result is not None
        
        # "메시지1"은 제거됨
        result = cache.get("메시지1", "session1")
        assert result is None
    
    def test_cache_different_sessions(self):
        """다른 세션의 같은 메시지 테스트 (메시지만 기준)"""
        cache = StreamCache()
        
        message = "같은 질문"
        content1 = "세션1 답변"
        content2 = "세션2 답변"
        
        # 세션1에 저장
        cache.set(message, content1, {}, "session1")
        result1 = cache.get(message, "session1")
        assert result1["content"] == content1
        
        # 세션2에 다른 내용 저장 (같은 메시지이므로 덮어쓰기)
        cache.set(message, content2, {}, "session2")
        
        # 세션1 조회 (메시지만 기준이므로 마지막 저장된 내용 반환)
        result1_again = cache.get(message, "session1")
        assert result1_again["content"] == content2
        
        # 세션2 조회
        result2 = cache.get(message, "session2")
        assert result2["content"] == content2
    
    def test_cache_clear(self):
        """캐시 전체 삭제 테스트"""
        cache = StreamCache()
        
        # 여러 항목 저장
        for i in range(5):
            cache.set(f"메시지{i}", f"답변{i}", {}, f"session{i}")
        
        assert len(cache.cache) == 5
        
        # 캐시 삭제
        cache.clear()
        
        assert len(cache.cache) == 0
    
    def test_cache_none_session_id(self):
        """세션 ID가 None인 경우 테스트"""
        cache = StreamCache()
        
        message = "세션 없는 메시지"
        content = "세션 없는 답변"
        
        # 세션 ID 없이 저장
        cache.set(message, content, {}, None)
        
        # 세션 ID 없이 조회
        result = cache.get(message, None)
        
        assert result is not None
        assert result["content"] == content
    
    def test_cache_metadata_handling(self):
        """메타데이터 처리 테스트"""
        cache = StreamCache()
        
        message = "메타데이터 테스트"
        content = "답변"
        metadata = {
            "sources": ["소스1"],
            "legal_references": ["참조1"],
            "sources_detail": [{"name": "소스1", "type": "case"}]
        }
        
        cache.set(message, content, metadata, "session1")
        result = cache.get(message, "session1")
        
        assert result["metadata"] == metadata
        assert result["metadata"]["sources"] == ["소스1"]
        assert len(result["metadata"]["sources_detail"]) == 1
    
    def test_cache_empty_metadata(self):
        """빈 메타데이터 처리 테스트"""
        cache = StreamCache()
        
        message = "빈 메타데이터 테스트"
        content = "답변"
        
        cache.set(message, content, None, "session1")
        result = cache.get(message, "session1")
        
        assert result["metadata"] == {}


class TestGetStreamCache:
    """get_stream_cache 함수 테스트"""
    
    def test_get_stream_cache_disabled(self):
        """캐시 비활성화 테스트"""
        with patch('api.routers.chat.get_api_config') as mock_config:
            mock_config.return_value.enable_stream_cache = False
            
            # 전역 인스턴스 초기화
            import api.routers.chat as chat_module
            chat_module._stream_cache_instance = None
            
            cache = get_stream_cache()
            assert cache is None
    
    def test_get_stream_cache_enabled(self):
        """캐시 활성화 테스트"""
        with patch('api.routers.chat.get_api_config') as mock_config:
            config = mock_config.return_value
            config.enable_stream_cache = True
            config.stream_cache_max_size = 50
            config.stream_cache_ttl_seconds = 1800
            
            # 전역 인스턴스 초기화
            import api.routers.chat as chat_module
            chat_module._stream_cache_instance = None
            
            cache = get_stream_cache()
            
            assert cache is not None
            assert isinstance(cache, StreamCache)
            assert cache.max_size == 50
            assert cache.ttl_seconds == 1800
    
    def test_get_stream_cache_singleton(self):
        """캐시 싱글톤 패턴 테스트"""
        with patch('api.routers.chat.get_api_config') as mock_config:
            config = mock_config.return_value
            config.enable_stream_cache = True
            config.stream_cache_max_size = 100
            config.stream_cache_ttl_seconds = 3600
            
            # 전역 인스턴스 초기화
            import api.routers.chat as chat_module
            chat_module._stream_cache_instance = None
            
            # 첫 번째 호출
            cache1 = get_stream_cache()
            
            # 두 번째 호출 (같은 인스턴스 반환)
            cache2 = get_stream_cache()
            
            assert cache1 is cache2
            assert cache1 is chat_module._stream_cache_instance

