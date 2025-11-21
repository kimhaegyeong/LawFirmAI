"""
스트리밍 캐시 단위 테스트
"""
import pytest
import time
from api.routers.chat import StreamCache, get_stream_cache


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
        
        # 만료 후 조회 (캐시 미스)
        result = cache.get(message, session_id)
        assert result is None
    
    def test_cache_max_size(self):
        """최대 크기 제한 테스트"""
        cache = StreamCache(max_size=2, ttl_seconds=60)
        
        # 2개 저장
        cache.set("메시지1", "답변1", {}, "session1")
        cache.set("메시지2", "답변2", {}, "session2")
        
        assert len(cache.cache) == 2
        
        # 3번째 저장 시 가장 오래된 항목 제거
        cache.set("메시지3", "답변3", {}, "session3")
        
        # 최대 크기 유지
        assert len(cache.cache) == 2
        
        # 첫 번째 메시지는 제거되었을 수 있음
        result1 = cache.get("메시지1", "session1")
        result2 = cache.get("메시지2", "session2")
        result3 = cache.get("메시지3", "session3")
        
        # 최소 2개는 유지되어야 함
        assert sum([r is not None for r in [result1, result2, result3]]) >= 2
    
    def test_cache_get_stream_cache_singleton(self):
        """get_stream_cache 싱글톤 패턴 테스트"""
        cache1 = get_stream_cache()
        cache2 = get_stream_cache()
        assert cache1 is cache2

