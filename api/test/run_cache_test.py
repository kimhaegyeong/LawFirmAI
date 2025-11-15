"""
캐시 테스트 직접 실행 스크립트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.routers.chat import StreamCache, get_stream_cache
import time


def test_cache_basic():
    """기본 캐시 테스트"""
    print("=" * 80)
    print("캐시 기본 테스트 시작")
    print("=" * 80)
    
    # 테스트 1: 캐시 초기화
    print("\n[테스트 1] 캐시 초기화")
    cache = StreamCache(max_size=10, ttl_seconds=60)
    assert cache.max_size == 10
    assert cache.ttl_seconds == 60
    assert len(cache.cache) == 0
    print("✅ 캐시 초기화 성공")
    
    # 테스트 2: 캐시 저장 및 조회
    print("\n[테스트 2] 캐시 저장 및 조회")
    message = "테스트 질문"
    content = "테스트 답변"
    metadata = {"sources": ["소스1", "소스2"]}
    session_id = "test_session_1"
    
    cache.set(message, content, metadata, session_id)
    result = cache.get(message, session_id)
    
    assert result is not None
    assert result["content"] == content
    assert result["metadata"] == metadata
    print(f"✅ 캐시 저장 및 조회 성공: {result['content']}")
    
    # 테스트 3: 캐시 키 생성 (메시지만 기준)
    print("\n[테스트 3] 캐시 키 생성 (메시지만 기준)")
    key1 = cache._generate_key("테스트", "session1")
    key2 = cache._generate_key("테스트", "session1")
    key3 = cache._generate_key("테스트", "session2")
    key4 = cache._generate_key("다른 테스트", "session1")
    
    # 같은 메시지는 세션과 무관하게 같은 키
    assert key1 == key2
    assert key1 == key3  # 세션이 달라도 같은 메시지면 같은 키
    # 다른 메시지는 다른 키
    assert key1 != key4
    print(f"✅ 캐시 키 생성 성공: 같은 메시지(다른 세션)={key1 == key3}, 다른 메시지={key1 != key4}")
    
    # 테스트 4: TTL 만료
    print("\n[테스트 4] TTL 만료 테스트")
    cache_ttl = StreamCache(max_size=10, ttl_seconds=1)
    cache_ttl.set("TTL 테스트", "TTL 답변", {}, "test_session")
    
    result = cache_ttl.get("TTL 테스트", "test_session")
    assert result is not None
    print("✅ TTL 만료 전 조회 성공")
    
    time.sleep(1.5)
    result = cache_ttl.get("TTL 테스트", "test_session")
    assert result is None
    print("✅ TTL 만료 후 조회 실패 (예상됨)")
    
    # 테스트 5: LRU 제거
    print("\n[테스트 5] LRU 캐시 제거")
    cache_lru = StreamCache(max_size=3, ttl_seconds=3600)
    
    for i in range(3):
        cache_lru.set(f"메시지{i}", f"답변{i}", {}, f"session{i}")
    
    assert len(cache_lru.cache) == 3
    print(f"✅ 캐시에 3개 항목 저장: {len(cache_lru.cache)}")
    
    cache_lru.set("메시지3", "답변3", {}, "session3")
    assert len(cache_lru.cache) == 3
    
    result = cache_lru.get("메시지0", "session0")
    assert result is None
    print("✅ LRU 제거 성공: 첫 번째 항목 제거됨")
    
    result = cache_lru.get("메시지1", "session1")
    assert result is not None
    print("✅ 나머지 항목 유지 확인")
    
    # 테스트 6: 다른 세션 (메시지만 기준이므로 같은 캐시 사용)
    print("\n[테스트 6] 다른 세션 테스트 (메시지만 기준)")
    cache_session = StreamCache()
    message = "같은 질문"
    content1 = "세션1 답변"
    content2 = "세션2 답변"
    
    # session1에 저장
    cache_session.set(message, content1, {}, "session1")
    result1 = cache_session.get(message, "session1")
    assert result1["content"] == content1
    print(f"✅ session1 저장 및 조회 성공: {result1['content']}")
    
    # session2에 다른 내용 저장 (같은 메시지이므로 덮어쓰기)
    cache_session.set(message, content2, {}, "session2")
    result2 = cache_session.get(message, "session2")
    # 메시지만 기준이므로 마지막에 저장한 내용이 반환됨
    assert result2["content"] == content2
    print(f"✅ session2 저장 후 조회 성공: {result2['content']}")
    
    # session1으로 다시 조회해도 같은 내용 (메시지만 기준)
    result1_again = cache_session.get(message, "session1")
    assert result1_again["content"] == content2  # 마지막 저장된 내용
    print(f"✅ 메시지만 기준 캐싱 확인: session1 조회={result1_again['content']}")
    
    # 테스트 7: 캐시 클리어
    print("\n[테스트 7] 캐시 클리어")
    cache_clear = StreamCache()
    for i in range(5):
        cache_clear.set(f"메시지{i}", f"답변{i}", {}, f"session{i}")
    
    assert len(cache_clear.cache) == 5
    cache_clear.clear()
    assert len(cache_clear.cache) == 0
    print("✅ 캐시 클리어 성공")
    
    print("\n" + "=" * 80)
    print("모든 테스트 통과! ✅")
    print("=" * 80)
    return True


def test_cache_integration():
    """캐시 통합 테스트 (get_stream_cache)"""
    print("\n" + "=" * 80)
    print("캐시 통합 테스트 시작")
    print("=" * 80)
    
    # 전역 인스턴스 초기화
    import api.routers.chat as chat_module
    chat_module._stream_cache_instance = None
    
    # 캐시 비활성화 테스트
    print("\n[테스트 1] 캐시 비활성화")
    from unittest.mock import patch
    with patch('api.routers.chat.get_api_config') as mock_config:
        mock_config.return_value.enable_stream_cache = False
        cache = get_stream_cache()
        assert cache is None
        print("✅ 캐시 비활성화 확인")
    
    # 캐시 활성화 테스트
    print("\n[테스트 2] 캐시 활성화")
    chat_module._stream_cache_instance = None
    with patch('api.routers.chat.get_api_config') as mock_config:
        config = mock_config.return_value
        config.enable_stream_cache = True
        config.stream_cache_max_size = 50
        config.stream_cache_ttl_seconds = 1800
        
        cache = get_stream_cache()
        assert cache is not None
        assert isinstance(cache, StreamCache)
        assert cache.max_size == 50
        assert cache.ttl_seconds == 1800
        print("✅ 캐시 활성화 및 설정 확인")
    
    # 싱글톤 패턴 테스트
    print("\n[테스트 3] 싱글톤 패턴")
    chat_module._stream_cache_instance = None
    with patch('api.routers.chat.get_api_config') as mock_config:
        config = mock_config.return_value
        config.enable_stream_cache = True
        config.stream_cache_max_size = 100
        config.stream_cache_ttl_seconds = 3600
        
        cache1 = get_stream_cache()
        cache2 = get_stream_cache()
        
        assert cache1 is cache2
        assert cache1 is chat_module._stream_cache_instance
        print("✅ 싱글톤 패턴 확인")
    
    print("\n" + "=" * 80)
    print("통합 테스트 통과! ✅")
    print("=" * 80)
    return True


if __name__ == "__main__":
    try:
        test_cache_basic()
        test_cache_integration()
        print("\n🎉 모든 테스트 성공!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

