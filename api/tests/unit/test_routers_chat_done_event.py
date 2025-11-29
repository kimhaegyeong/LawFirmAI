# -*- coding: utf-8 -*-
"""
Done 이벤트 전송 테스트
_generate_stream_response와 stream_with_quota_management에서 done 이벤트가 제대로 전송되는지 확인
"""

import sys
import os
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'api'))

from api.routers.chat import _generate_stream_response
from api.utils.sse_formatter import format_sse_event


async def test_done_event_in_generate_stream_response():
    """_generate_stream_response에서 done 이벤트가 전송되는지 테스트"""
    print("\n=== _generate_stream_response Done 이벤트 테스트 ===")
    
    # Mock ChatService 생성
    mock_chat_service = MagicMock()
    
    # stream_final_answer가 done 이벤트를 보내지 않는 경우 시뮬레이션
    async def mock_stream_final_answer(message, session_id):
        # stream 이벤트만 보내고 done 이벤트는 보내지 않음
        stream_event = {
            "type": "stream",
            "content": "테스트 답변",
            "timestamp": "2024-01-01T00:00:00"
        }
        yield format_sse_event(stream_event)
        
        # final 이벤트
        final_event = {
            "type": "final",
            "content": "테스트 답변",
            "metadata": {},
            "timestamp": "2024-01-01T00:00:00"
        }
        yield format_sse_event(final_event)
        # done 이벤트는 보내지 않음
    
    mock_chat_service.stream_final_answer = mock_stream_final_answer
    
    # 세션 서비스 모킹
    with patch('api.routers.chat.session_service') as mock_session_service:
        mock_session_service.add_message = MagicMock(return_value="test-message-id")
        mock_session_service.get_session = MagicMock(return_value={"user_id": None})
        
        # 캐시 모킹
        with patch('api.routers.chat.get_stream_cache') as mock_get_cache:
            mock_get_cache.return_value = None
            
            # _maybe_generate_session_title 모킹
            with patch('api.routers.chat._maybe_generate_session_title'):
                events = []
                done_received = False
                
                try:
                    async for chunk in _generate_stream_response(
                        chat_service=mock_chat_service,
                        message="테스트 메시지",
                        session_id="test-session"
                    ):
                        if chunk:
                            # SSE 이벤트 파싱
                            if chunk.startswith("data: "):
                                json_str = chunk[6:].strip()
                                try:
                                    event_data = json.loads(json_str)
                                    event_type = event_data.get("type", "")
                                    events.append(event_type)
                                    
                                    if event_type == "done":
                                        done_received = True
                                        print(f"✅ Done 이벤트 수신: {json.dumps(event_data, ensure_ascii=False)[:100]}")
                                except json.JSONDecodeError:
                                    pass
                except Exception as e:
                    print(f"❌ 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
                
                print(f"📊 수신된 이벤트 타입: {events}")
                
                if done_received:
                    print("✅ Done 이벤트가 정상적으로 전송되었습니다.")
                    return True
                else:
                    print("❌ Done 이벤트가 전송되지 않았습니다.")
                    return False


async def test_done_event_when_stream_final_answer_sends_done():
    """stream_final_answer가 이미 done 이벤트를 보낸 경우 중복 전송하지 않는지 테스트"""
    print("\n=== Done 이벤트 중복 방지 테스트 ===")
    
    mock_chat_service = MagicMock()
    
    # stream_final_answer가 done 이벤트를 보내는 경우 시뮬레이션
    async def mock_stream_final_answer_with_done(message, session_id):
        stream_event = {
            "type": "stream",
            "content": "테스트 답변",
            "timestamp": "2024-01-01T00:00:00"
        }
        yield format_sse_event(stream_event)
        
        # done 이벤트를 보냄
        done_event = {
            "type": "done",
            "content": "테스트 답변",
            "metadata": {},
            "timestamp": "2024-01-01T00:00:00"
        }
        yield format_sse_event(done_event)
    
    mock_chat_service.stream_final_answer = mock_stream_final_answer_with_done
    
    with patch('api.routers.chat.session_service') as mock_session_service:
        mock_session_service.add_message = MagicMock(return_value="test-message-id")
        mock_session_service.get_session = MagicMock(return_value={"user_id": None})
        
        with patch('api.routers.chat.get_stream_cache') as mock_get_cache:
            mock_get_cache.return_value = None
            
            with patch('api.routers.chat._maybe_generate_session_title'):
                events = []
                done_count = 0
                
                try:
                    async for chunk in _generate_stream_response(
                        chat_service=mock_chat_service,
                        message="테스트 메시지",
                        session_id="test-session"
                    ):
                        if chunk:
                            if chunk.startswith("data: "):
                                json_str = chunk[6:].strip()
                                try:
                                    event_data = json.loads(json_str)
                                    event_type = event_data.get("type", "")
                                    events.append(event_type)
                                    
                                    if event_type == "done":
                                        done_count += 1
                                except json.JSONDecodeError:
                                    pass
                except Exception as e:
                    print(f"❌ 오류 발생: {e}")
                    return False
                
                print(f"📊 수신된 이벤트 타입: {events}")
                print(f"📊 Done 이벤트 수: {done_count}")
                
                # done 이벤트가 1개만 있어야 함 (중복 방지)
                if done_count == 1:
                    print("✅ Done 이벤트가 중복되지 않고 정상적으로 전송되었습니다.")
                    return True
                elif done_count > 1:
                    print(f"❌ Done 이벤트가 {done_count}번 중복 전송되었습니다.")
                    return False
                else:
                    print("❌ Done 이벤트가 전송되지 않았습니다.")
                    return False


async def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("Done 이벤트 전송 테스트 시작")
    print("=" * 60)
    
    results = []
    
    # 테스트 1: done 이벤트가 전송되는지 확인
    print("\n[테스트 1] Done 이벤트 전송 확인")
    try:
        result1 = await test_done_event_in_generate_stream_response()
        results.append(("Done 이벤트 전송", result1))
    except Exception as e:
        print(f"❌ 테스트 1 실패: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Done 이벤트 전송", False))
    
    # 테스트 2: done 이벤트 중복 방지 확인
    print("\n[테스트 2] Done 이벤트 중복 방지 확인")
    try:
        result2 = await test_done_event_when_stream_final_answer_sends_done()
        results.append(("Done 이벤트 중복 방지", result2))
    except Exception as e:
        print(f"❌ 테스트 2 실패: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Done 이벤트 중복 방지", False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"{test_name}: {status}")
    
    print(f"\n총 {total_count}개 테스트 중 {passed_count}개 통과")
    
    if passed_count == total_count:
        print("🎉 모든 테스트 통과!")
        return 0
    else:
        print("⚠️ 일부 테스트 실패")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

