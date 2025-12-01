# -*- coding: utf-8 -*-
"""
스트리밍 완료 테스트 스크립트
ERR_INCOMPLETE_CHUNKED_ENCODING 오류가 발생하지 않는지 확인
"""

import sys
import os
import asyncio
import json
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

import requests
from api.services.chat_service import get_chat_service


def test_stream_complete():
    """스트리밍이 완전히 종료되는지 테스트"""
    print("\n=== 스트리밍 완료 테스트 ===")
    
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    stream_url = f"{base_url}/api/v1/chat/stream"
    
    # 테스트 요청
    test_request = {
        "message": "계약 해지 사유에 대해 알려주세요",
        "session_id": None  # 새 세션 생성
    }
    
    print(f"요청 URL: {stream_url}")
    print(f"요청 메시지: {test_request['message']}")
    
    events_received = []
    done_event_received = False
    error_occurred = False
    incomplete_error = False
    
    try:
        response = requests.post(
            stream_url,
            json=test_request,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            },
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ HTTP 오류: {response.status_code}")
            print(f"응답: {response.text}")
            return False
        
        print(f"✅ 응답 수신 시작 (Status: {response.status_code})")
        
        # SSE 스트림 읽기
        buffer = ""
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                buffer += chunk
                
                # SSE 이벤트 파싱 (줄 단위로 처리)
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    
                    if event.startswith("data: "):
                        json_str = event[6:].strip()
                        try:
                            event_data = json.loads(json_str)
                            event_type = event_data.get("type", "")
                            events_received.append(event_type)
                            
                            if event_type == "done":
                                done_event_received = True
                                print(f"✅ Done 이벤트 수신: {json.dumps(event_data, ensure_ascii=False)[:100]}")
                            elif event_type == "error":
                                error_occurred = True
                                print(f"⚠️ Error 이벤트 수신: {event_data.get('content', '')[:100]}")
                            elif event_type == "stream":
                                content = event_data.get("content", "")
                                if len(events_received) <= 3:  # 처음 몇 개만 출력
                                    print(f"📝 Stream 이벤트: {content[:50]}...")
                        except json.JSONDecodeError as e:
                            print(f"⚠️ JSON 파싱 오류: {e}, event: {event[:100]}")
        
        # 버퍼에 남은 데이터 처리
        if buffer.strip():
            if buffer.startswith("data: "):
                json_str = buffer[6:].strip()
                try:
                    event_data = json.loads(json_str)
                    event_type = event_data.get("type", "")
                    events_received.append(event_type)
                    if event_type == "done":
                        done_event_received = True
                except json.JSONDecodeError:
                    pass
        
        print(f"\n📊 수신된 이벤트 타입: {events_received}")
        print(f"📊 총 이벤트 수: {len(events_received)}")
        
        # 결과 확인
        if done_event_received:
            print("✅ Done 이벤트가 정상적으로 수신되었습니다.")
            return True
        else:
            print("❌ Done 이벤트가 수신되지 않았습니다.")
            print(f"수신된 이벤트: {set(events_received)}")
            return False
            
    except requests.exceptions.ChunkedEncodingError as e:
        incomplete_error = True
        print(f"❌ ERR_INCOMPLETE_CHUNKED_ENCODING 오류 발생: {e}")
        print(f"수신된 이벤트: {set(events_received)}")
        if done_event_received:
            print("⚠️ Done 이벤트는 수신되었지만 스트림이 완전히 종료되지 않았습니다.")
        return False
    except Exception as e:
        error_occurred = True
        print(f"❌ 오류 발생: {type(e).__name__}: {e}")
        if "incomplete" in str(e).lower() or "chunked" in str(e).lower():
            incomplete_error = True
            print("⚠️ ERR_INCOMPLETE_CHUNKED_ENCODING 관련 오류로 보입니다.")
        return False


def test_stream_with_service():
    """ChatService를 직접 사용하여 스트리밍 테스트"""
    print("\n=== ChatService 직접 스트리밍 테스트 ===")
    
    try:
        chat_service = get_chat_service()
        if not chat_service:
            print("❌ ChatService를 가져올 수 없습니다.")
            return False
        
        events_received = []
        done_event_received = False
        
        async def test_async():
            nonlocal events_received, done_event_received
            
            try:
                async for chunk in chat_service.stream_final_answer(
                    message="계약 해지 사유에 대해 알려주세요",
                    session_id=None
                ):
                    if chunk:
                        # SSE 형식 파싱
                        if chunk.startswith("data: "):
                            json_str = chunk[6:].strip()
                            try:
                                event_data = json.loads(json_str)
                                event_type = event_data.get("type", "")
                                events_received.append(event_type)
                                
                                if event_type == "done":
                                    done_event_received = True
                                    print(f"✅ Done 이벤트 수신")
                                elif event_type == "stream" and len(events_received) <= 3:
                                    content = event_data.get("content", "")
                                    print(f"📝 Stream 이벤트: {content[:50]}...")
                            except json.JSONDecodeError:
                                pass
                
                print(f"📊 수신된 이벤트 타입: {events_received}")
                print(f"📊 총 이벤트 수: {len(events_received)}")
                
                if done_event_received:
                    print("✅ Done 이벤트가 정상적으로 수신되었습니다.")
                    return True
                else:
                    print("❌ Done 이벤트가 수신되지 않았습니다.")
                    return False
                    
            except Exception as e:
                print(f"❌ 오류 발생: {type(e).__name__}: {e}")
                return False
        
        result = asyncio.run(test_async())
        return result
        
    except Exception as e:
        print(f"❌ 테스트 실패: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("스트리밍 완료 테스트 시작")
    print("=" * 60)
    
    results = []
    
    # 테스트 1: HTTP API를 통한 스트리밍 테스트
    print("\n[테스트 1] HTTP API 스트리밍 테스트")
    try:
        result1 = test_stream_complete()
        results.append(("HTTP API 스트리밍", result1))
    except Exception as e:
        print(f"❌ 테스트 1 실패: {e}")
        results.append(("HTTP API 스트리밍", False))
    
    # 테스트 2: ChatService 직접 테스트
    print("\n[테스트 2] ChatService 직접 스트리밍 테스트")
    try:
        result2 = test_stream_with_service()
        results.append(("ChatService 직접 스트리밍", result2))
    except Exception as e:
        print(f"❌ 테스트 2 실패: {e}")
        results.append(("ChatService 직접 스트리밍", False))
    
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
    sys.exit(main())

