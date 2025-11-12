# -*- coding: utf-8 -*-
"""
/chat/stream API SSE 스트리밍 테스트
"""
import asyncio
import json
import requests
import sys
import os
from typing import List, Dict, Any

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

BASE_URL = "http://localhost:8000/api/v1"
STREAM_ENDPOINT = f"{BASE_URL}/chat/stream"


def check_server_health():
    """서버 상태 확인"""
    try:
        health_url = f"{BASE_URL.replace('/api/v1', '')}/health"
        response = requests.get(health_url, timeout=2)
        return response.status_code == 200
    except:
        return False


def test_stream_api():
    """SSE 스트리밍 API 테스트"""
    print("=" * 80)
    print("SSE 스트리밍 API 테스트 시작")
    print("=" * 80)
    
    # 서버 상태 확인
    print("\n🔍 서버 상태 확인 중...")
    if not check_server_health():
        print("❌ 서버가 실행 중이지 않습니다.")
        print("\n서버를 시작하려면:")
        print("  1. 새 터미널에서: cd api")
        print("  2. python main.py 실행")
        print("\n또는 PowerShell에서:")
        print("  cd api; python main.py")
        return False
    
    print("✅ 서버가 실행 중입니다.\n")
    print(f"엔드포인트: {STREAM_ENDPOINT}")
    print(f"테스트 질문: '민법 제750조 손해배상에 대해 설명해주세요'\n")
    
    # 요청 데이터
    request_data = {
        "message": "민법 제750조 손해배상에 대해 설명해주세요",
        "session_id": "test-stream-session-001"
    }
    
    # 헤더 설정
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    chunk_count = 0
    stream_chunks = []
    progress_events = []
    final_events = []
    error_events = []
    full_content = ""
    
    try:
        print("📡 스트리밍 요청 전송 중...")
        print("-" * 80)
        
        # SSE 스트리밍 요청
        response = requests.post(
            STREAM_ENDPOINT,
            json=request_data,
            headers=headers,
            stream=True,
            timeout=60
        )
        
        # 응답 상태 확인
        if response.status_code != 200:
            print(f"❌ 오류: HTTP {response.status_code}")
            print(f"응답: {response.text}")
            return False
        
        print(f"✅ 연결 성공 (HTTP {response.status_code})")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        print("-" * 80)
        print("\n📥 스트리밍 데이터 수신 중...\n")
        
        # SSE 데이터 파싱
        buffer = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            
            # SSE 형식 파싱: "data: {...}"
            if line.startswith("data: "):
                data_str = line[6:]  # "data: " 제거
                try:
                    event_data = json.loads(data_str)
                    event_type = event_data.get("type", "unknown")
                    
                    chunk_count += 1
                    
                    if event_type == "progress":
                        progress_events.append(event_data)
                        print(f"📊 [{chunk_count}] Progress: {event_data.get('content', '')}")
                    
                    elif event_type == "stream":
                        content = event_data.get("content", "")
                        source = event_data.get("source", "unknown")
                        full_content += content
                        stream_chunks.append({
                            "chunk": chunk_count,
                            "content": content,
                            "source": source,
                            "length": len(content)
                        })
                        
                        # 처음 10개 청크만 상세 출력
                        if chunk_count <= 10:
                            print(f"📦 [{chunk_count}] Stream chunk (source: {source}): {content[:50]}{'...' if len(content) > 50 else ''}")
                        elif chunk_count == 11:
                            print("... (더 많은 청크 수신 중)")
                    
                    elif event_type == "final":
                        final_events.append(event_data)
                        print(f"\n✅ [{chunk_count}] Final event 수신")
                        if event_data.get("metadata"):
                            metadata = event_data.get("metadata", {})
                            print(f"   - Sources: {len(metadata.get('sources', []))}개")
                            print(f"   - Legal References: {len(metadata.get('legal_references', []))}개")
                    
                    elif event_type == "error":
                        error_events.append(event_data)
                        print(f"❌ [{chunk_count}] Error: {event_data.get('content', '')}")
                    
                    elif event_type == "done":
                        print(f"\n🏁 [{chunk_count}] Done event 수신")
                        break
                    
                    else:
                        print(f"❓ [{chunk_count}] Unknown event type: {event_type}")
                
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON 파싱 오류: {e}, 데이터: {data_str[:100]}")
            
            elif line.startswith("event: "):
                # SSE 이벤트 타입 (선택적)
                event_name = line[7:]
                print(f"📌 Event: {event_name}")
            
            elif line.strip() == "":
                # 빈 줄 (SSE 구분자)
                continue
            
            else:
                # 기타 데이터
                print(f"⚠️ 예상치 못한 형식: {line[:100]}")
        
        print("\n" + "=" * 80)
        print("📊 테스트 결과 요약")
        print("=" * 80)
        print(f"✅ 총 이벤트 수: {chunk_count}개")
        print(f"📦 Stream 청크: {len(stream_chunks)}개")
        print(f"📊 Progress 이벤트: {len(progress_events)}개")
        print(f"✅ Final 이벤트: {len(final_events)}개")
        print(f"❌ Error 이벤트: {len(error_events)}개")
        print(f"📝 전체 답변 길이: {len(full_content)}자")
        
        if stream_chunks:
            total_chunk_length = sum(c["length"] for c in stream_chunks)
            avg_chunk_length = total_chunk_length / len(stream_chunks) if stream_chunks else 0
            print(f"📏 평균 청크 크기: {avg_chunk_length:.1f}자")
            print(f"📏 최소 청크 크기: {min(c['length'] for c in stream_chunks)}자")
            print(f"📏 최대 청크 크기: {max(c['length'] for c in stream_chunks)}자")
            
            # 소스별 통계
            callback_chunks = [c for c in stream_chunks if c.get("source") == "callback"]
            event_chunks = [c for c in stream_chunks if c.get("source") != "callback"]
            print(f"📡 콜백 소스 청크: {len(callback_chunks)}개")
            print(f"📡 이벤트 소스 청크: {len(event_chunks)}개")
        
        print("\n" + "=" * 80)
        print("✅ 테스트 완료")
        print("=" * 80)
        
        # 성공 기준
        success = (
            chunk_count > 0 and
            len(stream_chunks) > 0 and
            len(full_content) > 0 and
            len(error_events) == 0
        )
        
        return success
        
    except requests.exceptions.ConnectionError:
        print("❌ 연결 실패: API 서버가 실행 중인지 확인하세요.")
        print("   실행 명령: cd api && python main.py")
        return False
    
    except requests.exceptions.Timeout:
        print("❌ 타임아웃: 스트리밍 응답이 너무 오래 걸립니다.")
        return False
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_stream_api()
    sys.exit(0 if success else 1)

