# -*- coding: utf-8 -*-
"""
간단한 SSE 스트리밍 테스트
"""
import requests
import json
import sys
import uuid
import time

def check_server():
    """서버 상태 확인"""
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def test_stream():
    """간단한 스트리밍 테스트"""
    # 서버 상태 확인
    if not check_server():
        print("❌ 서버가 실행 중이지 않습니다.")
        print("\n서버를 시작하려면:")
        print("  cd api")
        print("  python main.py")
        return False
    
    url = "http://localhost:8000/api/v1/chat/stream"
    
    # UUID 형식의 세션 ID 생성
    session_id = str(uuid.uuid4())
    
    data = {
        "message": "민법 제750조 손해배상에 대해 설명해주세요",
        "session_id": session_id
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    print("=" * 80)
    print("📡 SSE 스트리밍 테스트")
    print("=" * 80)
    print(f"\nURL: {url}")
    print(f"질문: {data['message']}")
    print(f"세션 ID: {session_id}\n")
    print("-" * 80)
    
    try:
        print("⏳ 요청 전송 중... (타임아웃: 120초)")
        response = requests.post(url, json=data, headers=headers, stream=True, timeout=120)
        
        if response.status_code != 200:
            print(f"❌ 오류: HTTP {response.status_code}")
            print(response.text)
            return False
        
        print(f"✅ 연결 성공 (HTTP {response.status_code})")
        print(f"📋 Content-Type: {response.headers.get('Content-Type')}\n")
        print("📥 스트리밍 데이터 수신 중...\n")
        
        chunk_count = 0
        stream_count = 0
        callback_count = 0
        event_count = 0
        full_content = ""
        start_time = time.time()
        last_chunk_time = start_time
        
        for line in response.iter_lines(decode_unicode=True):
            current_time = time.time()
            
            # 타임아웃 경고 (5초 이상 청크가 없으면)
            if current_time - last_chunk_time > 5 and chunk_count > 0:
                print(f"\n⚠️ {current_time - last_chunk_time:.1f}초 동안 청크가 수신되지 않았습니다...")
            
            if not line:
                continue
            
            if line.startswith("data: "):
                chunk_count += 1
                last_chunk_time = current_time
                data_str = line[6:]
                
                try:
                    event = json.loads(data_str)
                    event_type = event.get("type", "")
                    
                    if event_type == "stream":
                        stream_count += 1
                        content = event.get("content", "")
                        source = event.get("source", "")
                        full_content += content
                        
                        if source == "callback":
                            callback_count += 1
                            marker = "📡"
                        else:
                            event_count += 1
                            marker = "📦"
                        
                        # 처음 30개 청크만 출력
                        if stream_count <= 30:
                            print(f"{marker}[{stream_count}] {content}", end="", flush=True)
                        elif stream_count == 31:
                            print("\n... (더 많은 청크 수신 중) ...", end="", flush=True)
                    
                    elif event_type == "progress":
                        print(f"\n📊 Progress: {event.get('content', '')}")
                    
                    elif event_type == "final":
                        print(f"\n\n✅ Final event 수신")
                        if event.get("metadata"):
                            meta = event["metadata"]
                            print(f"   📚 Sources: {len(meta.get('sources', []))}개")
                            print(f"   ⚖️ Legal References: {len(meta.get('legal_references', []))}개")
                    
                    elif event_type == "done":
                        print(f"\n🏁 Done")
                        break
                    
                    elif event_type == "error":
                        print(f"\n❌ Error: {event.get('content', '')}")
                
                except json.JSONDecodeError as e:
                    print(f"\n⚠️ JSON 파싱 오류: {e}, 데이터: {data_str[:100]}")
                    pass
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("📊 테스트 결과")
        print("=" * 80)
        print(f"⏱️ 총 소요 시간: {total_time:.2f}초")
        print(f"✅ 총 이벤트: {chunk_count}개")
        print(f"📦 Stream 청크: {stream_count}개")
        print(f"   - 콜백 소스: {callback_count}개")
        print(f"   - 이벤트 소스: {event_count}개")
        print(f"📝 전체 답변 길이: {len(full_content)}자")
        
        if stream_count > 0:
            avg_time_per_chunk = total_time / stream_count
            print(f"⚡ 평균 청크 수신 간격: {avg_time_per_chunk:.3f}초")
        
        # 성공 여부 판단
        success = chunk_count > 0 and stream_count > 0 and len(full_content) > 0
        
        if success:
            print("\n✅ 테스트 성공: SSE 스트리밍이 정상적으로 작동합니다!")
            if callback_count > 0:
                print("   ✅ 콜백 기반 스트리밍이 작동 중입니다.")
            if event_count > 0:
                print("   ✅ 이벤트 기반 스트리밍이 작동 중입니다.")
        else:
            print("\n⚠️ 테스트 실패: 청크가 수신되지 않았습니다.")
            if chunk_count == 0:
                print("   - 이벤트가 전혀 수신되지 않았습니다.")
            elif stream_count == 0:
                print("   - Stream 타입 이벤트가 수신되지 않았습니다.")
        
        return success
        
    except requests.exceptions.Timeout:
        print("❌ 타임아웃: 스트리밍 응답이 너무 오래 걸립니다.")
        print("   서버 로그를 확인하세요.")
        return False
    
    except requests.exceptions.ConnectionError:
        print("❌ 연결 실패: API 서버를 먼저 시작하세요.")
        print("   cd api && python main.py")
        return False
    
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_stream()
    sys.exit(0 if success else 1)

