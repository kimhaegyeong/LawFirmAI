"""
스트리밍 API 테스트 스크립트
"""
import asyncio
import aiohttp
import json
import sys
from datetime import datetime

async def test_stream_api():
    """스트리밍 API 테스트"""
    url = "http://localhost:8000/api/v1/chat/stream"
    
    # 테스트 요청 데이터
    payload = {
        "message": "임대차 분쟁 시 해결 방법은?",
        "session_id": "435a63d5-0adb-475b-8ddf-8b4dde98b51d"
    }
    
    print(f"[{datetime.now()}] 스트리밍 API 테스트 시작")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    print("-" * 80)
    
    chunk_count = 0
    total_length = 0
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                print(f"Status: {response.status}")
                print(f"Headers: {dict(response.headers)}")
                print("-" * 80)
                
                if response.status != 200:
                    text = await response.text()
                    print(f"Error: {text}")
                    return
                
                # SSE 스트리밍 읽기
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if line_str.startswith('data: '):
                        content = line_str[6:]  # "data: " 제거
                        
                        # 완료 신호 체크
                        if content in ['[스트리밍 완료]', '[완료]']:
                            print(f"\n[{datetime.now()}] 스트리밍 완료 신호 수신")
                            break
                        
                        # 진행 상황 메시지 필터링
                        if '[답변 생성 중...]' in content:
                            print(f"[{datetime.now()}] 진행 상황 메시지 (필터링됨): {content}")
                            continue
                        
                        chunk_count += 1
                        total_length += len(content)
                        
                        # 청크 출력 (처음 10개만 상세 출력)
                        if chunk_count <= 10:
                            print(f"[{datetime.now()}] Chunk #{chunk_count}: {repr(content)}")
                        elif chunk_count == 11:
                            print(f"[{datetime.now()}] ... (더 많은 청크 수신 중)")
                        
                        # 실제 내용 출력 (처음 5개 청크)
                        if chunk_count <= 5:
                            print(f"  Content: {content}")
                        
                        # 전체 내용 수집 (최종 출력용)
                        if 'full_content' not in locals():
                            full_content = ""
                        full_content += content
                
                print("-" * 80)
                print(f"[{datetime.now()}] 테스트 완료")
                print(f"총 청크 수: {chunk_count}")
                print(f"총 길이: {total_length}자")
                print("-" * 80)
                print("전체 응답 내용:")
                if 'full_content' in locals():
                    print(full_content)
                    print(f"\n총 {len(full_content)}자")
                print("-" * 80)
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_stream_api())

