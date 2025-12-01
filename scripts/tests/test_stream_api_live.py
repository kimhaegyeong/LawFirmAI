"""
Stream API 실시간 테스트 및 로그 분석
"""
import sys
import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

BASE_URL = "http://localhost:8000/api/v1"
STREAM_ENDPOINT = f"{BASE_URL}/chat/stream"

def test_stream_api():
    """Stream API 테스트 및 로그 수집"""
    print("=" * 80)
    print("Stream API 테스트 시작")
    print("=" * 80)
    
    # 테스트 요청 데이터 (session_id는 None 또는 UUID 형식만 허용)
    import uuid
    request_data = {
        "message": "민법 제750조 손해배상에 대해 간단히 설명해주세요",
        "session_id": None  # None을 사용하거나 UUID 형식 사용: str(uuid.uuid4())
    }
    
    print(f"\n요청 URL: {STREAM_ENDPOINT}")
    print(f"요청 데이터: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
    print("\n" + "-" * 80)
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    events: List[Dict[str, Any]] = []
    event_types: Dict[str, int] = {}
    errors: List[str] = []
    warnings: List[str] = []
    start_time = time.time()
    
    try:
        print("스트리밍 시작...\n")
        response = requests.post(
            STREAM_ENDPOINT,
            json=request_data,
            headers=headers,
            stream=True,
            timeout=120
        )
        
        print(f"응답 상태 코드: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        print("-" * 80 + "\n")
        
        if response.status_code != 200:
            print(f"❌ 오류: HTTP {response.status_code}")
            print(f"응답 내용: {response.text[:500]}")
            return
        
        # SSE 이벤트 파싱
        buffer = ""
        event_count = 0
        chunk_count = 0
        
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                if buffer:
                    # 빈 줄은 이벤트 구분자
                    try:
                        event_data = json.loads(buffer)
                        events.append(event_data)
                        event_type = event_data.get("type", "unknown")
                        event_types[event_type] = event_types.get(event_type, 0) + 1
                        event_count += 1
                        
                        # 이벤트 타입별 로깅
                        if event_type == "stream":
                            chunk_count += 1
                            if chunk_count <= 5:  # 처음 5개만 상세 로깅
                                content = event_data.get("content", "")
                                print(f"[{event_count}] {event_type}: {content[:50]}...")
                        elif event_type in ["sources", "done", "error"]:
                            print(f"[{event_count}] {event_type}: {json.dumps(event_data, ensure_ascii=False)[:100]}...")
                        elif event_count <= 10:  # 처음 10개 이벤트만 로깅
                            print(f"[{event_count}] {event_type}")
                        
                        buffer = ""
                    except json.JSONDecodeError as e:
                        errors.append(f"JSON 파싱 오류: {buffer[:100]}... - {e}")
                        buffer = ""
                continue
            
            if line.startswith("data: "):
                buffer = line[6:]  # "data: " 제거
            elif line.startswith("event: "):
                # 이벤트 타입 (선택적)
                pass
            elif line.startswith(":"):
                # 주석 라인 (무시)
                pass
            else:
                # 연속된 데이터 라인
                if buffer:
                    buffer += "\n" + line
                else:
                    buffer = line
        
        # 마지막 버퍼 처리
        if buffer:
            try:
                event_data = json.loads(buffer)
                events.append(event_data)
                event_type = event_data.get("type", "unknown")
                event_types[event_type] = event_types.get(event_type, 0) + 1
                event_count += 1
            except json.JSONDecodeError as e:
                errors.append(f"JSON 파싱 오류 (마지막): {buffer[:100]}... - {e}")
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("테스트 결과 요약")
        print("=" * 80)
        print(f"총 소요 시간: {elapsed_time:.2f}초")
        print(f"총 이벤트 수: {event_count}")
        print(f"스트림 청크 수: {chunk_count}")
        print(f"\n이벤트 타입별 통계:")
        for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {event_type}: {count}개")
        
        # 최종 답변 추출
        full_answer = ""
        for event in events:
            if event.get("type") == "stream":
                full_answer += event.get("content", "")
        
        print(f"\n최종 답변 길이: {len(full_answer)}자")
        if full_answer:
            print(f"답변 미리보기: {full_answer[:200]}...")
        
        # sources 이벤트 확인
        sources_events = [e for e in events if e.get("type") == "sources"]
        if sources_events:
            print(f"\n✅ Sources 이벤트 수신: {len(sources_events)}개")
            for i, sources_event in enumerate(sources_events, 1):
                metadata = sources_event.get("metadata", {})
                sources_by_type = metadata.get("sources_by_type", {})
                print(f"  Sources 이벤트 #{i}:")
                for source_type, items in sources_by_type.items():
                    if items:
                        print(f"    - {source_type}: {len(items)}개")
        else:
            warnings.append("Sources 이벤트가 수신되지 않았습니다")
        
        # done 이벤트 확인
        done_events = [e for e in events if e.get("type") == "done"]
        if done_events:
            print(f"\n✅ Done 이벤트 수신: {len(done_events)}개")
        else:
            warnings.append("Done 이벤트가 수신되지 않았습니다")
        
        # 에러 이벤트 확인
        error_events = [e for e in events if e.get("type") == "error"]
        if error_events:
            print(f"\n⚠️ Error 이벤트 수신: {len(error_events)}개")
            for error_event in error_events:
                errors.append(f"Error 이벤트: {json.dumps(error_event, ensure_ascii=False)}")
        
        # 경고 및 오류 출력
        if warnings:
            print(f"\n⚠️ 경고 ({len(warnings)}개):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        
        if errors:
            print(f"\n❌ 오류 ({len(errors)}개):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
        
        # 개선점 분석
        print("\n" + "=" * 80)
        print("개선점 분석")
        print("=" * 80)
        
        improvements = []
        
        # 1. 응답 시간 분석
        if elapsed_time > 30:
            improvements.append("응답 시간이 30초를 초과했습니다. 성능 최적화가 필요합니다.")
        elif elapsed_time > 10:
            improvements.append("응답 시간이 10초를 초과했습니다. 성능 모니터링을 권장합니다.")
        
        # 2. 이벤트 타입 분석
        if "stream" not in event_types:
            improvements.append("Stream 이벤트가 수신되지 않았습니다. 스트리밍이 정상 작동하지 않을 수 있습니다.")
        
        if "done" not in event_types:
            improvements.append("Done 이벤트가 수신되지 않았습니다. 스트림 종료가 명확하지 않을 수 있습니다.")
        
        if "sources" not in event_types:
            improvements.append("Sources 이벤트가 수신되지 않았습니다. 참고자료 정보가 전달되지 않을 수 있습니다.")
        
        # 3. 청크 수 분석
        if chunk_count == 0:
            improvements.append("스트림 청크가 수신되지 않았습니다. 스트리밍이 작동하지 않을 수 있습니다.")
        elif chunk_count < 5:
            improvements.append(f"스트림 청크 수가 매우 적습니다 ({chunk_count}개). 답변 생성이 제대로 되지 않았을 수 있습니다.")
        
        # 4. 에러 분석
        if error_events:
            improvements.append(f"에러 이벤트가 {len(error_events)}개 수신되었습니다. 에러 처리 로직을 확인해야 합니다.")
        
        # 5. JSON 파싱 오류
        if errors:
            json_errors = [e for e in errors if "JSON 파싱" in e]
            if json_errors:
                improvements.append(f"JSON 파싱 오류가 {len(json_errors)}개 발생했습니다. SSE 형식이 올바른지 확인해야 합니다.")
        
        # 6. 답변 길이 분석
        if not full_answer:
            improvements.append("최종 답변이 비어있습니다. 답변 생성 로직을 확인해야 합니다.")
        elif len(full_answer) < 50:
            improvements.append(f"최종 답변이 매우 짧습니다 ({len(full_answer)}자). 답변 생성이 완료되지 않았을 수 있습니다.")
        
        # 7. 이벤트 순서 분석
        event_sequence = [e.get("type") for e in events]
        if event_sequence and event_sequence[-1] != "done":
            improvements.append("마지막 이벤트가 'done'이 아닙니다. 스트림이 정상적으로 종료되지 않았을 수 있습니다.")
        
        # 8. Sources 이벤트 내용 분석
        if sources_events:
            for sources_event in sources_events:
                metadata = sources_event.get("metadata", {})
                sources_by_type = metadata.get("sources_by_type", {})
                total_sources = sum(len(items) for items in sources_by_type.values())
                if total_sources == 0:
                    improvements.append("Sources 이벤트는 수신되었지만 참고자료가 비어있습니다. sources_detail 추출 로직을 확인해야 합니다.")
        
        # 개선점 출력
        if improvements:
            print(f"\n총 {len(improvements)}개 개선점 발견:\n")
            for i, improvement in enumerate(improvements, 1):
                print(f"{i}. {improvement}")
        else:
            print("\n✅ 특별한 개선점이 발견되지 않았습니다.")
        
        # 상세 로그 저장
        log_file = project_root / "logs" / "test" / f"stream_api_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "request": request_data,
            "response": {
                "status_code": response.status_code,
                "elapsed_time": elapsed_time,
                "event_count": event_count,
                "chunk_count": chunk_count,
                "event_types": event_types,
                "full_answer_length": len(full_answer),
                "full_answer_preview": full_answer[:500] if full_answer else ""
            },
            "events": events[:100],  # 처음 100개만 저장 (너무 크면 잘림)
            "warnings": warnings,
            "errors": errors,
            "improvements": improvements
        }
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n상세 로그 저장: {log_file}")
        
    except requests.exceptions.ConnectionError:
        print("❌ 오류: API 서버에 연결할 수 없습니다.")
        print("   서버가 실행 중인지 확인하세요: http://localhost:8000")
        improvements = [
            "API 서버가 실행 중이지 않습니다. 서버를 시작한 후 테스트를 실행하세요."
        ]
    except requests.exceptions.Timeout:
        print("❌ 오류: 요청 시간 초과 (120초)")
        improvements = [
            "스트리밍 응답 시간이 120초를 초과했습니다. 타임아웃 설정을 늘리거나 성능을 최적화해야 합니다."
        ]
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        improvements = [
            f"예상치 못한 오류가 발생했습니다: {str(e)}"
        ]
    
    return improvements

if __name__ == "__main__":
    improvements = test_stream_api()
    
    if improvements:
        print("\n" + "=" * 80)
        print("개선점 요약 (번호순)")
        print("=" * 80)
        for i, improvement in enumerate(improvements, 1):
            print(f"{i}. {improvement}")

