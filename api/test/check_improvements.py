# -*- coding: utf-8 -*-
"""
개선 사항 확인 스크립트
API 서버 로그를 분석하여 개선 사항이 적용되었는지 확인
"""
import requests
import json
import sys
import uuid
import time

def check_improvements():
    """개선 사항 확인"""
    url = "http://localhost:8000/api/v1/chat/stream"
    
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
    print("🔍 개선 사항 확인 테스트")
    print("=" * 80)
    print(f"\n질문: {data['message']}")
    print(f"세션 ID: {session_id}\n")
    
    improvements = {
        "콜백 기반 스트리밍": {"found": False, "count": 0},
        "retrieved_docs 복원": {"found": False, "count": 0},
        "sources 생성": {"found": False, "count": 0},
        "legal_references 생성": {"found": False, "count": 0},
        "query_type 유지": {"found": False, "value": None}
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, stream=True, timeout=120)
        
        if response.status_code != 200:
            print(f"❌ 오류: HTTP {response.status_code}")
            return False
        
        print("✅ 연결 성공\n")
        print("📥 스트리밍 데이터 수신 중...\n")
        
        callback_chunks = 0
        event_chunks = 0
        full_content = ""
        final_metadata = None
        
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    event = json.loads(data_str)
                    event_type = event.get("type", "")
                    
                    if event_type == "stream":
                        content = event.get("content", "")
                        source = event.get("source", "")
                        full_content += content
                        
                        if source == "callback":
                            callback_chunks += 1
                            improvements["콜백 기반 스트리밍"]["found"] = True
                            improvements["콜백 기반 스트리밍"]["count"] = callback_chunks
                        else:
                            event_chunks += 1
                    
                    elif event_type == "final":
                        final_metadata = event.get("metadata", {})
                        sources = final_metadata.get("sources", [])
                        legal_references = final_metadata.get("legal_references", [])
                        
                        if sources and len(sources) > 0:
                            improvements["sources 생성"]["found"] = True
                            improvements["sources 생성"]["count"] = len(sources)
                        
                        if legal_references and len(legal_references) > 0:
                            improvements["legal_references 생성"]["found"] = True
                            improvements["legal_references 생성"]["count"] = len(legal_references)
                    
                    elif event_type == "done":
                        break
                except json.JSONDecodeError:
                    pass
        
        print("\n" + "=" * 80)
        print("📊 개선 사항 확인 결과")
        print("=" * 80)
        
        all_passed = True
        for improvement_name, status in improvements.items():
            if status["found"]:
                if "count" in status:
                    print(f"✅ {improvement_name}: 성공 (개수: {status['count']})")
                elif "value" in status:
                    print(f"✅ {improvement_name}: 성공 (값: {status['value']})")
                else:
                    print(f"✅ {improvement_name}: 성공")
            else:
                print(f"❌ {improvement_name}: 실패")
                all_passed = False
        
        print("\n" + "=" * 80)
        print("📈 상세 통계")
        print("=" * 80)
        print(f"콜백 청크: {callback_chunks}개")
        print(f"이벤트 청크: {event_chunks}개")
        print(f"전체 답변 길이: {len(full_content)}자")
        if final_metadata:
            print(f"Sources: {len(final_metadata.get('sources', []))}개")
            print(f"Legal References: {len(final_metadata.get('legal_references', []))}개")
        
        print("\n" + "=" * 80)
        if all_passed:
            print("✅ 모든 개선 사항이 적용되었습니다!")
        else:
            print("⚠️ 일부 개선 사항이 아직 적용되지 않았습니다.")
        print("=" * 80)
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_improvements()
    sys.exit(0 if success else 1)

