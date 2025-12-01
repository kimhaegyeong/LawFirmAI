# -*- coding: utf-8 -*-
"""
Stream API Type 정보 보존 테스트 스크립트

Usage:
    python test_stream_api_type_check.py "질의 내용"
"""

import sys
import os
import requests
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

BASE_URL = "http://localhost:8000/api/v1"
STREAM_ENDPOINT = f"{BASE_URL}/chat/stream"

def check_server_health() -> bool:
    """서버가 실행 중인지 확인"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def test_stream_api_type_preservation(query: str) -> Dict[str, Any]:
    """Stream API에서 type 정보 보존 테스트"""
    print("=" * 80)
    print("Stream API Type 정보 보존 테스트")
    print("=" * 80)
    print(f"질의: {query}")
    print()
    
    # 서버 상태 확인
    print("1. 서버 상태 확인 중...")
    if not check_server_health():
        print("   ❌ 서버가 실행 중이지 않습니다.")
        print("   서버를 시작한 후 다시 시도하세요:")
        print("   cd api && python main.py")
        return {"success": False, "error": "서버가 실행 중이지 않습니다"}
    
    print("   ✅ 서버가 실행 중입니다.")
    print()
    
    # 세션 생성 (선택적)
    print("2. 세션 생성 중...")
    session_id = None
    try:
        import uuid
        session_response = requests.post(
            f"{BASE_URL}/sessions",
            json={"title": "Test Session"},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if session_response.status_code == 200:
            session_data = session_response.json()
            session_id = session_data.get("session_id")
            print(f"   ✅ 세션 생성 완료: {session_id}")
        else:
            # 세션 생성 실패 시 None으로 보내서 자동 생성되도록
            print(f"   ⚠️  세션 생성 실패 (상태 코드: {session_response.status_code}), session_id=None으로 전송")
    except Exception as e:
        print(f"   ⚠️  세션 생성 오류: {e}, session_id=None으로 전송")
    print()
    
    # Stream API 호출
    print("3. Stream API 호출 중...")
    print("   (이 작업은 몇 초에서 몇 분이 걸릴 수 있습니다)")
    print()
    
    request_data = {
        "message": query
    }
    if session_id:
        request_data["session_id"] = session_id
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    # API 키가 환경 변수에 있으면 추가
    api_key = os.getenv("API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    
    type_info_found = []
    type_unknown_count = 0
    events_received = []
    
    try:
        response = requests.post(
            STREAM_ENDPOINT,
            json=request_data,
            headers=headers,
            stream=True,
            timeout=180
        )
        
        if response.status_code != 200:
            print(f"   ❌ 응답 상태 코드: {response.status_code}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
        
        print(f"   ✅ 응답 수신 시작 (Content-Type: {response.headers.get('Content-Type', 'N/A')})")
        print()
        
        # SSE 데이터 파싱
        buffer = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            
            if line.startswith("data: "):
                data_str = line[6:]  # "data: " 제거
                try:
                    event_data = json.loads(data_str)
                    events_received.append(event_data)
                    
                    event_type = event_data.get("type", "")
                    
                    # 디버깅: 모든 이벤트 타입 출력
                    if event_type:
                        print(f"   📨 이벤트 수신: type={event_type}")
                    
                    # sources 이벤트에서 type 정보 확인
                    if event_type == "sources":
                        print(f"   ✅ Sources 이벤트 수신!")
                        metadata = event_data.get("metadata", {})
                        sources_by_type = metadata.get("sources_by_type", {})
                        
                        if sources_by_type:
                            print(f"   📊 sources_by_type 발견: {list(sources_by_type.keys())}")
                            
                            # 각 타입별로 문서 확인
                            for source_type_key, sources_list in sources_by_type.items():
                                if isinstance(sources_list, list) and len(sources_list) > 0:
                                    print(f"      - {source_type_key}: {len(sources_list)}개")
                                    for i, source in enumerate(sources_list):
                                        if isinstance(source, dict):
                                            # type 정보 추출 (여러 위치에서 확인)
                                            source_type = (
                                                source.get("type") or 
                                                source.get("source_type") or
                                                (source.get("metadata", {}).get("type") if isinstance(source.get("metadata"), dict) else None) or
                                                (source.get("metadata", {}).get("source_type") if isinstance(source.get("metadata"), dict) else None)
                                            )
                                            
                                            if source_type:
                                                type_info_found.append({
                                                    "index": i,
                                                    "type": source_type,
                                                    "source": source.get("name") or source.get("title") or source.get("case_name") or "N/A",
                                                    "source_type_key": source_type_key
                                                })
                                                if source_type.lower() == "unknown":
                                                    type_unknown_count += 1
                                            else:
                                                print(f"         ⚠️  문서 {i}: type 정보 없음, keys={list(source.keys())[:10]}")
                                elif isinstance(sources_list, list):
                                    print(f"      - {source_type_key}: 0개 (빈 리스트)")
                    
                    # done 이벤트 확인
                    if event_type == "done":
                        print("   ✅ 완료 이벤트 수신")
                        break
                        
                except json.JSONDecodeError:
                    pass
        
        print()
        print("4. Type 정보 분석:")
        print("=" * 80)
        
        if type_info_found:
            print(f"   발견된 type 정보: {len(type_info_found)}개")
            type_stats = {}
            for info in type_info_found:
                doc_type = info["type"]
                type_stats[doc_type] = type_stats.get(doc_type, 0) + 1
                print(f"   - [{doc_type}] {info['source'][:50]}...")
            
            print()
            print("   📊 Type 통계:")
            for doc_type, count in sorted(type_stats.items(), key=lambda x: x[1], reverse=True):
                type_names = {
                    "statute_article": "법령",
                    "precedent_content": "판례",
                    "unknown": "알 수 없음"
                }
                doc_type_display = type_names.get(doc_type, doc_type)
                print(f"      - {doc_type_display}: {count}개")
            
            if type_unknown_count > 0:
                print()
                print(f"   ⚠️  type=unknown인 문서가 {type_unknown_count}개 발견되었습니다!")
                return {
                    "success": False,
                    "type_unknown_count": type_unknown_count,
                    "type_info": type_info_found
                }
            else:
                print()
                print("   ✅ 모든 문서의 type 정보가 정상적으로 설정되었습니다!")
        else:
            print("   ⚠️  type 정보를 찾을 수 없습니다.")
            print("   (sources 이벤트가 없거나 type 필드가 없을 수 있습니다)")
        
        print()
        print("5. 결과 요약:")
        print("=" * 80)
        print(f"   - 수신된 이벤트 수: {len(events_received)}")
        print(f"   - type 정보 발견: {len(type_info_found)}개")
        print(f"   - type=unknown: {type_unknown_count}개")
        
        return {
            "success": type_unknown_count == 0,
            "events_count": len(events_received),
            "type_info_count": len(type_info_found),
            "type_unknown_count": type_unknown_count,
            "type_info": type_info_found
        }
        
    except requests.exceptions.Timeout:
        print("   ❌ 요청 시간 초과")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"   ❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def main():
    """메인 실행 함수"""
    # 질의 가져오기
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "전세금 반환 보증에 대해 알려주세요"
    
    if not query:
        print("질의를 입력해주세요.")
        print("\n사용법:")
        print("  python test_stream_api_type_check.py \"질의 내용\"")
        return 1
    
    # 테스트 실행
    result = test_stream_api_type_preservation(query)
    
    print()
    print("=" * 80)
    if result.get("success"):
        print("테스트 완료: ✅ 통과")
    else:
        print("테스트 완료: ❌ 실패")
        if result.get("error"):
            print(f"   오류: {result.get('error')}")
        if result.get("type_unknown_count", 0) > 0:
            print(f"   type=unknown 문서: {result.get('type_unknown_count')}개")
    print("=" * 80)
    
    return 0 if result.get("success") else 1

if __name__ == "__main__":
    sys.exit(main())

