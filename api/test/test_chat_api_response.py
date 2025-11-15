# -*- coding: utf-8 -*-
"""
Chat API 응답 테스트 스크립트
related_questions, sources_detail 등이 제대로 포함되는지 확인
"""

import requests
import json
import sys
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

def test_chat_api_response():
    """Chat API 응답 테스트"""
    print("\n" + "=" * 80)
    print("Chat API 응답 테스트")
    print("=" * 80)
    
    # 테스트 요청 데이터
    test_data = {
        "message": "전세금 반환 보증에 대해 설명해주세요",
        "session_id": None,  # 자동 생성
        "enable_checkpoint": False
    }
    
    try:
        print(f"\n📤 요청 전송: POST {API_BASE_URL}/api/v1/chat")
        print(f"   메시지: {test_data['message']}")
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2분 타임아웃
        )
        
        print(f"\n📥 응답 상태 코드: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ 오류 발생: {response.text}")
            return False
        
        # 응답 파싱
        result = response.json()
        
        print("\n" + "=" * 80)
        print("응답 분석")
        print("=" * 80)
        
        # 필수 필드 확인
        required_fields = ["answer", "sources", "sources_detail", "confidence", "related_questions"]
        print("\n✅ 필수 필드 확인:")
        for field in required_fields:
            if field in result:
                value = result[field]
                if isinstance(value, list):
                    print(f"   - {field}: {len(value)}개")
                elif isinstance(value, dict):
                    print(f"   - {field}: {len(value)}개 키")
                else:
                    print(f"   - {field}: {type(value).__name__}")
            else:
                print(f"   ❌ {field}: 없음")
        
        # sources_detail 상세 분석
        sources_detail = result.get("sources_detail", [])
        if sources_detail:
            print(f"\n📋 Sources Detail 분석 ({len(sources_detail)}개):")
            for idx, detail in enumerate(sources_detail[:5], 1):
                print(f"\n   [{idx}] {detail.get('name', 'N/A')}")
                print(f"       - type: {detail.get('type', 'N/A')}")
                print(f"       - case_name: {detail.get('case_name', 'N/A')}")
                print(f"       - case_number: {detail.get('case_number', 'N/A')}")
                print(f"       - court: {detail.get('court', 'N/A')}")
                print(f"       - url: {detail.get('url', 'N/A')[:50]}..." if detail.get('url') else "       - url: N/A")
                metadata = detail.get('metadata', {})
                if metadata:
                    print(f"       - metadata.court: {metadata.get('court', 'N/A')}")
                    print(f"       - metadata.doc_id: {metadata.get('doc_id', 'N/A')}")
                    print(f"       - metadata.casenames: {metadata.get('casenames', 'N/A')}")
        else:
            print("\n⚠️  Sources Detail이 없습니다!")
        
        # related_questions 확인
        related_questions = result.get("related_questions", [])
        if related_questions:
            print(f"\n❓ Related Questions ({len(related_questions)}개):")
            for idx, question in enumerate(related_questions[:5], 1):
                print(f"   {idx}. {question}")
        else:
            print("\n⚠️  Related Questions가 없습니다!")
            # metadata에서 확인
            metadata = result.get("metadata", {})
            if isinstance(metadata, dict):
                metadata_related_questions = metadata.get("related_questions", [])
                if metadata_related_questions:
                    print(f"   (metadata.related_questions에 {len(metadata_related_questions)}개 발견)")
                    for idx, question in enumerate(metadata_related_questions[:3], 1):
                        print(f"   {idx}. {question}")
        
        # sources 확인
        sources = result.get("sources", [])
        if sources:
            print(f"\n📚 Sources ({len(sources)}개):")
            for idx, source in enumerate(sources[:5], 1):
                print(f"   {idx}. {source}")
        else:
            print("\n⚠️  Sources가 없습니다!")
        
        # sources와 sources_detail 개수 비교
        if sources and sources_detail:
            if len(sources) == len(sources_detail):
                print(f"\n✅ Sources와 Sources Detail 개수 일치: {len(sources)}개")
            else:
                print(f"\n⚠️  Sources와 Sources Detail 개수 불일치: sources={len(sources)}, sources_detail={len(sources_detail)}")
        
        # 답변 길이 확인
        answer = result.get("answer", "")
        if answer:
            print(f"\n📝 답변 길이: {len(answer)}자")
            print(f"   답변 미리보기: {answer[:100]}...")
        
        # 신뢰도 확인
        confidence = result.get("confidence", 0.0)
        print(f"\n🎯 신뢰도: {confidence:.2f}")
        
        print("\n" + "=" * 80)
        print("✅ API 응답 테스트 완료!")
        print("=" * 80)
        
        # JSON 파일로 저장
        output_file = "api_test_response.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 응답이 {output_file}에 저장되었습니다.")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ API 서버에 연결할 수 없습니다. {API_BASE_URL}에서 서버가 실행 중인지 확인하세요.")
        print("   서버 실행 명령: python -m uvicorn api.main:app --reload")
        return False
    except requests.exceptions.Timeout:
        print("\n❌ 요청 시간 초과 (120초)")
        return False
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_chat_api_response()
    sys.exit(0 if success else 1)

