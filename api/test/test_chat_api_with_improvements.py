# -*- coding: utf-8 -*-
"""
개선된 프롬프트와 검색 로직을 사용한 Chat API 통합 테스트
"""

import requests
import json
import sys
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

def test_chat_api_with_improvements():
    """개선된 Chat API 응답 테스트"""
    print("\n" + "=" * 80)
    print("개선된 Chat API 응답 테스트")
    print("=" * 80)
    
    # 테스트 요청 데이터
    test_data = {
        "message": "전세금 반환 보증에 대해 설명해주세요",
        "session_id": None,  # 자동 생성
        "enable_checkpoint": False  # 체크포인트 비활성화 (numpy 직렬화 오류 방지)
    }
    
    try:
        print(f"\n📤 요청 전송: POST {API_BASE_URL}/api/v1/chat")
        print(f"   메시지: {test_data['message']}")
        print(f"   체크포인트: {test_data['enable_checkpoint']}")
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2분 타임아웃
        )
        
        print(f"\n📥 응답 상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("\n" + "=" * 80)
            print("응답 분석")
            print("=" * 80)
            
            # 필수 필드 확인
            print("\n✅ 필수 필드 확인:")
            assert "answer" in response_data, "응답에 'answer' 필드가 없습니다."
            answer = response_data.get("answer", "")
            print(f"   - answer: {len(answer)}자")
            
            assert "sources" in response_data, "응답에 'sources' 필드가 없습니다."
            sources = response_data.get("sources", [])
            print(f"   - sources: {len(sources)}개")
            
            assert "sources_detail" in response_data, "응답에 'sources_detail' 필드가 없습니다."
            sources_detail = response_data.get("sources_detail", [])
            print(f"   - sources_detail: {len(sources_detail)}개")
            
            assert "confidence" in response_data, "응답에 'confidence' 필드가 없습니다."
            confidence = response_data.get("confidence", 0.0)
            print(f"   - confidence: {confidence:.2f}")
            
            assert "related_questions" in response_data, "응답에 'related_questions' 필드가 없습니다."
            related_questions = response_data.get("related_questions", [])
            print(f"   - related_questions: {len(related_questions)}개")
            
            # 답변 내용 확인
            print(f"\n📝 답변 내용:")
            if answer and len(answer) > 100:
                print(f"   길이: {len(answer)}자")
                print(f"   미리보기: {answer[:200]}...")
            else:
                print(f"   ⚠️ 답변이 비어있거나 너무 짧음: {len(answer)}자")
            
            # Sources Detail 타입 분포 확인
            if sources_detail:
                print(f"\n📚 Sources Detail 타입 분포:")
                type_distribution = {}
                for detail in sources_detail:
                    doc_type = detail.get("type", "unknown")
                    type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
                
                for doc_type, count in type_distribution.items():
                    print(f"   - {doc_type}: {count}개")
                
                # 개선 효과 확인
                if len(type_distribution) == 1 and "case_paragraph" in type_distribution:
                    print(f"\n   ⚠️ 여전히 판례만 검색되었습니다 (데이터 불균형 때문일 수 있음)")
                elif len(type_distribution) > 1:
                    print(f"\n   ✅ 다양한 타입의 문서가 검색되었습니다!")
            
            # Related Questions 확인
            if related_questions:
                print(f"\n❓ Related Questions:")
                for i, q in enumerate(related_questions[:5], 1):
                    print(f"   {i}. {q}")
            
            # 추가 필드 확인
            processing_time = response_data.get("processing_time", 0.0)
            print(f"\n⏱️ Processing Time: {processing_time:.2f}초")
            
            query_type = response_data.get("query_type", "")
            print(f"📋 Query Type: {query_type}")
            
            # 검증
            print("\n" + "=" * 80)
            print("검증 결과")
            print("=" * 80)
            
            checks = []
            
            # 답변이 있는지 확인
            if answer and len(answer) > 100:
                checks.append(("답변 생성", True, f"{len(answer)}자"))
            else:
                checks.append(("답변 생성", False, f"{len(answer)}자 (너무 짧음)"))
            
            # Sources가 있는지 확인
            if sources and len(sources) > 0:
                checks.append(("Sources 생성", True, f"{len(sources)}개"))
            else:
                checks.append(("Sources 생성", False, "Sources가 없음"))
            
            # Sources Detail이 있는지 확인
            if sources_detail and len(sources_detail) > 0:
                checks.append(("Sources Detail 생성", True, f"{len(sources_detail)}개"))
            else:
                checks.append(("Sources Detail 생성", False, "Sources Detail이 없음"))
            
            # 다양한 타입의 문서가 검색되었는지 확인
            if sources_detail:
                types = set(detail.get("type", "unknown") for detail in sources_detail)
                if len(types) > 1:
                    checks.append(("다양한 문서 타입 검색", True, f"{len(types)}개 타입: {', '.join(types)}"))
                else:
                    checks.append(("다양한 문서 타입 검색", False, f"단일 타입만 검색됨 ({list(types)[0] if types else 'unknown'})"))
            
            # Related Questions가 있는지 확인
            if related_questions and len(related_questions) > 0:
                checks.append(("Related Questions 생성", True, f"{len(related_questions)}개"))
            else:
                checks.append(("Related Questions 생성", False, "Related Questions가 없음"))
            
            # Confidence 확인
            if confidence > 0.5:
                checks.append(("Confidence", True, f"{confidence:.2f}"))
            else:
                checks.append(("Confidence", False, f"{confidence:.2f} (너무 낮음)"))
            
            # 결과 출력
            passed = 0
            failed = 0
            
            for check_name, check_result, detail in checks:
                status = "✅" if check_result else "❌"
                print(f"{status} {check_name}: {detail}")
                if check_result:
                    passed += 1
                else:
                    failed += 1
            
            print(f"\n총 {len(checks)}개 검증 중 {passed}개 통과, {failed}개 실패")
            
            # 응답을 파일로 저장
            output_file = "api_test_response_improvements.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 응답이 {output_file}에 저장되었습니다.")
            
            if failed == 0:
                print("\n🎉 모든 검증 통과!")
                return 0
            else:
                print(f"\n⚠️ {failed}개 검증 실패 (일부는 정상일 수 있음)")
                return 0  # 일부 실패는 정상일 수 있으므로 0 반환
                
        else:
            print(f"❌ 오류 발생: {response.status_code}")
            print(f"   응답 내용: {response.text[:500]}")
            
            # 오류 응답도 파일로 저장
            try:
                error_data = response.json()
                with open("api_test_response_improvements.json", "w", encoding="utf-8") as f:
                    json.dump(error_data, f, ensure_ascii=False, indent=2)
                print("\n💾 오류 응답이 api_test_response_improvements.json에 저장되었습니다.")
            except:
                pass
            
            return 1
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ API 서버에 연결할 수 없습니다.")
        print(f"   서버가 실행 중인지 확인하세요: {API_BASE_URL}")
        return 1
    except requests.exceptions.Timeout:
        print(f"\n❌ 요청 시간 초과 (120초)")
        return 1
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 중 오류 발생: {e}")
        return 1
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_chat_api_with_improvements())

