#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatService 테스트 스크립트
"""

import sys
import os
import asyncio
from datetime import datetime

# Google Cloud 관련 경고 완전 억제 (가장 먼저 실행)
os.environ['GRPC_DNS_RESOLVER'] = 'native'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['GOOGLE_CLOUD_PROJECT'] = ''
os.environ['GCLOUD_PROJECT'] = ''
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
os.environ['GOOGLE_CLOUD_DISABLE_GRPC'] = 'true'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

# 로깅 억제 (더 포괄적)
import logging
logging.getLogger('grpc').setLevel(logging.CRITICAL)
logging.getLogger('google').setLevel(logging.CRITICAL)
logging.getLogger('google.auth').setLevel(logging.CRITICAL)
logging.getLogger('google.auth.transport').setLevel(logging.CRITICAL)
logging.getLogger('google.auth.transport.grpc').setLevel(logging.CRITICAL)
logging.getLogger('google.auth.transport.requests').setLevel(logging.CRITICAL)
logging.getLogger('google.cloud').setLevel(logging.CRITICAL)
logging.getLogger('google.api_core').setLevel(logging.CRITICAL)

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.utils.config import Config
from source.services.enhanced_chat_service import EnhancedChatService

def categorize_question(question: str) -> str:
    """질문을 카테고리별로 분류 (개선된 버전)"""
    question_lower = question.lower()
    
    # 우선순위 기반 분류 (더 구체적인 키워드부터)
    
    # 형사법 관련 (우선순위 높음)
    if any(word in question_lower for word in ['형법상', '형법', '살인죄', '절도죄', '사기죄', '강도죄', '강간죄', '형사', 'criminal', '구속', '보석', '피고인']):
        return "criminal"
    
    # 상속법 관련 (우선순위 높음)
    elif any(word in question_lower for word in ['상속법', '유류분', '상속세', '상속재산', '상속포기', '상속인']):
        return "inheritance"
    
    # 계약 관련
    elif any(word in question_lower for word in ['계약', 'contract']):
        return "contract"
    
    # 부동산 관련
    elif any(word in question_lower for word in ['부동산', 'real estate', '매매', '등기']):
        return "real_estate"
    
    # 가족법 관련 (상속 제외)
    elif any(word in question_lower for word in ['이혼', '가족', '양육', '유언', '입양']) and '상속' not in question_lower:
        return "family_law"
    
    # 민사법 관련
    elif any(word in question_lower for word in ['민사', 'civil', '손해배상', '소송', '강제집행', '가압류']):
        return "civil"
    
    # 노동법 관련
    elif any(word in question_lower for word in ['근로', '노동', 'labor', '해고', '임금', '산업재해']):
        return "labor"
    
    # 개인정보보호법 관련
    elif any(word in question_lower for word in ['개인정보', 'privacy', '정보보호']):
        return "privacy"
    
    # 지적재산권 관련
    elif any(word in question_lower for word in ['저작권', 'copyright', '지적재산권', '특허', '상표', '디자인']):
        return "ip"
    
    # 행정법 관련
    elif any(word in question_lower for word in ['행정', 'administrative', '행정심판', '행정소송']):
        return "administrative"
    
    # 국제법 관련
    elif any(word in question_lower for word in ['국제', 'international', '국제사법', '국제중재']):
        return "international"
    
    # 법인 관련
    elif any(word in question_lower for word in ['법인', 'corporate', '주식회사', '법인세']):
        return "corporate"
    
    # 환경법 관련
    elif any(word in question_lower for word in ['환경', 'environmental', '환경영향평가']):
        return "environmental"
    
    # 금융법 관련
    elif any(word in question_lower for word in ['금융', 'financial', '자본시장', '금융투자']):
        return "financial"
    
    # 세법 관련
    elif any(word in question_lower for word in ['세법', 'tax', '소득세', '부가가치세', '세무']):
        return "tax"
    
    # 의료법 관련
    elif any(word in question_lower for word in ['의료', 'medical', '의료사고', '의료법']):
        return "medical"
    
    # 교육법 관련
    elif any(word in question_lower for word in ['교육', 'education', '교원', '학생', '학교폭력']):
        return "education"
    
    # 건설법 관련
    elif any(word in question_lower for word in ['건설', 'construction', '건축', '건설사고']):
        return "construction"
    
    # 교통법 관련
    elif any(word in question_lower for word in ['교통', 'traffic', '교통사고', '면허']):
        return "traffic"
    
    else:
        return "other"

async def test_chat_service():
    """ChatService 테스트"""
    print("=" * 60)
    print("EnhancedChatService 테스트 시작")
    print("=" * 60)
    
    try:
        # 설정 로드
        config = Config()
        
        # EnhancedChatService 초기화
        print("EnhancedChatService 초기화 중...")
        chat_service = EnhancedChatService(config)
        print("EnhancedChatService 초기화 완료!")
        
        # 서비스 상태 확인
        print("\n" + "-" * 40)
        print("서비스 상태 확인")
        print("-" * 40)
        status = chat_service.get_system_status()
        print(f"서비스 이름: {status.get('service_name', 'EnhancedChatService')}")
        print(f"전체 상태: {status.get('overall_status', 'Unknown')}")
        
        # 컴포넌트 상태
        components = status.get('components', {})
        print(f"통합 검색 엔진: {'✓' if components.get('unified_search_engine') else '✗'}")
        print(f"통합 RAG 서비스: {'✓' if components.get('unified_rag_service') else '✗'}")
        print(f"통합 분류기: {'✓' if components.get('unified_classifier') else '✗'}")
        print(f"품질 향상 시스템: {'✓' if components.get('quality_enhancement_systems') else '✗'}")
        
        # 데이터베이스 상태
        database_status = status.get('database_status', {})
        print(f"데이터베이스: {'✓' if database_status.get('connected') else '✗'}")
        print(f"벡터 스토어: {'✓' if database_status.get('vector_store_ready') else '✗'}")
        
        # 테스트 질문들 (100개로 대폭 확장)
        test_questions = [
            # 기본 법률 상담 및 인사
            "안녕하세요! 법률 상담을 받고 싶습니다.",
            "법률 상담이 필요합니다.",
            "변호사 상담을 받고 싶어요.",
            "법적 도움이 필요합니다.",
            
            # 계약 관련
            "계약서 작성에 대해 알려주세요.",
            "계약서 작성 방법을 알고 싶습니다.",
            "계약서 검토를 받고 싶어요.",
            "계약 해지 절차는 어떻게 되나요?",
            "계약 위반 시 대응 방법은?",
            "계약서에 포함해야 할 필수 조항은?",
            
            # 부동산 관련
            "부동산 매매 계약 시 주의사항이 있나요?",
            "부동산 등기 절차는 어떻게 진행되나요?",
            "부동산 임대차 계약서 작성법은?",
            "부동산 매매 시 중개수수료는 얼마인가요?",
            "부동산 소유권 이전 절차는?",
            "부동산 담보대출 조건은 어떻게 되나요?",
            
            # 가족법 관련
            "이혼 절차는 어떻게 진행되나요?",
            "이혼 소송 비용은 얼마인가요?",
            "양육비 산정 기준은 무엇인가요?",
            "상속 포기 절차는 어떻게 되나요?",
            "상속세 계산 방법을 알려주세요.",
            "유언장 작성 방법은?",
            "가족법상 친권과 양육권의 차이는?",
            "입양 절차는 어떻게 되나요?",
            
            # 형사법 관련
            "형사 사건에서 변호사 선임은 필수인가요?",
            "형사소송에서 피고인의 권리는 무엇인가요?",
            "구속 영장 신청 절차는?",
            "보석 신청 방법은?",
            "형사 합의 절차는 어떻게 되나요?",
            "형사 피해자 보호 제도는?",
            
            # 민사법 관련
            "손해배상 청구 요건은 무엇인가요?",
            "민사소송에서 증거 수집 방법은?",
            "민사소송 비용은 얼마인가요?",
            "소송 제기 절차는 어떻게 되나요?",
            "강제집행 절차는?",
            "가압류 신청 방법은?",
            
            # 노동법 관련
            "근로기준법상 휴가 규정은 어떻게 되나요?",
            "노동법상 해고의 정당한 사유는?",
            "임금 체불 시 대응 방법은?",
            "근로시간 규정은 어떻게 되나요?",
            "산업재해 보상 절차는?",
            "노동조합 설립 절차는?",
            
            # 상속법 관련
            "상속법에서 유류분 제도는 무엇인가요?",
            "상속재산 분할 절차는?",
            "상속포기 기간은 언제까지인가요?",
            "유언의 효력은 언제부터인가요?",
            "상속인 순위는 어떻게 되나요?",
            
            # 개인정보보호법 관련
            "개인정보보호법의 주요 내용을 알려주세요.",
            "개인정보 유출 시 대응 방법은?",
            "개인정보 처리방침 작성법은?",
            "개인정보보호법 위반 시 처벌은?",
            
            # 저작권 관련
            "저작권 침해 시 법적 대응 방안은?",
            "저작권 등록 절차는 어떻게 되나요?",
            "저작권 사용료 산정 기준은?",
            "저작권 침해 금지 신청은?",
            
            # 행정법 관련
            "행정심판과 행정소송의 차이점은?",
            "행정처분 취소 소송은?",
            "행정심판 신청 절차는?",
            "행정소송 비용은 얼마인가요?",
            
            # 국제법 관련
            "국제사법의 적용 범위는 어떻게 되나요?",
            "국제계약 분쟁 해결 방법은?",
            "국제중재 절차는?",
            "국제사법재판소 관할은?",
            
            # 법인 관련
            "법인 설립 절차에 대해 설명해주세요.",
            "법인 등기 절차는?",
            "법인세 신고 방법은?",
            "법인 해산 절차는?",
            "주식회사 설립 비용은?",
            
            # 환경법 관련
            "환경법상 환경영향평가 제도는?",
            "환경오염 배상 책임은?",
            "환경법 위반 시 처벌은?",
            "환경영향평가 신청 절차는?",
            
            # 지적재산권 관련
            "지적재산권 보호 방법에 대해 알려주세요.",
            "특허 출원 절차는?",
            "상표 등록 방법은?",
            "디자인 등록 절차는?",
            "지적재산권 침해 금지 신청은?",
            
            # 금융법 관련
            "금융투자업법의 주요 내용은?",
            "자본시장법 규정은?",
            "금융감독원 신고 절차는?",
            "금융사고 대응 방법은?",
            
            # 세법 관련
            "소득세 신고 방법은?",
            "부가가치세 계산법은?",
            "세무조사 대응 방법은?",
            "세무대리인 자격 요건은?",
            
            # 의료법 관련
            "의료사고 배상 책임은?",
            "의료법 위반 시 처벌은?",
            "의료기관 개설 절차는?",
            "의료진 면책 사유는?",
            
            # 교육법 관련
            "교육법상 교원의 권리는?",
            "학생 인권 보호 규정은?",
            "교육청 신고 절차는?",
            "학교폭력 대응 방법은?",
            
            # 건설법 관련
            "건설업 등록 절차는?",
            "건설사고 배상 책임은?",
            "건축법 위반 시 처벌은?",
            "건설공사 계약서 작성법은?",
            
            # 교통법 관련
            "교통사고 처리 절차는?",
            "교통사고 합의 방법은?",
            "면허 취소 절차는?",
            "교통법 위반 시 처벌은?",
            
            # 기타 법률 분야
            "공정거래법 위반 시 처벌은?",
            "독점규제법 규정은?",
            "소비자보호법 내용은?",
            "전자상거래법 규정은?",
            "정보통신망법 위반 시 처벌은?",
            "방송통신법 규정은?",
            "언론중재법 절차는?",
            "공공기관 정보공개법은?",
            "국가보안법 위반 시 처벌은?",
            "국가보안법 규정은?",
            "형법상 살인죄 처벌은?",
            "형법상 절도죄 처벌은?",
            "형법상 사기죄 처벌은?",
            "형법상 강도죄 처벌은?",
            "형법상 강간죄 처벌은?"
        ]
        
        print("\n" + "-" * 40)
        print("질문 테스트 시작")
        print("-" * 40)
        
        session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        user_id = "test_user"
        
        # 테스트 통계 수집 (상세 분석용)
        test_stats = {
            "total_questions": len(test_questions),
            "successful_responses": 0,
            "failed_responses": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "total_sources": 0,
            "restricted_questions": 0,
            "warnings_generated": 0,
            "response_times": [],
            "confidences": [],
            "source_counts": [],
            "category_stats": {
                "contract": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "real_estate": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "family_law": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "criminal": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "civil": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "labor": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "inheritance": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "privacy": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "copyright": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "administrative": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "international": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "corporate": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "environmental": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "ip": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "financial": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "tax": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "medical": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "education": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "construction": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "traffic": {"count": 0, "restricted": 0, "avg_time": 0.0},
                "other": {"count": 0, "restricted": 0, "avg_time": 0.0}
            }
        }
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[테스트 {i}] 질문: {question}")
            print("-" * 30)
            
            try:
                # 메시지 처리
                start_time = datetime.now()
                result = await chat_service.process_message(
                    message=question,
                    session_id=session_id,
                    user_id=user_id
                )
                end_time = datetime.now()
                
                # 결과 출력
                response = result.get('response', '응답 없음')
                confidence = result.get('confidence', 0.0)
                processing_time = result.get('processing_time', 0.0)
                sources_count = len(result.get('sources', []))
                
                print(f"응답: {response[:200]}...")
                print(f"신뢰도: {confidence:.2f}")
                print(f"처리 시간: {processing_time:.2f}초")
                print(f"소스 수: {sources_count}")
                
                # 통계 수집
                test_stats["successful_responses"] += 1
                test_stats["total_processing_time"] += processing_time
                test_stats["total_sources"] += sources_count
                test_stats["response_times"].append(processing_time)
                test_stats["confidences"].append(confidence)
                test_stats["source_counts"].append(sources_count)
                
                # 카테고리별 통계 수집
                category = categorize_question(question)
                test_stats["category_stats"][category]["count"] += 1
                test_stats["category_stats"][category]["avg_time"] += processing_time
                
                # 제한 정보 확인
                restriction_info = result.get('restriction_info')
                if restriction_info:
                    is_restricted = restriction_info.get('is_restricted', False)
                    print(f"제한 여부: {'제한됨' if is_restricted else '허용됨'}")
                    if is_restricted:
                        print(f"제한 수준: {restriction_info.get('restriction_level', 'unknown')}")
                        test_stats["restricted_questions"] += 1
                        test_stats["category_stats"][category]["restricted"] += 1
                
                # Phase 정보 확인
                phase_info = result.get('phase_info', {})
                print(f"Phase 1 활성화: {'✓' if phase_info.get('phase1', {}).get('enabled') else '✗'}")
                print(f"Phase 2 활성화: {'✓' if phase_info.get('phase2', {}).get('enabled') else '✗'}")
                print(f"Phase 3 활성화: {'✓' if phase_info.get('phase3', {}).get('enabled') else '✗'}")
                
                # 오류 확인
                errors = result.get('errors', [])
                if errors:
                    print(f"오류: {errors}")
                    test_stats["failed_responses"] += 1
                
            except Exception as e:
                print(f"오류 발생: {str(e)}")
                test_stats["failed_responses"] += 1
        
        # 테스트 통계 계산 및 출력
        if test_stats["response_times"]:
            test_stats["average_processing_time"] = test_stats["total_processing_time"] / test_stats["successful_responses"]
            test_stats["average_confidence"] = sum(test_stats["confidences"]) / len(test_stats["confidences"])
            test_stats["min_response_time"] = min(test_stats["response_times"])
            test_stats["max_response_time"] = max(test_stats["response_times"])
            test_stats["average_sources"] = test_stats["total_sources"] / test_stats["successful_responses"]
            test_stats["min_sources"] = min(test_stats["source_counts"])
            test_stats["max_sources"] = max(test_stats["source_counts"])
            
            # 카테고리별 평균 시간 계산
            for category, stats in test_stats["category_stats"].items():
                if stats["count"] > 0:
                    stats["avg_time"] = stats["avg_time"] / stats["count"]
                    stats["restriction_rate"] = (stats["restricted"] / stats["count"]) * 100
        
        print("\n" + "=" * 80)
        print("대규모 테스트 통계 요약 (100개 질문)")
        print("=" * 80)
        print(f"총 질문 수: {test_stats['total_questions']}")
        print(f"성공한 응답: {test_stats['successful_responses']}")
        print(f"실패한 응답: {test_stats['failed_responses']}")
        print(f"성공률: {(test_stats['successful_responses'] / test_stats['total_questions']) * 100:.1f}%")
        print(f"제한된 질문: {test_stats['restricted_questions']}")
        print(f"제한률: {(test_stats['restricted_questions'] / test_stats['total_questions']) * 100:.1f}%")
        
        if test_stats["response_times"]:
            print(f"\n📊 성능 지표:")
            print(f"  평균 응답 시간: {test_stats['average_processing_time']:.2f}초")
            print(f"  최소 응답 시간: {test_stats['min_response_time']:.2f}초")
            print(f"  최대 응답 시간: {test_stats['max_response_time']:.2f}초")
            print(f"  평균 신뢰도: {test_stats['average_confidence']:.2f}")
            print(f"  평균 소스 수: {test_stats['average_sources']:.1f}")
            print(f"  최소 소스 수: {test_stats['min_sources']}")
            print(f"  최대 소스 수: {test_stats['max_sources']}")
            
            print(f"\n📈 카테고리별 분석:")
            for category, stats in test_stats["category_stats"].items():
                if stats["count"] > 0:
                    print(f"  {category}: {stats['count']}개 질문, 평균 {stats['avg_time']:.2f}초, 제한률 {stats['restriction_rate']:.1f}%")
        
        print("\n" + "-" * 40)
        print("\n" + "-" * 40)
        print("성능 메트릭")
        print("-" * 40)
        
        try:
            metrics = chat_service.get_performance_metrics()
            print(f"메트릭 수집 시간: {metrics.get('timestamp', 'Unknown')}")
            
            # 성능 메트릭 출력
            if 'performance' in metrics:
                perf = metrics['performance']
                print(f"평균 응답 시간: {perf.get('avg_response_time', 0.0):.2f}초")
                print(f"총 요청 수: {perf.get('total_requests', 0)}")
            
            # 메모리 사용량
            if 'memory' in metrics:
                memory = metrics['memory']
                print(f"메모리 사용량: {memory.get('used_mb', 0.0):.2f}MB")
                print(f"메모리 비율: {memory.get('percentage', 0.0):.1f}%")
        
        except Exception as e:
            print(f"성능 메트릭 조회 오류: {str(e)}")
        
        # 시스템 통계 확인
        print("\n" + "-" * 40)
        print("시스템 통계")
        print("-" * 40)
        
        try:
            stats = chat_service.get_system_statistics()
            for component, stat in stats.items():
                print(f"{component}: {'활성화' if stat.get('enabled') else '비활성화'}")
                if stat.get('enabled'):
                    for key, value in stat.items():
                        if key != 'enabled':
                            print(f"  {key}: {value}")
        
        except Exception as e:
            print(f"시스템 통계 조회 오류: {str(e)}")
        
        print("\n" + "=" * 60)
        print("EnhancedChatService 테스트 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Windows 콘솔 인코딩 설정
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    
    # 비동기 테스트 실행
    asyncio.run(test_chat_service())
