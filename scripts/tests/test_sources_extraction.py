"""
sources_detail 추출 및 sources_by_type 생성 테스트
문제 원인 파악을 위한 테스트 코드
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
from typing import Dict, Any, List
from api.services.sources_extractor import SourcesExtractor
from api.services.streaming.stream_handler import StreamHandler
from api.routers.chat import _create_sources_event

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_retrieved_docs() -> List[Dict[str, Any]]:
    """테스트용 retrieved_docs 생성"""
    return [
        {
            "content": "민법 제618조에 따르면 임대인은 임차인에게 목적물을 사용하게 할 의무가 있습니다.",
            "metadata": {
                "source_type": "statute_article",
                "statute_name": "민법",
                "article_no": "618",
                "law_id": "법률 제12345호"
            },
            "relevance_score": 0.95
        },
        {
            "content": "대법원 2020다12345 판결에 따르면 임대차 계약 해지 시 손해배상 책임이 발생할 수 있습니다.",
            "metadata": {
                "source_type": "case_paragraph",
                "case_name": "임대차 계약 해지 사건",
                "case_number": "2020다12345",
                "court": "대법원",
                "doc_id": "case_12345"
            },
            "relevance_score": 0.88
        },
        {
            "content": "임대차보호법 제3조에 따르면 임차인은 차임을 지급할 의무가 있습니다.",
            "metadata": {
                "source_type": "statute_article",
                "statute_name": "임대차보호법",
                "article_no": "3",
                "law_id": "법률 제67890호"
            },
            "relevance_score": 0.82
        }
    ]


def test_sources_extractor_extract_sources_detail():
    """sources_extractor._extract_sources_detail 테스트"""
    logger.info("=" * 80)
    logger.info("TEST 1: sources_extractor._extract_sources_detail 테스트")
    logger.info("=" * 80)
    
    try:
        from api.services.chat_service import get_chat_service
        
        chat_service = get_chat_service()
        if not chat_service or not hasattr(chat_service, 'sources_extractor'):
            logger.error("❌ chat_service 또는 sources_extractor를 찾을 수 없음")
            return False
        
        extractor = chat_service.sources_extractor
        retrieved_docs = create_mock_retrieved_docs()
        
        # state_values 생성
        state_values = {
            "retrieved_docs": retrieved_docs
        }
        
        logger.info(f"Input: retrieved_docs count = {len(retrieved_docs)}")
        
        # sources_detail 추출
        sources_detail = extractor._extract_sources_detail(state_values)
        
        logger.info(f"Result: sources_detail count = {len(sources_detail) if sources_detail else 0}")
        
        if sources_detail:
            logger.info("✅ sources_detail 추출 성공")
            for i, detail in enumerate(sources_detail[:3], 1):
                logger.info(f"  [{i}] type={detail.get('type')}, name={detail.get('name', 'N/A')[:50]}")
        else:
            logger.error("❌ sources_detail 추출 실패: 빈 리스트 반환")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ sources_detail 추출 중 예외 발생: {e}", exc_info=True)
        return False


def test_sources_extractor_get_sources_by_type():
    """sources_extractor._get_sources_by_type 테스트"""
    logger.info("=" * 80)
    logger.info("TEST 2: sources_extractor._get_sources_by_type 테스트")
    logger.info("=" * 80)
    
    try:
        from api.services.chat_service import get_chat_service
        
        chat_service = get_chat_service()
        if not chat_service or not hasattr(chat_service, 'sources_extractor'):
            logger.error("❌ chat_service 또는 sources_extractor를 찾을 수 없음")
            return False
        
        extractor = chat_service.sources_extractor
        retrieved_docs = create_mock_retrieved_docs()
        
        # state_values 생성
        state_values = {
            "retrieved_docs": retrieved_docs
        }
        
        # sources_detail 추출
        sources_detail = extractor._extract_sources_detail(state_values)
        
        if not sources_detail:
            logger.error("❌ sources_detail이 비어있어 테스트 불가")
            return False
        
        logger.info(f"Input: sources_detail count = {len(sources_detail)}")
        
        # sources_by_type 생성
        sources_by_type = extractor._get_sources_by_type(sources_detail)
        
        logger.info(f"Result: sources_by_type = {sources_by_type}")
        
        if sources_by_type:
            logger.info("✅ sources_by_type 생성 성공")
            for key, items in sources_by_type.items():
                logger.info(f"  {key}: {len(items)} items")
                if items:
                    logger.info(f"    First item keys: {list(items[0].keys())[:10]}")
        else:
            logger.error("❌ sources_by_type 생성 실패: None 반환")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ sources_by_type 생성 중 예외 발생: {e}", exc_info=True)
        return False


def test_sources_extractor_get_sources_by_type_with_reference_statutes():
    """sources_extractor._get_sources_by_type_with_reference_statutes 테스트"""
    logger.info("=" * 80)
    logger.info("TEST 3: sources_extractor._get_sources_by_type_with_reference_statutes 테스트")
    logger.info("=" * 80)
    
    try:
        from api.services.chat_service import get_chat_service
        
        chat_service = get_chat_service()
        if not chat_service or not hasattr(chat_service, 'sources_extractor'):
            logger.error("❌ chat_service 또는 sources_extractor를 찾을 수 없음")
            return False
        
        extractor = chat_service.sources_extractor
        retrieved_docs = create_mock_retrieved_docs()
        
        # state_values 생성
        state_values = {
            "retrieved_docs": retrieved_docs
        }
        
        # sources_detail 추출
        sources_detail = extractor._extract_sources_detail(state_values)
        
        if not sources_detail:
            logger.error("❌ sources_detail이 비어있어 테스트 불가")
            return False
        
        logger.info(f"Input: sources_detail count = {len(sources_detail)}")
        
        # sources_by_type 생성 (참조 법령 포함)
        sources_by_type = extractor._get_sources_by_type_with_reference_statutes(sources_detail)
        
        logger.info(f"Result: sources_by_type = {sources_by_type}")
        
        if sources_by_type:
            logger.info("✅ sources_by_type 생성 성공 (참조 법령 포함)")
            for key, items in sources_by_type.items():
                logger.info(f"  {key}: {len(items)} items")
                if items:
                    logger.info(f"    First item keys: {list(items[0].keys())[:10]}")
        else:
            logger.error("❌ sources_by_type 생성 실패: None 반환")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ sources_by_type 생성 중 예외 발생: {e}", exc_info=True)
        return False


def test_stream_handler_generate_sources_by_type():
    """stream_handler._generate_sources_by_type 테스트"""
    logger.info("=" * 80)
    logger.info("TEST 4: stream_handler._generate_sources_by_type 테스트")
    logger.info("=" * 80)
    
    try:
        from api.services.chat_service import get_chat_service
        
        chat_service = get_chat_service()
        if not chat_service or not hasattr(chat_service, 'sources_extractor'):
            logger.error("❌ chat_service 또는 sources_extractor를 찾을 수 없음")
            return False
        
        extractor = chat_service.sources_extractor
        retrieved_docs = create_mock_retrieved_docs()
        
        # state_values 생성
        state_values = {
            "retrieved_docs": retrieved_docs
        }
        
        # sources_detail 추출
        sources_detail = extractor._extract_sources_detail(state_values)
        
        if not sources_detail:
            logger.error("❌ sources_detail이 비어있어 테스트 불가")
            return False
        
        logger.info(f"Input: sources_detail count = {len(sources_detail)}")
        
        # StreamHandler 생성 (workflow_service와 sources_extractor 포함)
        stream_handler = StreamHandler(
            workflow_service=chat_service.workflow_service,
            sources_extractor=extractor,
            extract_related_questions_fn=None
        )
        
        # sources_by_type 생성
        sources_by_type = stream_handler._generate_sources_by_type(sources_detail)
        
        logger.info(f"Result: sources_by_type = {sources_by_type}")
        
        if sources_by_type:
            logger.info("✅ stream_handler._generate_sources_by_type 성공")
            for key, items in sources_by_type.items():
                logger.info(f"  {key}: {len(items)} items")
        else:
            logger.error("❌ stream_handler._generate_sources_by_type 실패: None 반환")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ stream_handler._generate_sources_by_type 중 예외 발생: {e}", exc_info=True)
        return False


def test_create_sources_event():
    """_create_sources_event 테스트"""
    logger.info("=" * 80)
    logger.info("TEST 5: _create_sources_event 테스트")
    logger.info("=" * 80)
    
    try:
        from api.services.chat_service import get_chat_service
        
        chat_service = get_chat_service()
        if not chat_service or not hasattr(chat_service, 'sources_extractor'):
            logger.error("❌ chat_service 또는 sources_extractor를 찾을 수 없음")
            return False
        
        extractor = chat_service.sources_extractor
        retrieved_docs = create_mock_retrieved_docs()
        
        # state_values 생성
        state_values = {
            "retrieved_docs": retrieved_docs
        }
        
        # sources_detail 추출
        sources_detail = extractor._extract_sources_detail(state_values)
        
        if not sources_detail:
            logger.error("❌ sources_detail이 비어있어 테스트 불가")
            return False
        
        logger.info(f"Input: sources_detail count = {len(sources_detail)}")
        
        # metadata 생성
        metadata = {
            "sources_detail": sources_detail,
            "message_id": "test-message-id-123"
        }
        
        # sources 이벤트 생성
        sources_event = _create_sources_event(metadata, "test-message-id-123")
        
        logger.info(f"Result: sources_event type = {sources_event.get('type')}")
        logger.info(f"Result: sources_event metadata keys = {list(sources_event.get('metadata', {}).keys())}")
        
        sources_by_type = sources_event.get("metadata", {}).get("sources_by_type")
        
        if sources_by_type:
            logger.info("✅ _create_sources_event 성공")
            for key, items in sources_by_type.items():
                logger.info(f"  {key}: {len(items)} items")
        else:
            logger.error("❌ _create_sources_event 실패: sources_by_type이 비어있음")
            logger.error(f"  Full event: {sources_event}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ _create_sources_event 중 예외 발생: {e}", exc_info=True)
        return False


def test_empty_sources_detail():
    """빈 sources_detail로 테스트 (실제 문제 상황 시뮬레이션)"""
    logger.info("=" * 80)
    logger.info("TEST 6: 빈 sources_detail로 테스트 (실제 문제 상황)")
    logger.info("=" * 80)
    
    try:
        # 빈 metadata로 테스트
        metadata = {
            "sources_detail": [],
            "sources_by_type": None,
            "message_id": "test-message-id-empty"
        }
        
        logger.info(f"Input: sources_detail = [] (빈 배열)")
        logger.info(f"Input: sources_by_type = None")
        
        # sources 이벤트 생성
        sources_event = _create_sources_event(metadata, "test-message-id-empty")
        
        logger.info(f"Result: sources_event type = {sources_event.get('type')}")
        
        sources_by_type = sources_event.get("metadata", {}).get("sources_by_type")
        
        if sources_by_type:
            logger.info("✅ 빈 sources_detail에서도 sources_by_type 생성됨 (기본 구조)")
            for key, items in sources_by_type.items():
                logger.info(f"  {key}: {len(items)} items")
        else:
            logger.error("❌ 빈 sources_detail에서 sources_by_type 생성 실패")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 빈 sources_detail 테스트 중 예외 발생: {e}", exc_info=True)
        return False


def main():
    """모든 테스트 실행"""
    logger.info("=" * 80)
    logger.info("sources_detail 추출 및 sources_by_type 생성 테스트 시작")
    logger.info("=" * 80)
    
    results = []
    
    # 테스트 실행
    results.append(("sources_detail 추출", test_sources_extractor_extract_sources_detail()))
    results.append(("sources_by_type 생성", test_sources_extractor_get_sources_by_type()))
    results.append(("sources_by_type 생성 (참조 법령 포함)", test_sources_extractor_get_sources_by_type_with_reference_statutes()))
    results.append(("stream_handler._generate_sources_by_type", test_stream_handler_generate_sources_by_type()))
    results.append(("_create_sources_event", test_create_sources_event()))
    results.append(("빈 sources_detail 테스트", test_empty_sources_detail()))
    
    # 결과 요약
    logger.info("=" * 80)
    logger.info("테스트 결과 요약")
    logger.info("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("=" * 80)
    logger.info(f"총 {len(results)}개 테스트: {passed}개 통과, {failed}개 실패")
    logger.info("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

