# -*- coding: utf-8 -*-
"""
RAG 검색 성능 개선 간단 검증 테스트
구현된 개선사항들이 제대로 작동하는지 간단히 검증
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
lawfirm_langgraph_dir = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(lawfirm_langgraph_dir))

import warnings
warnings.filterwarnings('ignore')

import logging
import hashlib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_hash_based_query_cache():
    """해시 기반 쿼리 캐시 테스트"""
    logger.info("\n=== 테스트 1: 해시 기반 쿼리 캐시 ===")
    
    try:
        from core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
        
        # DB 경로 찾기
        db_path = None
        for path in ["data/lawfirm_v2.db", "./data/lawfirm_v2.db", str(project_root / "data" / "lawfirm_v2.db")]:
            if Path(path).exists():
                db_path = path
                break
        
        if not db_path:
            logger.warning("   ⏭️  데이터베이스 없음 - 스킵")
            return False
        
        engine = SemanticSearchEngineV2(db_path=db_path)
        
        # 해시 캐시 사용 여부 확인
        assert hasattr(engine, '_use_hash_cache'), "해시 캐시 속성이 없습니다"
        logger.info(f"   ✅ 해시 캐시 사용: {engine._use_hash_cache}")
        
        # 캐시 크기 확인
        assert hasattr(engine, '_cache_max_size'), "캐시 크기 속성이 없습니다"
        assert engine._cache_max_size == 1000, f"캐시 크기가 1000이 아닙니다: {engine._cache_max_size}"
        logger.info(f"   ✅ 캐시 크기: {engine._cache_max_size}")
        
        # 쿼리 정규화 및 해시 생성 테스트
        test_query = "민법 제543조 계약 해지"
        normalized = engine._normalize_query(test_query)
        logger.info(f"   ✅ 쿼리 정규화: '{test_query}' -> '{normalized}'")
        
        if engine._use_hash_cache:
            query_hash = hashlib.md5(normalized.encode()).hexdigest()
            logger.info(f"   ✅ 해시 생성: {query_hash[:16]}...")
        
        logger.info("   ✅ 해시 기반 쿼리 캐시 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ 해시 기반 쿼리 캐시 테스트 실패: {e}")
        return False


def test_search_quality_metrics():
    """검색 성능 메트릭 수집 테스트"""
    logger.info("\n=== 테스트 2: 검색 성능 메트릭 수집 ===")
    
    try:
        from core.search.processors.result_merger import ResultRanker
        
        ranker = ResultRanker(use_cross_encoder=False)
        
        # evaluate_search_quality 메서드 존재 확인
        assert hasattr(ranker, 'evaluate_search_quality'), "evaluate_search_quality 메서드가 없습니다"
        logger.info("   ✅ evaluate_search_quality 메서드 존재")
        
        # 샘플 검색 결과 생성
        sample_results = [
            {
                "content": "민법 제543조 계약 해지 사유",
                "relevance_score": 0.85,
                "final_weighted_score": 0.85
            },
            {
                "content": "계약 해지 손해배상 범위",
                "relevance_score": 0.75,
                "final_weighted_score": 0.75
            },
            {
                "content": "계약 해지 절차",
                "relevance_score": 0.65,
                "final_weighted_score": 0.65
            }
        ]
        
        # 메트릭 계산
        metrics = ranker.evaluate_search_quality(
            query="계약 해지",
            results=sample_results,
            query_type="law_inquiry",
            extracted_keywords=["계약", "해지", "사유"]
        )
        
        # 메트릭 항목 확인
        required_metrics = ["avg_relevance", "min_relevance", "max_relevance", "diversity_score", "keyword_coverage"]
        for metric in required_metrics:
            assert metric in metrics, f"메트릭 '{metric}'이 없습니다"
            assert isinstance(metrics[metric], (int, float)), f"메트릭 '{metric}'이 숫자가 아닙니다"
            logger.info(f"   ✅ {metric}: {metrics[metric]:.3f}")
        
        # 값 검증
        assert metrics["avg_relevance"] > 0, "평균 관련성 점수가 0보다 커야 합니다"
        assert metrics["max_relevance"] >= metrics["min_relevance"], "최대 관련성 점수가 최소 관련성 점수보다 크거나 같아야 합니다"
        assert 0 <= metrics["diversity_score"] <= 1, "다양성 점수는 0-1 사이여야 합니다"
        assert 0 <= metrics["keyword_coverage"] <= 1, "키워드 커버리지는 0-1 사이여야 합니다"
        
        logger.info("   ✅ 검색 성능 메트릭 수집 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ 검색 성능 메트릭 수집 테스트 실패: {e}")
        return False


def test_semantic_chunking():
    """의미 단위 청킹 테스트"""
    logger.info("\n=== 테스트 3: 의미 단위 청킹 전략 ===")
    
    try:
        # text_chunker 모듈 import
        scripts_utils_path = project_root / "scripts" / "utils"
        if scripts_utils_path.exists():
            sys.path.insert(0, str(scripts_utils_path))
        from text_chunker import chunk_by_semantic_units, chunk_paragraphs
        
        # 법령 조문 청킹 테스트
        statute_text = """
        제1조 (목적)
        이 법은 계약에 관한 사항을 규정한다.
        
        제2조 (정의)
        이 법에서 사용하는 용어의 뜻은 다음과 같다.
        """
        
        statute_chunks = chunk_by_semantic_units(statute_text, doc_type="statute_article")
        assert len(statute_chunks) > 0, "법령 조문 청킹 결과가 없습니다"
        logger.info(f"   ✅ 법령 조문 청킹: {len(statute_chunks)}개 청크 생성")
        
        # 판례 청킹 테스트
        case_text = """
        【판결요지】
        계약 해지는 계약 당사자 일방이 상대방에게 의사표시를 하여 계약을 소급하여 소멸시키는 행위이다.
        
        【판단】
        원고의 계약 해지 의사표시는 유효하다.
        """
        
        case_chunks = chunk_by_semantic_units(case_text, doc_type="case_paragraph")
        assert len(case_chunks) > 0, "판례 청킹 결과가 없습니다"
        logger.info(f"   ✅ 판례 청킹: {len(case_chunks)}개 청크 생성")
        
        # 판결요지, 판단 구분 확인
        chunk_types = [chunk.get("metadata", {}).get("chunk_type", "") for chunk in case_chunks]
        assert "판결요지" in chunk_types or "판단" in chunk_types, "판결요지 또는 판단 청크가 없습니다"
        logger.info(f"   ✅ 청크 타입: {chunk_types}")
        
        logger.info("   ✅ 의미 단위 청킹 전략 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ 의미 단위 청킹 전략 테스트 실패: {e}")
        return False


def test_query_expansion_structured_info():
    """쿼리 확장 구조화된 정보 추출 테스트"""
    logger.info("\n=== 테스트 4: 쿼리 확장 구조화된 정보 추출 ===")
    
    try:
        from core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
        
        # DB 경로 찾기
        db_path = None
        for path in ["data/lawfirm_v2.db", "./data/lawfirm_v2.db", str(project_root / "data" / "lawfirm_v2.db")]:
            if Path(path).exists():
                db_path = path
                break
        
        if not db_path:
            logger.warning("   ⏭️  데이터베이스 없음 - 스킵")
            return False
        
        engine = SemanticSearchEngineV2(db_path=db_path)
        
        # _extract_structured_info 메서드 존재 확인
        assert hasattr(engine, '_extract_structured_info'), "_extract_structured_info 메서드가 없습니다"
        logger.info("   ✅ _extract_structured_info 메서드 존재")
        
        # 법령 조문 번호 추출 테스트
        query1 = "민법 제543조 계약 해지"
        structured_info1 = engine._extract_structured_info(query1)
        assert len(structured_info1) > 0, "법령 조문 번호를 추출하지 못했습니다"
        assert "민법 제543조" in structured_info1[0], "법령 조문 번호 추출 실패"
        logger.info(f"   ✅ 법령 조문 번호 추출: {structured_info1}")
        
        # 판례 사건번호 추출 테스트
        query2 = "대법원 2020다12345 판례"
        structured_info2 = engine._extract_structured_info(query2)
        assert len(structured_info2) > 0, "판례 사건번호를 추출하지 못했습니다"
        logger.info(f"   ✅ 판례 사건번호 추출: {structured_info2}")
        
        # 헌법 조문 추출 테스트
        query3 = "헌법 제10조 기본권"
        structured_info3 = engine._extract_structured_info(query3)
        assert len(structured_info3) > 0, "헌법 조문을 추출하지 못했습니다"
        assert "헌법 제10조" in structured_info3[0], "헌법 조문 추출 실패"
        logger.info(f"   ✅ 헌법 조문 추출: {structured_info3}")
        
        logger.info("   ✅ 쿼리 확장 구조화된 정보 추출 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ 쿼리 확장 구조화된 정보 추출 테스트 실패: {e}")
        return False


def test_nprobe_optimization():
    """nprobe 최적화 테스트"""
    logger.info("\n=== 테스트 5: nprobe 최적화 ===")
    
    try:
        from core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
        
        # DB 경로 찾기
        db_path = None
        for path in ["data/lawfirm_v2.db", "./data/lawfirm_v2.db", str(project_root / "data" / "lawfirm_v2.db")]:
            if Path(path).exists():
                db_path = path
                break
        
        if not db_path:
            logger.warning("   ⏭️  데이터베이스 없음 - 스킵")
            return False
        
        engine = SemanticSearchEngineV2(db_path=db_path)
        
        # _calculate_optimal_nprobe 메서드 존재 확인
        assert hasattr(engine, '_calculate_optimal_nprobe'), "_calculate_optimal_nprobe 메서드가 없습니다"
        logger.info("   ✅ _calculate_optimal_nprobe 메서드 존재")
        
        # 다양한 k 값에 대한 nprobe 계산 테스트
        test_cases = [
            (5, 10000),   # 작은 k
            (15, 10000),  # 중간 k
            (30, 10000),  # 큰 k
        ]
        
        for k, total_vectors in test_cases:
            nprobe = engine._calculate_optimal_nprobe(k, total_vectors)
            assert nprobe > 0, f"nprobe가 0보다 커야 합니다 (k={k})"
            assert nprobe <= 1024, f"nprobe가 1024를 초과하면 안 됩니다 (k={k}, nprobe={nprobe})"
            logger.info(f"   ✅ k={k}, total={total_vectors} -> nprobe={nprobe}")
        
        # k 값이 클수록 nprobe도 증가하는지 확인
        nprobe_small = engine._calculate_optimal_nprobe(5, 10000)
        nprobe_large = engine._calculate_optimal_nprobe(30, 10000)
        assert nprobe_large >= nprobe_small, "큰 k 값에 대해 nprobe가 증가해야 합니다"
        logger.info(f"   ✅ nprobe 증가 확인: k=5 -> {nprobe_small}, k=30 -> {nprobe_large}")
        
        logger.info("   ✅ nprobe 최적화 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ nprobe 최적화 테스트 실패: {e}")
        return False


def test_hybrid_search_dynamic_weights():
    """하이브리드 검색 가중치 동적 조정 테스트"""
    logger.info("\n=== 테스트 6: 하이브리드 검색 가중치 동적 조정 ===")
    
    try:
        from core.services.hybrid_search_engine_v2 import HybridSearchEngineV2
        from core.classification.classifiers.question_classifier import QuestionType
        
        # DB 경로 찾기
        db_path = None
        for path in ["data/lawfirm_v2.db", "./data/lawfirm_v2.db", str(project_root / "data" / "lawfirm_v2.db")]:
            if Path(path).exists():
                db_path = path
                break
        
        if not db_path:
            logger.warning("   ⏭️  데이터베이스 없음 - 스킵")
            return False
        
        engine = HybridSearchEngineV2(db_path=db_path)
        
        # _get_query_type_weights 메서드 존재 확인
        assert hasattr(engine, '_get_query_type_weights'), "_get_query_type_weights 메서드가 없습니다"
        logger.info("   ✅ _get_query_type_weights 메서드 존재")
        
        # 질문 유형별 가중치 확인
        test_cases = [
            (QuestionType.LAW_INQUIRY, "법령 문의"),
            (QuestionType.PRECEDENT_SEARCH, "판례 검색"),
            (QuestionType.COMPLEX_QUESTION, "복합 질문"),
            (QuestionType.GENERAL, "일반 질문"),
        ]
        
        for query_type, description in test_cases:
            weights = engine._get_query_type_weights(query_type)
            assert "exact" in weights, f"{description}: exact 가중치가 없습니다"
            assert "semantic" in weights, f"{description}: semantic 가중치가 없습니다"
            assert 0 <= weights["exact"] <= 1, f"{description}: exact 가중치가 0-1 사이여야 합니다"
            assert 0 <= weights["semantic"] <= 1, f"{description}: semantic 가중치가 0-1 사이여야 합니다"
            assert abs(weights["exact"] + weights["semantic"] - 1.0) < 0.01, f"{description}: 가중치 합이 1에 가까워야 합니다"
            logger.info(f"   ✅ {description}: exact={weights['exact']:.2f}, semantic={weights['semantic']:.2f}")
        
        # 법령 문의는 키워드 검색 가중치가 높아야 함
        law_weights = engine._get_query_type_weights(QuestionType.LAW_INQUIRY)
        assert law_weights["exact"] > law_weights["semantic"], "법령 문의는 키워드 검색 가중치가 높아야 합니다"
        logger.info("   ✅ 법령 문의: 키워드 검색 가중치 > 의미적 검색 가중치")
        
        # 판례 검색은 의미적 검색 가중치가 높아야 함
        precedent_weights = engine._get_query_type_weights(QuestionType.PRECEDENT_SEARCH)
        assert precedent_weights["semantic"] > precedent_weights["exact"], "판례 검색은 의미적 검색 가중치가 높아야 합니다"
        logger.info("   ✅ 판례 검색: 의미적 검색 가중치 > 키워드 검색 가중치")
        
        logger.info("   ✅ 하이브리드 검색 가중치 동적 조정 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ 하이브리드 검색 가중치 동적 조정 테스트 실패: {e}")
        return False


def main():
    """모든 테스트 실행"""
    logger.info("="*80)
    logger.info("RAG 검색 성능 개선 검증 테스트 시작")
    logger.info("="*80)
    
    tests = [
        ("해시 기반 쿼리 캐시", test_hash_based_query_cache),
        ("검색 성능 메트릭 수집", test_search_quality_metrics),
        ("의미 단위 청킹 전략", test_semantic_chunking),
        ("쿼리 확장 구조화된 정보 추출", test_query_expansion_structured_info),
        ("nprobe 최적화", test_nprobe_optimization),
        ("하이브리드 검색 가중치 동적 조정", test_hybrid_search_dynamic_weights),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"테스트 '{test_name}' 실행 중 오류: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    logger.info("\n" + "="*80)
    logger.info("테스트 결과 요약")
    logger.info("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

