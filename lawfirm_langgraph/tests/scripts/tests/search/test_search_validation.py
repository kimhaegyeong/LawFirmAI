# -*- coding: utf-8 -*-
"""
검색 결과 검증 로직 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
scripts_dir = script_dir.parent
tests_dir = scripts_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

import logging
from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

# 로깅 설정
import os
log_level = os.getenv('TEST_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_search_validation():
    """검색 결과 검증 로직 테스트"""
    logger.info("="*80)
    logger.info("검색 결과 검증 로직 테스트 시작")
    logger.info("="*80)
    
    # 데이터베이스 경로
    db_path = project_root / "data" / "lawfirm_v2.db"
    if not db_path.exists():
        logger.error(f"데이터베이스 파일을 찾을 수 없습니다: {db_path}")
        return
    
    # 검색 엔진 초기화
    logger.info("검색 엔진 초기화 중...")
    try:
        search_engine = SemanticSearchEngineV2(db_path=str(db_path))
        logger.info("✅ 검색 엔진 초기화 완료")
    except Exception as e:
        logger.error(f"검색 엔진 초기화 실패: {e}")
        return
    
    # 테스트 쿼리
    test_queries = [
        "민법 제750조 손해배상",
        "계약 위반",
        "임대차 보증금"
    ]
    
    for query in test_queries:
        logger.info("\n" + "="*80)
        logger.info(f"테스트 쿼리: {query}")
        logger.info("="*80)
        
        try:
            # 검색 실행
            results = search_engine.search(
                query=query,
                k=10,
                similarity_threshold=0.3
            )
            
            logger.info(f"\n검색 결과: {len(results)}개")
            
            # 검증 결과 분석
            if results:
                # embedding_version_id 확인
                version_ids = []
                missing_version_count = 0
                for i, result in enumerate(results, 1):
                    version_id = result.get('embedding_version_id') or result.get('metadata', {}).get('embedding_version_id')
                    if version_id:
                        version_ids.append(version_id)
                    else:
                        missing_version_count += 1
                        logger.warning(f"  결과 {i}: embedding_version_id 없음")
                
                if version_ids:
                    version_dist = {}
                    for vid in version_ids:
                        version_dist[vid] = version_dist.get(vid, 0) + 1
                    logger.info(f"  버전 분포: {version_dist}")
                
                if missing_version_count > 0:
                    logger.warning(f"  ⚠️  embedding_version_id 누락: {missing_version_count}개")
                else:
                    logger.info(f"  ✅ 모든 결과에 embedding_version_id 포함")
                
                # 메타데이터 완전성 확인
                missing_metadata_count = 0
                for i, result in enumerate(results, 1):
                    source_type = result.get('type') or result.get('source_type') or result.get('metadata', {}).get('source_type')
                    if not source_type:
                        missing_metadata_count += 1
                        logger.warning(f"  결과 {i}: source_type 없음")
                
                if missing_metadata_count > 0:
                    logger.warning(f"  ⚠️  source_type 누락: {missing_metadata_count}개")
                else:
                    logger.info(f"  ✅ 모든 결과에 source_type 포함")
                
                # 텍스트 품질 확인
                short_text_count = 0
                empty_text_count = 0
                for i, result in enumerate(results, 1):
                    text = result.get('text') or result.get('content') or result.get('metadata', {}).get('text') or result.get('metadata', {}).get('content')
                    if not text or len(str(text).strip()) == 0:
                        empty_text_count += 1
                        logger.warning(f"  결과 {i}: 텍스트 없음")
                    elif len(str(text).strip()) < 100:
                        short_text_count += 1
                        logger.warning(f"  결과 {i}: 텍스트 너무 짧음 ({len(str(text).strip())}자)")
                
                if empty_text_count > 0 or short_text_count > 0:
                    logger.warning(f"  ⚠️  텍스트 품질 문제: 빈 텍스트 {empty_text_count}개, 짧은 텍스트 {short_text_count}개")
                else:
                    logger.info(f"  ✅ 모든 결과의 텍스트 품질 양호")
                
                # 상위 3개 결과 상세 정보
                logger.info("\n상위 3개 결과 상세:")
                for i, result in enumerate(results[:3], 1):
                    logger.info(f"  {i}. ID: {result.get('id')}")
                    logger.info(f"     Type: {result.get('type')}")
                    logger.info(f"     Version ID: {result.get('embedding_version_id')}")
                    logger.info(f"     Score: {result.get('score', 0):.4f}")
                    logger.info(f"     Text length: {len(str(result.get('text', '') or result.get('content', '')).strip())}자")
            else:
                logger.warning("검색 결과가 없습니다")
                
        except Exception as e:
            logger.error(f"검색 실행 중 오류 발생: {e}", exc_info=True)
    
    logger.info("\n" + "="*80)
    logger.info("검색 결과 검증 로직 테스트 완료")
    logger.info("="*80)

if __name__ == "__main__":
    test_search_validation()

