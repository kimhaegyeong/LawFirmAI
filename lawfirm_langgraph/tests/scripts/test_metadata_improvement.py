# -*- coding: utf-8 -*-
"""메타데이터 개선 확인 테스트"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from lawfirm_langgraph.core.utils.config import Config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 설정
os.environ['USE_EXTERNAL_VECTOR_STORE'] = 'true'
os.environ['EXTERNAL_VECTOR_STORE_BASE_PATH'] = str(project_root / "data" / "vector_store" / "v2.0.0-dynamic-dynamic-ivfpq")

logger.info("=" * 60)
logger.info("메타데이터 개선 확인 테스트")
logger.info("=" * 60)

# SemanticSearchEngineV2 초기화
config = Config()
search_engine = SemanticSearchEngineV2(
    db_path=config.database_path,
    use_external_index=config.use_external_vector_store,
    external_index_path=config.external_vector_store_base_path
)

# 검색 쿼리
query = "임대차 보증금 반환"
logger.info(f"\n검색 쿼리: {query}")

# 검색 실행
results = search_engine.search(query, k=10, similarity_threshold=0.15)

if not results:
    logger.error("❌ 검색 결과가 없습니다")
    sys.exit(1)

logger.info(f"\n✅ 검색 결과: {len(results)}개\n")

# 메타데이터 완전성 확인
metadata_stats = {
    'case_paragraph': {
        'total': 0,
        'has_doc_id': 0,
        'has_casenames': 0,
        'has_court': 0,
        'complete': 0
    },
    'decision_paragraph': {
        'total': 0,
        'has_org': 0,
        'has_doc_id': 0,
        'complete': 0
    },
    'statute_article': {
        'total': 0,
        'has_statute_name': 0,
        'has_article_no': 0,
        'complete': 0
    },
    'interpretation_paragraph': {
        'total': 0,
        'has_org': 0,
        'has_doc_id': 0,
        'has_title': 0,
        'complete': 0
    }
}

for i, result in enumerate(results, 1):
    source_type = result.get('type') or result.get('metadata', {}).get('source_type')
    metadata = result.get('metadata', {})
    
    logger.info(f"결과 {i}: source_type={source_type}, score={result.get('score', 0):.4f}")
    
    if source_type == 'case_paragraph':
        stats = metadata_stats['case_paragraph']
        stats['total'] += 1
        if result.get('doc_id') or metadata.get('doc_id'):
            stats['has_doc_id'] += 1
        if result.get('casenames') or metadata.get('casenames'):
            stats['has_casenames'] += 1
        if result.get('court') or metadata.get('court'):
            stats['has_court'] += 1
        if (result.get('doc_id') or metadata.get('doc_id')) and \
           (result.get('casenames') or metadata.get('casenames')) and \
           (result.get('court') or metadata.get('court')):
            stats['complete'] += 1
            logger.info(f"  ✅ 완전한 메타데이터: doc_id={result.get('doc_id') or metadata.get('doc_id')}, "
                       f"casenames={result.get('casenames') or metadata.get('casenames')}, "
                       f"court={result.get('court') or metadata.get('court')}")
        else:
            missing = []
            if not (result.get('doc_id') or metadata.get('doc_id')):
                missing.append('doc_id')
            if not (result.get('casenames') or metadata.get('casenames')):
                missing.append('casenames')
            if not (result.get('court') or metadata.get('court')):
                missing.append('court')
            logger.warning(f"  ⚠️  누락된 메타데이터: {', '.join(missing)}")
    
    elif source_type == 'decision_paragraph':
        stats = metadata_stats['decision_paragraph']
        stats['total'] += 1
        if result.get('org') or metadata.get('org'):
            stats['has_org'] += 1
        if result.get('doc_id') or metadata.get('doc_id'):
            stats['has_doc_id'] += 1
        if (result.get('org') or metadata.get('org')) and \
           (result.get('doc_id') or metadata.get('doc_id')):
            stats['complete'] += 1
            logger.info(f"  ✅ 완전한 메타데이터: org={result.get('org') or metadata.get('org')}, "
                       f"doc_id={result.get('doc_id') or metadata.get('doc_id')}")
        else:
            missing = []
            if not (result.get('org') or metadata.get('org')):
                missing.append('org')
            if not (result.get('doc_id') or metadata.get('doc_id')):
                missing.append('doc_id')
            logger.warning(f"  ⚠️  누락된 메타데이터: {', '.join(missing)}")
    
    elif source_type == 'statute_article':
        stats = metadata_stats['statute_article']
        stats['total'] += 1
        if result.get('statute_name') or metadata.get('statute_name'):
            stats['has_statute_name'] += 1
        if result.get('article_no') or metadata.get('article_no'):
            stats['has_article_no'] += 1
        if (result.get('statute_name') or metadata.get('statute_name')) and \
           (result.get('article_no') or metadata.get('article_no')):
            stats['complete'] += 1
            logger.info(f"  ✅ 완전한 메타데이터: statute_name={result.get('statute_name') or metadata.get('statute_name')}, "
                       f"article_no={result.get('article_no') or metadata.get('article_no')}")
        else:
            missing = []
            if not (result.get('statute_name') or metadata.get('statute_name')):
                missing.append('statute_name')
            if not (result.get('article_no') or metadata.get('article_no')):
                missing.append('article_no')
            logger.warning(f"  ⚠️  누락된 메타데이터: {', '.join(missing)}")
    
    elif source_type == 'interpretation_paragraph':
        stats = metadata_stats['interpretation_paragraph']
        stats['total'] += 1
        if result.get('org') or metadata.get('org'):
            stats['has_org'] += 1
        if result.get('doc_id') or metadata.get('doc_id'):
            stats['has_doc_id'] += 1
        if result.get('title') or metadata.get('title'):
            stats['has_title'] += 1
        if (result.get('org') or metadata.get('org')) and \
           (result.get('doc_id') or metadata.get('doc_id')) and \
           (result.get('title') or metadata.get('title')):
            stats['complete'] += 1
            logger.info(f"  ✅ 완전한 메타데이터: org={result.get('org') or metadata.get('org')}, "
                       f"doc_id={result.get('doc_id') or metadata.get('doc_id')}, "
                       f"title={result.get('title') or metadata.get('title')}")
        else:
            missing = []
            if not (result.get('org') or metadata.get('org')):
                missing.append('org')
            if not (result.get('doc_id') or metadata.get('doc_id')):
                missing.append('doc_id')
            if not (result.get('title') or metadata.get('title')):
                missing.append('title')
            logger.warning(f"  ⚠️  누락된 메타데이터: {', '.join(missing)}")

# 통계 출력
logger.info("\n" + "=" * 60)
logger.info("메타데이터 완전성 통계")
logger.info("=" * 60)

for source_type, stats in metadata_stats.items():
    if stats['total'] > 0:
        logger.info(f"\n{source_type}:")
        logger.info(f"  총 결과 수: {stats['total']}")
        
        if source_type == 'case_paragraph':
            logger.info(f"  doc_id 보유: {stats['has_doc_id']}/{stats['total']} ({stats['has_doc_id']/stats['total']*100:.1f}%)")
            logger.info(f"  casenames 보유: {stats['has_casenames']}/{stats['total']} ({stats['has_casenames']/stats['total']*100:.1f}%)")
            logger.info(f"  court 보유: {stats['has_court']}/{stats['total']} ({stats['has_court']/stats['total']*100:.1f}%)")
            logger.info(f"  완전한 메타데이터: {stats['complete']}/{stats['total']} ({stats['complete']/stats['total']*100:.1f}%)")
        elif source_type == 'decision_paragraph':
            logger.info(f"  org 보유: {stats['has_org']}/{stats['total']} ({stats['has_org']/stats['total']*100:.1f}%)")
            logger.info(f"  doc_id 보유: {stats['has_doc_id']}/{stats['total']} ({stats['has_doc_id']/stats['total']*100:.1f}%)")
            logger.info(f"  완전한 메타데이터: {stats['complete']}/{stats['total']} ({stats['complete']/stats['total']*100:.1f}%)")
        elif source_type == 'statute_article':
            logger.info(f"  statute_name 보유: {stats['has_statute_name']}/{stats['total']} ({stats['has_statute_name']/stats['total']*100:.1f}%)")
            logger.info(f"  article_no 보유: {stats['has_article_no']}/{stats['total']} ({stats['has_article_no']/stats['total']*100:.1f}%)")
            logger.info(f"  완전한 메타데이터: {stats['complete']}/{stats['total']} ({stats['complete']/stats['total']*100:.1f}%)")
        elif source_type == 'interpretation_paragraph':
            logger.info(f"  org 보유: {stats['has_org']}/{stats['total']} ({stats['has_org']/stats['total']*100:.1f}%)")
            logger.info(f"  doc_id 보유: {stats['has_doc_id']}/{stats['total']} ({stats['has_doc_id']/stats['total']*100:.1f}%)")
            logger.info(f"  title 보유: {stats['has_title']}/{stats['total']} ({stats['has_title']/stats['total']*100:.1f}%)")
            logger.info(f"  완전한 메타데이터: {stats['complete']}/{stats['total']} ({stats['complete']/stats['total']*100:.1f}%)")

# 전체 요약
total_results = len(results)
total_complete = sum(stats['complete'] for stats in metadata_stats.values())
overall_completeness = (total_complete / total_results * 100) if total_results > 0 else 0

logger.info("\n" + "=" * 60)
logger.info("전체 요약")
logger.info("=" * 60)
logger.info(f"총 검색 결과: {total_results}개")
logger.info(f"완전한 메타데이터를 가진 결과: {total_complete}개")
logger.info(f"전체 완전성: {overall_completeness:.1f}%")

if overall_completeness >= 90:
    logger.info("✅ 메타데이터 개선 성공! 90% 이상 완전성 달성")
elif overall_completeness >= 70:
    logger.info("⚠️  메타데이터 개선 부분 성공. 추가 개선 필요")
else:
    logger.info("❌ 메타데이터 개선 필요. 많은 필드가 누락됨")

