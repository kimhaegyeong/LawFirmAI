# -*- coding: utf-8 -*-
"""
전체 워크플로우 테스트 스크립트
- LangGraph 워크플로우 전체 테스트
- 검색 결과 품질 확인
- 성능 모니터링
"""

import sys
import os
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_full_workflow(query: str):
    """전체 워크플로우 테스트"""
    logger.info("="*80)
    logger.info("전체 워크플로우 테스트")
    logger.info("="*80)
    logger.info(f"\n📋 질의: {query}\n")
    
    start_time = time.time()
    
    try:
        # 환경 변수 설정
        if not os.getenv('USE_EXTERNAL_VECTOR_STORE'):
            os.environ['USE_EXTERNAL_VECTOR_STORE'] = 'true'
        
        if not os.getenv('EXTERNAL_VECTOR_STORE_BASE_PATH'):
            possible_paths = [
                "data/vector_store/v2.0.0-dynamic-dynamic-ivfpq",
                "./data/vector_store/v2.0.0-dynamic-dynamic-ivfpq",
                str(project_root / "data" / "vector_store" / "v2.0.0-dynamic-dynamic-ivfpq")
            ]
            for path in possible_paths:
                if Path(path).exists():
                    os.environ['EXTERNAL_VECTOR_STORE_BASE_PATH'] = path
                    logger.info(f"✅ IndexIVFPQ 인덱스 경로: {path}")
                    break
        
        # LangGraph 설정 및 서비스 초기화
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
        
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False
        
        logger.info("1️⃣  LangGraph 서비스 초기화 중...")
        init_start = time.time()
        service = LangGraphWorkflowService(config)
        init_time = time.time() - init_start
        logger.info(f"   ✅ 초기화 완료 ({init_time:.2f}초)")
        
        # 검색 엔진 정보 확인
        if hasattr(service, 'workflow') and hasattr(service.workflow, 'semantic_search'):
            search_engine = service.workflow.semantic_search
            if search_engine and hasattr(search_engine, 'index'):
                index_type = type(search_engine.index).__name__ if search_engine.index else 'None'
                index_size = search_engine.index.ntotal if search_engine.index else 0
                logger.info(f"   📊 검색 엔진: {index_type} ({index_size:,} vectors)")
        
        # 질의 처리
        logger.info("\n2️⃣  질의 처리 중...")
        query_start = time.time()
        
        result = await service.process_query(query)
        
        query_time = time.time() - query_start
        total_time = time.time() - start_time
        
        logger.info(f"   ✅ 질의 처리 완료 ({query_time:.2f}초)")
        
        # 결과 분석
        logger.info("\n3️⃣  결과 분석")
        logger.info("="*80)
        
        if result:
            # 검색 결과 확인
            retrieved_docs = result.get('retrieved_docs', [])
            logger.info(f"\n📊 검색 결과 통계:")
            logger.info(f"   검색 결과 수: {len(retrieved_docs)}개")
            
            if retrieved_docs:
                # 메타데이터 완전성 확인
                metadata_stats = {
                    'has_doc_id': 0,
                    'has_casenames': 0,
                    'has_court': 0,
                    'has_org': 0,
                    'has_statute_name': 0,
                    'has_article_no': 0,
                    'has_title': 0,
                    'complete_metadata': 0,
                    'text_lengths': []
                }
                
                source_type_counts = {}
                
                for i, doc in enumerate(retrieved_docs[:10], 1):
                    source_type = doc.get('source_type') or doc.get('type') or doc.get('metadata', {}).get('source_type')
                    if source_type:
                        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
                    
                    # 메타데이터 확인
                    metadata = doc.get('metadata', {})
                    doc_id = doc.get('doc_id') or metadata.get('doc_id')
                    casenames = doc.get('casenames') or metadata.get('casenames')
                    court = doc.get('court') or metadata.get('court')
                    org = doc.get('org') or metadata.get('org')
                    statute_name = doc.get('statute_name') or metadata.get('statute_name') or doc.get('law_name')
                    article_no = doc.get('article_no') or metadata.get('article_no')
                    title = doc.get('title') or metadata.get('title')
                    
                    if doc_id:
                        metadata_stats['has_doc_id'] += 1
                    if casenames:
                        metadata_stats['has_casenames'] += 1
                    if court:
                        metadata_stats['has_court'] += 1
                    if org:
                        metadata_stats['has_org'] += 1
                    if statute_name:
                        metadata_stats['has_statute_name'] += 1
                    if article_no:
                        metadata_stats['has_article_no'] += 1
                    if title:
                        metadata_stats['has_title'] += 1
                    
                    # source_type별 완전성 확인
                    is_complete = False
                    if source_type == 'case_paragraph':
                        is_complete = bool(doc_id and casenames and court)
                    elif source_type == 'decision_paragraph':
                        is_complete = bool(doc_id and org)
                    elif source_type == 'statute_article':
                        is_complete = bool(statute_name and article_no)
                    elif source_type == 'interpretation_paragraph':
                        is_complete = bool(doc_id and org and title)
                    else:
                        is_complete = bool(doc_id)
                    
                    if is_complete:
                        metadata_stats['complete_metadata'] += 1
                    
                    # 텍스트 길이 확인
                    text = doc.get('text') or doc.get('content') or ''
                    if text:
                        metadata_stats['text_lengths'].append(len(text))
                
                logger.info(f"\n📋 메타데이터 완전성:")
                logger.info(f"   doc_id: {metadata_stats['has_doc_id']}/{len(retrieved_docs)} ({metadata_stats['has_doc_id']/len(retrieved_docs)*100:.1f}%)")
                if metadata_stats['has_casenames'] > 0:
                    logger.info(f"   casenames: {metadata_stats['has_casenames']}/{len(retrieved_docs)}")
                if metadata_stats['has_court'] > 0:
                    logger.info(f"   court: {metadata_stats['has_court']}/{len(retrieved_docs)}")
                if metadata_stats['has_org'] > 0:
                    logger.info(f"   org: {metadata_stats['has_org']}/{len(retrieved_docs)}")
                if metadata_stats['has_statute_name'] > 0:
                    logger.info(f"   statute_name: {metadata_stats['has_statute_name']}/{len(retrieved_docs)}")
                if metadata_stats['has_article_no'] > 0:
                    logger.info(f"   article_no: {metadata_stats['has_article_no']}/{len(retrieved_docs)}")
                if metadata_stats['has_title'] > 0:
                    logger.info(f"   title: {metadata_stats['has_title']}/{len(retrieved_docs)}")
                logger.info(f"   완전한 메타데이터: {metadata_stats['complete_metadata']}/{len(retrieved_docs)} ({metadata_stats['complete_metadata']/len(retrieved_docs)*100:.1f}%)")
                
                logger.info(f"\n📊 source_type 분포:")
                for stype, count in source_type_counts.items():
                    logger.info(f"   {stype}: {count}개")
                
                if metadata_stats['text_lengths']:
                    avg_length = sum(metadata_stats['text_lengths']) / len(metadata_stats['text_lengths'])
                    min_length = min(metadata_stats['text_lengths'])
                    max_length = max(metadata_stats['text_lengths'])
                    logger.info(f"\n📝 텍스트 길이:")
                    logger.info(f"   평균: {avg_length:.0f}자")
                    logger.info(f"   최소: {min_length}자")
                    logger.info(f"   최대: {max_length}자")
                    logger.info(f"   100자 미만: {sum(1 for l in metadata_stats['text_lengths'] if l < 100)}개")
            
            # 최종 답변 확인
            final_answer = result.get('final_answer') or result.get('answer') or result.get('response')
            if final_answer:
                logger.info(f"\n💬 최종 답변:")
                logger.info(f"   길이: {len(final_answer)}자")
                logger.info(f"   미리보기: {final_answer[:200]}...")
            else:
                logger.warning("   ⚠️  최종 답변이 없습니다")
        
        # 성능 통계
        logger.info(f"\n⏱️  성능 통계:")
        logger.info(f"   초기화 시간: {init_time:.2f}초")
        logger.info(f"   질의 처리 시간: {query_time:.2f}초")
        logger.info(f"   총 소요 시간: {total_time:.2f}초")
        
        logger.info("\n" + "="*80)
        logger.info("테스트 완료")
        logger.info("="*80)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        raise


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='전체 워크플로우 테스트')
    parser.add_argument('query', nargs='?', default='임대차 보증금 반환', help='테스트할 질의')
    
    args = parser.parse_args()
    
    try:
        result = asyncio.run(test_full_workflow(args.query))
        return 0
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"\n\n❌ 테스트 실패: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

