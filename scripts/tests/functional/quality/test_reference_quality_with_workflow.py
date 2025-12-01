#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph 워크플로우를 통한 참조 자료 품질 분석

실제 워크플로우를 실행하고 retrieved_docs의 품질을 분석합니다.
"""
import sys
import asyncio
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 환경 변수 설정
os.environ['USE_STREAMING_MODE'] = 'false'

# 로깅 설정
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_workflow_and_analyze(query: str):
    """워크플로우 테스트 및 참조 자료 품질 분석"""
    try:
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
        
        # 설정 로드
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False
        
        # 서비스 초기화
        service = LangGraphWorkflowService(config)
        
        # 질의 처리
        logger.info(f"질의 처리 중: {query}")
        result = await service.process_query(
            query=query,
            session_id="quality_test",
            enable_checkpoint=False
        )
        
        # 참조 자료 분석
        retrieved_docs = result.get("retrieved_docs", [])
        sources = result.get("sources", [])
        sources_detail = result.get("sources_detail", [])
        
        print("\n" + "="*80)
        print("참조 자료 품질 분석 리포트")
        print("="*80)
        print(f"\n검색 쿼리: {query}")
        print(f"답변 길이: {len(str(result.get('answer', '')))}자\n")
        
        # retrieved_docs 분석
        print("📚 Retrieved Docs 분석")
        print("-" * 80)
        print(f"총 검색 결과: {len(retrieved_docs)}개")
        
        if retrieved_docs:
            # 타입별 분포
            type_counts = {}
            for doc in retrieved_docs:
                doc_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "unknown")
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            print(f"\n타입별 분포:")
            for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {doc_type}: {count}개")
            
            # 유사도 분석
            similarities = []
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    score = doc.get("score") or doc.get("similarity") or doc.get("relevance_score", 0.0)
                    similarities.append(score)
            
            if similarities:
                print(f"\n유사도 통계:")
                print(f"  평균: {sum(similarities) / len(similarities):.4f}")
                print(f"  최고: {max(similarities):.4f}")
                print(f"  최저: {min(similarities):.4f}")
                print(f"  고품질 (≥0.7): {sum(1 for s in similarities if s >= 0.7)}개")
                print(f"  중품질 (0.5-0.7): {sum(1 for s in similarities if 0.5 <= s < 0.7)}개")
                print(f"  저품질 (<0.5): {sum(1 for s in similarities if s < 0.5)}개")
            
            # 청킹 전략 분석
            strategy_counts = {}
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    strategy = doc.get("metadata", {}).get("chunking_strategy") or "unknown"
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            if strategy_counts:
                print(f"\n청킹 전략 분포:")
                for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {strategy}: {count}개")
            
            # 상위 결과 샘플
            print(f"\n상위 결과 샘플 (상위 5개):")
            for i, doc in enumerate(retrieved_docs[:5], 1):
                if isinstance(doc, dict):
                    doc_type = doc.get("type") or doc.get("source_type", "unknown")
                    score = (doc.get("score") or 
                            doc.get("similarity") or 
                            doc.get("relevance_score") or 
                            doc.get("hybrid_score") or 
                            doc.get("metadata", {}).get("score") or
                            doc.get("metadata", {}).get("similarity") or
                            0.0)
                    text_preview = (doc.get("text") or doc.get("content", ""))[:100]
                    print(f"\n  {i}. [{doc_type}] (유사도: {score:.4f})")
                    print(f"     {text_preview}...")
        else:
            print("  ⚠️  검색 결과가 없습니다.")
        
        # sources 분석
        print(f"\n📋 Sources 분석")
        print("-" * 80)
        print(f"총 소스 수: {len(sources)}개")
        if sources:
            for i, source in enumerate(sources[:5], 1):
                if isinstance(source, dict):
                    name = source.get("name") or source.get("title", "제목 없음")
                    print(f"  {i}. {name}")
        
        # sources_detail 분석
        print(f"\n📄 Sources Detail 분석")
        print("-" * 80)
        print(f"총 상세 소스 수: {len(sources_detail)}개")
        if sources_detail:
            for i, detail in enumerate(sources_detail[:5], 1):
                if isinstance(detail, dict):
                    name = detail.get("name") or detail.get("title", "제목 없음")
                    source_type = detail.get("type") or detail.get("source_type", "unknown")
                    print(f"  {i}. [{source_type}] {name}")
        
        # 품질 평가
        print(f"\n🎯 품질 평가")
        print("-" * 80)
        
        issues = []
        if not retrieved_docs:
            issues.append("검색 결과가 없습니다.")
        elif len(retrieved_docs) < 3:
            issues.append(f"검색 결과가 너무 적습니다: {len(retrieved_docs)}개")
        
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            if avg_sim < 0.6:
                issues.append(f"평균 유사도가 낮습니다: {avg_sim:.4f}")
            if sum(1 for s in similarities if s < 0.5) > len(similarities) * 0.3:
                issues.append("저품질 결과가 많습니다.")
        
        if not sources:
            issues.append("Sources가 생성되지 않았습니다.")
        
        if issues:
            print("⚠️  발견된 문제점:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("✅ 특별한 문제점이 발견되지 않았습니다.")
        
        print("\n" + "="*80)
        
        return result
        
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        raise


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='워크플로우 참조 자료 품질 분석')
    parser.add_argument('--query', default='전세금 반환 보증에 대해 알려주세요', help='검색 쿼리')
    
    args = parser.parse_args()
    
    result = asyncio.run(test_workflow_and_analyze(args.query))
    
    print("\n✅ 분석 완료!")


if __name__ == '__main__':
    main()

