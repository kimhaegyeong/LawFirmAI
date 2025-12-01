#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
참조 자료 품질 분석 스크립트

검색 결과의 참조 자료 품질을 분석합니다.
"""
import sys
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# 프로젝트 루트를 sys.path에 추가
from scripts.utils.path_utils import setup_project_path
setup_project_path()

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from scripts.utils.text_utils import extract_keywords


def analyze_reference_quality(
    query: str,
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """참조 자료 품질 분석"""
    analysis = {
        'total_results': len(results),
        'by_strategy': {},
        'by_category': {},
        'quality_metrics': {},
        'issues': []
    }
    
    if not results:
        analysis['issues'].append('검색 결과가 없습니다.')
        return analysis
    
    # 키워드 추출
    keywords = extract_keywords(query)
    
    # 전략별 분석
    for result in results:
        strategy = result.get('metadata', {}).get('chunking_strategy') or 'unknown'
        if strategy not in analysis['by_strategy']:
            analysis['by_strategy'][strategy] = {
                'count': 0,
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 1.0
            }
        
        strategy_data = analysis['by_strategy'][strategy]
        strategy_data['count'] += 1
        similarity = result.get('similarity', 0.0)
        strategy_data['avg_similarity'] += similarity
        strategy_data['max_similarity'] = max(strategy_data['max_similarity'], similarity)
        strategy_data['min_similarity'] = min(strategy_data['min_similarity'], similarity)
    
    # 평균 계산
    for strategy_data in analysis['by_strategy'].values():
        if strategy_data['count'] > 0:
            strategy_data['avg_similarity'] /= strategy_data['count']
    
    # 카테고리별 분석
    for result in results:
        category = result.get('metadata', {}).get('chunk_size_category') or 'unknown'
        if category not in analysis['by_category']:
            analysis['by_category'][category] = {
                'count': 0,
                'avg_similarity': 0.0
            }
        
        category_data = analysis['by_category'][category]
        category_data['count'] += 1
        category_data['avg_similarity'] += result.get('similarity', 0.0)
    
    for category_data in analysis['by_category'].values():
        if category_data['count'] > 0:
            category_data['avg_similarity'] /= category_data['count']
    
    # 품질 메트릭
    similarities = [r.get('similarity', 0.0) for r in results]
    analysis['quality_metrics'] = {
        'avg_similarity': sum(similarities) / len(similarities) if similarities else 0.0,
        'max_similarity': max(similarities) if similarities else 0.0,
        'min_similarity': min(similarities) if similarities else 0.0,
        'high_quality_count': sum(1 for s in similarities if s >= 0.7),
        'medium_quality_count': sum(1 for s in similarities if 0.5 <= s < 0.7),
        'low_quality_count': sum(1 for s in similarities if s < 0.5)
    }
    
    # 키워드 매칭 분석
    keyword_matches = []
    for result in results:
        text = result.get('text', '')
        matched_keywords = [kw for kw in keywords if kw in text]
        keyword_matches.append({
            'matched_count': len(matched_keywords),
            'total_keywords': len(keywords),
            'match_ratio': len(matched_keywords) / len(keywords) if keywords else 0.0
        })
    
    analysis['keyword_analysis'] = {
        'avg_match_ratio': sum(m['match_ratio'] for m in keyword_matches) / len(keyword_matches) if keyword_matches else 0.0,
        'full_match_count': sum(1 for m in keyword_matches if m['match_ratio'] >= 0.8)
    }
    
    # 문제점 식별
    if analysis['quality_metrics']['avg_similarity'] < 0.6:
        analysis['issues'].append(f'평균 유사도가 낮습니다: {analysis["quality_metrics"]["avg_similarity"]:.4f}')
    
    if analysis['quality_metrics']['low_quality_count'] > len(results) * 0.5:
        analysis['issues'].append(f'저품질 결과가 많습니다: {analysis["quality_metrics"]["low_quality_count"]}/{len(results)}')
    
    if analysis['keyword_analysis']['avg_match_ratio'] < 0.5:
        analysis['issues'].append(f'키워드 매칭률이 낮습니다: {analysis["keyword_analysis"]["avg_match_ratio"]:.4f}')
    
    return analysis




def print_analysis_report(query: str, analysis: Dict[str, Any], results: List[Dict[str, Any]]):
    """분석 결과 리포트 출력"""
    print("\n" + "="*80)
    print("참조 자료 품질 분석 리포트")
    print("="*80)
    print(f"\n검색 쿼리: {query}")
    print(f"총 검색 결과: {analysis['total_results']}개\n")
    
    # 품질 메트릭
    print("📊 품질 메트릭")
    print("-" * 80)
    metrics = analysis['quality_metrics']
    print(f"  평균 유사도: {metrics['avg_similarity']:.4f}")
    print(f"  최고 유사도: {metrics['max_similarity']:.4f}")
    print(f"  최저 유사도: {metrics['min_similarity']:.4f}")
    print(f"  고품질 (≥0.7): {metrics['high_quality_count']}개")
    print(f"  중품질 (0.5-0.7): {metrics['medium_quality_count']}개")
    print(f"  저품질 (<0.5): {metrics['low_quality_count']}개")
    
    # 전략별 분석
    if analysis['by_strategy']:
        print("\n📈 청킹 전략별 분석")
        print("-" * 80)
        for strategy, data in analysis['by_strategy'].items():
            strategy_name = (strategy or 'unknown').upper()
            print(f"\n  [{strategy_name}]")
            print(f"    결과 수: {data['count']}개")
            print(f"    평균 유사도: {data['avg_similarity']:.4f}")
            print(f"    최고 유사도: {data['max_similarity']:.4f}")
            print(f"    최저 유사도: {data['min_similarity']:.4f}")
    
    # 카테고리별 분석
    if analysis['by_category']:
        print("\n📦 크기 카테고리별 분석")
        print("-" * 80)
        for category, data in analysis['by_category'].items():
            print(f"  {category}: {data['count']}개 (평균 유사도: {data['avg_similarity']:.4f})")
    
    # 키워드 분석
    if 'keyword_analysis' in analysis:
        print("\n🔑 키워드 매칭 분석")
        print("-" * 80)
        kw_analysis = analysis['keyword_analysis']
        print(f"  평균 키워드 매칭률: {kw_analysis['avg_match_ratio']:.4f}")
        print(f"  완전 매칭 결과: {kw_analysis['full_match_count']}개")
    
    # 문제점
    if analysis['issues']:
        print("\n⚠️  발견된 문제점")
        print("-" * 80)
        for i, issue in enumerate(analysis['issues'], 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✅ 특별한 문제점이 발견되지 않았습니다.")
    
    # 상위 결과 샘플
    print("\n📋 상위 결과 샘플 (상위 5개)")
    print("-" * 80)
    for i, result in enumerate(results[:5], 1):
        print(f"\n  결과 {i}:")
        print(f"    유사도: {result.get('similarity', 0):.4f}")
        print(f"    청킹 전략: {result.get('metadata', {}).get('chunking_strategy', 'N/A')}")
        print(f"    크기 카테고리: {result.get('metadata', {}).get('chunk_size_category', 'N/A')}")
        print(f"    텍스트 미리보기: {result.get('text', '')[:150]}...")
    
    print("\n" + "="*80)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='참조 자료 품질 분석')
    parser.add_argument('--query', default='전세금 반환 보증에 대해 알려주세요', help='검색 쿼리')
    parser.add_argument('--db', default='data/lawfirm_v2.db', help='데이터베이스 경로')
    parser.add_argument('--k', type=int, default=10, help='검색 결과 수')
    
    args = parser.parse_args()
    
    # 검색 엔진 초기화
    engine = SemanticSearchEngineV2(db_path=args.db)
    
    if not engine.is_available():
        print("❌ 검색 엔진을 사용할 수 없습니다.")
        return
    
    # 검색 수행
    print(f"검색 중: {args.query}")
    results = engine.search(
        query=args.query,
        k=args.k,
        similarity_threshold=0.4,
        deduplicate_by_group=True
    )
    
    if not results:
        print("⚠️  검색 결과가 없습니다.")
        return
    
    # 품질 분석
    analysis = analyze_reference_quality(args.query, results)
    
    # 리포트 출력
    print_analysis_report(args.query, analysis, results)


if __name__ == '__main__':
    main()

