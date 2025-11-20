#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
하이브리드 청킹 검색 품질 검증 테스트

다양한 청킹 전략으로 생성된 청크를 사용한 검색 품질을 비교합니다.
"""
import sys
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.embeddings import SentenceEmbedder


def test_search_with_chunking_strategy(
    db_path: str,
    query: str,
    chunking_strategy: str,
    k: int = 5
) -> List[Dict[str, Any]]:
    """특정 청킹 전략으로 검색 테스트"""
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    
    try:
        # 쿼리 벡터 생성
        embedder = SentenceEmbedder("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        query_vector = embedder.encode([query])[0]
        
        # 해당 청킹 전략의 청크만 검색
        cursor = conn.execute("""
            SELECT 
                tc.id as chunk_id,
                tc.text,
                tc.chunking_strategy,
                tc.chunk_size_category,
                tc.chunk_group_id,
                tc.source_type,
                tc.source_id,
                e.vector
            FROM text_chunks tc
            JOIN embeddings e ON tc.id = e.chunk_id
            WHERE tc.chunking_strategy = ?
            ORDER BY tc.id
            LIMIT 1000
        """, (chunking_strategy,))
        
        chunks = cursor.fetchall()
        if not chunks:
            print(f"⚠️  {chunking_strategy} 전략의 청크를 찾을 수 없습니다.")
            return []
        
        # 유사도 계산
        results = []
        for chunk in chunks:
            import numpy as np
            vector_bytes = chunk['vector']
            chunk_vector = np.frombuffer(vector_bytes, dtype=np.float32)
            
            # 코사인 유사도 계산
            similarity = np.dot(query_vector, chunk_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector)
            )
            
            results.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'similarity': float(similarity),
                'chunking_strategy': chunk['chunking_strategy'],
                'chunk_size_category': chunk['chunk_size_category'],
                'chunk_group_id': chunk['chunk_group_id'],
                'source_type': chunk['source_type'],
                'source_id': chunk['source_id'],
            })
        
        # 유사도 기준 정렬
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 하이브리드 청킹의 경우 그룹별 중복 제거 (최고 점수만 유지)
        if chunking_strategy == 'hybrid':
            seen_groups = {}
            deduplicated_results = []
            for result in results:
                group_id = result.get('chunk_group_id')
                if group_id and group_id != 'N/A':
                    if group_id not in seen_groups:
                        seen_groups[group_id] = result
                        deduplicated_results.append(result)
                    elif result['similarity'] > seen_groups[group_id]['similarity']:
                        # 더 높은 점수로 교체
                        idx = deduplicated_results.index(seen_groups[group_id])
                        deduplicated_results[idx] = result
                        seen_groups[group_id] = result
                else:
                    # 그룹 ID가 없으면 그대로 추가
                    deduplicated_results.append(result)
            results = deduplicated_results[:k]
        else:
            results = results[:k]
        
        return results
        
    finally:
        conn.close()


def compare_search_quality(
    db_path: str,
    test_queries: List[str],
    strategies: List[str] = ['standard', 'dynamic', 'hybrid']
):
    """검색 품질 비교"""
    print(f"\n{'='*80}")
    print("하이브리드 청킹 검색 품질 검증")
    print(f"{'='*80}\n")
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"검색 쿼리: {query}")
        print(f"{'='*80}\n")
        
        all_results = {}
        
        for strategy in strategies:
            print(f"\n[{strategy.upper()} 전략]")
            print("-" * 80)
            
            results = test_search_with_chunking_strategy(
                db_path, query, strategy, k=5
            )
            
            if not results:
                print("  검색 결과 없음")
                continue
            
            all_results[strategy] = results
            
            print(f"  검색 결과: {len(results)}개")
            for i, result in enumerate(results, 1):
                print(f"\n  결과 {i}:")
                print(f"    유사도: {result['similarity']:.4f}")
                print(f"    크기 카테고리: {result.get('chunk_size_category', 'N/A')}")
                print(f"    그룹 ID: {result.get('chunk_group_id', 'N/A')[:20] if result.get('chunk_group_id') else 'N/A'}")
                print(f"    텍스트 미리보기: {result['text'][:100]}...")
        
        # 전략별 비교 요약
        print(f"\n{'='*80}")
        print("전략별 비교 요약")
        print(f"{'='*80}")
        for strategy in strategies:
            if strategy in all_results and all_results[strategy]:
                avg_similarity = sum(r['similarity'] for r in all_results[strategy]) / len(all_results[strategy])
                max_similarity = max(r['similarity'] for r in all_results[strategy])
                print(f"\n{strategy.upper()}:")
                print(f"  평균 유사도: {avg_similarity:.4f}")
                print(f"  최고 유사도: {max_similarity:.4f}")
                print(f"  결과 수: {len(all_results[strategy])}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='하이브리드 청킹 검색 품질 검증')
    parser.add_argument('--db', default='data/lawfirm_v2.db', help='데이터베이스 경로')
    parser.add_argument('--query', help='테스트할 검색 쿼리')
    parser.add_argument('--queries-file', help='테스트 쿼리 파일 (한 줄에 하나씩)')
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
        sys.exit(1)
    
    # 테스트 쿼리 준비
    if args.queries_file:
        with open(args.queries_file, 'r', encoding='utf-8') as f:
            test_queries = [line.strip() for line in f if line.strip()]
    elif args.query:
        test_queries = [args.query]
    else:
        # 기본 테스트 쿼리
        test_queries = [
            "전세금 반환 보증에 대해 알려주세요",
            "계약서 해지 조건",
            "손해배상 책임 범위",
        ]
    
    compare_search_quality(str(db_path), test_queries)
    
    print(f"\n{'='*80}")
    print("✅ 검색 품질 검증 완료!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

