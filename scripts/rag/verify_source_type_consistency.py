#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS 인덱스와 DB의 source_type 일치성 검증 스크립트

Usage:
    python scripts/rag/verify_source_type_consistency.py
    python scripts/rag/verify_source_type_consistency.py --sample-size 1000
    python scripts/rag/verify_source_type_consistency.py --mlflow-run-id <run_id>
"""

import sys
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import defaultdict

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "lawfirm_langgraph"))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not available. Install with: pip install faiss-cpu")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not available. Install with: pip install mlflow")


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """데이터베이스 연결"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_mlflow_index_via_engine(run_id: Optional[str] = None) -> Tuple[Optional[faiss.Index], Optional[List[int]], Optional[Dict]]:
    """SemanticSearchEngineV2를 통해 FAISS 인덱스 로드"""
    try:
        from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
        from lawfirm_langgraph.core.utils.config import Config
        
        # Config 설정
        config = Config()
        if run_id:
            os.environ['MLFLOW_RUN_ID'] = run_id
        
        # SemanticSearchEngineV2 초기화 (인덱스 로드)
        print("🔄 Initializing SemanticSearchEngineV2 to load index...")
        engine = SemanticSearchEngineV2(
            db_path=config.database_path,
            use_mlflow_index=True
        )
        
        if engine.index is None:
            print("❌ Failed to load index from SemanticSearchEngineV2")
            return None, None, None
        
        index = engine.index
        chunk_ids = engine._chunk_ids if hasattr(engine, '_chunk_ids') and engine._chunk_ids else None
        
        print(f"✅ Loaded FAISS index via SemanticSearchEngineV2: {index.ntotal} vectors")
        if chunk_ids:
            print(f"✅ Loaded chunk_ids: {len(chunk_ids)} chunks")
        else:
            print("⚠️  chunk_ids not available, will use sequential IDs")
            chunk_ids = list(range(index.ntotal))
        
        # version_info는 engine에서 가져올 수 없으므로 None
        version_info = None
        
        return index, chunk_ids, version_info
        
    except Exception as e:
        print(f"❌ Failed to load index via SemanticSearchEngineV2: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def load_mlflow_index(run_id: Optional[str] = None) -> Tuple[Optional[faiss.Index], Optional[List[int]], Optional[Dict]]:
    """MLflow에서 FAISS 인덱스 로드 (fallback: SemanticSearchEngineV2 사용)"""
    if not MLFLOW_AVAILABLE:
        print("⚠️  MLflow not available, using SemanticSearchEngineV2")
        return load_mlflow_index_via_engine(run_id)
    
    try:
        # MLflow tracking URI 설정
        mlflow_uri = str(project_root / "mlflow" / "mlruns")
        os.environ['MLFLOW_TRACKING_URI'] = f"file:///{mlflow_uri.replace(chr(92), '/')}"
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        
        # run_id가 없으면 프로덕션 run 찾기, 없으면 최근 run 사용
        if not run_id:
            try:
                client = mlflow.tracking.MlflowClient()
                # 먼저 프로덕션 run 찾기
                runs = client.search_runs(
                    experiment_ids=["0"],
                    filter_string="tags.status='production_ready'",
                    max_results=1,
                    order_by=["start_time DESC"]
                )
                if runs:
                    run_id = runs[0].info.run_id
                    print(f"✅ Found production run: {run_id}")
                else:
                    # 프로덕션 run이 없으면 최근 run 사용
                    runs = client.search_runs(
                        experiment_ids=["0"],
                        max_results=1,
                        order_by=["start_time DESC"]
                    )
                    if runs:
                        run_id = runs[0].info.run_id
                        print(f"⚠️  No production run found. Using most recent run: {run_id}")
            except Exception as e:
                print(f"⚠️  Failed to search MLflow runs: {e}")
                print("   Using SemanticSearchEngineV2 instead...")
                return load_mlflow_index_via_engine(None)
        
        if not run_id:
            print("⚠️  No run_id specified, using SemanticSearchEngineV2...")
            return load_mlflow_index_via_engine(None)
        
        # FAISS 인덱스 로드
        try:
            index_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="faiss_index"
            )
        except Exception as e:
            print(f"⚠️  Failed to download artifacts from MLflow: {e}")
            print("   Using SemanticSearchEngineV2 instead...")
            return load_mlflow_index_via_engine(run_id)
        
        if not os.path.exists(index_path):
            print(f"⚠️  FAISS index not found at: {index_path}")
            print("   Using SemanticSearchEngineV2 instead...")
            return load_mlflow_index_via_engine(run_id)
        
        index = faiss.read_index(index_path)
        print(f"✅ Loaded FAISS index: {index.ntotal} vectors")
        
        # chunk_ids 로드
        chunk_ids_path = os.path.join(os.path.dirname(index_path), "chunk_ids.npy")
        if os.path.exists(chunk_ids_path):
            import numpy as np
            chunk_ids = np.load(chunk_ids_path).tolist()
            print(f"✅ Loaded chunk_ids: {len(chunk_ids)} chunks")
        else:
            print("⚠️  chunk_ids.npy not found. Using sequential IDs")
            chunk_ids = list(range(index.ntotal))
        
        # version_info 로드
        version_info = None
        try:
            version_info = mlflow.artifacts.load_dict(f"runs:/{run_id}/version_info.json")
            print(f"✅ Loaded version_info.json")
        except Exception as e:
            print(f"⚠️  Failed to load version_info.json: {e}")
        
        return index, chunk_ids, version_info
        
    except Exception as e:
        print(f"⚠️  Failed to load MLflow index: {e}")
        print("   Using SemanticSearchEngineV2 instead...")
        return load_mlflow_index_via_engine(run_id)


def verify_source_type_consistency(
    db_path: str,
    chunk_ids: List[int],
    sample_size: int = 1000
) -> Dict[str, any]:
    """source_type 일치성 검증"""
    conn = get_db_connection(db_path)
    
    # 샘플링
    if len(chunk_ids) > sample_size:
        import random
        sampled_chunk_ids = random.sample(chunk_ids, sample_size)
        print(f"📊 Sampling {sample_size} chunks from {len(chunk_ids)} total chunks")
    else:
        sampled_chunk_ids = chunk_ids
        print(f"📊 Verifying all {len(chunk_ids)} chunks")
    
    # DB에서 source_type 조회
    results = {
        'total_checked': len(sampled_chunk_ids),
        'found_in_db': 0,
        'not_found_in_db': 0,
        'source_type_distribution': defaultdict(int),
        'missing_chunks': [],
        'type_mismatches': []
    }
    
    # 배치로 조회 (성능 최적화)
    batch_size = 100
    for i in range(0, len(sampled_chunk_ids), batch_size):
        batch_ids = sampled_chunk_ids[i:i+batch_size]
        placeholders = ','.join(['?'] * len(batch_ids))
        
        cursor = conn.execute(
            f"SELECT id, source_type FROM text_chunks WHERE id IN ({placeholders})",
            batch_ids
        )
        rows = cursor.fetchall()
        
        found_ids = {row['id'] for row in rows}
        results['found_in_db'] += len(found_ids)
        results['not_found_in_db'] += len(batch_ids) - len(found_ids)
        
        for row in rows:
            chunk_id = row['id']
            source_type = row['source_type']
            results['source_type_distribution'][source_type] += 1
        
        # DB에 없는 chunk_id 기록
        for chunk_id in batch_ids:
            if chunk_id not in found_ids:
                results['missing_chunks'].append(chunk_id)
    
    conn.close()
    
    return results


def analyze_type_distribution(results: Dict[str, any]) -> None:
    """타입 분포 분석 및 출력"""
    print("\n" + "="*80)
    print("📊 source_type 분포 분석")
    print("="*80)
    
    print(f"\n✅ DB에서 찾은 chunk: {results['found_in_db']}/{results['total_checked']}")
    print(f"❌ DB에서 찾지 못한 chunk: {results['not_found_in_db']}/{results['total_checked']}")
    
    if results['not_found_in_db'] > 0:
        print(f"\n⚠️  DB에 없는 chunk_id 샘플 (최대 10개):")
        for chunk_id in results['missing_chunks'][:10]:
            print(f"   - chunk_id: {chunk_id}")
        if len(results['missing_chunks']) > 10:
            print(f"   ... (총 {len(results['missing_chunks'])}개)")
    
    print(f"\n📈 source_type 분포:")
    type_dist = results['source_type_distribution']
    total = sum(type_dist.values())
    
    for source_type, count in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"   - {source_type}: {count}개 ({percentage:.1f}%)")
    
    # 타입별 검색 가능성 분석
    print(f"\n🔍 타입별 검색 가능성 분석:")
    type_mapping = {
        'statute_article': '법령 조문',
        'case_paragraph': '판례',
        'decision_paragraph': '결정례',
        'interpretation_paragraph': '해석례'
    }
    
    for source_type, korean_name in type_mapping.items():
        count = type_dist.get(source_type, 0)
        if count == 0:
            print(f"   ⚠️  {korean_name} ({source_type}): 0개 - 이 타입으로 검색하면 결과가 없을 수 있음")
        elif count < 10:
            print(f"   ⚠️  {korean_name} ({source_type}): {count}개 - 매우 적음, 검색 결과가 제한적일 수 있음")
        else:
            print(f"   ✅ {korean_name} ({source_type}): {count}개")


def check_type_search_feasibility(results: Dict[str, any]) -> None:
    """타입별 검색 가능성 확인"""
    print("\n" + "="*80)
    print("🔍 타입별 검색 가능성 확인")
    print("="*80)
    
    type_dist = results['source_type_distribution']
    required_types = ['statute_article', 'case_paragraph', 'decision_paragraph', 'interpretation_paragraph']
    
    print(f"\n타입별 검색 요청 시 예상 결과:")
    for req_type in required_types:
        count = type_dist.get(req_type, 0)
        total = results['found_in_db']
        percentage = (count / total * 100) if total > 0 else 0
        
        if count == 0:
            print(f"   ❌ {req_type}: 0개 - 검색 시 모든 결과가 필터링됨")
        elif count < total * 0.01:  # 1% 미만
            print(f"   ⚠️  {req_type}: {count}개 ({percentage:.2f}%) - 매우 적음, 대부분 필터링될 가능성")
        elif count < total * 0.05:  # 5% 미만
            print(f"   ⚠️  {req_type}: {count}개 ({percentage:.2f}%) - 적음, 일부 필터링될 가능성")
        else:
            print(f"   ✅ {req_type}: {count}개 ({percentage:.2f}%) - 충분함")


def main():
    parser = argparse.ArgumentParser(description='FAISS 인덱스와 DB의 source_type 일치성 검증')
    parser.add_argument('--sample-size', type=int, default=1000, help='검증할 chunk 샘플 수 (기본값: 1000)')
    parser.add_argument('--mlflow-run-id', type=str, default=None, help='MLflow run ID (없으면 프로덕션 run 사용)')
    parser.add_argument('--db-path', type=str, default=None, help='데이터베이스 경로 (기본값: .env에서 읽음)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FAISS 인덱스와 DB의 source_type 일치성 검증")
    print("="*80)
    
    # 데이터베이스 경로 확인
    if args.db_path:
        db_path = args.db_path
    else:
        # .env에서 DATABASE_PATH 읽기
        try:
            from dotenv import load_dotenv
            load_dotenv()
            db_path = os.getenv("DATABASE_PATH")
            if not db_path:
                # 기본 경로
                db_path = str(project_root / "data" / "lawfirm_v2.db")
        except Exception:
            db_path = str(project_root / "data" / "lawfirm_v2.db")
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return 1
    
    print(f"\n📁 Database: {db_path}")
    
    # MLflow 인덱스 로드
    if not FAISS_AVAILABLE:
        print("❌ FAISS not available")
        return 1
    
    index, chunk_ids, version_info = load_mlflow_index(args.mlflow_run_id)
    if index is None or chunk_ids is None:
        print("❌ Failed to load FAISS index")
        return 1
    
    print(f"\n📊 FAISS Index Info:")
    print(f"   - Total vectors: {index.ntotal}")
    print(f"   - Chunk IDs: {len(chunk_ids)}")
    print(f"   - Dimension: {index.d}")
    
    if version_info:
        embedding_config = version_info.get('embedding_config', {})
        if embedding_config:
            print(f"   - Embedding model: {embedding_config.get('model', 'N/A')}")
            print(f"   - Dimension: {embedding_config.get('dimension', 'N/A')}")
    
    # source_type 일치성 검증
    print(f"\n🔍 Verifying source_type consistency...")
    results = verify_source_type_consistency(db_path, chunk_ids, args.sample_size)
    
    # 결과 분석
    analyze_type_distribution(results)
    check_type_search_feasibility(results)
    
    # 요약
    print("\n" + "="*80)
    print("📋 검증 요약")
    print("="*80)
    
    if results['not_found_in_db'] > 0:
        missing_ratio = results['not_found_in_db'] / results['total_checked']
        if missing_ratio > 0.1:  # 10% 이상
            print(f"❌ CRITICAL: {missing_ratio:.1%}의 chunk가 DB에 없습니다!")
            print(f"   → FAISS 인덱스와 DB가 동기화되지 않았을 수 있습니다.")
            print(f"   → 인덱스를 재빌드하는 것을 권장합니다.")
        else:
            print(f"⚠️  {missing_ratio:.1%}의 chunk가 DB에 없습니다.")
            print(f"   → 일부 chunk가 삭제되었거나 인덱스와 DB가 불일치할 수 있습니다.")
    else:
        print(f"✅ 모든 chunk가 DB에 존재합니다.")
    
    # 타입별 검색 가능성 요약
    type_dist = results['source_type_distribution']
    required_types = ['statute_article', 'case_paragraph', 'decision_paragraph', 'interpretation_paragraph']
    missing_types = [t for t in required_types if type_dist.get(t, 0) == 0]
    
    if missing_types:
        print(f"\n❌ CRITICAL: 다음 타입의 데이터가 없습니다:")
        for t in missing_types:
            print(f"   - {t}")
        print(f"   → 이 타입으로 검색하면 모든 결과가 필터링됩니다.")
        print(f"   → source_type 필터를 완화하거나 데이터를 추가해야 합니다.")
    else:
        low_count_types = [(t, type_dist.get(t, 0)) for t in required_types if type_dist.get(t, 0) < 10]
        if low_count_types:
            print(f"\n⚠️  다음 타입의 데이터가 매우 적습니다:")
            for t, count in low_count_types:
                print(f"   - {t}: {count}개")
            print(f"   → 이 타입으로 검색하면 대부분 필터링될 수 있습니다.")
        else:
            print(f"\n✅ 모든 필수 타입의 데이터가 충분합니다.")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

