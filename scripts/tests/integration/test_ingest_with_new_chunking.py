#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
새로운 청킹 전략으로 소량 데이터 ingest 테스트

이 스크립트는 새로운 청킹 전략(dynamic, hybrid)이 제대로 작동하는지 확인합니다.
"""
import sys
import sqlite3
import json
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.embeddings import SentenceEmbedder
from scripts.utils.embedding_version_manager import EmbeddingVersionManager


def create_test_case_data() -> dict:
    """테스트용 판례 데이터 생성"""
    return {
        "doc_id": "TEST_CASE_001",
        "normalized_court": "대법원",
        "casetype": "민사",
        "casenames": "전세금 반환 보증 관련 사건",
        "announce_date": "2024-01-01",
        "sentences": [
            "제1조 전세금 반환 보증에 관한 사항을 규정한다.",
            "전세금 반환 보증은 임대인이 전세금을 반환하지 못할 경우를 대비하여 보증인을 선정하는 제도이다.",
            "보증인은 전세금 반환 의무를 이행하지 못한 임대인을 대신하여 전세금을 반환할 의무를 진다.",
            "제2조 보증인의 자격 및 선정 방법에 관하여 규정한다.",
            "보증인은 신용이 있는 자로서 전세금 반환 능력이 있어야 한다.",
            "보증인은 임대인과 별도로 보증계약을 체결하여야 하며, 이 계약은 공증을 받아야 한다.",
            "제3조 보증인의 책임 범위에 관하여 규정한다.",
            "보증인은 전세금 원금 및 이자, 그리고 지연 손해금을 포함한 모든 금액에 대해 책임을 진다.",
            "보증인의 책임은 전세계약 종료 시점부터 시작되며, 임대인이 전세금을 반환하지 못한 경우 즉시 발생한다.",
        ]
    }


def test_ingest_with_strategy(
    db_path: str,
    chunking_strategy: str,
    query_type: Optional[str] = None
):
    """특정 청킹 전략으로 ingest 테스트"""
    print(f"\n{'='*60}")
    print(f"청킹 전략 테스트: {chunking_strategy}")
    if query_type:
        print(f"질문 유형: {query_type}")
    print(f"{'='*60}")
    
    # 테스트 데이터 생성
    test_data = create_test_case_data()
    
    # 데이터베이스 연결 (타임아웃 설정)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    
    try:
        # Domain 생성
        domain_id = conn.execute(
            "INSERT OR IGNORE INTO domains(name) VALUES(?)",
            ("테스트",)
        ).lastrowid
        if not domain_id:
            domain_id = conn.execute(
                "SELECT id FROM domains WHERE name=?", ("테스트",)
            ).fetchone()[0]
        
        # Case 생성
        case_id = conn.execute(
            """INSERT OR IGNORE INTO cases(domain_id, doc_id, court, case_type, casenames, announce_date)
               VALUES(?,?,?,?,?,?)""",
            (
                domain_id,
                test_data["doc_id"],
                test_data["normalized_court"],
                test_data["casetype"],
                test_data["casenames"],
                test_data["announce_date"]
            )
        ).lastrowid
        
        if not case_id:
            case_id = conn.execute(
                "SELECT id FROM cases WHERE doc_id=?", (test_data["doc_id"],)
            ).fetchone()[0]
        
        # Paragraphs 삽입
        paragraphs = test_data["sentences"]
        for i, para in enumerate(paragraphs):
            conn.execute(
                "INSERT OR REPLACE INTO case_paragraphs(case_id, para_index, text) VALUES(?,?,?)",
                (case_id, i, para)
            )
        
        # Embedder 생성
        embedder = SentenceEmbedder("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        
        # Version Manager 생성 (별도 연결 사용)
        version_manager = EmbeddingVersionManager(db_path)
        
        # 버전 조회 (활성 버전 우선, 없으면 비활성 버전 사용)
        active_version = version_manager.get_active_version(chunking_strategy)
        if not active_version:
            # 비활성 버전 조회
            all_versions = version_manager.list_versions(chunking_strategy)
            if all_versions:
                active_version = all_versions[0]  # 가장 최근 버전 사용
                print(f"⚠️  활성 버전이 없어 최근 버전(ID: {active_version['id']})을 사용합니다.")
            else:
                print(f"⚠️  {chunking_strategy} 전략의 버전이 없습니다. 먼저 마이그레이션을 실행하세요.")
                return
        version_id = active_version['id']
        
        # 기존 청크 삭제 (완전 교체 방식) - 연결 충돌 방지를 위해 건너뛰기
        # 실제 운영 환경에서는 ingest 스크립트에서 처리됨
        print("기존 청크 삭제는 건너뜁니다 (테스트 환경)")
        
        # 청킹 전략 사용
        from scripts.utils.chunking.factory import ChunkingFactory
        strategy = ChunkingFactory.create_strategy(
            strategy_name=chunking_strategy,
            query_type=query_type
        )
        
        chunk_results = strategy.chunk(
            content=paragraphs,
            source_type="case_paragraph",
            source_id=case_id
        )
        
        if not chunk_results:
            print("⚠️  청크가 생성되지 않았습니다.")
            return
        
        print(f"생성된 청크 수: {len(chunk_results)}")
        
        # 청크 및 임베딩 저장
        chunk_ids = []
        texts_to_embed = []
        
        max_idx = conn.execute(
            "SELECT COALESCE(MAX(chunk_index), -1) FROM text_chunks WHERE source_type='case_paragraph' AND source_id=?",
            (case_id,)
        ).fetchone()[0]
        next_chunk_index = int(max_idx) + 1
        
        for i, chunk_result in enumerate(chunk_results):
            chunk_idx = next_chunk_index + i
            metadata = chunk_result.metadata
            
            cursor = conn.execute(
                """INSERT INTO text_chunks(
                    source_type, source_id, level, chunk_index, 
                    start_char, end_char, overlap_chars, text, token_count, meta,
                    chunking_strategy, chunk_size_category, chunk_group_id, 
                    query_type, original_document_id, embedding_version_id
                ) VALUES(?,?,?,?,?,?,?,?,?,NULL,?,?,?,?,?,?)""",
                (
                    "case_paragraph",
                    case_id,
                    metadata.get("level", "paragraph"),
                    chunk_idx,
                    None, None, None,
                    chunk_result.text,
                    None,
                    metadata.get("chunking_strategy"),
                    metadata.get("chunk_size_category"),
                    metadata.get("chunk_group_id"),
                    metadata.get("query_type"),
                    metadata.get("original_document_id"),
                    version_id
                )
            )
            
            chunk_id = cursor.lastrowid
            chunk_ids.append(chunk_id)
            texts_to_embed.append(chunk_result.text)
            
            print(f"  청크 {i+1}: 길이={len(chunk_result.text)}자, "
                  f"카테고리={metadata.get('chunk_size_category')}, "
                  f"그룹ID={metadata.get('chunk_group_id', 'N/A')[:8] if metadata.get('chunk_group_id') else 'N/A'}")
        
        # Embeddings 생성
        if texts_to_embed:
            vecs = embedder.encode(texts_to_embed, batch_size=128)
            dim = vecs.shape[1] if len(vecs.shape) > 1 else vecs.shape[0]
            model_name = getattr(embedder.model, 'name_or_path', 'snunlp/KR-SBERT-V40K-klueNLI-augSTS')
            
            embedding_data = [
                (chunk_id, model_name, dim, vec.tobytes(), version_id)
                for chunk_id, vec in zip(chunk_ids, vecs)
            ]
            
            conn.executemany(
                "INSERT INTO embeddings(chunk_id, model, dim, vector, version_id) VALUES(?,?,?,?,?)",
                embedding_data
            )
        
        conn.commit()
        
        # 결과 확인
        cursor = conn.execute(
            """SELECT chunking_strategy, chunk_size_category, COUNT(*) as cnt
               FROM text_chunks 
               WHERE source_type='case_paragraph' AND source_id=?
               GROUP BY chunking_strategy, chunk_size_category""",
            (case_id,)
        )
        
        print(f"\n저장된 청크 통계:")
        for row in cursor.fetchall():
            print(f"  {row['chunking_strategy']} / {row['chunk_size_category']}: {row['cnt']}개")
        
        print(f"✅ {chunking_strategy} 전략 테스트 완료")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='새로운 청킹 전략으로 ingest 테스트')
    parser.add_argument('--db', default='data/lawfirm_v2.db', help='데이터베이스 경로')
    parser.add_argument('--strategy', choices=['standard', 'dynamic', 'hybrid'], 
                       default='hybrid', help='테스트할 청킹 전략')
    parser.add_argument('--query-type', help='질문 유형 (동적 청킹용)')
    parser.add_argument('--all', action='store_true', help='모든 전략 테스트')
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
        sys.exit(1)
    
    if args.all:
        # 모든 전략 테스트
        strategies = [
            ('standard', None),
            ('dynamic', 'law_inquiry'),
            ('dynamic', 'precedent_search'),
            ('hybrid', None),
        ]
        
        for strategy, query_type in strategies:
            test_ingest_with_strategy(str(db_path), strategy, query_type)
    else:
        # 단일 전략 테스트
        test_ingest_with_strategy(str(db_path), args.strategy, args.query_type)
    
    print(f"\n{'='*60}")
    print("✅ 모든 테스트 완료!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

