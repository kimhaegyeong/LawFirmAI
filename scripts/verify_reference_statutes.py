"""
참조조문 추출 품질 검증 스크립트
"""
import sqlite3
import json
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]  # scripts의 부모 = 프로젝트 루트
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 직접 경로로 import
import importlib.util
extractor_path = _PROJECT_ROOT / "scripts" / "utils" / "reference_statute_extractor.py"
spec = importlib.util.spec_from_file_location(
    "reference_statute_extractor",
    extractor_path
)
reference_statute_extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_statute_extractor)
ReferenceStatuteExtractor = reference_statute_extractor.ReferenceStatuteExtractor


def verify_cases(conn: sqlite3.Connection, extractor: ReferenceStatuteExtractor):
    """Cases 검증"""
    print("=" * 60)
    print("Cases (판례) 검증")
    print("=" * 60)
    
    # 추출된 케이스 샘플
    cursor = conn.execute(
        "SELECT doc_id, reference_statutes FROM cases WHERE reference_statutes IS NOT NULL LIMIT 5"
    )
    rows = cursor.fetchall()
    
    print("\n[추출 성공 샘플]")
    for doc_id, ref_json in rows:
        refs = json.loads(ref_json) if ref_json else []
        print(f"  Doc ID: {doc_id}")
        print(f"  추출된 법령 수: {len(refs)}")
        for ref in refs[:3]:  # 최대 3개만 표시
            print(f"    - {ref.get('statute_name')} 제{ref.get('article_no')}조")
        print()
    
    # 키워드는 있지만 추출 실패한 케이스 샘플
    cursor = conn.execute(
        """SELECT c.doc_id, GROUP_CONCAT(cp.text, '\n') as full_text
           FROM cases c
           JOIN case_paragraphs cp ON c.id = cp.case_id
           WHERE c.reference_statutes IS NULL
           GROUP BY c.id
           HAVING full_text LIKE '%참조조문%'
           LIMIT 5"""
    )
    rows = cursor.fetchall()
    
    print("\n[키워드는 있지만 추출 실패한 샘플]")
    for doc_id, full_text in rows:
        print(f"  Doc ID: {doc_id}")
        print(f"  전체 텍스트 길이: {len(full_text)} 문자")
        
        # "참조조문" 키워드 검색
        idx = full_text.find("참조조문")
        snippet = full_text[idx:idx+500]
        print(f"  '참조조문' 발견 위치: {idx}")
        print(f"  주변 텍스트: {snippet[:300]}...")
        
        # 수동 추출 시도
        extracted = extractor.extract_from_content(full_text)
        if extracted:
            print(f"  ⚠️ 수동 추출 시도 결과: {len(extracted)}개 법령 추출됨!")
            for ref in extracted[:3]:
                print(f"    - {ref.get('statute_name')} 제{ref.get('article_no')}조")
        else:
            print(f"  ❌ 수동 추출 시도 결과: 추출 실패")
            # 실패 원인 분석
            if "참조조문:" in full_text:
                print(f"    - '참조조문:' 형식 발견")
            elif "참조조문：" in full_text:
                print(f"    - '참조조문：' 형식 발견")
            else:
                print(f"    - '참조조문' 뒤에 콜론 없음")
        print()
    
    # 추출 실패 케이스 샘플 (키워드 없음)
    cursor = conn.execute(
        """SELECT c.doc_id, GROUP_CONCAT(cp.text, '\n') as full_text
           FROM cases c
           JOIN case_paragraphs cp ON c.id = cp.case_id
           WHERE c.reference_statutes IS NULL
           GROUP BY c.id
           HAVING full_text NOT LIKE '%참조조문%'
           LIMIT 3"""
    )
    rows = cursor.fetchall()
    
    print("\n[키워드 없는 샘플]")
    for doc_id, full_text in rows:
        print(f"  Doc ID: {doc_id}")
        print(f"  전체 텍스트 길이: {len(full_text)} 문자")
        print(f"  '참조조문' 키워드 없음")
        print()


def verify_decisions(conn: sqlite3.Connection, extractor: ReferenceStatuteExtractor):
    """Decisions 검증"""
    print("=" * 60)
    print("Decisions (결정례) 검증")
    print("=" * 60)
    
    # 샘플 확인
    cursor = conn.execute(
        """SELECT d.doc_id, GROUP_CONCAT(dp.text, '\n') as full_text
           FROM decisions d
           JOIN decision_paragraphs dp ON d.id = dp.decision_id
           WHERE d.reference_statutes IS NULL
           GROUP BY d.id
           LIMIT 3"""
    )
    rows = cursor.fetchall()
    
    print("\n[샘플 분석]")
    for doc_id, full_text in rows:
        print(f"  Doc ID: {doc_id}")
        print(f"  전체 텍스트 길이: {len(full_text)} 문자")
        
        # "참조조문" 키워드 검색
        if "참조조문" in full_text:
            idx = full_text.find("참조조문")
            snippet = full_text[idx:idx+500]
            print(f"  '참조조문' 발견 위치: {idx}")
            print(f"  주변 텍스트: {snippet[:200]}...")
            
            # 수동 추출 시도
            extracted = extractor.extract_from_content(full_text)
            if extracted:
                print(f"  ⚠️ 수동 추출 시도 결과: {len(extracted)}개 법령 추출됨!")
                for ref in extracted[:2]:
                    print(f"    - {ref.get('statute_name')} 제{ref.get('article_no')}조")
            else:
                print(f"  수동 추출 시도 결과: 추출 실패")
        else:
            print(f"  '참조조문' 키워드 없음")
        print()


def verify_interpretations(conn: sqlite3.Connection, extractor: ReferenceStatuteExtractor):
    """Interpretations 검증"""
    print("=" * 60)
    print("Interpretations (해석례) 검증")
    print("=" * 60)
    
    # 샘플 확인
    cursor = conn.execute(
        """SELECT i.doc_id, GROUP_CONCAT(ip.text, '\n') as full_text
           FROM interpretations i
           JOIN interpretation_paragraphs ip ON i.id = ip.interpretation_id
           WHERE i.reference_statutes IS NULL
           GROUP BY i.id
           LIMIT 3"""
    )
    rows = cursor.fetchall()
    
    print("\n[샘플 분석]")
    for doc_id, full_text in rows:
        print(f"  Doc ID: {doc_id}")
        print(f"  전체 텍스트 길이: {len(full_text)} 문자")
        
        # "참조조문" 키워드 검색
        if "참조조문" in full_text:
            idx = full_text.find("참조조문")
            snippet = full_text[idx:idx+500]
            print(f"  '참조조문' 발견 위치: {idx}")
            print(f"  주변 텍스트: {snippet[:200]}...")
            
            # 수동 추출 시도
            extracted = extractor.extract_from_content(full_text)
            if extracted:
                print(f"  ⚠️ 수동 추출 시도 결과: {len(extracted)}개 법령 추출됨!")
                for ref in extracted[:2]:
                    print(f"    - {ref.get('statute_name')} 제{ref.get('article_no')}조")
            else:
                print(f"  수동 추출 시도 결과: 추출 실패")
        else:
            print(f"  '참조조문' 키워드 없음")
        print()


def check_keyword_statistics(conn: sqlite3.Connection):
    """키워드 통계 확인"""
    print("=" * 60)
    print("키워드 통계")
    print("=" * 60)
    
    # Cases
    cursor = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT c.id, GROUP_CONCAT(cp.text, '\n') as text 
            FROM cases c 
            JOIN case_paragraphs cp ON c.id = cp.case_id 
            GROUP BY c.id
        ) WHERE text LIKE '%참조조문%'
    """)
    cases_with_keyword = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM cases")
    total_cases = cursor.fetchone()[0]
    print(f"Cases with '참조조문' keyword: {cases_with_keyword}/{total_cases} ({cases_with_keyword*100/total_cases:.1f}%)")
    
    # Decisions
    cursor = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT d.id, GROUP_CONCAT(dp.text, '\n') as text 
            FROM decisions d 
            JOIN decision_paragraphs dp ON d.id = dp.decision_id 
            GROUP BY d.id
        ) WHERE text LIKE '%참조조문%'
    """)
    decisions_with_keyword = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM decisions")
    total_decisions = cursor.fetchone()[0]
    print(f"Decisions with '참조조문' keyword: {decisions_with_keyword}/{total_decisions} ({decisions_with_keyword*100/total_decisions if total_decisions > 0 else 0:.1f}%)")
    
    # Interpretations
    cursor = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT i.id, GROUP_CONCAT(ip.text, '\n') as text 
            FROM interpretations i 
            JOIN interpretation_paragraphs ip ON i.id = ip.interpretation_id 
            GROUP BY i.id
        ) WHERE text LIKE '%참조조문%'
    """)
    interps_with_keyword = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM interpretations")
    total_interps = cursor.fetchone()[0]
    print(f"Interpretations with '참조조문' keyword: {interps_with_keyword}/{total_interps} ({interps_with_keyword*100/total_interps if total_interps > 0 else 0:.1f}%)")
    
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="참조조문 추출 품질 검증")
    parser.add_argument("--db", default="./data/lawfirm_v2.db", help="Database path")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)
    
    extractor = ReferenceStatuteExtractor()
    
    with sqlite3.connect(str(db_path)) as conn:
        check_keyword_statistics(conn)
        verify_cases(conn, extractor)
        verify_decisions(conn, extractor)
        verify_interpretations(conn, extractor)
    
    print("=" * 60)
    print("검증 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()

