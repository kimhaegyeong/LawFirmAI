"""추출 품질 검증 스크립트"""
import sqlite3
import json
import sys
from pathlib import Path
from collections import Counter

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "reference_statute_extractor",
    _PROJECT_ROOT / "scripts" / "utils" / "reference_statute_extractor.py"
)
reference_statute_extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_statute_extractor)
ReferenceStatuteExtractor = reference_statute_extractor.ReferenceStatuteExtractor


def analyze_extraction_quality(conn: sqlite3.Connection):
    """추출 품질 분석"""
    print("=" * 60)
    print("추출 품질 분석")
    print("=" * 60)
    
    # 추출된 케이스 통계
    cursor = conn.execute("""
        SELECT reference_statutes 
        FROM cases 
        WHERE reference_statutes IS NOT NULL
    """)
    
    all_statutes = []
    statute_names = Counter()
    total_cases = 0
    cases_with_multiple = 0
    
    for row in cursor.fetchall():
        try:
            statutes = json.loads(row[0])
            total_cases += 1
            if len(statutes) > 1:
                cases_with_multiple += 1
            
            for statute in statutes:
                all_statutes.append(statute)
                statute_name = statute.get('statute_name', '')
                if statute_name:
                    statute_names[statute_name] += 1
        except:
            pass
    
    print(f"\n[추출 통계]")
    print(f"추출된 케이스: {total_cases}개")
    print(f"추출된 법령 총 수: {len(all_statutes)}개")
    print(f"케이스당 평균 법령 수: {len(all_statutes)/total_cases:.2f}개")
    print(f"여러 법령이 추출된 케이스: {cases_with_multiple}개 ({cases_with_multiple*100/total_cases:.1f}%)")
    
    print(f"\n[법령명 분포 (상위 20개)]")
    for statute_name, count in statute_names.most_common(20):
        print(f"  {statute_name}: {count}회")
    
    # 비정상적인 법령명 확인
    print(f"\n[비정상적인 법령명 샘플]")
    abnormal_names = []
    for statute in all_statutes:
        name = statute.get('statute_name', '')
        if name:
            # 너무 짧거나 이상한 패턴
            if len(name) < 3 or '관한' in name[:5] or name.startswith('제') or '판시사항' in name:
                abnormal_names.append(name)
    
    if abnormal_names:
        abnormal_counter = Counter(abnormal_names)
        for name, count in abnormal_counter.most_common(10):
            print(f"  {name}: {count}회")
    else:
        print("  비정상적인 법령명 없음")


def analyze_failed_cases(conn: sqlite3.Connection, extractor: ReferenceStatuteExtractor, sample_size: int = 50):
    """추출 실패 케이스 분석"""
    print("\n" + "=" * 60)
    print("추출 실패 케이스 분석")
    print("=" * 60)
    
    # 키워드는 있지만 추출 실패한 케이스
    cursor = conn.execute("""
        SELECT c.doc_id, GROUP_CONCAT(cp.text, '\n') as full_text
        FROM cases c
        JOIN case_paragraphs cp ON c.id = cp.case_id
        WHERE c.reference_statutes IS NULL
        GROUP BY c.id
        HAVING full_text LIKE '%참조조문%'
        LIMIT ?
    """, (sample_size,))
    
    rows = cursor.fetchall()
    print(f"\n분석할 샘플 수: {len(rows)}")
    
    patterns = {
        "참조조문_직후_참조판례": 0,
        "참조조문_직후_전문": 0,
        "참조조문_콜론_없음": 0,
        "법령정보_없음": 0,
        "기타": 0
    }
    
    for doc_id, full_text in rows:
        ref_pos = full_text.find("참조조문")
        if ref_pos == -1:
            continue
        
        snippet = full_text[ref_pos:ref_pos+100]
        
        # 패턴 분류
        if "참조조문" in snippet and "참조판례" in snippet[:50]:
            if snippet.find("참조판례") - snippet.find("참조조문") < 20:
                patterns["참조조문_직후_참조판례"] += 1
        elif "참조조문" in snippet and "전문" in snippet[:50]:
            if snippet.find("전문") - snippet.find("참조조문") < 20:
                patterns["참조조문_직후_전문"] += 1
        elif "참조조문:" not in snippet and "참조조문：" not in snippet:
            patterns["참조조문_콜론_없음"] += 1
        
        # 법령 정보가 있는지 확인
        import re
        if not re.search(r'제\s*\d+\s*조', snippet[:200]):
            patterns["법령정보_없음"] += 1
        else:
            patterns["기타"] += 1
    
    print("\n패턴 분류:")
    for pattern, count in patterns.items():
        if count > 0:
            print(f"  {pattern}: {count}개")


def check_duplicates(conn: sqlite3.Connection):
    """중복 추출 확인"""
    print("\n" + "=" * 60)
    print("중복 추출 확인")
    print("=" * 60)
    
    cursor = conn.execute("""
        SELECT doc_id, reference_statutes 
        FROM cases 
        WHERE reference_statutes IS NOT NULL
        LIMIT 1000
    """)
    
    duplicate_cases = []
    
    for row in cursor.fetchall():
        doc_id, ref_statutes = row
        try:
            statutes = json.loads(ref_statutes)
            # 같은 법령이 중복 추출되었는지 확인
            seen = set()
            duplicates = []
            for statute in statutes:
                key = f"{statute.get('statute_name')}_{statute.get('article_no')}_{statute.get('clause_no', '')}_{statute.get('item_no', '')}"
                if key in seen:
                    duplicates.append(statute)
                seen.add(key)
            
            if duplicates:
                duplicate_cases.append((doc_id, duplicates))
        except:
            pass
    
    if duplicate_cases:
        print(f"\n중복 추출된 케이스: {len(duplicate_cases)}개")
        print("\n[샘플]")
        for doc_id, duplicates in duplicate_cases[:5]:
            print(f"  {doc_id}: {len(duplicates)}개 중복")
            for dup in duplicates[:2]:
                print(f"    - {dup.get('statute_name')} 제{dup.get('article_no')}조")
    else:
        print("\n중복 추출된 케이스 없음")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="추출 품질 검증")
    parser.add_argument("--db", default="./data/lawfirm_v2.db", help="Database path")
    parser.add_argument("--sample-size", type=int, default=50, help="Sample size for analysis")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)
    
    extractor = ReferenceStatuteExtractor()
    
    with sqlite3.connect(str(db_path)) as conn:
        analyze_extraction_quality(conn)
        analyze_failed_cases(conn, extractor, args.sample_size)
        check_duplicates(conn)
    
    print("\n" + "=" * 60)
    print("검증 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()

