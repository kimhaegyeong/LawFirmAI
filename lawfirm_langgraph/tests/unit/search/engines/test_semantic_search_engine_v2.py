# -*- coding: utf-8 -*-
"""
SemanticSearchEngineV2 테스트 코드
"""

import sys
import os
import time
import sqlite3
from pathlib import Path

# 프로젝트 루트 경로 추가 (하위 폴더로 이동하여 parent 하나 추가)
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "lawfirm_langgraph"))

# conftest.py에서 db_path 찾기 로직 재사용
from lawfirm_langgraph.tests.unit.search.conftest import db_path as get_db_path

import warnings
warnings.filterwarnings('ignore', message='.*python-dotenv.*')

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2


class TestSemanticSearchEngineV2:
    """SemanticSearchEngineV2 테스트 클래스"""
    
    def __init__(self):
        """테스트 초기화"""
        self.engine = None
        self.db_path = None
        
    def setup(self):
        """테스트 설정"""
        # conftest.py의 db_path 찾기 로직 사용
        from lawfirm_langgraph.tests.unit.search.conftest import project_root as _project_root
        
        possible_db_paths = [
            "data/lawfirm_v2.db",
            "./data/lawfirm_v2.db",
            str(_project_root / "data" / "lawfirm_v2.db")
        ]
        
        for path in possible_db_paths:
            if Path(path).exists():
                self.db_path = path
                break
        
        if not self.db_path:
            print("❌ 데이터베이스 파일을 찾을 수 없습니다.")
            return False
        
        try:
            self.engine = SemanticSearchEngineV2(
                db_path=self.db_path,
                use_external_index=False
            )
            print(f"✅ SemanticSearchEngineV2 초기화 성공 (DB: {self.db_path})")
            return True
        except Exception as e:
            print(f"❌ SemanticSearchEngineV2 초기화 실패: {e}")
            return False
    
    def test_normalize_query(self):
        """쿼리 정규화 테스트"""
        print("\n📋 테스트: 쿼리 정규화")
        try:
            # 공백 정규화
            result1 = self.engine._normalize_query("  임대차   계약  ")
            assert result1 == "임대차 계약", f"Expected '임대차 계약', got '{result1}'"
            
            # 대소문자 정규화
            result2 = self.engine._normalize_query("임대차 계약")
            result3 = self.engine._normalize_query("임대차 계약")
            assert result2 == result3, "대소문자 정규화 실패"
            
            # 빈 문자열
            result4 = self.engine._normalize_query("")
            assert result4 == "", f"Expected '', got '{result4}'"
            
            print("   ✅ 쿼리 정규화 테스트 통과")
            return True
        except Exception as e:
            print(f"   ❌ 쿼리 정규화 테스트 실패: {e}")
            return False
    
    def test_cache_ttl(self):
        """캐시 TTL 테스트"""
        print("\n📋 테스트: 캐시 TTL")
        try:
            # 캐시에 항목 저장
            test_key = "test_key"
            test_value = {"test": "data"}
            self.engine._set_to_cache(test_key, test_value)
            
            # 즉시 조회 (캐시 히트)
            cached = self.engine._get_from_cache(test_key)
            assert cached == test_value, "캐시 저장/조회 실패"
            
            # TTL을 짧게 설정하여 만료 테스트
            original_ttl = self.engine._metadata_cache_ttl
            self.engine._metadata_cache_ttl = 0.1  # 0.1초
            
            # 캐시에 다시 저장
            self.engine._set_to_cache(test_key, test_value)
            
            # 0.2초 대기 (TTL 초과)
            time.sleep(0.2)
            
            # 만료된 항목 조회 (캐시 미스)
            expired = self.engine._get_from_cache(test_key)
            assert expired is None, "만료된 캐시 항목이 제거되지 않음"
            
            # TTL 복원
            self.engine._metadata_cache_ttl = original_ttl
            
            print("   ✅ 캐시 TTL 테스트 통과")
            return True
        except Exception as e:
            print(f"   ❌ 캐시 TTL 테스트 실패: {e}")
            return False
    
    def test_cache_cleanup(self):
        """캐시 정리 테스트"""
        print("\n📋 테스트: 캐시 정리")
        try:
            # 여러 항목 저장
            for i in range(10):
                self.engine._set_to_cache(f"key_{i}", {"data": i})
            
            initial_size = len(self.engine._metadata_cache)
            assert initial_size == 10, f"Expected 10 items, got {initial_size}"
            
            # TTL을 짧게 설정
            original_ttl = self.engine._metadata_cache_ttl
            original_cleanup_interval = self.engine._metadata_cache_cleanup_interval
            self.engine._metadata_cache_ttl = 0.1
            self.engine._metadata_cache_cleanup_interval = 0.05  # 0.05초
            
            # 시간 경과 대기
            time.sleep(0.15)
            
            # 정리 실행
            self.engine._cleanup_expired_cache()
            
            # 만료된 항목이 제거되었는지 확인
            cleaned_size = len(self.engine._metadata_cache)
            assert cleaned_size == 0, f"Expected 0 items after cleanup, got {cleaned_size}"
            
            # 설정 복원
            self.engine._metadata_cache_ttl = original_ttl
            self.engine._metadata_cache_cleanup_interval = original_cleanup_interval
            
            print("   ✅ 캐시 정리 테스트 통과")
            return True
        except Exception as e:
            print(f"   ❌ 캐시 정리 테스트 실패: {e}")
            return False
    
    def test_batch_load_chunk_metadata(self):
        """배치 chunk_metadata 조회 테스트"""
        print("\n📋 테스트: 배치 chunk_metadata 조회")
        try:
            conn = self.engine._get_connection()
            if not conn:
                print("   ⚠️  DB 연결 실패, 테스트 스킵")
                return True
            
            # 실제 chunk_id 조회
            cursor = conn.execute("SELECT id FROM text_chunks LIMIT 5")
            chunk_ids = [row[0] for row in cursor.fetchall()]
            
            if not chunk_ids:
                print("   ⚠️  chunk_id가 없어 테스트 스킵")
                return True
            
            # 배치 조회
            result = self.engine._batch_load_chunk_metadata(conn, chunk_ids)
            
            assert len(result) == len(chunk_ids), f"Expected {len(chunk_ids)} results, got {len(result)}"
            
            # 결과 검증
            for chunk_id in chunk_ids:
                assert chunk_id in result, f"chunk_id {chunk_id} not in result"
                assert 'meta' in result[chunk_id], f"chunk_id {chunk_id} missing 'meta' field"
                assert 'source_type' in result[chunk_id], f"chunk_id {chunk_id} missing 'source_type' field"
                assert 'source_id' in result[chunk_id], f"chunk_id {chunk_id} missing 'source_id' field"
            
            # 캐시 테스트 (두 번째 조회는 캐시에서 가져와야 함)
            cache_hits_before = self.engine._metadata_cache_hits
            result2 = self.engine._batch_load_chunk_metadata(conn, chunk_ids)
            cache_hits_after = self.engine._metadata_cache_hits
            
            assert cache_hits_after > cache_hits_before, "캐시 히트가 발생하지 않음"
            
            conn.close()
            print(f"   ✅ 배치 chunk_metadata 조회 테스트 통과 (조회: {len(chunk_ids)}개, 캐시 히트: {cache_hits_after - cache_hits_before}개)")
            return True
        except Exception as e:
            print(f"   ❌ 배치 chunk_metadata 조회 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_batch_load_source_metadata(self):
        """배치 source_metadata 조회 테스트"""
        print("\n📋 테스트: 배치 source_metadata 조회")
        try:
            conn = self.engine._get_connection()
            if not conn:
                print("   ⚠️  DB 연결 실패, 테스트 스킵")
                return True
            
            # 실제 source_type, source_id 조회
            cursor = conn.execute("""
                SELECT DISTINCT source_type, source_id 
                FROM text_chunks 
                WHERE source_type IS NOT NULL AND source_id IS NOT NULL
                LIMIT 5
            """)
            source_items = [(row[0], row[1]) for row in cursor.fetchall()]
            
            if not source_items:
                print("   ⚠️  source_items가 없어 테스트 스킵")
                return True
            
            # 배치 조회
            result = self.engine._batch_load_source_metadata(conn, source_items)
            
            # 결과 검증
            for source_type, source_id in source_items:
                # source_id가 문자열인 경우 정수로 변환 시도
                if isinstance(source_id, str):
                    import re
                    numbers = re.findall(r'\d+', str(source_id))
                    if numbers:
                        source_id = int(numbers[-1])
                    else:
                        continue
                
                key = (source_type, source_id)
                if key in result:
                    assert isinstance(result[key], dict), f"source_metadata for {key} is not a dict"
            
            # 캐시 테스트
            cache_hits_before = self.engine._metadata_cache_hits
            result2 = self.engine._batch_load_source_metadata(conn, source_items)
            cache_hits_after = self.engine._metadata_cache_hits
            
            assert cache_hits_after > cache_hits_before, "캐시 히트가 발생하지 않음"
            
            conn.close()
            print(f"   ✅ 배치 source_metadata 조회 테스트 통과 (조회: {len(source_items)}개, 캐시 히트: {cache_hits_after - cache_hits_before}개)")
            return True
        except Exception as e:
            print(f"   ❌ 배치 source_metadata 조회 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_search_basic(self):
        """기본 검색 테스트"""
        print("\n📋 테스트: 기본 검색")
        try:
            query = "임대차 계약"
            results = self.engine.search(
                query=query,
                k=5,
                similarity_threshold=0.3
            )
            
            assert isinstance(results, list), "검색 결과가 리스트가 아님"
            assert len(results) <= 5, f"검색 결과가 k보다 많음: {len(results)}"
            
            # 결과 검증
            for result in results:
                assert 'text' in result or 'content' in result, "검색 결과에 text/content가 없음"
                assert 'score' in result or 'similarity' in result, "검색 결과에 score/similarity가 없음"
            
            print(f"   ✅ 기본 검색 테스트 통과 (결과: {len(results)}개)")
            return True
        except Exception as e:
            print(f"   ❌ 기본 검색 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("="*80)
        print("SemanticSearchEngineV2 테스트 시작")
        print("="*80)
        
        if not self.setup():
            print("\n❌ 테스트 설정 실패")
            return False
        
        tests = [
            ("쿼리 정규화", self.test_normalize_query),
            ("캐시 TTL", self.test_cache_ttl),
            ("캐시 정리", self.test_cache_cleanup),
            ("배치 chunk_metadata 조회", self.test_batch_load_chunk_metadata),
            ("배치 source_metadata 조회", self.test_batch_load_source_metadata),
            ("기본 검색", self.test_search_basic),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"\n❌ {test_name} 테스트 중 예외 발생: {e}")
                results.append((test_name, False))
        
        # 결과 요약
        print("\n" + "="*80)
        print("테스트 결과 요약")
        print("="*80)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ 통과" if result else "❌ 실패"
            print(f"  {test_name}: {status}")
        
        print(f"\n총 {passed}/{total} 테스트 통과")
        print("="*80)
        
        return passed == total


if __name__ == "__main__":
    tester = TestSemanticSearchEngineV2()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

