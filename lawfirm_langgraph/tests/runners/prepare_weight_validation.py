# -*- coding: utf-8 -*-
"""
가중치 검증 실험 사전 준비 스크립트

실험 계획서에 따른 사전 준비 작업을 수행합니다.

Usage:
    python lawfirm_langgraph/tests/runners/prepare_weight_validation.py
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로
script_dir = Path(__file__).parent
runners_dir = script_dir.parent
tests_dir = runners_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent
sys.path.insert(0, str(lawfirm_langgraph_dir))
sys.path.insert(0, str(project_root))

# 환경 변수 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(project_root)
except ImportError:
    pass

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = get_logger(__name__)


class ExperimentPreparator:
    """실험 준비 클래스"""
    
    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
    
    def check_environment_variables(self):
        """환경 변수 설정 확인"""
        logger.info("="*80)
        logger.info("1단계: 환경 변수 설정 확인")
        logger.info("="*80)
        
        checks = {
            "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
            "USE_MLFLOW_INDEX": os.getenv("USE_MLFLOW_INDEX"),
            "MLFLOW_RUN_ID": os.getenv("MLFLOW_RUN_ID"),
        }
        
        for key, value in checks.items():
            if value:
                logger.info(f"   ✅ {key}: {value}")
                self.checks_passed.append(key)
            else:
                logger.info(f"   ⚠️  {key}: 설정되지 않음 (기본값 사용)")
        
        # MLflow 기본 경로 확인
        mlflow_path = project_root / "mlflow" / "mlruns"
        if not mlflow_path.exists():
            mlflow_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"   ✅ MLflow 디렉토리 생성: {mlflow_path}")
        else:
            logger.info(f"   ✅ MLflow 디렉토리 존재: {mlflow_path}")
        
        return len(self.checks_failed) == 0
    
    def check_database(self):
        """데이터베이스 연결 확인"""
        logger.info("\n" + "="*80)
        logger.info("2단계: 데이터베이스 연결 확인")
        logger.info("="*80)
        
        try:
            # 데이터베이스 경로 확인
            db_path = os.getenv("DATABASE_PATH", "data/lawfirm_v2.db")
            if not Path(db_path).is_absolute():
                db_path = project_root / db_path
            
            if not db_path or not Path(db_path).exists():
                logger.error(f"   ❌ 데이터베이스 파일이 없습니다: {db_path}")
                self.checks_failed.append("database_file")
                return False
            
            logger.info(f"   ✅ 데이터베이스 파일 존재: {db_path}")
            
            # 데이터베이스 연결 테스트
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 테이블 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"   ✅ 테이블 수: {len(tables)}개")
            
            # 주요 테이블 데이터 확인
            key_tables = {
                "statutes": "SELECT COUNT(*) FROM statutes",
                "statute_articles": "SELECT COUNT(*) FROM statute_articles",
                "cases": "SELECT COUNT(*) FROM cases",
                "case_paragraphs": "SELECT COUNT(*) FROM case_paragraphs"
            }
            
            logger.info("\n   주요 테이블 데이터:")
            for table_name, query in key_tables.items():
                try:
                    cursor.execute(query)
                    count = cursor.fetchone()[0]
                    logger.info(f"      - {table_name}: {count:,}개")
                    if count == 0:
                        logger.warning(f"      ⚠️  {table_name}에 데이터가 없습니다")
                except Exception as e:
                    logger.warning(f"      ⚠️  {table_name} 확인 실패: {e}")
            
            conn.close()
            self.checks_passed.append("database")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ 데이터베이스 확인 실패: {e}")
            self.checks_failed.append("database")
            return False
    
    def check_search_index(self):
        """검색 인덱스 상태 확인"""
        logger.info("\n" + "="*80)
        logger.info("3단계: 검색 인덱스 상태 확인")
        logger.info("="*80)
        
        try:
            # MLflow 인덱스 확인
            use_mlflow = os.getenv("USE_MLFLOW_INDEX", "true").lower() == "true"
            if use_mlflow:
                logger.info("   ✅ MLflow 인덱스 사용 설정됨")
                
                try:
                    from scripts.rag.mlflow_manager import MLflowFAISSManager
                    mlflow_manager = MLflowFAISSManager()
                    run_id = mlflow_manager.get_production_run()
                    if run_id:
                        logger.info(f"   ✅ 프로덕션 MLflow run ID: {run_id}")
                        self.checks_passed.append("mlflow_index")
                    else:
                        logger.warning("   ⚠️  프로덕션 MLflow run을 찾을 수 없습니다")
                        self.checks_failed.append("mlflow_production_run")
                except Exception as e:
                    logger.warning(f"   ⚠️  MLflow 인덱스 확인 실패: {e}")
                    self.checks_failed.append("mlflow_index")
            else:
                logger.info("   ℹ️  MLflow 인덱스 비활성화됨 (DB 기반 인덱스 사용)")
            
            # 검색 엔진 초기화 테스트
            try:
                from core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
                engine = SemanticSearchEngineV2()
                logger.info("   ✅ 검색 엔진 초기화 성공")
                
                # 간단한 검색 테스트
                test_query = "계약 해지"
                results = engine.search(test_query, k=3, similarity_threshold=0.2)
                logger.info(f"   ✅ 검색 테스트 성공: '{test_query}' → {len(results)}개 결과")
                
                self.checks_passed.append("search_engine")
                return True
            except Exception as e:
                logger.error(f"   ❌ 검색 엔진 확인 실패: {e}")
                self.checks_failed.append("search_engine")
                return False
                
        except Exception as e:
            logger.error(f"   ❌ 검색 인덱스 확인 실패: {e}")
            self.checks_failed.append("search_index")
            return False
    
    def check_test_queries(self):
        """테스트 쿼리 검증"""
        logger.info("\n" + "="*80)
        logger.info("4단계: 테스트 쿼리 검증")
        logger.info("="*80)
        
        # 직접 쿼리 세트 정의 (순환 import 방지)
        queries_dict = {
            "law_inquiry": [
                "민법 제750조 손해배상에 대해 설명해주세요",
                "계약 위약금에 대해 설명해주세요"
            ],
            "precedent_search": [
                "계약 해지 관련 판례를 찾아주세요",
                "손해배상 청구 사례를 알려주세요"
            ],
            "general": [
                "민사법 자문이 필요합니다",
                "계약서 작성 시 주의사항을 알려주세요"
            ]
        }
        
        total_queries = sum(len(q) for q in queries_dict.values())
        
        logger.info(f"   ✅ 테스트 쿼리 세트:")
        for q_type, queries in queries_dict.items():
            logger.info(f"      - {q_type}: {len(queries)}개")
        logger.info(f"   ✅ 총 테스트 쿼리: {total_queries}개")
        
        # 샘플 쿼리 테스트
        logger.info("\n   샘플 쿼리 테스트:")
        sample_queries = [
            queries_dict["law_inquiry"][0] if queries_dict.get("law_inquiry") else None,
            queries_dict["precedent_search"][0] if queries_dict.get("precedent_search") else None,
            queries_dict["general"][0] if queries_dict.get("general") else None
        ]
        
        try:
            from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
            db_path = os.getenv("DATABASE_PATH", "data/lawfirm_v2.db")
            if not Path(db_path).is_absolute():
                db_path = str(project_root / db_path)
            
            engine = SemanticSearchEngineV2(db_path=db_path)
            
            for query in sample_queries:
                if query:
                    try:
                        results = engine.search(query, k=3, similarity_threshold=0.2)
                        logger.info(f"      ✅ '{query[:30]}...' → {len(results)}개 결과")
                    except Exception as e:
                        logger.warning(f"      ⚠️  '{query[:30]}...' 테스트 실패: {e}")
            
            self.checks_passed.append("test_queries")
            return True
        except Exception as e:
            logger.warning(f"   ⚠️  테스트 쿼리 검증 실패: {e} (실험 진행에는 문제 없을 수 있음)")
            self.checks_failed.append("test_queries")
            return False
    
    def run_all_checks(self):
        """모든 확인 작업 실행"""
        logger.info("\n" + "="*80)
        logger.info("가중치 검증 실험 사전 준비")
        logger.info("="*80)
        
        results = {
            "environment": self.check_environment_variables(),
            "database": self.check_database(),
            "search_index": self.check_search_index(),
            "test_queries": self.check_test_queries()
        }
        
        # 결과 요약
        logger.info("\n" + "="*80)
        logger.info("사전 준비 결과 요약")
        logger.info("="*80)
        
        all_passed = all(results.values())
        
        logger.info(f"\n   통과한 확인: {len(self.checks_passed)}개")
        for check in self.checks_passed:
            logger.info(f"      ✅ {check}")
        
        if self.checks_failed:
            logger.info(f"\n   실패한 확인: {len(self.checks_failed)}개")
            for check in self.checks_failed:
                logger.info(f"      ❌ {check}")
        
        if all_passed:
            logger.info("\n   ✅ 모든 사전 준비 완료! 실험을 진행할 수 있습니다.")
        else:
            logger.warning("\n   ⚠️  일부 확인이 실패했습니다. 실험 전에 문제를 해결해주세요.")
        
        return all_passed


def main():
    """메인 함수"""
    preparator = ExperimentPreparator()
    success = preparator.run_all_checks()
    
    if success:
        logger.info("\n다음 단계: 베이스라인 측정")
        logger.info("   python lawfirm_langgraph/tests/runners/validate_weight_configurations.py --quick")
    else:
        logger.error("\n사전 준비가 완료되지 않았습니다. 문제를 해결한 후 다시 시도해주세요.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

