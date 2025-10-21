#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
품질 검증 및 최적화 시스템
데이터베이스의 품질을 검증하고 최적화합니다.
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quality_validation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """검증 결과 데이터 클래스"""
    total_laws: int = 0
    total_articles: int = 0
    total_cases: int = 0
    quality_distribution: Dict[str, int] = None
    low_quality_items: List[Dict[str, Any]] = None
    recommendations: List[str] = None
    errors: List[str] = None
    validation_time: float = 0.0
    
    def __post_init__(self):
        if self.quality_distribution is None:
            self.quality_distribution = {}
        if self.low_quality_items is None:
            self.low_quality_items = []
        if self.recommendations is None:
            self.recommendations = []
        if self.errors is None:
            self.errors = []


class QualityValidationSystem:
    """품질 검증 시스템"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        self.db_path = db_path
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.0
        }
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def validate_imported_data(self) -> ValidationResult:
        """
        임포트된 데이터 품질 검증
        
        Returns:
            ValidationResult: 검증 결과
        """
        logger.info("🔍 Starting data quality validation...")
        start_time = datetime.now()
        
        result = ValidationResult()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 기본 통계 수집
                result = self._collect_basic_statistics(cursor, result)
                
                # 품질 분포 분석
                result = self._analyze_quality_distribution(cursor, result)
                
                # 저품질 항목 식별
                result = self._identify_low_quality_items(cursor, result)
                
                # 개선 권장사항 생성
                result = self._generate_recommendations(result)
                
        except Exception as e:
            error_msg = f"Error during validation: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        # 검증 시간 계산
        end_time = datetime.now()
        result.validation_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Validation completed:")
        logger.info(f"  - Total laws: {result.total_laws}")
        logger.info(f"  - Total articles: {result.total_articles}")
        logger.info(f"  - Total cases: {result.total_cases}")
        logger.info(f"  - Quality distribution: {result.quality_distribution}")
        logger.info(f"  - Low quality items: {len(result.low_quality_items)}")
        logger.info(f"  - Validation time: {result.validation_time:.2f} seconds")
        
        return result
    
    def _collect_basic_statistics(self, cursor: sqlite3.Cursor, result: ValidationResult) -> ValidationResult:
        """기본 통계 수집"""
        try:
            # 법률 통계
            cursor.execute("SELECT COUNT(*) FROM assembly_laws")
            result.total_laws = cursor.fetchone()[0]
            
            # 조문 통계
            cursor.execute("SELECT COUNT(*) FROM assembly_articles")
            result.total_articles = cursor.fetchone()[0]
            
            # 판례 통계
            cursor.execute("SELECT COUNT(*) FROM precedent_cases")
            result.total_cases = cursor.fetchone()[0]
            
            logger.info(f"Basic statistics collected: {result.total_laws} laws, {result.total_articles} articles, {result.total_cases} cases")
            
        except Exception as e:
            error_msg = f"Error collecting basic statistics: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return result
    
    def _analyze_quality_distribution(self, cursor: sqlite3.Cursor, result: ValidationResult) -> ValidationResult:
        """품질 분포 분석"""
        try:
            # 법률 품질 분포
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN parsing_quality_score >= 0.9 THEN 'excellent'
                        WHEN parsing_quality_score >= 0.7 THEN 'good'
                        WHEN parsing_quality_score >= 0.5 THEN 'fair'
                        ELSE 'poor'
                    END as quality_level,
                    COUNT(*) as count
                FROM assembly_laws
                GROUP BY quality_level
            """)
            
            quality_dist = {}
            for row in cursor.fetchall():
                quality_dist[row['quality_level']] = row['count']
            
            result.quality_distribution = quality_dist
            
            # 평균 품질 점수
            cursor.execute("SELECT AVG(parsing_quality_score) FROM assembly_laws")
            avg_quality = cursor.fetchone()[0]
            logger.info(f"Average law quality score: {avg_quality:.3f}")
            
            # 조문 품질 분포
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN parsing_quality_score >= 0.9 THEN 'excellent'
                        WHEN parsing_quality_score >= 0.7 THEN 'good'
                        WHEN parsing_quality_score >= 0.5 THEN 'fair'
                        ELSE 'poor'
                    END as quality_level,
                    COUNT(*) as count
                FROM assembly_articles
                GROUP BY quality_level
            """)
            
            article_quality_dist = {}
            for row in cursor.fetchall():
                article_quality_dist[row['quality_level']] = row['count']
            
            logger.info(f"Article quality distribution: {article_quality_dist}")
            
        except Exception as e:
            error_msg = f"Error analyzing quality distribution: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return result
    
    def _identify_low_quality_items(self, cursor: sqlite3.Cursor, result: ValidationResult) -> ValidationResult:
        """저품질 항목 식별"""
        try:
            # 저품질 법률 식별
            cursor.execute("""
                SELECT 
                    law_name,
                    parsing_quality_score,
                    parsing_method,
                    article_count,
                    created_at
                FROM assembly_laws
                WHERE parsing_quality_score < 0.5
                ORDER BY parsing_quality_score ASC
                LIMIT 20
            """)
            
            low_quality_laws = []
            for row in cursor.fetchall():
                low_quality_laws.append({
                    'type': 'law',
                    'name': row['law_name'],
                    'quality_score': row['parsing_quality_score'],
                    'parsing_method': row['parsing_method'],
                    'article_count': row['article_count'],
                    'created_at': row['created_at']
                })
            
            # 저품질 조문 식별
            cursor.execute("""
                SELECT 
                    al.law_name,
                    aa.article_number,
                    aa.parsing_quality_score,
                    aa.parsing_method,
                    aa.word_count,
                    aa.char_count
                FROM assembly_articles aa
                JOIN assembly_laws al ON aa.law_id = al.law_id
                WHERE aa.parsing_quality_score < 0.5
                ORDER BY aa.parsing_quality_score ASC
                LIMIT 20
            """)
            
            low_quality_articles = []
            for row in cursor.fetchall():
                low_quality_articles.append({
                    'type': 'article',
                    'law_name': row['law_name'],
                    'article_number': row['article_number'],
                    'quality_score': row['parsing_quality_score'],
                    'parsing_method': row['parsing_method'],
                    'word_count': row['word_count'],
                    'char_count': row['char_count']
                })
            
            result.low_quality_items = low_quality_laws + low_quality_articles
            
            logger.info(f"Identified {len(low_quality_laws)} low-quality laws and {len(low_quality_articles)} low-quality articles")
            
        except Exception as e:
            error_msg = f"Error identifying low quality items: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return result
    
    def _generate_recommendations(self, result: ValidationResult) -> ValidationResult:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 품질 분포 기반 권장사항
        total_items = sum(result.quality_distribution.values())
        if total_items > 0:
            poor_ratio = result.quality_distribution.get('poor', 0) / total_items
            fair_ratio = result.quality_distribution.get('fair', 0) / total_items
            
            if poor_ratio > 0.1:  # 10% 이상이 poor
                recommendations.append("10% 이상의 법률이 저품질입니다. 파싱 알고리즘 개선이 필요합니다.")
            
            if fair_ratio > 0.3:  # 30% 이상이 fair
                recommendations.append("30% 이상의 법률이 보통 품질입니다. 품질 향상 작업을 권장합니다.")
            
            excellent_ratio = result.quality_distribution.get('excellent', 0) / total_items
            if excellent_ratio < 0.2:  # 20% 미만이 excellent
                recommendations.append("우수 품질 법률이 20% 미만입니다. 전체적인 품질 개선이 필요합니다.")
        
        # 저품질 항목 기반 권장사항
        if len(result.low_quality_items) > 0:
            recommendations.append(f"{len(result.low_quality_items)}개의 저품질 항목이 식별되었습니다. 수동 검토가 필요합니다.")
            
            # 파싱 방법별 분석
            parsing_methods = {}
            for item in result.low_quality_items:
                method = item.get('parsing_method', 'unknown')
                parsing_methods[method] = parsing_methods.get(method, 0) + 1
            
            for method, count in parsing_methods.items():
                if count > 5:
                    recommendations.append(f"{method} 파싱 방법으로 처리된 항목 중 {count}개가 저품질입니다. 해당 방법의 개선이 필요합니다.")
        
        # 데이터 완성도 기반 권장사항
        if result.total_laws == 0:
            recommendations.append("법률 데이터가 없습니다. 데이터 임포트를 확인하세요.")
        
        if result.total_articles == 0:
            recommendations.append("조문 데이터가 없습니다. 조문 파싱을 확인하세요.")
        
        if result.total_cases == 0:
            recommendations.append("판례 데이터가 없습니다. 판례 임포트를 확인하세요.")
        
        result.recommendations = recommendations
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return result
    
    def generate_quality_report(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """품질 리포트 생성"""
        report = {
            'validation_summary': {
                'total_laws': validation_result.total_laws,
                'total_articles': validation_result.total_articles,
                'total_cases': validation_result.total_cases,
                'validation_time': validation_result.validation_time,
                'total_errors': len(validation_result.errors)
            },
            'quality_distribution': validation_result.quality_distribution,
            'low_quality_items': validation_result.low_quality_items[:10],  # 상위 10개만
            'recommendations': validation_result.recommendations,
            'errors': validation_result.errors,
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def optimize_database_performance(self) -> bool:
        """데이터베이스 성능 최적화"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                logger.info("⚡ Optimizing database performance...")
                
                # 인덱스 생성 및 최적화
                indexes_to_create = [
                    "CREATE INDEX IF NOT EXISTS idx_assembly_laws_quality ON assembly_laws(parsing_quality_score)",
                    "CREATE INDEX IF NOT EXISTS idx_assembly_laws_method ON assembly_laws(parsing_method)",
                    "CREATE INDEX IF NOT EXISTS idx_assembly_laws_type ON assembly_laws(law_type)",
                    "CREATE INDEX IF NOT EXISTS idx_assembly_articles_quality ON assembly_articles(parsing_quality_score)",
                    "CREATE INDEX IF NOT EXISTS idx_assembly_articles_method ON assembly_articles(parsing_method)",
                    "CREATE INDEX IF NOT EXISTS idx_precedent_cases_field ON precedent_cases(field)",
                    "CREATE INDEX IF NOT EXISTS idx_precedent_cases_court ON precedent_cases(court)"
                ]
                
                for index_sql in indexes_to_create:
                    try:
                        cursor.execute(index_sql)
                        logger.debug(f"Created index: {index_sql}")
                    except sqlite3.OperationalError as e:
                        logger.warning(f"Index creation failed: {e}")
                
                # 통계 업데이트
                cursor.execute("ANALYZE")
                logger.info("Database statistics updated")
                
                # VACUUM 실행 (선택적)
                cursor.execute("VACUUM")
                logger.info("Database vacuumed")
                
                conn.commit()
                logger.info("✅ Database performance optimization completed")
                return True
                
        except Exception as e:
            logger.error(f"Error optimizing database performance: {e}")
            return False
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """데이터 무결성 검증"""
        integrity_results = {
            'foreign_key_violations': [],
            'orphaned_records': [],
            'duplicate_records': [],
            'missing_required_fields': [],
            'data_type_violations': []
        }
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                logger.info("🔍 Validating data integrity...")
                
                # 외래키 위반 검사
                cursor.execute("""
                    SELECT aa.law_id, COUNT(*) as count
                    FROM assembly_articles aa
                    LEFT JOIN assembly_laws al ON aa.law_id = al.law_id
                    WHERE al.law_id IS NULL
                    GROUP BY aa.law_id
                """)
                
                orphaned_articles = cursor.fetchall()
                if orphaned_articles:
                    integrity_results['orphaned_records'].append({
                        'type': 'orphaned_articles',
                        'count': len(orphaned_articles),
                        'details': [dict(row) for row in orphaned_articles]
                    })
                
                # 중복 레코드 검사
                cursor.execute("""
                    SELECT law_id, COUNT(*) as count
                    FROM assembly_laws
                    GROUP BY law_id
                    HAVING COUNT(*) > 1
                """)
                
                duplicate_laws = cursor.fetchall()
                if duplicate_laws:
                    integrity_results['duplicate_records'].append({
                        'type': 'duplicate_laws',
                        'count': len(duplicate_laws),
                        'details': [dict(row) for row in duplicate_laws]
                    })
                
                # 필수 필드 누락 검사
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM assembly_laws
                    WHERE law_name IS NULL OR law_name = ''
                """)
                
                missing_law_names = cursor.fetchone()[0]
                if missing_law_names > 0:
                    integrity_results['missing_required_fields'].append({
                        'type': 'missing_law_names',
                        'count': missing_law_names
                    })
                
                logger.info("✅ Data integrity validation completed")
                
        except Exception as e:
            logger.error(f"Error validating data integrity: {e}")
            integrity_results['errors'] = [str(e)]
        
        return integrity_results


def main():
    """메인 함수"""
    logger.info("🚀 Starting quality validation and optimization...")
    
    # 검증 시스템 초기화
    validator = QualityValidationSystem()
    
    # 데이터 품질 검증
    logger.info("\n📋 Phase 1: Validating data quality...")
    validation_result = validator.validate_imported_data()
    
    # 데이터 무결성 검증
    logger.info("\n📋 Phase 2: Validating data integrity...")
    integrity_result = validator.validate_data_integrity()
    
    # 데이터베이스 성능 최적화
    logger.info("\n📋 Phase 3: Optimizing database performance...")
    optimization_success = validator.optimize_database_performance()
    
    # 품질 리포트 생성
    logger.info("\n📋 Phase 4: Generating quality report...")
    quality_report = validator.generate_quality_report(validation_result)
    
    # 통합 결과 생성
    final_result = {
        'quality_validation': quality_report,
        'data_integrity': integrity_result,
        'performance_optimization': {
            'optimization_success': optimization_success,
            'optimization_time': datetime.now().isoformat()
        },
        'summary': {
            'total_laws': validation_result.total_laws,
            'total_articles': validation_result.total_articles,
            'total_cases': validation_result.total_cases,
            'quality_distribution': validation_result.quality_distribution,
            'low_quality_count': len(validation_result.low_quality_items),
            'recommendations_count': len(validation_result.recommendations),
            'total_errors': len(validation_result.errors)
        }
    }
    
    # 리포트 저장
    with open("data/quality_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n📊 Detailed report saved to: data/quality_validation_report.json")
    logger.info("✅ Quality validation and optimization completed successfully!")
    
    # 주요 결과 출력
    print(f"\n📊 Quality Validation Summary:")
    print(f"  - Total Laws: {validation_result.total_laws:,}")
    print(f"  - Total Articles: {validation_result.total_articles:,}")
    print(f"  - Total Cases: {validation_result.total_cases:,}")
    print(f"  - Quality Distribution: {validation_result.quality_distribution}")
    print(f"  - Low Quality Items: {len(validation_result.low_quality_items)}")
    print(f"  - Recommendations: {len(validation_result.recommendations)}")
    
    if validation_result.errors:
        print(f"\n⚠️ Validation completed with {len(validation_result.errors)} errors")
        print("Check logs for details.")
    else:
        print("\n🎉 Quality validation completed successfully!")
    
    return final_result


if __name__ == "__main__":
    result = main()
    print("\n✅ Quality validation and optimization process completed!")
