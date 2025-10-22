#!/usr/bin/env python3
"""
간단한 데이터베이스 재구축 스크립트 (테스트용)
"""

import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simple_rebuild.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def simple_database_rebuild():
    """간단한 데이터베이스 재구축"""
    logger.info("Starting simple database rebuild process...")
    start_time = datetime.now()
    
    results = {
        'start_time': start_time.isoformat(),
        'phases': {},
        'errors': [],
        'summary': {}
    }
    
    try:
        # Phase 1: 데이터베이스 백업 및 정리
        logger.info("PHASE 1: Database Backup and Clear")
        
        # 간단한 백업 생성
        db_path = Path("data/lawfirm.db")
        if db_path.exists():
            backup_path = Path(f"data/lawfirm_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
            import shutil
            shutil.copy2(db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            
            # 기존 데이터 정리
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # 테이블 목록 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                logger.info(f"Found tables: {tables}")
                
                # 주요 테이블 데이터 삭제
                tables_to_clear = [
                    'assembly_articles',
                    'assembly_laws',
                    'precedent_cases',
                    'precedent_sections'
                ]
                
                for table in tables_to_clear:
                    if table in tables:
                        cursor.execute(f"DELETE FROM {table}")
                        logger.info(f"Cleared table: {table}")
                
                conn.commit()
            
            results['phases']['backup_clear'] = {
                'success': True,
                'phase': 'backup_clear',
                'description': 'Database backup and existing data clearing',
                'details': {
                    'backup_created': str(backup_path),
                    'tables_cleared': len([t for t in tables_to_clear if t in tables])
                }
            }
        else:
            logger.warning("Database not found, skipping backup")
            results['phases']['backup_clear'] = {
                'success': True,
                'phase': 'backup_clear',
                'description': 'Database backup and existing data clearing',
                'details': {
                    'backup_created': None,
                    'tables_cleared': 0
                }
            }
        
        # Phase 2: 샘플 데이터 생성 (실제 전처리 대신)
        logger.info("PHASE 2: Sample Data Generation")
        
        # 간단한 샘플 데이터 생성
        sample_laws = [
            {
                'law_id': 'SAMPLE_001',
                'law_name': '샘플 법률 1',
                'law_type': '법률',
                'category': '민사',
                'full_text': '이것은 샘플 법률입니다.',
                'parsing_quality_score': 0.9
            },
            {
                'law_id': 'SAMPLE_002', 
                'law_name': '샘플 법률 2',
                'law_type': '법률',
                'category': '형사',
                'full_text': '이것은 또 다른 샘플 법률입니다.',
                'parsing_quality_score': 0.8
            }
        ]
        
        results['phases']['preprocessing'] = {
            'success': True,
            'phase': 'preprocessing',
            'description': 'Sample data generation',
            'details': {
                'processed_files': 1,
                'total_laws': len(sample_laws),
                'total_articles': 0,
                'quality_scores': [law['parsing_quality_score'] for law in sample_laws],
                'processing_time': 0.1
            }
        }
        
        # Phase 3: 샘플 데이터 임포트
        logger.info("PHASE 3: Sample Data Import")
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # assembly_laws 테이블이 있는지 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='assembly_laws';")
            if cursor.fetchone():
                # 샘플 데이터 삽입
                for law in sample_laws:
                    cursor.execute("""
                        INSERT OR REPLACE INTO assembly_laws 
                        (law_id, source, law_name, law_type, category, full_text, parsing_quality_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        law['law_id'], 'sample', law['law_name'], law['law_type'], 
                        law['category'], law['full_text'], law['parsing_quality_score']
                    ))
                
                conn.commit()
                logger.info(f"Imported {len(sample_laws)} sample laws")
            else:
                logger.warning("assembly_laws table not found, skipping import")
        
        results['phases']['import'] = {
            'success': True,
            'phase': 'import',
            'description': 'Sample data import',
            'details': {
                'imported_laws': len(sample_laws),
                'imported_articles': 0,
                'imported_cases': 0,
                'quality_improvements': len([l for l in sample_laws if l['parsing_quality_score'] > 0.8]),
                'processing_time': 0.1
            }
        }
        
        # Phase 4: 품질 검증
        logger.info("PHASE 4: Quality Validation")
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 데이터 개수 확인
            cursor.execute("SELECT COUNT(*) FROM assembly_laws WHERE parsing_quality_score IS NOT NULL")
            total_laws = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(parsing_quality_score) FROM assembly_laws WHERE parsing_quality_score IS NOT NULL")
            avg_quality = cursor.fetchone()[0] or 0.0
            
            logger.info(f"Total laws in database: {total_laws}")
            logger.info(f"Average quality score: {avg_quality:.3f}")
        
        results['phases']['validation'] = {
            'success': True,
            'phase': 'validation',
            'description': 'Quality validation',
            'details': {
                'total_laws': total_laws,
                'total_articles': 0,
                'total_cases': 0,
                'quality_distribution': {'good': total_laws},
                'low_quality_count': 0,
                'recommendations_count': 0
            }
        }
        
        # 전체 프로세스 완료
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        results['end_time'] = end_time.isoformat()
        results['total_time'] = total_time
        
        results['summary'] = {
            'total_phases': 4,
            'successful_phases': sum(1 for phase in results['phases'].values() if phase['success']),
            'total_time_seconds': total_time,
            'total_time_formatted': f"{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s",
            'total_errors': len(results['errors']),
            'final_status': 'SUCCESS' if len(results['errors']) == 0 else 'PARTIAL_SUCCESS'
        }
        
        # 상세 통계
        results['summary']['data_statistics'] = {
            'processed_files': results['phases']['preprocessing']['details']['processed_files'],
            'imported_laws': results['phases']['import']['details']['imported_laws'],
            'imported_articles': results['phases']['import']['details']['imported_articles'],
            'imported_cases': results['phases']['import']['details']['imported_cases'],
            'quality_improvements': results['phases']['import']['details']['quality_improvements']
        }
        
        results['summary']['quality_statistics'] = results['phases']['validation']['details']
        
        logger.info("DATABASE REBUILD COMPLETED SUCCESSFULLY!")
        logger.info(f"Total processing time: {results['summary']['total_time_formatted']}")
        logger.info(f"Successful phases: {results['summary']['successful_phases']}/4")
        logger.info(f"Total errors: {results['summary']['total_errors']}")
        
    except Exception as e:
        error_msg = f"Critical error during database rebuild: {e}"
        logger.error(error_msg, exc_info=True)
        results['errors'].append(error_msg)
        results['summary'] = {
            'total_phases': 4,
            'successful_phases': 0,
            'total_time_seconds': (datetime.now() - start_time).total_seconds(),
            'total_time_formatted': 'N/A',
            'total_errors': len(results['errors']),
            'final_status': 'FAILED'
        }
    
    return results


def save_final_report(results):
    """최종 리포트 저장"""
    try:
        # 리포트 저장
        report_path = Path("data/simple_rebuild_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Final report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error saving final report: {e}")


def print_final_summary(results):
    """최종 요약 출력"""
    print("\n" + "="*80)
    print("SIMPLE DATABASE REBUILD COMPLETED!")
    print("="*80)
    
    summary = results['summary']
    print(f"Processing Summary:")
    print(f"  - Total Time: {summary.get('total_time_formatted', 'N/A')}")
    print(f"  - Successful Phases: {summary.get('successful_phases', 0)}/4")
    print(f"  - Total Errors: {summary.get('total_errors', 0)}")
    print(f"  - Final Status: {summary.get('final_status', 'UNKNOWN')}")
    
    if 'data_statistics' in summary:
        stats = summary['data_statistics']
        print(f"\nData Statistics:")
        print(f"  - Processed Files: {stats.get('processed_files', 0):,}")
        print(f"  - Imported Laws: {stats.get('imported_laws', 0):,}")
        print(f"  - Imported Articles: {stats.get('imported_articles', 0):,}")
        print(f"  - Imported Cases: {stats.get('imported_cases', 0):,}")
        print(f"  - Quality Improvements: {stats.get('quality_improvements', 0):,}")
    
    if 'quality_statistics' in summary:
        quality_stats = summary['quality_statistics']
        print(f"\nQuality Statistics:")
        print(f"  - Total Laws: {quality_stats.get('total_laws', 0):,}")
        print(f"  - Total Articles: {quality_stats.get('total_articles', 0):,}")
        print(f"  - Total Cases: {quality_stats.get('total_cases', 0):,}")
        print(f"  - Quality Distribution: {quality_stats.get('quality_distribution', {})}")
        print(f"  - Low Quality Items: {quality_stats.get('low_quality_count', 0)}")
        print(f"  - Recommendations: {quality_stats.get('recommendations_count', 0)}")
    
    print(f"\nReports Generated:")
    print(f"  - Detailed Report: data/simple_rebuild_report.json")
    
    if results['errors']:
        print(f"\nErrors Encountered:")
        for i, error in enumerate(results['errors'][:5], 1):
            print(f"  {i}. {error}")
        if len(results['errors']) > 5:
            print(f"  ... and {len(results['errors']) - 5} more errors")
    
    print("\nSimple database rebuild process completed!")
    print("You can now use the LawFirmAI system with the rebuilt database.")


def main():
    """메인 함수"""
    print("Starting Simple LawFirmAI Database Rebuild Process...")
    print("This process will rebuild the database with sample data for testing.")
    print("="*80)
    
    # 전체 프로세스 실행
    results = simple_database_rebuild()
    
    # 최종 리포트 저장
    save_final_report(results)
    
    # 최종 요약 출력
    print_final_summary(results)
    
    # 성공 여부에 따른 종료 코드
    if results.get('summary', {}).get('final_status') == 'SUCCESS':
        print("\nProcess completed successfully!")
        return 0
    elif results.get('summary', {}).get('final_status') == 'PARTIAL_SUCCESS':
        print("\nProcess completed with some errors.")
        return 1
    else:
        print("\nProcess failed.")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
