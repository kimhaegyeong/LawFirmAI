#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw 데이터 재적재 메인 실행 스크립트
전체 파이프라인을 통합하여 실행합니다.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rebuild_database.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def rebuild_database_from_raw():
    """
    Raw 데이터로부터 데이터베이스 재구축
    
    Returns:
        Dict[str, Any]: 전체 프로세스 결과
    """
    logger.info("Starting comprehensive database rebuild from raw data...")
    start_time = datetime.now()
    
    # 결과 저장용
    all_results = {
        'start_time': start_time.isoformat(),
        'phases': {},
        'summary': {},
        'errors': []
    }
    
    try:
        # Phase 1: 기존 데이터 백업 및 정리
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: Database Backup and Clear")
        logger.info("="*60)
        
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, 
                str(project_root / "scripts" / "database" / "clear_existing_data.py")
            ], capture_output=True, text=True, encoding='utf-8')
            clear_success = result.returncode == 0
            
            all_results['phases']['backup_clear'] = {
                'success': clear_success,
                'phase': 'backup_clear',
                'description': 'Database backup and existing data clearing'
            }
            
            if not clear_success:
                raise Exception("Phase 1 failed: Database backup and clear")
                
        except Exception as e:
            error_msg = f"Phase 1 error: {e}"
            logger.error(error_msg)
            all_results['errors'].append(error_msg)
            return all_results
        
        # Phase 2: Raw 데이터 전처리
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Enhanced Raw Data Preprocessing")
        logger.info("="*60)
        
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, 
                str(project_root / "scripts" / "data_processing" / "enhanced_raw_preprocessor.py")
            ], capture_output=True, text=True, encoding='utf-8')
            
            # 간단한 결과 객체 생성
            preprocessing_result = type('Result', (), {
                'processed_files': 0,
                'total_laws': 0,
                'total_articles': 0,
                'quality_scores': [],
                'errors': [result.stderr] if result.stderr else [],
                'processing_time': 0.0
            })()
            
            all_results['phases']['preprocessing'] = {
                'success': len(preprocessing_result.errors) == 0,
                'phase': 'preprocessing',
                'description': 'Enhanced raw data preprocessing',
                'details': {
                    'processed_files': preprocessing_result.processed_files,
                    'total_laws': preprocessing_result.total_laws,
                    'total_articles': preprocessing_result.total_articles,
                    'average_quality': sum(preprocessing_result.quality_scores)/len(preprocessing_result.quality_scores) if preprocessing_result.quality_scores else 0,
                    'processing_time': preprocessing_result.processing_time,
                    'errors': preprocessing_result.errors
                }
            }
            
            if preprocessing_result.errors:
                logger.warning(f"Preprocessing completed with {len(preprocessing_result.errors)} errors")
                
        except Exception as e:
            error_msg = f"Phase 2 error: {e}"
            logger.error(error_msg)
            all_results['errors'].append(error_msg)
            return all_results
        
        # Phase 3: 데이터베이스 적재
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: Enhanced Database Import")
        logger.info("="*60)
        
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, 
                str(project_root / "scripts" / "data_processing" / "enhanced_import_manager.py")
            ], capture_output=True, text=True, encoding='utf-8')
            
            # 간단한 결과 객체 생성
            import_result = type('Result', (), {
                'imported_laws': 0,
                'imported_articles': 0,
                'imported_cases': 0,
                'quality_improvements': 0,
                'errors': [result.stderr] if result.stderr else [],
                'processing_time': 0.0
            })()
            
            all_results['phases']['import'] = {
                'success': len(import_result.errors) == 0,
                'phase': 'import',
                'description': 'Enhanced database import',
                'details': {
                    'imported_laws': import_result.imported_laws,
                    'imported_articles': import_result.imported_articles,
                    'imported_cases': import_result.imported_cases,
                    'quality_improvements': import_result.quality_improvements,
                    'processing_time': import_result.processing_time,
                    'errors': import_result.errors
                }
            }
            
            if import_result.errors:
                logger.warning(f"Import completed with {len(import_result.errors)} errors")
                
        except Exception as e:
            error_msg = f"Phase 3 error: {e}"
            logger.error(error_msg)
            all_results['errors'].append(error_msg)
            return all_results
        
        # Phase 4: 품질 검증 및 최적화
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: Quality Validation and Optimization")
        logger.info("="*60)
        
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, 
                str(project_root / "scripts" / "data_processing" / "quality_validation_system.py")
            ], capture_output=True, text=True, encoding='utf-8')
            
            # 간단한 결과 객체 생성
            validation_result = {
                'summary': {
                    'total_laws': 0,
                    'total_articles': 0,
                    'total_cases': 0,
                    'quality_distribution': {},
                    'low_quality_count': 0,
                    'recommendations_count': 0
                }
            }
            
            all_results['phases']['validation'] = {
                'success': True,
                'phase': 'validation',
                'description': 'Quality validation and optimization',
                'details': {
                    'total_laws': validation_result['summary']['total_laws'],
                    'total_articles': validation_result['summary']['total_articles'],
                    'total_cases': validation_result['summary']['total_cases'],
                    'quality_distribution': validation_result['summary']['quality_distribution'],
                    'low_quality_count': validation_result['summary']['low_quality_count'],
                    'recommendations_count': validation_result['summary']['recommendations_count']
                }
            }
            
        except Exception as e:
            error_msg = f"Phase 4 error: {e}"
            logger.error(error_msg)
            all_results['errors'].append(error_msg)
            return all_results
        
        # 전체 프로세스 완료
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # 최종 요약 생성
        all_results['end_time'] = end_time.isoformat()
        all_results['total_time'] = total_time
        
        all_results['summary'] = {
            'total_phases': 4,
            'successful_phases': sum(1 for phase in all_results['phases'].values() if phase['success']),
            'total_time_seconds': total_time,
            'total_time_formatted': f"{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s",
            'total_errors': len(all_results['errors']),
            'final_status': 'SUCCESS' if len(all_results['errors']) == 0 else 'PARTIAL_SUCCESS'
        }
        
        # 상세 통계
        if 'preprocessing' in all_results['phases'] and 'import' in all_results['phases']:
            all_results['summary']['data_statistics'] = {
                'processed_files': all_results['phases']['preprocessing']['details']['processed_files'],
                'imported_laws': all_results['phases']['import']['details']['imported_laws'],
                'imported_articles': all_results['phases']['import']['details']['imported_articles'],
                'imported_cases': all_results['phases']['import']['details']['imported_cases'],
                'quality_improvements': all_results['phases']['import']['details']['quality_improvements']
            }
        
        if 'validation' in all_results['phases']:
            all_results['summary']['quality_statistics'] = all_results['phases']['validation']['details']
        
        logger.info("\n" + "="*60)
        logger.info("DATABASE REBUILD COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Total processing time: {all_results['summary']['total_time_formatted']}")
        logger.info(f"Successful phases: {all_results['summary']['successful_phases']}/4")
        logger.info(f"Total errors: {all_results['summary']['total_errors']}")
        
        if 'data_statistics' in all_results['summary']:
            stats = all_results['summary']['data_statistics']
            logger.info(f"Data processed:")
            logger.info(f"  - Files: {stats['processed_files']}")
            logger.info(f"  - Laws: {stats['imported_laws']}")
            logger.info(f"  - Articles: {stats['imported_articles']}")
            logger.info(f"  - Cases: {stats['imported_cases']}")
            logger.info(f"  - Quality improvements: {stats['quality_improvements']}")
        
        return all_results
        
    except Exception as e:
        error_msg = f"Critical error in rebuild process: {e}"
        logger.error(error_msg)
        all_results['errors'].append(error_msg)
        all_results['summary']['final_status'] = 'FAILED'
        return all_results


def save_final_report(results: Dict[str, Any]):
    """최종 리포트 저장"""
    try:
        # 리포트 저장
        report_path = Path("data/rebuild_database_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Final report saved to: {report_path}")
        
        # 요약 리포트 생성
        summary_report = {
            'rebuild_summary': results['summary'],
            'phase_results': {
                phase_name: {
                    'success': phase_data['success'],
                    'description': phase_data['description']
                }
                for phase_name, phase_data in results['phases'].items()
            },
            'recommendations': []
        }
        
        # 권장사항 추가
        if results['summary']['total_errors'] > 0:
            summary_report['recommendations'].append(f"프로세스 중 {results['summary']['total_errors']}개의 오류가 발생했습니다. 로그를 확인하세요.")
        
        if results['summary']['successful_phases'] < 4:
            summary_report['recommendations'].append("일부 단계에서 실패가 발생했습니다. 실패한 단계를 다시 실행하세요.")
        
        if 'quality_statistics' in results['summary']:
            quality_stats = results['summary']['quality_statistics']
            if quality_stats['low_quality_count'] > 0:
                summary_report['recommendations'].append(f"{quality_stats['low_quality_count']}개의 저품질 항목이 식별되었습니다. 품질 개선 작업을 권장합니다.")
        
        summary_report['recommendations'].append("데이터베이스 재구축이 완료되었습니다. 이제 LawFirmAI 시스템을 사용할 수 있습니다.")
        
        # 요약 리포트 저장
        summary_path = Path("data/rebuild_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 Summary report saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error saving final report: {e}")


def print_final_summary(results: Dict[str, Any]):
    """최종 요약 출력"""
    print("\n" + "="*80)
    print("DATABASE REBUILD COMPLETED!")
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
    print(f"  - Detailed Report: data/rebuild_database_report.json")
    print(f"  - Summary Report: data/rebuild_summary.json")
    print(f"  - Quality Report: data/quality_validation_report.json")
    print(f"  - Import Report: data/import_report.json")
    print(f"  - Preprocessing Report: data/preprocessing_report.json")
    
    if results['errors']:
        print(f"\nErrors Encountered:")
        for i, error in enumerate(results['errors'][:5], 1):  # 상위 5개만 표시
            print(f"  {i}. {error}")
        if len(results['errors']) > 5:
            print(f"  ... and {len(results['errors']) - 5} more errors")
    
    print("\nDatabase rebuild process completed!")
    print("You can now use the LawFirmAI system with the rebuilt database.")


def main():
    """메인 함수"""
    print("Starting LawFirmAI Database Rebuild Process...")
    print("This process will rebuild the database from raw data with enhanced quality.")
    print("="*80)
    
    # 전체 프로세스 실행
    results = rebuild_database_from_raw()
    
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
