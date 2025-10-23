#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법령용어 수집 상태 모니터링 스크립트

법령용어 수집 시스템의 상태를 모니터링하고 보고서를 생성합니다.
- 수집 상태 조회
- 로그 파일 분석
- 통계 정보 제공
- 상태 보고서 생성
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from scripts.data_collection.law_open_api.utils import (
    TimestampManager, 
    get_log_files,
    cleanup_old_logs
)
from scripts.data_collection.law_open_api.collectors import IncrementalLegalTermCollector
from source.data.law_open_api_client import LawOpenAPIClient

logger = logging.getLogger(__name__)


def print_header(title: str):
    """헤더 출력"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_section(title: str):
    """섹션 헤더 출력"""
    print(f"\n📋 {title}")
    print("-" * 40)


def get_collection_status() -> Dict[str, Any]:
    """수집 상태 조회"""
    try:
        timestamp_manager = TimestampManager()
        stats = timestamp_manager.get_all_stats()
        
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def get_log_analysis(log_dir: str = "logs/legal_term_collection") -> Dict[str, Any]:
    """로그 파일 분석"""
    try:
        log_files = get_log_files(log_dir)
        
        analysis = {
            "log_files": log_files,
            "total_files": sum(len(files) for files in log_files.values()),
            "analysis_time": datetime.now().isoformat()
        }
        
        # 최근 로그 파일 정보
        recent_files = {}
        for log_type, files in log_files.items():
            if files:
                recent_file = Path(files[0])
                if recent_file.exists():
                    stat = recent_file.stat()
                    recent_files[log_type] = {
                        "file": str(recent_file),
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }
        
        analysis["recent_files"] = recent_files
        
        return analysis
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def get_data_directory_info(data_dir: str = "data/raw/law_open_api/legal_terms") -> Dict[str, Any]:
    """데이터 디렉토리 정보 조회"""
    try:
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return {
                "status": "not_found",
                "message": f"데이터 디렉토리가 존재하지 않습니다: {data_dir}"
            }
        
        info = {
            "directory": str(data_path),
            "exists": True,
            "total_size_mb": 0,
            "file_count": 0,
            "subdirectories": [],
            "recent_files": []
        }
        
        # 디렉토리 크기 및 파일 수 계산
        for item in data_path.rglob("*"):
            if item.is_file():
                info["file_count"] += 1
                info["total_size_mb"] += item.stat().st_size
        
        info["total_size_mb"] = round(info["total_size_mb"] / (1024 * 1024), 2)
        
        # 하위 디렉토리 목록
        for item in data_path.iterdir():
            if item.is_dir():
                info["subdirectories"].append(str(item.name))
        
        # 최근 파일들 (최대 10개)
        recent_files = []
        for item in sorted(data_path.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            stat = item.stat()
            recent_files.append({
                "file": str(item.relative_to(data_path)),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        info["recent_files"] = recent_files
        
        return info
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def test_api_connection() -> Dict[str, Any]:
    """API 연결 테스트"""
    try:
        client = LawOpenAPIClient()
        is_connected = client.test_connection()
        
        return {
            "status": "success" if is_connected else "failed",
            "connected": is_connected,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def generate_status_report() -> Dict[str, Any]:
    """상태 보고서 생성"""
    report = {
        "generated_at": datetime.now().isoformat(),
        "collection_status": get_collection_status(),
        "log_analysis": get_log_analysis(),
        "data_directory": get_data_directory_info(),
        "api_connection": test_api_connection()
    }
    
    return report


def print_collection_status(status_data: Dict[str, Any]):
    """수집 상태 출력"""
    if status_data["status"] == "error":
        print(f"❌ 수집 상태 조회 실패: {status_data['error']}")
        return
    
    data = status_data["data"]
    
    if not data:
        print("📊 수집 이력이 없습니다.")
        return
    
    for data_type, stats in data.items():
        print(f"\n📊 {data_type}")
        print(f"  수집 횟수: {stats['collection_count']}회")
        print(f"  성공 횟수: {stats['success_count']}회")
        print(f"  실패 횟수: {stats['error_count']}회")
        print(f"  성공률: {stats['success_rate']}%")
        
        if stats['last_collection']:
            last_time = datetime.fromisoformat(stats['last_collection'])
            time_diff = datetime.now() - last_time
            print(f"  마지막 수집: {last_time.strftime('%Y-%m-%d %H:%M:%S')} ({time_diff.days}일 전)")
        
        if stats['last_successful_collection']:
            last_success = datetime.fromisoformat(stats['last_successful_collection'])
            success_diff = datetime.now() - last_success
            print(f"  마지막 성공: {last_success.strftime('%Y-%m-%d %H:%M:%S')} ({success_diff.days}일 전)")


def print_log_analysis(log_data: Dict[str, Any]):
    """로그 분석 결과 출력"""
    if "error" in log_data:
        print(f"❌ 로그 분석 실패: {log_data['error']}")
        return
    
    print(f"📁 총 로그 파일: {log_data['total_files']}개")
    
    recent_files = log_data.get("recent_files", {})
    if recent_files:
        print(f"\n📄 최근 로그 파일:")
        for log_type, file_info in recent_files.items():
            print(f"  {log_type}: {file_info['file']} ({file_info['size_mb']}MB, {file_info['modified'][:19]})")


def print_data_directory_info(data_info: Dict[str, Any]):
    """데이터 디렉토리 정보 출력"""
    if data_info["status"] == "not_found":
        print(f"❌ {data_info['message']}")
        return
    
    if data_info["status"] == "error":
        print(f"❌ 데이터 디렉토리 정보 조회 실패: {data_info['error']}")
        return
    
    print(f"📁 디렉토리: {data_info['directory']}")
    print(f"📊 총 크기: {data_info['total_size_mb']}MB")
    print(f"📄 파일 수: {data_info['file_count']}개")
    
    if data_info['subdirectories']:
        print(f"📂 하위 디렉토리: {', '.join(data_info['subdirectories'])}")
    
    recent_files = data_info.get("recent_files", [])
    if recent_files:
        print(f"\n📄 최근 파일 (최대 10개):")
        for file_info in recent_files[:5]:  # 최대 5개만 표시
            print(f"  {file_info['file']} ({file_info['size_mb']}MB, {file_info['modified'][:19]})")


def print_api_connection_status(api_data: Dict[str, Any]):
    """API 연결 상태 출력"""
    if api_data["status"] == "error":
        print(f"❌ API 연결 테스트 실패: {api_data['error']}")
        return
    
    if api_data["connected"]:
        print("✅ API 연결 정상")
    else:
        print("❌ API 연결 실패")


def save_report(report: Dict[str, Any], output_file: str = None):
    """보고서 저장"""
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"reports/legal_term_status_{timestamp}.json"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 보고서 저장 완료: {output_path}")
        
    except Exception as e:
        print(f"❌ 보고서 저장 실패: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='법령용어 수집 상태 모니터링')
    parser.add_argument('--output', '-o', type=str,
                       help='보고서 출력 파일 경로')
    parser.add_argument('--log-dir', type=str, default='logs/legal_term_collection',
                       help='로그 디렉토리 경로')
    parser.add_argument('--data-dir', type=str, default='data/raw/law_open_api/legal_terms',
                       help='데이터 디렉토리 경로')
    parser.add_argument('--cleanup', action='store_true',
                       help='오래된 로그 파일 정리')
    parser.add_argument('--days', type=int, default=30,
                       help='로그 보관 일수 (기본값: 30일)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='상세 출력 없이 보고서만 생성')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print_header("법령용어 수집 상태 모니터링")
        print(f"모니터링 시간: {datetime.now()}")
    
    try:
        # 보고서 생성
        report = generate_status_report()
        
        if not args.quiet:
            # 수집 상태 출력
            print_section("수집 상태")
            print_collection_status(report["collection_status"])
            
            # 로그 분석 출력
            print_section("로그 분석")
            print_log_analysis(report["log_analysis"])
            
            # 데이터 디렉토리 정보 출력
            print_section("데이터 디렉토리")
            print_data_directory_info(report["data_directory"])
            
            # API 연결 상태 출력
            print_section("API 연결 상태")
            print_api_connection_status(report["api_connection"])
        
        # 보고서 저장
        if args.output:
            save_report(report, args.output)
        
        # 로그 정리
        if args.cleanup:
            if not args.quiet:
                print_section("로그 정리")
            cleanup_old_logs(args.log_dir, args.days)
            if not args.quiet:
                print(f"✅ {args.days}일 이상 된 로그 파일 정리 완료")
        
        if not args.quiet:
            print_header("모니터링 완료")
        
        return 0
        
    except Exception as e:
        print(f"❌ 모니터링 실패: {e}")
        logger.error(f"모니터링 스크립트 실행 실패: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)




