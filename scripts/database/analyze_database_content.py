#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터베이스 내용 분석 스크립트

SQLite 데이터베이스의 내용을 분석하여 어떤 데이터가 있는지 확인합니다.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

# 직접 DatabaseManager 클래스 정의 (import 오류 방지)
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any

class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        """데이터베이스 관리자 초기화"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"DatabaseManager initialized with path: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """쿼리 실행"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/database_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatabaseAnalyzer:
    """데이터베이스 분석 클래스"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        """데이터베이스 분석기 초기화"""
        self.db_path = Path(db_path)
        self.db_manager = DatabaseManager(str(self.db_path))
        self.analysis_result = {}
        
        logger.info(f"DatabaseAnalyzer initialized with path: {self.db_path}")
    
    def analyze_database_content(self) -> Dict[str, Any]:
        """데이터베이스 내용 분석"""
        logger.info("데이터베이스 내용 분석 시작...")
        
        analysis = {
            "database_path": str(self.db_path),
            "analysis_timestamp": datetime.now().isoformat(),
            "table_counts": {},
            "document_types": {},
            "content_samples": {},
            "metadata_analysis": {},
            "total_documents": 0,
            "analysis_summary": {}
        }
        
        try:
            # 1. 테이블별 데이터 수 확인
            analysis["table_counts"] = self._analyze_table_counts()
            
            # 2. 문서 타입별 분석
            analysis["document_types"] = self._analyze_document_types()
            
            # 3. 메타데이터 분석
            analysis["metadata_analysis"] = self._analyze_metadata()
            
            # 4. 콘텐츠 샘플 추출
            analysis["content_samples"] = self._extract_content_samples()
            
            # 5. 전체 문서 수 계산
            analysis["total_documents"] = sum(analysis["table_counts"].values())
            
            # 6. 분석 요약 생성
            analysis["analysis_summary"] = self._generate_analysis_summary(analysis)
            
            logger.info(f"데이터베이스 분석 완료: 총 {analysis['total_documents']}개 문서")
            
        except Exception as e:
            logger.error(f"데이터베이스 분석 중 오류: {e}")
            analysis["error"] = str(e)
        
        self.analysis_result = analysis
        return analysis
    
    def _analyze_table_counts(self) -> Dict[str, int]:
        """테이블별 데이터 수 분석"""
        table_counts = {}
        
        tables = [
            "documents", "law_metadata", "precedent_metadata", 
            "constitutional_metadata", "interpretation_metadata",
            "administrative_rule_metadata", "local_ordinance_metadata",
            "chat_history"
        ]
        
        for table in tables:
            try:
                count_query = f"SELECT COUNT(*) FROM {table}"
                result = self.db_manager.execute_query(count_query)
                table_counts[table] = result[0][0] if result else 0
                logger.info(f"{table} 테이블: {table_counts[table]}개 레코드")
            except Exception as e:
                logger.warning(f"{table} 테이블 분석 실패: {e}")
                table_counts[table] = 0
        
        return table_counts
    
    def _analyze_document_types(self) -> Dict[str, Any]:
        """문서 타입별 분석"""
        document_types = {}
        
        try:
            # 문서 타입별 개수 조회
            type_query = """
                SELECT document_type, COUNT(*) as count
                FROM documents 
                GROUP BY document_type
            """
            results = self.db_manager.execute_query(type_query)
            
            for row in results:
                doc_type = row["document_type"]
                count = row["count"]
                document_types[doc_type] = {
                    "count": count,
                    "percentage": 0  # 나중에 계산
                }
                logger.info(f"{doc_type} 문서: {count}개")
            
            # 비율 계산
            total = sum(doc_type["count"] for doc_type in document_types.values())
            if total > 0:
                for doc_type in document_types.values():
                    doc_type["percentage"] = round((doc_type["count"] / total) * 100, 2)
            
        except Exception as e:
            logger.warning(f"문서 타입 분석 실패: {e}")
        
        return document_types
    
    def _analyze_metadata(self) -> Dict[str, Any]:
        """메타데이터 분석"""
        metadata_analysis = {}
        
        try:
            # 법령 메타데이터 분석
            law_query = """
                SELECT law_name, COUNT(*) as count
                FROM law_metadata 
                WHERE law_name IS NOT NULL
                GROUP BY law_name
                ORDER BY count DESC
                LIMIT 10
            """
            law_results = self.db_manager.execute_query(law_query)
            metadata_analysis["top_laws"] = [dict(row) for row in law_results]
            
            # 판례 메타데이터 분석
            precedent_query = """
                SELECT court_name, COUNT(*) as count
                FROM precedent_metadata 
                WHERE court_name IS NOT NULL
                GROUP BY court_name
                ORDER BY count DESC
                LIMIT 10
            """
            precedent_results = self.db_manager.execute_query(precedent_query)
            metadata_analysis["top_courts"] = [dict(row) for row in precedent_results]
            
        except Exception as e:
            logger.warning(f"메타데이터 분석 실패: {e}")
        
        return metadata_analysis
    
    def _extract_content_samples(self) -> Dict[str, List[str]]:
        """콘텐츠 샘플 추출"""
        content_samples = {}
        
        try:
            # 각 문서 타입별 샘플 추출
            sample_query = """
                SELECT document_type, title, content
                FROM documents 
                WHERE content IS NOT NULL AND LENGTH(content) > 50
                ORDER BY RANDOM()
                LIMIT 3
            """
            results = self.db_manager.execute_query(sample_query)
            
            for row in results:
                doc_type = row["document_type"]
                if doc_type not in content_samples:
                    content_samples[doc_type] = []
                
                content_samples[doc_type].append({
                    "title": row["title"],
                    "content_preview": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"]
                })
            
        except Exception as e:
            logger.warning(f"콘텐츠 샘플 추출 실패: {e}")
        
        return content_samples
    
    def _generate_analysis_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """분석 요약 생성"""
        summary = {
            "has_data": analysis["total_documents"] > 0,
            "primary_data_types": [],
            "recommended_approach": "",
            "estimated_qa_potential": 0
        }
        
        if analysis["total_documents"] > 0:
            # 주요 데이터 타입 식별
            doc_types = analysis["document_types"]
            if doc_types:
                sorted_types = sorted(doc_types.items(), key=lambda x: x[1]["count"], reverse=True)
                summary["primary_data_types"] = [doc_type for doc_type, _ in sorted_types[:3]]
            
            # 추천 접근법 결정
            if "law" in doc_types and doc_types["law"]["count"] > 0:
                summary["recommended_approach"] = "law_and_precedent_focused"
            elif "precedent" in doc_types and doc_types["precedent"]["count"] > 0:
                summary["recommended_approach"] = "precedent_focused"
            else:
                summary["recommended_approach"] = "mixed_approach"
            
            # 예상 Q&A 생성 가능 수
            summary["estimated_qa_potential"] = analysis["total_documents"] * 2  # 문서당 평균 2개 Q&A
        
        else:
            summary["recommended_approach"] = "template_based_generation"
            summary["estimated_qa_potential"] = 200  # 템플릿 기반 생성 가능 수
        
        return summary
    
    def save_analysis_report(self, output_path: str = "logs/database_analysis_report.json"):
        """분석 보고서 저장"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"분석 보고서 저장 완료: {output_file}")
        return str(output_file)
    
    def print_summary(self):
        """분석 결과 요약 출력"""
        if not self.analysis_result:
            logger.warning("분석 결과가 없습니다.")
            return
        
        summary = self.analysis_result.get("analysis_summary", {})
        table_counts = self.analysis_result.get("table_counts", {})
        doc_types = self.analysis_result.get("document_types", {})
        
        print("\n" + "="*60)
        print("📊 데이터베이스 분석 결과 요약")
        print("="*60)
        
        print(f"📁 데이터베이스 경로: {self.analysis_result.get('database_path', 'N/A')}")
        print(f"📅 분석 시간: {self.analysis_result.get('analysis_timestamp', 'N/A')}")
        print(f"📄 총 문서 수: {self.analysis_result.get('total_documents', 0)}개")
        
        print("\n📋 테이블별 데이터 수:")
        for table, count in table_counts.items():
            print(f"  - {table}: {count}개")
        
        print("\n📚 문서 타입별 분포:")
        for doc_type, info in doc_types.items():
            print(f"  - {doc_type}: {info['count']}개 ({info['percentage']}%)")
        
        print(f"\n🎯 추천 접근법: {summary.get('recommended_approach', 'N/A')}")
        print(f"🔢 예상 Q&A 생성 가능 수: {summary.get('estimated_qa_potential', 0)}개")
        
        if summary.get("primary_data_types"):
            print(f"⭐ 주요 데이터 타입: {', '.join(summary['primary_data_types'])}")
        
        print("="*60)


def main():
    """메인 실행 함수"""
    logger.info("데이터베이스 분석 시작...")
    
    # 데이터베이스 분석기 초기화
    analyzer = DatabaseAnalyzer()
    
    # 데이터베이스 내용 분석
    analysis_result = analyzer.analyze_database_content()
    
    # 분석 보고서 저장
    report_path = analyzer.save_analysis_report()
    
    # 분석 결과 요약 출력
    analyzer.print_summary()
    
    logger.info("데이터베이스 분석 완료!")
    logger.info(f"상세 보고서: {report_path}")
    
    return analysis_result


if __name__ == "__main__":
    main()
