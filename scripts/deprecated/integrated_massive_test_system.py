#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 대규모 테스트 시스템
3000개의 테스트 질의를 생성하고 실행하여 시스템 성능을 종합적으로 평가합니다.
"""

import sys
import os
import json
import time
import glob
from typing import Dict, List, Any, Optional
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 모듈 import
from scripts.massive_test_query_generator import MassiveTestQueryGenerator
from scripts.massive_test_runner import MassiveTestRunner

class IntegratedMassiveTestSystem:
    """통합 대규모 테스트 시스템"""
    
    def __init__(self, total_queries: int = 3000, max_workers: int = 8, enable_chat: bool = False, store_details: bool = False, batch_size: int = 200, method: str = "parallel"):
        self.total_queries = total_queries
        self.max_workers = max_workers
        self.enable_chat = enable_chat
        self.store_details = store_details
        self.batch_size = batch_size
        self.method = method
        self.generator = MassiveTestQueryGenerator()
        self.runner = MassiveTestRunner(max_workers=max_workers, enable_chat=enable_chat, store_details=store_details)
        
    def run_complete_test(self) -> Dict[str, Any]:
        """완전한 테스트 실행"""
        print("🚀 통합 대규모 테스트 시스템 시작")
        print("=" * 80)
        
        start_time = time.time()
        
        # 1단계: 테스트 질의 생성
        print("\n📝 1단계: 테스트 질의 생성")
        print("-" * 40)
        
        queries = self.generator.generate_massive_test_queries(self.total_queries)
        queries_file = self.generator.save_queries_to_file(queries)
        stats = self.generator.generate_statistics(queries)
        
        print(f"✅ {len(queries)}개의 테스트 질의 생성 완료")
        print(f"📁 질의 파일: {queries_file}")
        
        # 2단계: 대규모 테스트 실행
        print("\n🧪 2단계: 대규모 테스트 실행")
        print("-" * 40)
        
        # 질의를 딕셔너리 형태로 변환
        queries_data = []
        for query in queries:
            queries_data.append({
                "query": query.query,
                "category": query.category,
                "subcategory": query.subcategory,
                "expected_restricted": query.expected_restricted,
                "difficulty_level": query.difficulty_level,
                "context_type": query.context_type,
                "legal_area": query.legal_area,
                "keywords": query.keywords,
                "description": query.description
            })
        
        # 테스트 실행
        results = self.runner.run_massive_test(queries_data, method=self.method, batch_size=self.batch_size)
        summary = self.runner.generate_summary()
        
        print(f"✅ {len(results)}개의 테스트 완료")
        
        # 3단계: 결과 분석 및 저장
        print("\n📊 3단계: 결과 분석 및 저장")
        print("-" * 40)
        
        # 결과 저장
        results_file = self.runner.save_results(results, summary)
        
        # 보고서 생성
        report = self.runner.generate_report(summary)
        
        # 보고서 파일 저장
        report_file = results_file.replace('.json', '_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📁 결과 파일: {results_file}")
        print(f"📄 보고서 파일: {report_file}")
        
        # 4단계: 상세 분석
        print("\n🔍 4단계: 상세 분석")
        print("-" * 40)
        
        detailed_analysis = self._perform_detailed_analysis(results, summary, stats)
        
        # 상세 분석 결과 저장
        analysis_file = results_file.replace('.json', '_analysis.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"📊 상세 분석 파일: {analysis_file}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 최종 결과
        final_results = {
            "metadata": {
                "total_queries": self.total_queries,
                "total_time": total_time,
                "queries_file": queries_file,
                "results_file": results_file,
                "report_file": report_file,
                "analysis_file": analysis_file,
                "completed_at": datetime.now().isoformat()
            },
            "generation_stats": stats,
            "test_summary": summary.__dict__,
            "detailed_analysis": detailed_analysis,
            "report": report
        }
        
        print(f"\n🎉 통합 테스트 완료! 총 소요 시간: {total_time:.2f}초")
        
        return final_results
    
    def _perform_detailed_analysis(self, results: List, summary, generation_stats: Dict) -> Dict[str, Any]:
        """상세 분석 수행"""
        analysis = {
            "performance_metrics": {},
            "category_analysis": {},
            "error_analysis": {},
            "recommendations": []
        }
        
        # 성능 메트릭
        analysis["performance_metrics"] = {
            "queries_per_second": summary.total_tests / summary.processing_time if summary.processing_time > 0 else 0,
            "average_processing_time": summary.processing_time / summary.total_tests if summary.total_tests > 0 else 0,
            "error_rate": summary.error_count / summary.total_tests if summary.total_tests > 0 else 0,
            "confidence_distribution": self._analyze_confidence_distribution(results),
            "score_distribution": self._analyze_score_distribution(results)
        }
        
        # 카테고리별 상세 분석
        analysis["category_analysis"] = self._analyze_categories(results)
        
        # 오류 분석
        analysis["error_analysis"] = self._analyze_errors(results)
        
        # 개선 권장사항
        analysis["recommendations"] = self._generate_recommendations(summary, analysis)
        
        return analysis
    
    def _analyze_confidence_distribution(self, results: List) -> Dict[str, Any]:
        """신뢰도 분포 분석"""
        confidences = [r.confidence for r in results if r.confidence > 0]
        
        if not confidences:
            return {"error": "신뢰도 데이터 없음"}
        
        return {
            "min": min(confidences),
            "max": max(confidences),
            "mean": sum(confidences) / len(confidences),
            "median": sorted(confidences)[len(confidences) // 2],
            "high_confidence_count": sum(1 for c in confidences if c >= 0.8),
            "low_confidence_count": sum(1 for c in confidences if c < 0.5)
        }
    
    def _analyze_score_distribution(self, results: List) -> Dict[str, Any]:
        """점수 분포 분석"""
        scores = [r.total_score for r in results if r.total_score > 0]
        
        if not scores:
            return {"error": "점수 데이터 없음"}
        
        return {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
            "median": sorted(scores)[len(scores) // 2],
            "high_score_count": sum(1 for s in scores if s >= 0.8),
            "low_score_count": sum(1 for s in scores if s < 0.3)
        }
    
    def _analyze_categories(self, results: List) -> Dict[str, Any]:
        """카테고리별 상세 분석"""
        category_analysis = {}
        
        # 카테고리별 그룹화
        categories = {}
        for result in results:
            category = result.category
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # 각 카테고리 분석
        for category, category_results in categories.items():
            total = len(category_results)
            correct = sum(1 for r in category_results if r.is_correct)
            errors = sum(1 for r in category_results if r.error_message)
            
            category_analysis[category] = {
                "total_queries": total,
                "correct_predictions": correct,
                "accuracy": correct / total if total > 0 else 0,
                "error_count": errors,
                "error_rate": errors / total if total > 0 else 0,
                "average_confidence": sum(r.confidence for r in category_results) / total if total > 0 else 0,
                "average_score": sum(r.total_score for r in category_results) / total if total > 0 else 0,
                "average_processing_time": sum(r.processing_time for r in category_results) / total if total > 0 else 0
            }
        
        return category_analysis
    
    def _analyze_errors(self, results: List) -> Dict[str, Any]:
        """오류 분석"""
        error_results = [r for r in results if r.error_message]
        
        if not error_results:
            return {"error_count": 0, "error_types": {}}
        
        # 오류 유형별 분류
        error_types = {}
        for result in error_results:
            error_type = self._classify_error(result.error_message)
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(result)
        
        # 오류 유형별 통계
        error_analysis = {
            "error_count": len(error_results),
            "error_rate": len(error_results) / len(results) if results else 0,
            "error_types": {}
        }
        
        for error_type, type_results in error_types.items():
            error_analysis["error_types"][error_type] = {
                "count": len(type_results),
                "percentage": len(type_results) / len(error_results) * 100,
                "categories": list(set(r.category for r in type_results))
            }
        
        return error_analysis
    
    def _classify_error(self, error_message: str) -> str:
        """오류 유형 분류"""
        error_message = error_message.lower()
        
        if "timeout" in error_message or "time" in error_message:
            return "timeout_error"
        elif "memory" in error_message or "out of memory" in error_message:
            return "memory_error"
        elif "connection" in error_message or "network" in error_message:
            return "connection_error"
        elif "validation" in error_message or "invalid" in error_message:
            return "validation_error"
        elif "import" in error_message or "module" in error_message:
            return "import_error"
        else:
            return "unknown_error"
    
    def _generate_recommendations(self, summary, analysis: Dict) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 정확도 기반 권장사항
        if summary.overall_accuracy < 0.90:
            recommendations.append("전체 정확도가 90% 미만입니다. 시스템 튜닝이 필요합니다.")
        
        if summary.overall_accuracy < 0.80:
            recommendations.append("전체 정확도가 80% 미만입니다. 시스템 재설계를 고려해야 합니다.")
        
        # 카테고리별 권장사항
        low_accuracy_categories = [
            category for category, accuracy in summary.category_accuracies.items()
            if accuracy < 0.80
        ]
        
        if low_accuracy_categories:
            recommendations.append(f"정확도가 낮은 카테고리 ({', '.join(low_accuracy_categories)})의 패턴과 로직을 재검토해야 합니다.")
        
        # 오류 기반 권장사항
        if analysis["error_analysis"]["error_count"] > 0:
            error_rate = analysis["error_analysis"]["error_rate"]
            if error_rate > 0.05:  # 5% 이상
                recommendations.append(f"오류율이 {error_rate:.1%}로 높습니다. 오류 처리 로직을 개선해야 합니다.")
        
        # 성능 기반 권장사항
        performance = analysis["performance_metrics"]
        if performance.get("queries_per_second", 0) < 10:
            recommendations.append("처리 성능이 낮습니다. 병렬 처리나 캐싱을 고려해야 합니다.")
        
        # 신뢰도 기반 권장사항
        confidence_dist = analysis["performance_metrics"].get("confidence_distribution", {})
        if confidence_dist.get("low_confidence_count", 0) > confidence_dist.get("high_confidence_count", 0):
            recommendations.append("낮은 신뢰도 질의가 많습니다. 모델의 확신도를 높이는 튜닝이 필요합니다.")
        
        return recommendations

def main():
    """메인 함수"""
    try:
        print("🎯 LawFirmAI 대규모 테스트 시스템")
        print("=" * 80)
        
        # 테스트 시스템 초기화
        # 환경변수 TOTAL_QUERIES가 있으면 우선 사용
        import os as _os
        _total = int(_os.getenv("TOTAL_QUERIES", "13000"))
        test_system = IntegratedMassiveTestSystem(
            total_queries=_total,
            max_workers=min(os.cpu_count() or 8, 12),
            enable_chat=False,
            store_details=False,
            batch_size=200,
            method="parallel"
        )
        
        # 완전한 테스트 실행
        results = test_system.run_complete_test()
        
        if results:
            print("\n📋 최종 결과 요약:")
            print(f"  총 질의 수: {results['metadata']['total_queries']:,}")
            print(f"  전체 정확도: {results['test_summary']['overall_accuracy']:.1%}")
            print(f"  총 소요 시간: {results['metadata']['total_time']:.2f}초")
            print(f"  처리 성능: {results['test_summary']['total_tests']/results['metadata']['total_time']:.1f} 질의/초")
            
            print(f"\n📁 생성된 파일들:")
            print(f"  질의 파일: {results['metadata']['queries_file']}")
            print(f"  결과 파일: {results['metadata']['results_file']}")
            print(f"  보고서 파일: {results['metadata']['report_file']}")
            print(f"  분석 파일: {results['metadata']['analysis_file']}")
            
            # 보고서 출력
            print("\n" + "=" * 80)
            print(results['report'])
            
        else:
            print("❌ 테스트 실행 실패")
        
        return results
        
    except Exception as e:
        print(f"❌ 통합 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
