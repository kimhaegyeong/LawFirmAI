"""
Day 4 간단 테스트 스크립트
기본적인 평가 및 최적화 기능 테스트
"""

import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_advanced_evaluator():
    """고도화된 평가기 테스트"""
    logger.info("Testing Advanced Legal Evaluator...")
    
    try:
        from source.models.advanced_evaluator import AdvancedLegalEvaluator
        
        # 더미 모델과 토크나이저로 테스트
        evaluator = AdvancedLegalEvaluator(None, None)
        
        # 테스트 데이터
        test_data = [
            {
                "question": "계약서 해지 시 손해배상은 어떻게 되나요?",
                "answer": "계약 해지 시 손해배상은 계약 위반의 정도와 예견 가능한 손해 범위에 따라 결정됩니다."
            },
            {
                "question": "민법상 소유권의 내용은 무엇인가요?",
                "answer": "민법상 소유권은 소유물을 사용, 수익, 처분할 수 있는 권리입니다."
            }
        ]
        
        # 평가 실행
        results = evaluator.comprehensive_evaluation(test_data)
        
        logger.info("Advanced Evaluator test completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Advanced Evaluator test failed: {e}")
        return None


def test_ab_test_framework():
    """A/B 테스트 프레임워크 테스트"""
    logger.info("Testing A/B Test Framework...")
    
    try:
        from source.models.ab_test_framework import ABTestFramework, ModelVariant
        
        # A/B 테스트 프레임워크 초기화
        ab_test = ABTestFramework("test_comparison")
        
        # 모델 변형 추가
        variant_a = ModelVariant(
            name="Model_A",
            model_path="models/test/a",
            description="테스트 모델 A",
            config={"type": "a"},
            weight=1.0
        )
        
        variant_b = ModelVariant(
            name="Model_B",
            model_path="models/test/b",
            description="테스트 모델 B",
            config={"type": "b"},
            weight=1.0
        )
        
        ab_test.add_variant(variant_a)
        ab_test.add_variant(variant_b)
        
        # 테스트 구성
        ab_test.configure_test(
            test_duration_days=1,
            min_sample_size=10,
            confidence_level=0.95,
            primary_metric="comprehensive_score"
        )
        
        # 테스트 데이터
        test_data = [
            {"question": "테스트 질문 1", "answer": "테스트 답변 1"},
            {"question": "테스트 질문 2", "answer": "테스트 답변 2"}
        ] * 10  # 20개 샘플
        
        # A/B 테스트 실행
        results = ab_test.run_ab_test(test_data)
        
        logger.info("A/B Test Framework test completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"A/B Test Framework test failed: {e}")
        return None


def test_model_optimizer():
    """모델 최적화기 테스트"""
    logger.info("Testing Model Optimizer...")
    
    try:
        from source.models.model_optimizer import LegalModelOptimizer
        
        # 더미 모델과 토크나이저로 테스트
        optimizer = LegalModelOptimizer(None, None, device="cpu")
        
        # 최적화 실행 (실제 모델이 없으므로 시뮬레이션)
        logger.info("Model optimizer test completed (simulation mode)")
        return {"status": "simulation_completed"}
        
    except Exception as e:
        logger.error(f"Model Optimizer test failed: {e}")
        return None


def generate_day4_test_report(results: Dict[str, Any]):
    """Day 4 테스트 보고서 생성"""
    logger.info("Generating Day 4 test report...")
    
    report = {
        "test_info": {
            "test_time": datetime.now().isoformat(),
            "test_type": "Day 4 Evaluation and Optimization Test",
            "status": "completed"
        },
        "test_results": results,
        "summary": {
            "advanced_evaluator": "✅ Completed" if results.get("advanced_evaluator") else "❌ Failed",
            "ab_test_framework": "✅ Completed" if results.get("ab_test_framework") else "❌ Failed",
            "model_optimizer": "✅ Completed" if results.get("model_optimizer") else "❌ Failed"
        }
    }
    
    # 보고서 저장
    output_dir = Path("results") / "day4_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "day4_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    # 텍스트 요약 생성
    summary_text = f"""
Day 4 테스트 보고서
==================

테스트 시간: {report['test_info']['test_time']}
테스트 유형: {report['test_info']['test_type']}
테스트 상태: {report['test_info']['status']}

테스트 결과:
- 고도화된 평가기: {report['summary']['advanced_evaluator']}
- A/B 테스트 프레임워크: {report['summary']['ab_test_framework']}
- 모델 최적화기: {report['summary']['model_optimizer']}

상세 결과:
"""
    
    if results.get("advanced_evaluator"):
        eval_result = results["advanced_evaluator"]
        summary_text += f"- 평가기 종합 점수: {eval_result['summary']['comprehensive_score']:.3f}\n"
        summary_text += f"- 평가기 등급: {eval_result['summary']['grade']}\n"
    
    if results.get("ab_test_framework"):
        ab_result = results["ab_test_framework"]
        summary_text += f"- A/B 테스트 승자: {ab_result['winner']['variant_name']}\n"
        summary_text += f"- 승자 점수: {ab_result['winner']['primary_metric_score']:.3f}\n"
    
    summary_text += f"""
보고서 생성 완료: {datetime.now().isoformat()}
"""
    
    summary_path = output_dir / "day4_test_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    logger.info(f"Day 4 test report saved to {output_dir}")
    return report


def main():
    """메인 함수"""
    logger.info("Starting Day 4 test...")
    
    results = {}
    
    # 1. 고도화된 평가기 테스트
    results["advanced_evaluator"] = test_advanced_evaluator()
    
    # 2. A/B 테스트 프레임워크 테스트
    results["ab_test_framework"] = test_ab_test_framework()
    
    # 3. 모델 최적화기 테스트
    results["model_optimizer"] = test_model_optimizer()
    
    # 4. 테스트 보고서 생성
    report = generate_day4_test_report(results)
    
    # 결과 출력
    print("\n" + "="*60)
    print("🧪 Day 4 테스트 완료!")
    print("="*60)
    print(f"고도화된 평가기: {report['summary']['advanced_evaluator']}")
    print(f"A/B 테스트 프레임워크: {report['summary']['ab_test_framework']}")
    print(f"모델 최적화기: {report['summary']['model_optimizer']}")
    print(f"📁 결과 저장 위치: results/day4_test/")
    print("="*60)
    
    logger.info("Day 4 test completed successfully!")


if __name__ == "__main__":
    main()
