"""
Day 4 모델 평가 및 최적화 통합 실행 스크립트
고도화된 평가, 최적화, A/B 테스트를 통합 실행
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import torch

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.models.legal_finetuner import LegalModelFineTuner, LegalModelEvaluator
from source.models.advanced_evaluator import AdvancedLegalEvaluator
from source.models.model_optimizer import LegalModelOptimizer
from source.models.ab_test_framework import ABTestFramework, ModelVariant

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/day4_evaluation_optimization.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class Day4EvaluationOptimizationPipeline:
    """Day 4 평가 및 최적화 파이프라인"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.evaluation_results = {}
        self.optimization_results = {}
        self.ab_test_results = {}
        
        # 로그 디렉토리 생성
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("Day4EvaluationOptimizationPipeline initialized")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """테스트 데이터 로드"""
        logger.info("Loading test data...")
        
        try:
            test_path = Path(self.config["data"]["test_path"])
            with open(test_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            
            logger.info(f"Test data loaded successfully: {len(test_data)} samples")
            return test_data
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            raise
    
    def load_models(self):
        """모델들 로드"""
        logger.info("Loading models...")
        
        try:
            # 원본 모델 로드
            original_model_path = self.config["models"]["original_path"]
            if Path(original_model_path).exists():
                self.models["original"] = LegalModelFineTuner(device="cpu")
                self.models["original"].load_model(original_model_path)
                logger.info("Original model loaded successfully")
            
            # LoRA 파인튜닝된 모델 로드
            lora_model_path = self.config["models"]["lora_path"]
            if Path(lora_model_path).exists():
                self.models["lora"] = LegalModelFineTuner(device="cpu")
                self.models["lora"].load_model(lora_model_path)
                logger.info("LoRA model loaded successfully")
            
            # 최적화된 모델 로드 (있는 경우)
            optimized_model_path = self.config["models"].get("optimized_path")
            if optimized_model_path and Path(optimized_model_path).exists():
                self.models["optimized"] = LegalModelFineTuner(device="cpu")
                self.models["optimized"].load_model(optimized_model_path)
                logger.info("Optimized model loaded successfully")
            
            logger.info(f"Total models loaded: {len(self.models)}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def run_comprehensive_evaluation(self, test_data: List[Dict[str, Any]]):
        """종합 평가 실행"""
        logger.info("Starting comprehensive evaluation...")
        
        try:
            for model_name, model in self.models.items():
                logger.info(f"Evaluating model: {model_name}")
                
                # 고도화된 평가 실행
                evaluator = AdvancedLegalEvaluator(model.model, model.tokenizer)
                evaluation_results = evaluator.comprehensive_evaluation(test_data)
                
                self.evaluation_results[model_name] = evaluation_results
                
                # 결과 저장
                self._save_evaluation_results(model_name, evaluation_results)
            
            logger.info("Comprehensive evaluation completed")
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            raise
    
    def run_model_optimization(self):
        """모델 최적화 실행"""
        logger.info("Starting model optimization...")
        
        try:
            for model_name, model in self.models.items():
                logger.info(f"Optimizing model: {model_name}")
                
                # 모델 최적화 실행
                optimizer = LegalModelOptimizer(model.model, model.tokenizer, device="cpu")
                optimization_results = optimizer.comprehensive_optimization(
                    f"models/optimized/{model_name}"
                )
                
                self.optimization_results[model_name] = optimization_results
                
                # 결과 저장
                self._save_optimization_results(model_name, optimization_results)
            
            logger.info("Model optimization completed")
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            raise
    
    def run_ab_testing(self, test_data: List[Dict[str, Any]]):
        """A/B 테스트 실행"""
        logger.info("Starting A/B testing...")
        
        try:
            # A/B 테스트 프레임워크 초기화
            ab_test = ABTestFramework("legal_model_comparison")
            
            # 모델 변형 추가
            for model_name, model in self.models.items():
                variant = ModelVariant(
                    name=model_name,
                    model_path=f"models/{model_name}",
                    description=f"{model_name} 모델",
                    config={"model_type": model_name},
                    weight=1.0
                )
                ab_test.add_variant(variant)
            
            # 테스트 구성
            ab_test.configure_test(
                test_duration_days=self.config["ab_test"]["duration_days"],
                min_sample_size=self.config["ab_test"]["min_sample_size"],
                confidence_level=self.config["ab_test"]["confidence_level"],
                primary_metric=self.config["ab_test"]["primary_metric"]
            )
            
            # A/B 테스트 실행
            ab_test_results = ab_test.run_ab_test(test_data)
            self.ab_test_results = ab_test_results
            
            logger.info("A/B testing completed")
            
        except Exception as e:
            logger.error(f"A/B testing failed: {e}")
            raise
    
    def generate_comprehensive_report(self):
        """종합 보고서 생성"""
        logger.info("Generating comprehensive report...")
        
        try:
            report = {
                "report_info": {
                    "generation_time": datetime.now().isoformat(),
                    "total_models": len(self.models),
                    "evaluation_completed": len(self.evaluation_results),
                    "optimization_completed": len(self.optimization_results),
                    "ab_test_completed": bool(self.ab_test_results)
                },
                "evaluation_summary": self._summarize_evaluations(),
                "optimization_summary": self._summarize_optimizations(),
                "ab_test_summary": self._summarize_ab_test(),
                "recommendations": self._generate_final_recommendations()
            }
            
            # 보고서 저장
            self._save_comprehensive_report(report)
            
            logger.info("Comprehensive report generated")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            raise
    
    def _summarize_evaluations(self) -> Dict[str, Any]:
        """평가 결과 요약"""
        summary = {
            "model_rankings": {},
            "best_performing_model": None,
            "performance_comparison": {}
        }
        
        if not self.evaluation_results:
            return summary
        
        # 각 모델의 종합 점수 수집
        comprehensive_scores = {}
        for model_name, results in self.evaluation_results.items():
            comprehensive_score = results["summary"]["comprehensive_score"]
            comprehensive_scores[model_name] = comprehensive_score
        
        # 모델 순위 결정
        sorted_models = sorted(comprehensive_scores.items(), key=lambda x: x[1], reverse=True)
        summary["model_rankings"] = {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}
        summary["best_performing_model"] = sorted_models[0][0] if sorted_models else None
        
        # 성능 비교
        summary["performance_comparison"] = comprehensive_scores
        
        return summary
    
    def _summarize_optimizations(self) -> Dict[str, Any]:
        """최적화 결과 요약"""
        summary = {
            "optimization_methods": set(),
            "performance_improvements": {},
            "deployment_readiness": {}
        }
        
        if not self.optimization_results:
            return summary
        
        for model_name, results in self.optimization_results.items():
            optimization_summary = results["optimization_summary"]
            
            # 최적화 방법 수집
            summary["optimization_methods"].update(optimization_summary["optimization_methods"])
            
            # 성능 개선 사항
            summary["performance_improvements"][model_name] = optimization_summary["performance_improvements"]
            
            # 배포 준비도
            summary["deployment_readiness"][model_name] = optimization_summary["deployment_readiness"]
        
        summary["optimization_methods"] = list(summary["optimization_methods"])
        
        return summary
    
    def _summarize_ab_test(self) -> Dict[str, Any]:
        """A/B 테스트 결과 요약"""
        if not self.ab_test_results:
            return {"status": "not_completed"}
        
        return {
            "winner": self.ab_test_results["winner"],
            "statistical_analysis": self.ab_test_results["statistical_analysis"],
            "recommendations": self.ab_test_results["recommendations"]
        }
    
    def _generate_final_recommendations(self) -> List[str]:
        """최종 권장사항 생성"""
        recommendations = []
        
        # 평가 결과 기반 권장사항
        if self.evaluation_results:
            best_model = self._summarize_evaluations()["best_performing_model"]
            if best_model:
                recommendations.append(f"{best_model} 모델이 최고 성능을 보입니다. 프로덕션 배포를 권장합니다.")
        
        # 최적화 결과 기반 권장사항
        if self.optimization_results:
            recommendations.append("모델 최적화가 완료되었습니다. ONNX 변환과 양자화를 통해 배포 성능을 향상시킬 수 있습니다.")
        
        # A/B 테스트 결과 기반 권장사항
        if self.ab_test_results:
            winner = self.ab_test_results["winner"]
            if winner["statistically_significant"]:
                recommendations.append(f"A/B 테스트에서 {winner['variant_name']}이 통계적으로 유의하게 우수한 성능을 보입니다.")
            else:
                recommendations.append("A/B 테스트에서 통계적으로 유의한 차이가 없습니다. 더 큰 샘플 크기로 재테스트를 권장합니다.")
        
        # 일반적인 권장사항
        recommendations.extend([
            "정기적인 모델 성능 모니터링을 설정하세요.",
            "새로운 법률 데이터로 모델을 지속적으로 업데이트하세요.",
            "사용자 피드백을 수집하여 모델 성능을 개선하세요."
        ])
        
        return recommendations
    
    def _save_evaluation_results(self, model_name: str, results: Dict[str, Any]):
        """평가 결과 저장"""
        output_dir = Path("results") / "evaluations" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    
    def _save_optimization_results(self, model_name: str, results: Dict[str, Any]):
        """최적화 결과 저장"""
        output_dir = Path("results") / "optimizations" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "optimization_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    
    def _save_comprehensive_report(self, report: Dict[str, Any]):
        """종합 보고서 저장"""
        output_dir = Path("results") / "day4_comprehensive"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON 보고서 저장
        report_path = output_dir / "comprehensive_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        
        # 텍스트 요약 보고서 생성
        summary_report = self._generate_text_summary_report(report)
        summary_path = output_dir / "summary_report.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_report)
        
        logger.info(f"Comprehensive report saved to {output_dir}")
    
    def _generate_text_summary_report(self, report: Dict[str, Any]) -> str:
        """텍스트 요약 보고서 생성"""
        report_info = report["report_info"]
        eval_summary = report["evaluation_summary"]
        opt_summary = report["optimization_summary"]
        ab_summary = report["ab_test_summary"]
        
        summary = f"""
Day 4 모델 평가 및 최적화 종합 보고서
=====================================

생성 시간: {report_info['generation_time']}
총 모델 수: {report_info['total_models']}
평가 완료: {report_info['evaluation_completed']}
최적화 완료: {report_info['optimization_completed']}
A/B 테스트 완료: {'Yes' if report_info['ab_test_completed'] else 'No'}

평가 결과 요약:
"""
        
        if eval_summary["best_performing_model"]:
            summary += f"- 최고 성능 모델: {eval_summary['best_performing_model']}\n"
            summary += "- 모델 순위:\n"
            for model, rank in eval_summary["model_rankings"].items():
                score = eval_summary["performance_comparison"].get(model, 0)
                summary += f"  {rank}. {model}: {score:.3f}\n"
        
        summary += f"""
최적화 결과 요약:
- 적용된 최적화 방법: {', '.join(opt_summary.get('optimization_methods', []))}
"""
        
        if ab_summary.get("status") != "not_completed":
            winner = ab_summary["winner"]
            summary += f"""
A/B 테스트 결과:
- 승자: {winner['variant_name']}
- 주요 메트릭 점수: {winner['primary_metric_score']:.3f}
- 통계적 유의성: {'Yes' if winner['statistically_significant'] else 'No'}
- 신뢰도 점수: {winner['confidence_score']:.3f}
"""
        
        summary += """
최종 권장사항:
"""
        for i, recommendation in enumerate(report["recommendations"], 1):
            summary += f"{i}. {recommendation}\n"
        
        summary += f"""
보고서 생성 완료: {datetime.now().isoformat()}
"""
        
        return summary
    
    def run_complete_pipeline(self):
        """전체 파이프라인 실행"""
        logger.info("Starting Day 4 complete pipeline...")
        
        start_time = datetime.now()
        
        try:
            # 1. 테스트 데이터 로드
            test_data = self.load_test_data()
            
            # 2. 모델들 로드
            self.load_models()
            
            # 3. 종합 평가 실행
            self.run_comprehensive_evaluation(test_data)
            
            # 4. 모델 최적화 실행
            self.run_model_optimization()
            
            # 5. A/B 테스트 실행
            self.run_ab_testing(test_data)
            
            # 6. 종합 보고서 생성
            report = self.generate_comprehensive_report()
            
            # 완료 보고
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "="*60)
            print("🎉 Day 4 모델 평가 및 최적화 완료!")
            print("="*60)
            print(f"⏱️  총 소요 시간: {duration.total_seconds()/60:.1f}분")
            print(f"📊 평가된 모델: {len(self.models)}개")
            print(f"🔧 최적화 완료: {len(self.optimization_results)}개")
            print(f"🧪 A/B 테스트: {'완료' if self.ab_test_results else '미완료'}")
            
            if report["evaluation_summary"]["best_performing_model"]:
                best_model = report["evaluation_summary"]["best_performing_model"]
                print(f"🏆 최고 성능 모델: {best_model}")
            
            print(f"📁 결과 저장 위치: results/day4_comprehensive/")
            print("="*60)
            
            logger.info("Day 4 complete pipeline finished successfully!")
            
        except Exception as e:
            logger.error(f"Day 4 pipeline failed: {e}")
            raise


def create_default_config() -> Dict[str, Any]:
    """기본 설정 생성"""
    return {
        "data": {
            "test_path": "data/training/test_split.json"
        },
        "models": {
            "original_path": "models/original",
            "lora_path": "models/test/kogpt2-legal-lora-test",
            "optimized_path": "models/optimized"
        },
        "ab_test": {
            "duration_days": 7,
            "min_sample_size": 50,
            "confidence_level": 0.95,
            "primary_metric": "comprehensive_score"
        }
    }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Day 4 모델 평가 및 최적화")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--test-data", type=str, default="data/training/test_split.json", help="테스트 데이터 경로")
    parser.add_argument("--models", type=str, nargs="+", help="평가할 모델 경로들")
    parser.add_argument("--skip-optimization", action="store_true", help="최적화 단계 건너뛰기")
    parser.add_argument("--skip-ab-test", action="store_true", help="A/B 테스트 단계 건너뛰기")
    
    args = parser.parse_args()
    
    # 설정 로드 또는 생성
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = create_default_config()
        
        # 명령행 인수로 설정 업데이트
        config["data"]["test_path"] = args.test_data
        if args.models:
            config["models"]["lora_path"] = args.models[0] if len(args.models) > 0 else config["models"]["lora_path"]
    
    logger.info(f"Starting Day 4 evaluation and optimization with config: {config}")
    
    try:
        # 파이프라인 실행
        pipeline = Day4EvaluationOptimizationPipeline(config)
        pipeline.run_complete_pipeline()
        
        logger.info("Day 4 evaluation and optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"Day 4 evaluation and optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
