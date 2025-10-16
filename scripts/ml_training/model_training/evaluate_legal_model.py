"""
법률 모델 평가 스크립트
훈련된 법률 모델의 성능을 종합적으로 평가
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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/evaluate_legal_model.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class LegalModelEvaluationPipeline:
    """법률 모델 평가 파이프라인"""
    
    def __init__(self, model_path: str, test_data_path: str):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.fine_tuner = None
        self.evaluator = None
        
        # 로그 디렉토리 생성
        Path("logs").mkdir(exist_ok=True)
        
        logger.info(f"LegalModelEvaluationPipeline initialized for model: {model_path}")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """테스트 데이터 로드"""
        logger.info(f"Loading test data from {self.test_data_path}")
        
        try:
            with open(self.test_data_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            
            logger.info(f"Test data loaded successfully: {len(test_data)} samples")
            return test_data
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            raise
    
    def load_model(self):
        """모델 로드"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # 기본 모델 초기화
            self.fine_tuner = LegalModelFineTuner(device="cpu")
            
            # 훈련된 LoRA 어댑터 로드
            self.fine_tuner.load_model(self.model_path)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """모델 종합 평가"""
        logger.info("Starting comprehensive model evaluation...")
        
        try:
            # 평가기 초기화
            self.evaluator = LegalModelEvaluator(self.fine_tuner.model, self.fine_tuner.tokenizer)
            
            # 기본 성능 평가
            basic_results = self.evaluator.evaluate_legal_qa(test_data)
            
            # 상세 평가 수행
            detailed_results = self._detailed_evaluation(test_data)
            
            # 결과 통합
            evaluation_results = {
                "basic_metrics": basic_results,
                "detailed_metrics": detailed_results,
                "evaluation_info": {
                    "model_path": self.model_path,
                    "test_data_path": self.test_data_path,
                    "evaluation_time": datetime.now().isoformat(),
                    "total_test_samples": len(test_data)
                }
            }
            
            logger.info("Model evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def _detailed_evaluation(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """상세 평가 수행"""
        logger.info("Performing detailed evaluation...")
        
        detailed_results = {
            "by_question_type": {},
            "by_law_domain": {},
            "response_quality": {},
            "sample_analysis": []
        }
        
        # 질문 유형별 평가
        question_types = {}
        law_domains = {}
        
        for sample in test_data:
            q_type = sample.get("type", "unknown")
            source = sample.get("source", "unknown")
            
            if q_type not in question_types:
                question_types[q_type] = []
            question_types[q_type].append(sample)
            
            # 법령 도메인 분류
            law_domain = self._classify_law_domain(source)
            if law_domain not in law_domains:
                law_domains[law_domain] = []
            law_domains[law_domain].append(sample)
        
        # 질문 유형별 성능 평가
        for q_type, samples in question_types.items():
            if len(samples) > 0:
                type_results = self.evaluator.evaluate_legal_qa(samples)
                detailed_results["by_question_type"][q_type] = {
                    "sample_count": len(samples),
                    "metrics": type_results
                }
        
        # 법령 도메인별 성능 평가
        for domain, samples in law_domains.items():
            if len(samples) > 0:
                domain_results = self.evaluator.evaluate_legal_qa(samples)
                detailed_results["by_law_domain"][domain] = {
                    "sample_count": len(samples),
                    "metrics": domain_results
                }
        
        # 응답 품질 분석
        detailed_results["response_quality"] = self._analyze_response_quality(test_data)
        
        # 샘플 분석 (처음 5개)
        detailed_results["sample_analysis"] = self._analyze_samples(test_data[:5])
        
        return detailed_results
    
    def _classify_law_domain(self, source: str) -> str:
        """법령 도메인 분류"""
        source_lower = source.lower()
        
        if "민법" in source_lower:
            return "민법"
        elif "형법" in source_lower:
            return "형법"
        elif "상법" in source_lower:
            return "상법"
        elif "민사소송법" in source_lower:
            return "민사소송법"
        elif "형사소송법" in source_lower:
            return "형사소송법"
        elif "노동법" in source_lower:
            return "노동법"
        elif "부동산법" in source_lower:
            return "부동산법"
        elif "대법원" in source_lower or "판례" in source_lower:
            return "판례"
        elif "헌재" in source_lower:
            return "헌재"
        elif "법제처" in source_lower or "해석" in source_lower:
            return "법령해석"
        elif "행정규칙" in source_lower:
            return "행정규칙"
        else:
            return "기타"
    
    def _analyze_response_quality(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """응답 품질 분석"""
        logger.info("Analyzing response quality...")
        
        quality_metrics = {
            "average_response_length": 0.0,
            "response_length_distribution": {},
            "keyword_coverage": 0.0,
            "legal_term_usage": 0.0
        }
        
        response_lengths = []
        keyword_coverage_scores = []
        legal_term_scores = []
        
        # 법률 전문 용어 목록
        legal_terms = [
            "법률", "조항", "규정", "판례", "법원", "재판", "소송", "계약", "손해", "배상",
            "책임", "권리", "의무", "위반", "민법", "형법", "상법", "행정법", "헌법"
        ]
        
        for sample in test_data[:20]:  # 처음 20개 샘플만 분석
            try:
                question = sample.get("question", "")
                ground_truth = sample.get("answer", "")
                
                if not question or not ground_truth:
                    continue
                
                # 응답 생성
                prompt = f"<|startoftext|>질문: {question}\n답변:"
                predicted = self.fine_tuner.generate_response(prompt, max_length=200)
                
                # 응답 길이 분석
                response_lengths.append(len(predicted.split()))
                
                # 키워드 커버리지 분석
                question_words = set(question.lower().split())
                response_words = set(predicted.lower().split())
                coverage = len(question_words.intersection(response_words)) / len(question_words) if question_words else 0
                keyword_coverage_scores.append(coverage)
                
                # 법률 용어 사용 분석
                legal_term_count = sum(1 for term in legal_terms if term in predicted.lower())
                legal_term_scores.append(legal_term_count / len(predicted.split()) if predicted.split() else 0)
                
            except Exception as e:
                logger.warning(f"Error analyzing sample: {e}")
                continue
        
        if response_lengths:
            quality_metrics["average_response_length"] = sum(response_lengths) / len(response_lengths)
            quality_metrics["response_length_distribution"] = {
                "min": min(response_lengths),
                "max": max(response_lengths),
                "median": sorted(response_lengths)[len(response_lengths)//2]
            }
        
        if keyword_coverage_scores:
            quality_metrics["keyword_coverage"] = sum(keyword_coverage_scores) / len(keyword_coverage_scores)
        
        if legal_term_scores:
            quality_metrics["legal_term_usage"] = sum(legal_term_scores) / len(legal_term_scores)
        
        return quality_metrics
    
    def _analyze_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """샘플 분석"""
        logger.info("Analyzing sample responses...")
        
        sample_analysis = []
        
        for i, sample in enumerate(samples):
            try:
                question = sample.get("question", "")
                ground_truth = sample.get("answer", "")
                
                if not question or not ground_truth:
                    continue
                
                # 응답 생성
                prompt = f"<|startoftext|>질문: {question}\n답변:"
                predicted = self.fine_tuner.generate_response(prompt, max_length=200)
                
                # 유사도 계산
                similarity = self.evaluator._calculate_semantic_similarity(predicted, ground_truth)
                bleu_score = self.evaluator._calculate_bleu(predicted, ground_truth)
                rouge_score = self.evaluator._calculate_rouge(predicted, ground_truth)
                legal_relevance = self.evaluator._calculate_legal_relevance(predicted, question)
                
                sample_analysis.append({
                    "sample_id": i + 1,
                    "question": question,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "metrics": {
                        "similarity": similarity,
                        "bleu_score": bleu_score,
                        "rouge_score": rouge_score,
                        "legal_relevance": legal_relevance
                    }
                })
                
            except Exception as e:
                logger.warning(f"Error analyzing sample {i+1}: {e}")
                continue
        
        return sample_analysis
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """평가 보고서 생성"""
        logger.info("Generating evaluation report...")
        
        try:
            # 보고서 저장
            report_path = Path(self.model_path) / "evaluation_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
            
            # 요약 보고서 생성
            summary_report = self._generate_summary_report(evaluation_results)
            summary_path = Path(self.model_path) / "evaluation_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_report)
            
            logger.info(f"Evaluation report saved to {report_path}")
            logger.info(f"Summary report saved to {summary_path}")
            
            return summary_report
            
        except Exception as e:
            logger.error(f"Failed to generate evaluation report: {e}")
            raise
    
    def _generate_summary_report(self, evaluation_results: Dict[str, Any]) -> str:
        """요약 보고서 생성"""
        basic_metrics = evaluation_results["basic_metrics"]
        detailed_metrics = evaluation_results["detailed_metrics"]
        
        report = f"""
법률 모델 평가 보고서
====================

평가 정보:
- 모델 경로: {evaluation_results['evaluation_info']['model_path']}
- 테스트 데이터: {evaluation_results['evaluation_info']['test_data_path']}
- 평가 시간: {evaluation_results['evaluation_info']['evaluation_time']}
- 총 테스트 샘플: {evaluation_results['evaluation_info']['total_test_samples']}개

기본 성능 지표:
- 정확도: {basic_metrics.get('accuracy', 0.0):.3f}
- BLEU 점수: {basic_metrics.get('bleu_score', 0.0):.3f}
- ROUGE 점수: {basic_metrics.get('rouge_score', 0.0):.3f}
- 법률 관련성: {basic_metrics.get('legal_relevance', 0.0):.3f}

질문 유형별 성능:
"""
        
        for q_type, results in detailed_metrics["by_question_type"].items():
            metrics = results["metrics"]
            report += f"- {q_type}: 정확도 {metrics.get('accuracy', 0.0):.3f}, BLEU {metrics.get('bleu_score', 0.0):.3f} ({results['sample_count']}개 샘플)\n"
        
        report += "\n법령 도메인별 성능:\n"
        for domain, results in detailed_metrics["by_law_domain"].items():
            metrics = results["metrics"]
            report += f"- {domain}: 정확도 {metrics.get('accuracy', 0.0):.3f}, BLEU {metrics.get('bleu_score', 0.0):.3f} ({results['sample_count']}개 샘플)\n"
        
        response_quality = detailed_metrics["response_quality"]
        report += f"""
응답 품질 분석:
- 평균 응답 길이: {response_quality.get('average_response_length', 0.0):.1f} 단어
- 키워드 커버리지: {response_quality.get('keyword_coverage', 0.0):.3f}
- 법률 용어 사용률: {response_quality.get('legal_term_usage', 0.0):.3f}

샘플 분석 (상위 5개):
"""
        
        for sample in detailed_metrics["sample_analysis"]:
            report += f"""
샘플 {sample['sample_id']}:
질문: {sample['question'][:100]}...
정답: {sample['ground_truth'][:100]}...
예측: {sample['predicted'][:100]}...
점수: 유사도 {sample['metrics']['similarity']:.3f}, BLEU {sample['metrics']['bleu_score']:.3f}, 법률관련성 {sample['metrics']['legal_relevance']:.3f}
"""
        
        return report
    
    def run_evaluation_pipeline(self):
        """평가 파이프라인 실행"""
        logger.info("Starting legal model evaluation pipeline...")
        
        try:
            # 1. 테스트 데이터 로드
            test_data = self.load_test_data()
            
            # 2. 모델 로드
            self.load_model()
            
            # 3. 모델 평가
            evaluation_results = self.evaluate_model(test_data)
            
            # 4. 평가 보고서 생성
            summary_report = self.generate_evaluation_report(evaluation_results)
            
            # 콘솔에 요약 출력
            print("\n" + "="*60)
            print("📊 법률 모델 평가 완료!")
            print("="*60)
            print(summary_report)
            print("="*60)
            
            logger.info("Legal model evaluation pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {e}")
            raise


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="법률 모델 평가")
    parser.add_argument("--model", type=str, required=True, help="평가할 모델 경로")
    parser.add_argument("--test-data", type=str, default="data/training/test_split.json", help="테스트 데이터 경로")
    parser.add_argument("--output", type=str, help="출력 디렉토리 (기본값: 모델 경로)")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output:
        output_dir = args.output
    else:
        output_dir = args.model
    
    logger.info(f"Starting legal model evaluation for model: {args.model}")
    
    try:
        # 평가 파이프라인 실행
        pipeline = LegalModelEvaluationPipeline(args.model, args.test_data)
        pipeline.run_evaluation_pipeline()
        
        logger.info("Legal model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Legal model evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
