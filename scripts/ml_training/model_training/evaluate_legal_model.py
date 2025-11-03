"""
ë²•ë¥  ëª¨ë¸ ?‰ê? ?¤í¬ë¦½íŠ¸
?ˆë ¨??ë²•ë¥  ëª¨ë¸???±ëŠ¥??ì¢…í•©?ìœ¼ë¡??‰ê?
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

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.models.legal_finetuner import LegalModelFineTuner, LegalModelEvaluator

# ë¡œê¹… ?¤ì •
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
    """ë²•ë¥  ëª¨ë¸ ?‰ê? ?Œì´?„ë¼??""
    
    def __init__(self, model_path: str, test_data_path: str):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.fine_tuner = None
        self.evaluator = None
        
        # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
        Path("logs").mkdir(exist_ok=True)
        
        logger.info(f"LegalModelEvaluationPipeline initialized for model: {model_path}")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """?ŒìŠ¤???°ì´??ë¡œë“œ"""
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
        """ëª¨ë¸ ë¡œë“œ"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # ê¸°ë³¸ ëª¨ë¸ ì´ˆê¸°??
            self.fine_tuner = LegalModelFineTuner(device="cpu")
            
            # ?ˆë ¨??LoRA ?´ëŒ‘??ë¡œë“œ
            self.fine_tuner.load_model(self.model_path)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ëª¨ë¸ ì¢…í•© ?‰ê?"""
        logger.info("Starting comprehensive model evaluation...")
        
        try:
            # ?‰ê?ê¸?ì´ˆê¸°??
            self.evaluator = LegalModelEvaluator(self.fine_tuner.model, self.fine_tuner.tokenizer)
            
            # ê¸°ë³¸ ?±ëŠ¥ ?‰ê?
            basic_results = self.evaluator.evaluate_legal_qa(test_data)
            
            # ?ì„¸ ?‰ê? ?˜í–‰
            detailed_results = self._detailed_evaluation(test_data)
            
            # ê²°ê³¼ ?µí•©
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
        """?ì„¸ ?‰ê? ?˜í–‰"""
        logger.info("Performing detailed evaluation...")
        
        detailed_results = {
            "by_question_type": {},
            "by_law_domain": {},
            "response_quality": {},
            "sample_analysis": []
        }
        
        # ì§ˆë¬¸ ? í˜•ë³??‰ê?
        question_types = {}
        law_domains = {}
        
        for sample in test_data:
            q_type = sample.get("type", "unknown")
            source = sample.get("source", "unknown")
            
            if q_type not in question_types:
                question_types[q_type] = []
            question_types[q_type].append(sample)
            
            # ë²•ë ¹ ?„ë©”??ë¶„ë¥˜
            law_domain = self._classify_law_domain(source)
            if law_domain not in law_domains:
                law_domains[law_domain] = []
            law_domains[law_domain].append(sample)
        
        # ì§ˆë¬¸ ? í˜•ë³??±ëŠ¥ ?‰ê?
        for q_type, samples in question_types.items():
            if len(samples) > 0:
                type_results = self.evaluator.evaluate_legal_qa(samples)
                detailed_results["by_question_type"][q_type] = {
                    "sample_count": len(samples),
                    "metrics": type_results
                }
        
        # ë²•ë ¹ ?„ë©”?¸ë³„ ?±ëŠ¥ ?‰ê?
        for domain, samples in law_domains.items():
            if len(samples) > 0:
                domain_results = self.evaluator.evaluate_legal_qa(samples)
                detailed_results["by_law_domain"][domain] = {
                    "sample_count": len(samples),
                    "metrics": domain_results
                }
        
        # ?‘ë‹µ ?ˆì§ˆ ë¶„ì„
        detailed_results["response_quality"] = self._analyze_response_quality(test_data)
        
        # ?˜í”Œ ë¶„ì„ (ì²˜ìŒ 5ê°?
        detailed_results["sample_analysis"] = self._analyze_samples(test_data[:5])
        
        return detailed_results
    
    def _classify_law_domain(self, source: str) -> str:
        """ë²•ë ¹ ?„ë©”??ë¶„ë¥˜"""
        source_lower = source.lower()
        
        if "ë¯¼ë²•" in source_lower:
            return "ë¯¼ë²•"
        elif "?•ë²•" in source_lower:
            return "?•ë²•"
        elif "?ë²•" in source_lower:
            return "?ë²•"
        elif "ë¯¼ì‚¬?Œì†¡ë²? in source_lower:
            return "ë¯¼ì‚¬?Œì†¡ë²?
        elif "?•ì‚¬?Œì†¡ë²? in source_lower:
            return "?•ì‚¬?Œì†¡ë²?
        elif "?¸ë™ë²? in source_lower:
            return "?¸ë™ë²?
        elif "ë¶€?™ì‚°ë²? in source_lower:
            return "ë¶€?™ì‚°ë²?
        elif "?€ë²•ì›" in source_lower or "?ë?" in source_lower:
            return "?ë?"
        elif "?Œì¬" in source_lower:
            return "?Œì¬"
        elif "ë²•ì œì²? in source_lower or "?´ì„" in source_lower:
            return "ë²•ë ¹?´ì„"
        elif "?‰ì •ê·œì¹™" in source_lower:
            return "?‰ì •ê·œì¹™"
        else:
            return "ê¸°í?"
    
    def _analyze_response_quality(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """?‘ë‹µ ?ˆì§ˆ ë¶„ì„"""
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
        
        # ë²•ë¥  ?„ë¬¸ ?©ì–´ ëª©ë¡
        legal_terms = [
            "ë²•ë¥ ", "ì¡°í•­", "ê·œì •", "?ë?", "ë²•ì›", "?¬íŒ", "?Œì†¡", "ê³„ì•½", "?í•´", "ë°°ìƒ",
            "ì±…ì„", "ê¶Œë¦¬", "?˜ë¬´", "?„ë°˜", "ë¯¼ë²•", "?•ë²•", "?ë²•", "?‰ì •ë²?, "?Œë²•"
        ]
        
        for sample in test_data[:20]:  # ì²˜ìŒ 20ê°??˜í”Œë§?ë¶„ì„
            try:
                question = sample.get("question", "")
                ground_truth = sample.get("answer", "")
                
                if not question or not ground_truth:
                    continue
                
                # ?‘ë‹µ ?ì„±
                prompt = f"<|startoftext|>ì§ˆë¬¸: {question}\n?µë?:"
                predicted = self.fine_tuner.generate_response(prompt, max_length=200)
                
                # ?‘ë‹µ ê¸¸ì´ ë¶„ì„
                response_lengths.append(len(predicted.split()))
                
                # ?¤ì›Œ??ì»¤ë²„ë¦¬ì? ë¶„ì„
                question_words = set(question.lower().split())
                response_words = set(predicted.lower().split())
                coverage = len(question_words.intersection(response_words)) / len(question_words) if question_words else 0
                keyword_coverage_scores.append(coverage)
                
                # ë²•ë¥  ?©ì–´ ?¬ìš© ë¶„ì„
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
        """?˜í”Œ ë¶„ì„"""
        logger.info("Analyzing sample responses...")
        
        sample_analysis = []
        
        for i, sample in enumerate(samples):
            try:
                question = sample.get("question", "")
                ground_truth = sample.get("answer", "")
                
                if not question or not ground_truth:
                    continue
                
                # ?‘ë‹µ ?ì„±
                prompt = f"<|startoftext|>ì§ˆë¬¸: {question}\n?µë?:"
                predicted = self.fine_tuner.generate_response(prompt, max_length=200)
                
                # ? ì‚¬??ê³„ì‚°
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
        """?‰ê? ë³´ê³ ???ì„±"""
        logger.info("Generating evaluation report...")
        
        try:
            # ë³´ê³ ???€??
            report_path = Path(self.model_path) / "evaluation_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
            
            # ?”ì•½ ë³´ê³ ???ì„±
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
        """?”ì•½ ë³´ê³ ???ì„±"""
        basic_metrics = evaluation_results["basic_metrics"]
        detailed_metrics = evaluation_results["detailed_metrics"]
        
        report = f"""
ë²•ë¥  ëª¨ë¸ ?‰ê? ë³´ê³ ??
====================

?‰ê? ?•ë³´:
- ëª¨ë¸ ê²½ë¡œ: {evaluation_results['evaluation_info']['model_path']}
- ?ŒìŠ¤???°ì´?? {evaluation_results['evaluation_info']['test_data_path']}
- ?‰ê? ?œê°„: {evaluation_results['evaluation_info']['evaluation_time']}
- ì´??ŒìŠ¤???˜í”Œ: {evaluation_results['evaluation_info']['total_test_samples']}ê°?

ê¸°ë³¸ ?±ëŠ¥ ì§€??
- ?•í™•?? {basic_metrics.get('accuracy', 0.0):.3f}
- BLEU ?ìˆ˜: {basic_metrics.get('bleu_score', 0.0):.3f}
- ROUGE ?ìˆ˜: {basic_metrics.get('rouge_score', 0.0):.3f}
- ë²•ë¥  ê´€?¨ì„±: {basic_metrics.get('legal_relevance', 0.0):.3f}

ì§ˆë¬¸ ? í˜•ë³??±ëŠ¥:
"""
        
        for q_type, results in detailed_metrics["by_question_type"].items():
            metrics = results["metrics"]
            report += f"- {q_type}: ?•í™•??{metrics.get('accuracy', 0.0):.3f}, BLEU {metrics.get('bleu_score', 0.0):.3f} ({results['sample_count']}ê°??˜í”Œ)\n"
        
        report += "\në²•ë ¹ ?„ë©”?¸ë³„ ?±ëŠ¥:\n"
        for domain, results in detailed_metrics["by_law_domain"].items():
            metrics = results["metrics"]
            report += f"- {domain}: ?•í™•??{metrics.get('accuracy', 0.0):.3f}, BLEU {metrics.get('bleu_score', 0.0):.3f} ({results['sample_count']}ê°??˜í”Œ)\n"
        
        response_quality = detailed_metrics["response_quality"]
        report += f"""
?‘ë‹µ ?ˆì§ˆ ë¶„ì„:
- ?‰ê·  ?‘ë‹µ ê¸¸ì´: {response_quality.get('average_response_length', 0.0):.1f} ?¨ì–´
- ?¤ì›Œ??ì»¤ë²„ë¦¬ì?: {response_quality.get('keyword_coverage', 0.0):.3f}
- ë²•ë¥  ?©ì–´ ?¬ìš©ë¥? {response_quality.get('legal_term_usage', 0.0):.3f}

?˜í”Œ ë¶„ì„ (?ìœ„ 5ê°?:
"""
        
        for sample in detailed_metrics["sample_analysis"]:
            report += f"""
?˜í”Œ {sample['sample_id']}:
ì§ˆë¬¸: {sample['question'][:100]}...
?•ë‹µ: {sample['ground_truth'][:100]}...
?ˆì¸¡: {sample['predicted'][:100]}...
?ìˆ˜: ? ì‚¬??{sample['metrics']['similarity']:.3f}, BLEU {sample['metrics']['bleu_score']:.3f}, ë²•ë¥ ê´€?¨ì„± {sample['metrics']['legal_relevance']:.3f}
"""
        
        return report
    
    def run_evaluation_pipeline(self):
        """?‰ê? ?Œì´?„ë¼???¤í–‰"""
        logger.info("Starting legal model evaluation pipeline...")
        
        try:
            # 1. ?ŒìŠ¤???°ì´??ë¡œë“œ
            test_data = self.load_test_data()
            
            # 2. ëª¨ë¸ ë¡œë“œ
            self.load_model()
            
            # 3. ëª¨ë¸ ?‰ê?
            evaluation_results = self.evaluate_model(test_data)
            
            # 4. ?‰ê? ë³´ê³ ???ì„±
            summary_report = self.generate_evaluation_report(evaluation_results)
            
            # ì½˜ì†”???”ì•½ ì¶œë ¥
            print("\n" + "="*60)
            print("?“Š ë²•ë¥  ëª¨ë¸ ?‰ê? ?„ë£Œ!")
            print("="*60)
            print(summary_report)
            print("="*60)
            
            logger.info("Legal model evaluation pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {e}")
            raise


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    parser = argparse.ArgumentParser(description="ë²•ë¥  ëª¨ë¸ ?‰ê?")
    parser.add_argument("--model", type=str, required=True, help="?‰ê???ëª¨ë¸ ê²½ë¡œ")
    parser.add_argument("--test-data", type=str, default="data/training/test_split.json", help="?ŒìŠ¤???°ì´??ê²½ë¡œ")
    parser.add_argument("--output", type=str, help="ì¶œë ¥ ?”ë ‰? ë¦¬ (ê¸°ë³¸ê°? ëª¨ë¸ ê²½ë¡œ)")
    
    args = parser.parse_args()
    
    # ì¶œë ¥ ?”ë ‰? ë¦¬ ?¤ì •
    if args.output:
        output_dir = args.output
    else:
        output_dir = args.model
    
    logger.info(f"Starting legal model evaluation for model: {args.model}")
    
    try:
        # ?‰ê? ?Œì´?„ë¼???¤í–‰
        pipeline = LegalModelEvaluationPipeline(args.model, args.test_data)
        pipeline.run_evaluation_pipeline()
        
        logger.info("Legal model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Legal model evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
