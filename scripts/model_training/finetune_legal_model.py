"""
법률 모델 LoRA 파인튜닝 실행 스크립트
KoGPT-2 기반 법률 특화 모델 훈련
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

from source.models.legal_finetuner import LegalModelFineTuner, LegalModelEvaluator, LegalQADataset

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/finetune_legal_model.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class LegalModelTrainingPipeline:
    """법률 모델 훈련 파이프라인"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fine_tuner = None
        self.evaluator = None
        
        # 로그 디렉토리 생성
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("LegalModelTrainingPipeline initialized")
    
    def load_training_data(self) -> tuple:
        """훈련 데이터 로드"""
        logger.info("Loading training data...")
        
        try:
            # 훈련 데이터 로드
            train_path = Path(self.config["data"]["train_path"])
            val_path = Path(self.config["data"]["val_path"])
            test_path = Path(self.config["data"]["test_path"])
            
            with open(train_path, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            
            with open(val_path, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            
            with open(test_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"  Train: {len(train_data)} samples")
            logger.info(f"  Validation: {len(val_data)} samples")
            logger.info(f"  Test: {len(test_data)} samples")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise
    
    def initialize_model(self):
        """모델 초기화"""
        logger.info("Initializing model...")
        
        try:
            self.fine_tuner = LegalModelFineTuner(
                model_name=self.config["model"]["name"],
                device=self.config["model"]["device"]
            )
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def prepare_datasets(self, train_data: List[Dict], val_data: List[Dict]) -> tuple:
        """데이터셋 준비"""
        logger.info("Preparing datasets...")
        
        try:
            train_dataset, val_dataset = self.fine_tuner.prepare_training_data(train_data, val_data)
            logger.info("Datasets prepared successfully")
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare datasets: {e}")
            raise
    
    def setup_training(self):
        """훈련 설정"""
        logger.info("Setting up training configuration...")
        
        try:
            training_args = self.fine_tuner.setup_training_args(
                output_dir=self.config["training"]["output_dir"],
                num_train_epochs=self.config["training"]["epochs"],
                per_device_train_batch_size=self.config["training"]["batch_size"],
                per_device_eval_batch_size=self.config["training"]["eval_batch_size"],
                gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
                learning_rate=self.config["training"]["learning_rate"],
                warmup_steps=self.config["training"]["warmup_steps"],
                logging_steps=self.config["training"]["logging_steps"],
                save_steps=self.config["training"]["save_steps"],
                eval_steps=self.config["training"]["eval_steps"]
            )
            
            logger.info("Training configuration setup completed")
            return training_args
            
        except Exception as e:
            logger.error(f"Failed to setup training: {e}")
            raise
    
    def train_model(self, train_dataset: LegalQADataset, val_dataset: LegalQADataset):
        """모델 훈련"""
        logger.info("Starting model training...")
        
        try:
            trainer = self.fine_tuner.train(train_dataset, val_dataset)
            
            # 모델 저장
            self.fine_tuner.save_model(self.config["training"]["output_dir"])
            
            logger.info("Model training completed successfully")
            return trainer
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def evaluate_model(self, test_data: List[Dict]):
        """모델 평가"""
        logger.info("Evaluating model...")
        
        try:
            self.evaluator = LegalModelEvaluator(self.fine_tuner.model, self.fine_tuner.tokenizer)
            
            evaluation_results = self.evaluator.evaluate_legal_qa(test_data)
            
            # 평가 결과 저장
            eval_path = Path(self.config["training"]["output_dir"]) / "evaluation_results.json"
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Model evaluation completed: {evaluation_results}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def generate_sample_responses(self, test_data: List[Dict], num_samples: int = 5):
        """샘플 응답 생성"""
        logger.info(f"Generating {num_samples} sample responses...")
        
        try:
            sample_responses = []
            
            for i, sample in enumerate(test_data[:num_samples]):
                question = sample.get("question", "")
                ground_truth = sample.get("answer", "")
                
                if not question:
                    continue
                
                # 응답 생성
                prompt = f"<|startoftext|>질문: {question}\n답변:"
                predicted = self.fine_tuner.generate_response(prompt, max_length=200)
                
                sample_responses.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "sample_id": i + 1
                })
            
            # 샘플 응답 저장
            samples_path = Path(self.config["training"]["output_dir"]) / "sample_responses.json"
            with open(samples_path, "w", encoding="utf-8") as f:
                json.dump(sample_responses, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Sample responses generated and saved to {samples_path}")
            return sample_responses
            
        except Exception as e:
            logger.error(f"Failed to generate sample responses: {e}")
            raise
    
    def run_training_pipeline(self):
        """전체 훈련 파이프라인 실행"""
        logger.info("Starting legal model training pipeline...")
        
        start_time = datetime.now()
        
        try:
            # 1. 데이터 로드
            train_data, val_data, test_data = self.load_training_data()
            
            # 2. 모델 초기화
            self.initialize_model()
            
            # 3. 데이터셋 준비
            train_dataset, val_dataset = self.prepare_datasets(train_data, val_data)
            
            # 4. 훈련 설정
            self.setup_training()
            
            # 5. 모델 훈련
            trainer = self.train_model(train_dataset, val_dataset)
            
            # 6. 모델 평가
            evaluation_results = self.evaluate_model(test_data)
            
            # 7. 샘플 응답 생성
            sample_responses = self.generate_sample_responses(test_data)
            
            # 훈련 완료 보고서 생성
            self.generate_training_report(evaluation_results, sample_responses, start_time)
            
            logger.info("Legal model training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def generate_training_report(self, evaluation_results: Dict, sample_responses: List[Dict], start_time: datetime):
        """훈련 보고서 생성"""
        logger.info("Generating training report...")
        
        try:
            end_time = datetime.now()
            training_duration = end_time - start_time
            
            report = {
                "training_info": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_minutes": training_duration.total_seconds() / 60,
                    "model_name": self.config["model"]["name"],
                    "device": self.config["model"]["device"]
                },
                "training_config": self.config["training"],
                "evaluation_results": evaluation_results,
                "sample_responses": sample_responses[:3],  # 처음 3개만 포함
                "summary": {
                    "total_samples": evaluation_results.get("total_samples", 0),
                    "accuracy": evaluation_results.get("accuracy", 0.0),
                    "bleu_score": evaluation_results.get("bleu_score", 0.0),
                    "rouge_score": evaluation_results.get("rouge_score", 0.0),
                    "legal_relevance": evaluation_results.get("legal_relevance", 0.0)
                }
            }
            
            # 보고서 저장
            report_path = Path(self.config["training"]["output_dir"]) / "training_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Training report saved to {report_path}")
            
            # 콘솔에 요약 출력
            print("\n" + "="*60)
            print("🎉 법률 모델 훈련 완료!")
            print("="*60)
            print(f"⏱️  훈련 시간: {training_duration.total_seconds()/60:.1f}분")
            print(f"📊 총 샘플 수: {evaluation_results.get('total_samples', 0)}개")
            print(f"🎯 정확도: {evaluation_results.get('accuracy', 0.0):.3f}")
            print(f"📝 BLEU 점수: {evaluation_results.get('bleu_score', 0.0):.3f}")
            print(f"📄 ROUGE 점수: {evaluation_results.get('rouge_score', 0.0):.3f}")
            print(f"⚖️  법률 관련성: {evaluation_results.get('legal_relevance', 0.0):.3f}")
            print(f"💾 모델 저장 위치: {self.config['training']['output_dir']}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Failed to generate training report: {e}")
            raise


def create_default_config() -> Dict[str, Any]:
    """기본 설정 생성"""
    return {
        "model": {
            "name": "skt/kogpt2-base-v2",
            "device": "cpu"
        },
        "data": {
            "train_path": "data/training/train_split.json",
            "val_path": "data/training/validation_split.json",
            "test_path": "data/training/test_split.json"
        },
        "training": {
            "output_dir": "models/finetuned/kogpt2-legal-lora",
            "epochs": 3,
            "batch_size": 1,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 5e-5,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500
        }
    }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="법률 모델 LoRA 파인튜닝")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--model", type=str, default="skt/kogpt2-base-v2", help="모델 이름")
    parser.add_argument("--data", type=str, default="data/training", help="데이터 디렉토리")
    parser.add_argument("--output", type=str, default="models/finetuned/kogpt2-legal-lora", help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=3, help="훈련 에포크 수")
    parser.add_argument("--batch-size", type=int, default=1, help="배치 크기")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="학습률")
    parser.add_argument("--device", type=str, default="cpu", help="디바이스 (cpu/cuda)")
    parser.add_argument("--test-only", action="store_true", help="테스트 모드 (훈련 없이 평가만)")
    
    args = parser.parse_args()
    
    # 설정 로드 또는 생성
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = create_default_config()
        
        # 명령행 인수로 설정 업데이트
        config["model"]["name"] = args.model
        config["model"]["device"] = args.device
        config["data"]["train_path"] = str(Path(args.data) / "train_split.json")
        config["data"]["val_path"] = str(Path(args.data) / "validation_split.json")
        config["data"]["test_path"] = str(Path(args.data) / "test_split.json")
        config["training"]["output_dir"] = args.output
        config["training"]["epochs"] = args.epochs
        config["training"]["batch_size"] = args.batch_size
        config["training"]["learning_rate"] = args.learning_rate
    
    logger.info(f"Starting legal model fine-tuning with config: {config}")
    
    try:
        # 훈련 파이프라인 실행
        pipeline = LegalModelTrainingPipeline(config)
        
        if args.test_only:
            logger.info("Running in test mode (evaluation only)")
            # 테스트 모드: 기존 모델 로드 후 평가만 수행
            pipeline.initialize_model()
            train_data, val_data, test_data = pipeline.load_training_data()
            evaluation_results = pipeline.evaluate_model(test_data)
            sample_responses = pipeline.generate_sample_responses(test_data)
        else:
            # 전체 훈련 파이프라인 실행
            pipeline.run_training_pipeline()
        
        logger.info("Legal model fine-tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Legal model fine-tuning failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
