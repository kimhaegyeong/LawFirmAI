#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 파인튜닝 환경 설정 및 검증 스크립트
LawFirmAI 프로젝트 - TASK 3.1 훈련 환경 구성
"""

import sys
import os
import torch
import logging
from typing import Dict, List, Optional
import json

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_pytorch_installation() -> Dict:
    """PyTorch 설치 및 버전 확인"""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        
        logger.info(f"PyTorch version: {version}")
        logger.info(f"CUDA available: {cuda_available}")
        if cuda_available:
            logger.info(f"CUDA version: {cuda_version}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f}GB)")
        
        return {
            "status": "success",
            "pytorch_version": version,
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "gpu_count": torch.cuda.device_count() if cuda_available else 0
        }
    except ImportError as e:
        logger.error(f"PyTorch not installed: {e}")
        return {"status": "error", "error": str(e)}

def check_transformers_installation() -> Dict:
    """Transformers 라이브러리 확인"""
    logger = logging.getLogger(__name__)
    
    try:
        import transformers
        version = transformers.__version__
        logger.info(f"Transformers version: {version}")
        
        return {
            "status": "success",
            "transformers_version": version
        }
    except ImportError as e:
        logger.error(f"Transformers not installed: {e}")
        return {"status": "error", "error": str(e)}

def check_peft_installation() -> Dict:
    """PEFT 라이브러리 확인"""
    logger = logging.getLogger(__name__)
    
    try:
        import peft
        version = peft.__version__
        logger.info(f"PEFT version: {version}")
        
        # LoRA 관련 클래스 확인
        from peft import LoraConfig, get_peft_model, TaskType
        logger.info("PEFT LoRA classes imported successfully")
        
        return {
            "status": "success",
            "peft_version": version,
            "lora_available": True
        }
    except ImportError as e:
        logger.error(f"PEFT not installed: {e}")
        return {"status": "error", "error": str(e)}

def check_accelerate_installation() -> Dict:
    """Accelerate 라이브러리 확인"""
    logger = logging.getLogger(__name__)
    
    try:
        import accelerate
        version = accelerate.__version__
        logger.info(f"Accelerate version: {version}")
        
        return {
            "status": "success",
            "accelerate_version": version
        }
    except ImportError as e:
        logger.error(f"Accelerate not installed: {e}")
        return {"status": "error", "error": str(e)}

def check_bitsandbytes_installation() -> Dict:
    """BitsAndBytes 라이브러리 확인 (QLoRA용)"""
    logger = logging.getLogger(__name__)
    
    try:
        import bitsandbytes
        version = bitsandbytes.__version__
        logger.info(f"BitsAndBytes version: {version}")
        
        # QLoRA 관련 기능 확인
        try:
            from bitsandbytes.nn import Linear4bit
            logger.info("BitsAndBytes 4-bit quantization available")
            quantization_available = True
        except ImportError:
            logger.warning("BitsAndBytes 4-bit quantization not available")
            quantization_available = False
        
        return {
            "status": "success",
            "bitsandbytes_version": version,
            "quantization_available": quantization_available
        }
    except ImportError as e:
        logger.error(f"BitsAndBytes not installed: {e}")
        return {"status": "error", "error": str(e)}

def test_kogpt2_loading() -> Dict:
    """KoGPT-2 모델 로딩 테스트"""
    logger = logging.getLogger(__name__)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "skt/kogpt2-base-v2"
        logger.info(f"Testing KoGPT-2 model loading: {model_name}")
        
        # 토크나이저 로딩 테스트
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded successfully")
        
        # 모델 로딩 테스트 (CPU에서)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU용
            device_map="cpu"
        )
        logger.info("Model loaded successfully")
        
        # 간단한 추론 테스트
        test_text = "안녕하세요"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=20,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test generation successful: {generated_text}")
        
        return {
            "status": "success",
            "model_name": model_name,
            "test_generation": generated_text
        }
    except Exception as e:
        logger.error(f"KoGPT-2 loading test failed: {e}")
        return {"status": "error", "error": str(e)}

def test_lora_configuration() -> Dict:
    """LoRA 설정 테스트"""
    logger = logging.getLogger(__name__)
    
    try:
        from peft import LoraConfig, TaskType
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "skt/kogpt2-base-v2"
        
        # LoRA 설정 생성
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        logger.info("LoRA configuration created successfully")
        logger.info(f"LoRA rank: {lora_config.r}")
        logger.info(f"LoRA alpha: {lora_config.lora_alpha}")
        logger.info(f"Target modules: {lora_config.target_modules}")
        
        # 모델 로딩 및 LoRA 적용 테스트
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        from peft import get_peft_model
        peft_model = get_peft_model(model, lora_config)
        
        logger.info("LoRA model created successfully")
        
        # 파라미터 수 확인
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
        
        return {
            "status": "success",
            "lora_config": {
                "r": lora_config.r,
                "alpha": lora_config.lora_alpha,
                "dropout": lora_config.lora_dropout,
                "target_modules": lora_config.target_modules
            },
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_ratio": trainable_params/total_params
        }
    except Exception as e:
        logger.error(f"LoRA configuration test failed: {e}")
        return {"status": "error", "error": str(e)}

def run_environment_check() -> Dict:
    """전체 환경 검사 실행"""
    logger = setup_logging()
    logger.info("Starting LoRA fine-tuning environment check...")
    
    results = {
        "timestamp": torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else "2025-01-25T00:00:00",
        "checks": {}
    }
    
    # 각 라이브러리 확인
    checks = [
        ("pytorch", check_pytorch_installation),
        ("transformers", check_transformers_installation),
        ("peft", check_peft_installation),
        ("accelerate", check_accelerate_installation),
        ("bitsandbytes", check_bitsandbytes_installation),
        ("kogpt2_loading", test_kogpt2_loading),
        ("lora_configuration", test_lora_configuration)
    ]
    
    for check_name, check_func in checks:
        logger.info(f"Running {check_name} check...")
        results["checks"][check_name] = check_func()
    
    # 전체 상태 요약
    success_count = sum(1 for check in results["checks"].values() if check.get("status") == "success")
    total_count = len(results["checks"])
    
    results["summary"] = {
        "total_checks": total_count,
        "successful_checks": success_count,
        "failed_checks": total_count - success_count,
        "overall_status": "success" if success_count == total_count else "partial" if success_count > 0 else "failed"
    }
    
    logger.info(f"Environment check completed: {success_count}/{total_count} checks passed")
    
    return results

def save_check_report(results: Dict, output_file: str = "logs/lora_environment_check.json"):
    """검사 결과 보고서 저장"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Environment check report saved to {output_file}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Environment Check")
    parser.add_argument("--output", default="logs/lora_environment_check.json", help="Output report file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 환경 검사 실행
    results = run_environment_check()
    
    # 보고서 저장
    save_check_report(results, args.output)
    
    # 결과 출력
    summary = results["summary"]
    print(f"\n=== Environment Check Summary ===")
    print(f"Total checks: {summary['total_checks']}")
    print(f"Successful: {summary['successful_checks']}")
    print(f"Failed: {summary['failed_checks']}")
    print(f"Overall status: {summary['overall_status']}")
    
    if summary['overall_status'] == 'success':
        print("\n✅ All checks passed! LoRA fine-tuning environment is ready.")
    elif summary['overall_status'] == 'partial':
        print("\n⚠️ Some checks failed. Please review the report for details.")
    else:
        print("\n❌ Environment check failed. Please install missing dependencies.")

if __name__ == "__main__":
    main()
