#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KoGPT-2 모델 구조 분석 �??�바�?LoRA target modules 찾기
LawFirmAI ?�로?�트 - TASK 3.1 ?�련 ?�경 구성
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

def setup_logging():
    """로깅 ?�정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def analyze_model_structure(model_name: str = "skt/kogpt2-base-v2"):
    """모델 구조 분석"""
    logger = setup_logging()
    
    try:
        # 모델 로딩
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        logger.info(f"Analyzing model structure: {model_name}")
        logger.info(f"Model type: {type(model).__name__}")
        
        # 모델??모든 ?�이???�름 출력
        logger.info("\n=== Model Layer Names ===")
        for name, module in model.named_modules():
            if len(name) > 0:  # 루트 모듈 ?�외
                logger.info(f"{name}: {type(module).__name__}")
        
        # Transformer 블록??attention ?�이??찾기
        logger.info("\n=== Looking for Attention Layers ===")
        attention_layers = []
        
        for name, module in model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                logger.info(f"Found attention layer: {name} - {type(module).__name__}")
                attention_layers.append(name)
        
        # Linear ?�이??찾기 (LoRA ?�용 ?�??
        logger.info("\n=== Looking for Linear Layers ===")
        linear_layers = []
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                logger.info(f"Found Linear layer: {name} - {module}")
                linear_layers.append(name)
        
        # GPT-2 ?��??�의 attention ?�이??찾기
        logger.info("\n=== Looking for GPT-2 Style Attention Layers ===")
        gpt2_attention_layers = []
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(x in name for x in ['c_attn', 'c_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
                logger.info(f"Found GPT-2 style layer: {name} - {module}")
                gpt2_attention_layers.append(name)
        
        # 모델??�?번째 transformer 블록 분석
        logger.info("\n=== First Transformer Block Analysis ===")
        if hasattr(model, 'transformer'):
            transformer = model.transformer
            logger.info(f"Transformer type: {type(transformer).__name__}")
            
            if hasattr(transformer, 'h'):
                first_block = transformer.h[0]
                logger.info(f"First block type: {type(first_block).__name__}")
                
                for name, module in first_block.named_modules():
                    if len(name) > 0:
                        logger.info(f"Block layer: {name} - {type(module).__name__}")
        
        return {
            "model_name": model_name,
            "attention_layers": attention_layers,
            "linear_layers": linear_layers,
            "gpt2_style_layers": gpt2_attention_layers
        }
        
    except Exception as e:
        logger.error(f"Model analysis failed: {e}")
        return {"error": str(e)}

def find_optimal_lora_targets(model_name: str = "skt/kogpt2-base-v2"):
    """최적??LoRA target modules 찾기"""
    logger = setup_logging()
    
    try:
        # 모델 로딩
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        logger.info(f"Finding optimal LoRA targets for: {model_name}")
        
        # ?�반?�인 GPT-2 ?��???target modules ?�도
        common_targets = [
            ["c_attn", "c_proj"],  # GPT-2 ?��???
            ["q_proj", "k_proj", "v_proj", "o_proj"],  # LLaMA ?��???
            ["query", "key", "value", "dense"],  # BERT ?��???
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],  # ?�체 경로
        ]
        
        for targets in common_targets:
            logger.info(f"\nTrying targets: {targets}")
            
            # �?target??모델??존재?�는지 ?�인
            found_targets = []
            for target in targets:
                for name, module in model.named_modules():
                    if target in name and isinstance(module, torch.nn.Linear):
                        found_targets.append(name)
                        logger.info(f"Found: {name}")
                        break
            
            if found_targets:
                logger.info(f"??Found {len(found_targets)} matching targets: {found_targets}")
                return found_targets
            else:
                logger.info(f"??No matching targets found for: {targets}")
        
        # 모든 Linear ?�이??중에??attention 관??찾기
        logger.info("\n=== Searching for Attention-related Linear Layers ===")
        attention_linear_layers = []
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # attention 관???�워?��? ?�함???�이??찾기
                if any(keyword in name.lower() for keyword in ['attn', 'attention', 'query', 'key', 'value', 'proj']):
                    attention_linear_layers.append(name)
                    logger.info(f"Attention-related Linear layer: {name}")
        
        if attention_linear_layers:
            logger.info(f"??Found {len(attention_linear_layers)} attention-related Linear layers")
            return attention_linear_layers[:4]  # 최�? 4�?반환
        
        # 모든 Linear ?�이??반환 (최후???�단)
        logger.info("\n=== Fallback: All Linear Layers ===")
        all_linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                all_linear_layers.append(name)
        
        logger.info(f"Found {len(all_linear_layers)} total Linear layers")
        return all_linear_layers[:8]  # 최�? 8�?반환
        
    except Exception as e:
        logger.error(f"Target finding failed: {e}")
        return []

def test_lora_with_correct_targets(model_name: str = "skt/kogpt2-base-v2"):
    """?�바�?target modules�?LoRA ?�스??""
    logger = setup_logging()
    
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        # 최적??target modules 찾기
        target_modules = find_optimal_lora_targets(model_name)
        
        if not target_modules:
            logger.error("No suitable target modules found")
            return {"status": "error", "error": "No target modules found"}
        
        logger.info(f"Using target modules: {target_modules}")
        
        # 모델 로딩
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # LoRA ?�정
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        
        # LoRA 모델 ?�성
        peft_model = get_peft_model(model, lora_config)
        
        logger.info("??LoRA model created successfully!")
        
        # ?�라미터 ???�인
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
        
        return {
            "status": "success",
            "target_modules": target_modules,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_ratio": trainable_params/total_params
        }
        
    except Exception as e:
        logger.error(f"LoRA test failed: {e}")
        return {"status": "error", "error": str(e)}

def main():
    """메인 ?�수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="KoGPT-2 Model Structure Analysis")
    parser.add_argument("--model", default="skt/kogpt2-base-v2", help="Model name to analyze")
    parser.add_argument("--test-lora", action="store_true", help="Test LoRA with correct targets")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    if args.test_lora:
        # LoRA ?�스???�행
        logger.info("Testing LoRA with correct target modules...")
        result = test_lora_with_correct_targets(args.model)
        
        if result["status"] == "success":
            print(f"\n??LoRA configuration successful!")
            print(f"Target modules: {result['target_modules']}")
            print(f"Trainable parameters: {result['trainable_params']:,}")
            print(f"Trainable ratio: {result['trainable_ratio']:.2%}")
        else:
            print(f"\n??LoRA configuration failed: {result['error']}")
    else:
        # 모델 구조 분석
        logger.info("Analyzing model structure...")
        result = analyze_model_structure(args.model)
        
        if "error" not in result:
            print(f"\n=== Analysis Results ===")
            print(f"Attention layers found: {len(result['attention_layers'])}")
            print(f"Linear layers found: {len(result['linear_layers'])}")
            print(f"GPT-2 style layers found: {len(result['gpt2_style_layers'])}")
            
            if result['gpt2_style_layers']:
                print(f"\nRecommended target modules: {result['gpt2_style_layers']}")
        else:
            print(f"\n??Analysis failed: {result['error']}")

if __name__ == "__main__":
    main()
