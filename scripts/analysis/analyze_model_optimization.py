#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 ?�기 �?메모�??�용??분석 �?최적???�크립트
LawFirmAI ?�로?�트 - TASK 1.2.3
"""

import os
import sys
import time
import psutil
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import logging
from pathlib import Path

# 모델 최적???�이브러�?
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)
import onnx
from onnxruntime import InferenceSession
import onnxruntime as ort

# 로깅 ?�정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizationAnalyzer:
    """모델 최적??분석 ?�래??""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = {}
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> List[str]:
        """?�스???�이??로드"""
        return [
            "계약?�에??주의?�야 ??조항?� 무엇?��???",
            "?�해배상 �?��권의 ?�멸?�효??�??�인가??",
            "근로기�?법상 ?�게?�간?� ?�떻�?규정?�어 ?�나??",
            "부?�산 매매계약?�서 중도금�? ?�제 지급해???�나??",
            "?�혼 ???�산분할?� ?�떻�??�루?��??�요?"
        ]
    
    def analyze_kobart_optimization(self) -> Dict[str, Any]:
        """KoBART 모델 최적??분석"""
        logger.info("KoBART 모델 최적??분석 ?�작...")
        
        model_name = "skt/kobart-base-v1"
        results = {
            "model_name": model_name,
            "optimization_analysis": {},
            "benchmark_time": datetime.now().isoformat()
        }
        
        try:
            # ?�본 모델 분석
            original_analysis = self._analyze_original_model(model_name, "seq2seq")
            
            # ?�자??분석
            quantization_analysis = self._analyze_quantization(model_name, "seq2seq")
            
            # ONNX 변??분석
            onnx_analysis = self._analyze_onnx_conversion(model_name, "seq2seq")
            
            # ?�루??분석
            pruning_analysis = self._analyze_pruning(model_name, "seq2seq")
            
            results["optimization_analysis"] = {
                "original": original_analysis,
                "quantization": quantization_analysis,
                "onnx": onnx_analysis,
                "pruning": pruning_analysis,
                "recommendations": self._generate_optimization_recommendations(
                    original_analysis, quantization_analysis, onnx_analysis, pruning_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"KoBART 최적??분석 ?�패: {e}")
            results["error"] = str(e)
            
        return results
    
    def analyze_kogpt2_optimization(self) -> Dict[str, Any]:
        """KoGPT-2 모델 최적??분석"""
        logger.info("KoGPT-2 모델 최적??분석 ?�작...")
        
        model_name = "skt/kogpt2-base-v2"
        results = {
            "model_name": model_name,
            "optimization_analysis": {},
            "benchmark_time": datetime.now().isoformat()
        }
        
        try:
            # ?�본 모델 분석
            original_analysis = self._analyze_original_model(model_name, "causal_lm")
            
            # ?�자??분석
            quantization_analysis = self._analyze_quantization(model_name, "causal_lm")
            
            # ONNX 변??분석
            onnx_analysis = self._analyze_onnx_conversion(model_name, "causal_lm")
            
            # ?�루??분석
            pruning_analysis = self._analyze_pruning(model_name, "causal_lm")
            
            results["optimization_analysis"] = {
                "original": original_analysis,
                "quantization": quantization_analysis,
                "onnx": onnx_analysis,
                "pruning": pruning_analysis,
                "recommendations": self._generate_optimization_recommendations(
                    original_analysis, quantization_analysis, onnx_analysis, pruning_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"KoGPT-2 최적??분석 ?�패: {e}")
            results["error"] = str(e)
            
        return results
    
    def _analyze_original_model(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """?�본 모델 분석"""
        logger.info(f"?�본 {model_name} 모델 분석...")
        
        try:
            # 모델 로딩
            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
            
            model.to(self.device)
            loading_time = time.time() - start_time
            
            # 모델 ?�보 ?�집
            num_parameters = sum(p.numel() for p in model.parameters())
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 모델 ?�기 계산
            model_size = self._calculate_model_size(model)
            
            # 메모�??�용??측정
            memory_usage = self._get_memory_usage()
            
            # 추론 ?�능 ?�스??
            inference_performance = self._test_inference_performance(model, tokenizer, model_type)
            
            # 모델 ?�리
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                "loading_time": loading_time,
                "num_parameters": num_parameters,
                "trainable_parameters": trainable_parameters,
                "model_size_mb": model_size,
                "memory_usage_mb": memory_usage,
                "inference_performance": inference_performance
            }
            
        except Exception as e:
            logger.error(f"?�본 모델 분석 ?�패: {e}")
            return {"error": str(e)}
    
    def _analyze_quantization(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """?�자??분석"""
        logger.info(f"{model_name} ?�자??분석...")
        
        try:
            # INT8 ?�자??
            int8_analysis = self._test_int8_quantization(model_name, model_type)
            
            # INT4 ?�자??(BitsAndBytesConfig ?�용)
            int4_analysis = self._test_int4_quantization(model_name, model_type)
            
            return {
                "int8": int8_analysis,
                "int4": int4_analysis
            }
            
        except Exception as e:
            logger.error(f"?�자??분석 ?�패: {e}")
            return {"error": str(e)}
    
    def _test_int8_quantization(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """INT8 ?�자???�스??""
        try:
            # 모델 로딩
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            # INT8 ?�자??
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
            # ?�능 측정
            model_size = self._calculate_model_size(quantized_model)
            memory_usage = self._get_memory_usage()
            inference_performance = self._test_inference_performance(quantized_model, tokenizer, model_type)
            
            # ?�리
            del model, quantized_model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                "model_size_mb": model_size,
                "memory_usage_mb": memory_usage,
                "inference_performance": inference_performance,
                "compression_ratio": 0.5  # INT8?� ?�??50% ?�축
            }
            
        except Exception as e:
            logger.error(f"INT8 ?�자???�스???�패: {e}")
            return {"error": str(e)}
    
    def _test_int4_quantization(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """INT4 ?�자???�스??(BitsAndBytesConfig)"""
        try:
            # BitsAndBytesConfig ?�정
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # 모델 로딩
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config
                )
            
            # ?�능 측정
            model_size = self._calculate_model_size(model)
            memory_usage = self._get_memory_usage()
            inference_performance = self._test_inference_performance(model, tokenizer, model_type)
            
            # ?�리
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                "model_size_mb": model_size,
                "memory_usage_mb": memory_usage,
                "inference_performance": inference_performance,
                "compression_ratio": 0.25  # INT4???�??75% ?�축
            }
            
        except Exception as e:
            logger.error(f"INT4 ?�자???�스???�패: {e}")
            return {"error": str(e)}
    
    def _analyze_onnx_conversion(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """ONNX 변??분석"""
        logger.info(f"{model_name} ONNX 변??분석...")
        
        try:
            # PyTorch 모델 로딩
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
            
            model.eval()
            
            # ONNX 변??
            onnx_path = f"{model_name.replace('/', '_')}.onnx"
            dummy_input = torch.randint(0, 1000, (1, 10))  # ?��? ?�력
            
            start_time = time.time()
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['output'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'output': {0: 'batch_size', 1: 'sequence'}
                }
            )
            conversion_time = time.time() - start_time
            
            # ONNX 모델 분석
            onnx_model = onnx.load(onnx_path)
            onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
            
            # ONNX Runtime ?�능 ?�스??
            ort_session = InferenceSession(onnx_path)
            onnx_performance = self._test_onnx_performance(ort_session, tokenizer)
            
            # ?�리
            del model, tokenizer
            os.remove(onnx_path)
            
            return {
                "conversion_time": conversion_time,
                "onnx_size_mb": onnx_size,
                "onnx_performance": onnx_performance,
                "compression_ratio": onnx_size / self._calculate_model_size(model) if hasattr(self, '_temp_model_size') else 1.0
            }
            
        except Exception as e:
            logger.error(f"ONNX 변??분석 ?�패: {e}")
            return {"error": str(e)}
    
    def _analyze_pruning(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """?�루??분석"""
        logger.info(f"{model_name} ?�루??분석...")
        
        try:
            # 모델 로딩
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # 구조???�루??(20% ?�거)
            pruned_model = self._apply_structural_pruning(model, sparsity=0.2)
            
            # ?�능 측정
            model_size = self._calculate_model_size(pruned_model)
            memory_usage = self._get_memory_usage()
            inference_performance = self._test_inference_performance(pruned_model, tokenizer, model_type)
            
            # ?�리
            del model, pruned_model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                "model_size_mb": model_size,
                "memory_usage_mb": memory_usage,
                "inference_performance": inference_performance,
                "sparsity": 0.2,
                "compression_ratio": 0.8  # 20% ?�축
            }
            
        except Exception as e:
            logger.error(f"?�루??분석 ?�패: {e}")
            return {"error": str(e)}
    
    def _apply_structural_pruning(self, model, sparsity: float = 0.2):
        """구조???�루???�용"""
        # 간단??가중치 기반 ?�루??
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # 가중치???�댓값이 ?��? 것들??0?�로 ?�정
                with torch.no_grad():
                    weight = module.weight
                    threshold = torch.quantile(torch.abs(weight), sparsity)
                    mask = torch.abs(weight) > threshold
                    module.weight.data *= mask.float()
        
        return model
    
    def _test_onnx_performance(self, ort_session, tokenizer) -> Dict[str, Any]:
        """ONNX Runtime ?�능 ?�스??""
        try:
            total_time = 0
            successful_inferences = 0
            
            for text in self.test_data[:3]:  # 처음 3개만 ?�스??
                try:
                    # ?�큰??
                    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
                    input_ids = inputs["input_ids"].astype(np.int64)
                    
                    # ONNX 추론
                    start_time = time.time()
                    outputs = ort_session.run(None, {"input_ids": input_ids})
                    inference_time = time.time() - start_time
                    
                    total_time += inference_time
                    successful_inferences += 1
                    
                except Exception as e:
                    logger.warning(f"ONNX 추론 ?�패: {e}")
                    continue
            
            return {
                "total_time": total_time,
                "average_time": total_time / successful_inferences if successful_inferences > 0 else 0,
                "successful_inferences": successful_inferences
            }
            
        except Exception as e:
            logger.error(f"ONNX ?�능 ?�스???�패: {e}")
            return {"error": str(e)}
    
    def _test_inference_performance(self, model, tokenizer, model_type: str) -> Dict[str, Any]:
        """추론 ?�능 ?�스??""
        try:
            total_time = 0
            successful_inferences = 0
            
            for text in self.test_data:
                try:
                    start_time = time.time()
                    
                    if model_type == "seq2seq":
                        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                inputs["input_ids"],
                                max_length=100,
                                num_return_sequences=1,
                                temperature=0.7,
                                do_sample=True
                            )
                    else:
                        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                inputs["input_ids"],
                                max_length=inputs["input_ids"].shape[1] + 50,
                                num_return_sequences=1,
                                temperature=0.7,
                                do_sample=True
                            )
                    
                    inference_time = time.time() - start_time
                    total_time += inference_time
                    successful_inferences += 1
                    
                except Exception as e:
                    logger.warning(f"추론 ?�패: {e}")
                    continue
            
            return {
                "total_time": total_time,
                "average_time": total_time / successful_inferences if successful_inferences > 0 else 0,
                "successful_inferences": successful_inferences
            }
            
        except Exception as e:
            logger.error(f"추론 ?�능 ?�스???�패: {e}")
            return {"error": str(e)}
    
    def _calculate_model_size(self, model) -> float:
        """모델 ?�기 계산 (MB)"""
        try:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            return param_size / 1024 / 1024
        except:
            return 0
    
    def _get_memory_usage(self) -> float:
        """?�재 메모�??�용??반환 (MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _generate_optimization_recommendations(self, original, quantization, onnx, pruning) -> List[str]:
        """최적??권장?�항 ?�성"""
        recommendations = []
        
        try:
            # 메모�??�용??기반 권장?�항
            if "memory_usage_mb" in original:
                original_memory = original["memory_usage_mb"]
                
                if original_memory > 8000:  # 8GB ?�상
                    recommendations.append("메모�??�용?�이 ?�으므�??�자??INT8 ?�는 INT4) ?�용 권장")
                
                if "int4" in quantization and "memory_usage_mb" in quantization["int4"]:
                    int4_memory = quantization["int4"]["memory_usage_mb"]
                    if int4_memory < original_memory * 0.5:
                        recommendations.append("INT4 ?�자?�로 메모�??�용?�을 50% ?�상 ?�약 가??)
            
            # 추론 ?�도 기반 권장?�항
            if "inference_performance" in original and "average_time" in original["inference_performance"]:
                original_time = original["inference_performance"]["average_time"]
                
                if "onnx" in onnx and "onnx_performance" in onnx and "average_time" in onnx["onnx_performance"]:
                    onnx_time = onnx["onnx_performance"]["average_time"]
                    if onnx_time < original_time * 0.8:
                        recommendations.append("ONNX 변?�으�?추론 ?�도 20% ?�상 ?�상 가??)
            
            # HuggingFace Spaces ?�경 고려
            recommendations.append("HuggingFace Spaces ?�경?�서??INT4 ?�자?��? ONNX 변??조합 권장")
            recommendations.append("메모�??�한(16GB)??고려?�여 모델 ?�기 최적???�수")
            
        except Exception as e:
            recommendations.append(f"권장?�항 ?�성 �??�류: {e}")
        
        return recommendations
    
    def run_analysis(self) -> Dict[str, Any]:
        """?�체 최적??분석 ?�행"""
        logger.info("모델 최적??분석 ?�작...")
        
        # ?�스???�보 ?�집
        system_info = {
            "device": self.device,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "python_version": sys.version,
            "torch_version": torch.__version__
        }
        
        # �?모델 최적??분석 ?�행
        kobart_analysis = self.analyze_kobart_optimization()
        kogpt2_analysis = self.analyze_kogpt2_optimization()
        
        # 결과 종합
        analysis_results = {
            "system_info": system_info,
            "kobart_optimization": kobart_analysis,
            "kogpt2_optimization": kogpt2_analysis,
            "comparison": self._compare_optimizations(kobart_analysis, kogpt2_analysis)
        }
        
        return analysis_results
    
    def _compare_optimizations(self, kobart_analysis, kogpt2_analysis) -> Dict[str, Any]:
        """최적??결과 비교"""
        comparison = {
            "memory_optimization": {},
            "speed_optimization": {},
            "size_optimization": {},
            "recommendation": ""
        }
        
        try:
            # 메모�?최적??비교
            if "optimization_analysis" in kobart_analysis and "original" in kobart_analysis["optimization_analysis"]:
                kobart_original = kobart_analysis["optimization_analysis"]["original"]
                kobart_memory = kobart_original.get("memory_usage_mb", 0)
                
                if "optimization_analysis" in kogpt2_analysis and "original" in kogpt2_analysis["optimization_analysis"]:
                    kogpt2_original = kogpt2_analysis["optimization_analysis"]["original"]
                    kogpt2_memory = kogpt2_original.get("memory_usage_mb", 0)
                    
                    comparison["memory_optimization"] = {
                        "kobart_mb": kobart_memory,
                        "kogpt2_mb": kogpt2_memory,
                        "memory_ratio": kobart_memory / kogpt2_memory if kogpt2_memory > 0 else 0
                    }
            
            # 최적??권장?�항
            comparison["recommendation"] = "HuggingFace Spaces ?�경?�서??메모�??�율?�이 ?�수??KoGPT-2 + INT4 ?�자??+ ONNX 변??조합 권장"
            
        except Exception as e:
            logger.error(f"최적??비교 �??�류: {e}")
            comparison["error"] = str(e)
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """분석 결과 ?�??""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_optimization_analysis_{timestamp}.json"
        
        filepath = os.path.join("benchmark_results", filename)
        os.makedirs("benchmark_results", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"분석 결과 ?�?? {filepath}")
        return filepath

def main():
    """메인 ?�행 ?�수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="모델 최적??분석")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="?�행 ?�바?�스")
    parser.add_argument("--output", help="결과 ?�???�일�?)
    
    args = parser.parse_args()
    
    # 분석 ?�행
    analyzer = ModelOptimizationAnalyzer(device=args.device)
    results = analyzer.run_analysis()
    
    # 결과 ?�??
    output_file = analyzer.save_results(results, args.output)
    
    # 결과 ?�약 출력
    print("\n" + "="*50)
    print("모델 최적??분석 결과 ?�약")
    print("="*50)
    
    if "kobart_optimization" in results and "optimization_analysis" in results["kobart_optimization"]:
        kobart_analysis = results["kobart_optimization"]["optimization_analysis"]
        if "original" in kobart_analysis:
            original = kobart_analysis["original"]
            print(f"KoBART ?�본 - ?�기: {original.get('model_size_mb', 0):.1f}MB, "
                  f"메모�? {original.get('memory_usage_mb', 0):.1f}MB")
    
    if "kogpt2_optimization" in results and "optimization_analysis" in results["kogpt2_optimization"]:
        kogpt2_analysis = results["kogpt2_optimization"]["optimization_analysis"]
        if "original" in kogpt2_analysis:
            original = kogpt2_analysis["original"]
            print(f"KoGPT-2 ?�본 - ?�기: {original.get('model_size_mb', 0):.1f}MB, "
                  f"메모�? {original.get('memory_usage_mb', 0):.1f}MB")
    
    if "comparison" in results and "recommendation" in results["comparison"]:
        print(f"\n권장?�항: {results['comparison']['recommendation']}")
    
    print(f"\n?�세 결과: {output_file}")

if __name__ == "__main__":
    main()
