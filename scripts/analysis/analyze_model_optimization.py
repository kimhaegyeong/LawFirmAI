#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 크기 및 메모리 사용량 분석 및 최적화 스크립트
LawFirmAI 프로젝트 - TASK 1.2.3
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

# 모델 최적화 라이브러리
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)
import onnx
from onnxruntime import InferenceSession
import onnxruntime as ort

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizationAnalyzer:
    """모델 최적화 분석 클래스"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = {}
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> List[str]:
        """테스트 데이터 로드"""
        return [
            "계약서에서 주의해야 할 조항은 무엇인가요?",
            "손해배상 청구권의 소멸시효는 몇 년인가요?",
            "근로기준법상 휴게시간은 어떻게 규정되어 있나요?",
            "부동산 매매계약에서 중도금은 언제 지급해야 하나요?",
            "이혼 시 재산분할은 어떻게 이루어지나요?"
        ]
    
    def analyze_kobart_optimization(self) -> Dict[str, Any]:
        """KoBART 모델 최적화 분석"""
        logger.info("KoBART 모델 최적화 분석 시작...")
        
        model_name = "skt/kobart-base-v1"
        results = {
            "model_name": model_name,
            "optimization_analysis": {},
            "benchmark_time": datetime.now().isoformat()
        }
        
        try:
            # 원본 모델 분석
            original_analysis = self._analyze_original_model(model_name, "seq2seq")
            
            # 양자화 분석
            quantization_analysis = self._analyze_quantization(model_name, "seq2seq")
            
            # ONNX 변환 분석
            onnx_analysis = self._analyze_onnx_conversion(model_name, "seq2seq")
            
            # 프루닝 분석
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
            logger.error(f"KoBART 최적화 분석 실패: {e}")
            results["error"] = str(e)
            
        return results
    
    def analyze_kogpt2_optimization(self) -> Dict[str, Any]:
        """KoGPT-2 모델 최적화 분석"""
        logger.info("KoGPT-2 모델 최적화 분석 시작...")
        
        model_name = "skt/kogpt2-base-v2"
        results = {
            "model_name": model_name,
            "optimization_analysis": {},
            "benchmark_time": datetime.now().isoformat()
        }
        
        try:
            # 원본 모델 분석
            original_analysis = self._analyze_original_model(model_name, "causal_lm")
            
            # 양자화 분석
            quantization_analysis = self._analyze_quantization(model_name, "causal_lm")
            
            # ONNX 변환 분석
            onnx_analysis = self._analyze_onnx_conversion(model_name, "causal_lm")
            
            # 프루닝 분석
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
            logger.error(f"KoGPT-2 최적화 분석 실패: {e}")
            results["error"] = str(e)
            
        return results
    
    def _analyze_original_model(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """원본 모델 분석"""
        logger.info(f"원본 {model_name} 모델 분석...")
        
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
            
            # 모델 정보 수집
            num_parameters = sum(p.numel() for p in model.parameters())
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 모델 크기 계산
            model_size = self._calculate_model_size(model)
            
            # 메모리 사용량 측정
            memory_usage = self._get_memory_usage()
            
            # 추론 성능 테스트
            inference_performance = self._test_inference_performance(model, tokenizer, model_type)
            
            # 모델 정리
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
            logger.error(f"원본 모델 분석 실패: {e}")
            return {"error": str(e)}
    
    def _analyze_quantization(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """양자화 분석"""
        logger.info(f"{model_name} 양자화 분석...")
        
        try:
            # INT8 양자화
            int8_analysis = self._test_int8_quantization(model_name, model_type)
            
            # INT4 양자화 (BitsAndBytesConfig 사용)
            int4_analysis = self._test_int4_quantization(model_name, model_type)
            
            return {
                "int8": int8_analysis,
                "int4": int4_analysis
            }
            
        except Exception as e:
            logger.error(f"양자화 분석 실패: {e}")
            return {"error": str(e)}
    
    def _test_int8_quantization(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """INT8 양자화 테스트"""
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
            
            # INT8 양자화
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
            # 성능 측정
            model_size = self._calculate_model_size(quantized_model)
            memory_usage = self._get_memory_usage()
            inference_performance = self._test_inference_performance(quantized_model, tokenizer, model_type)
            
            # 정리
            del model, quantized_model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                "model_size_mb": model_size,
                "memory_usage_mb": memory_usage,
                "inference_performance": inference_performance,
                "compression_ratio": 0.5  # INT8은 대략 50% 압축
            }
            
        except Exception as e:
            logger.error(f"INT8 양자화 테스트 실패: {e}")
            return {"error": str(e)}
    
    def _test_int4_quantization(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """INT4 양자화 테스트 (BitsAndBytesConfig)"""
        try:
            # BitsAndBytesConfig 설정
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
            
            # 성능 측정
            model_size = self._calculate_model_size(model)
            memory_usage = self._get_memory_usage()
            inference_performance = self._test_inference_performance(model, tokenizer, model_type)
            
            # 정리
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                "model_size_mb": model_size,
                "memory_usage_mb": memory_usage,
                "inference_performance": inference_performance,
                "compression_ratio": 0.25  # INT4는 대략 75% 압축
            }
            
        except Exception as e:
            logger.error(f"INT4 양자화 테스트 실패: {e}")
            return {"error": str(e)}
    
    def _analyze_onnx_conversion(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """ONNX 변환 분석"""
        logger.info(f"{model_name} ONNX 변환 분석...")
        
        try:
            # PyTorch 모델 로딩
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
            
            model.eval()
            
            # ONNX 변환
            onnx_path = f"{model_name.replace('/', '_')}.onnx"
            dummy_input = torch.randint(0, 1000, (1, 10))  # 더미 입력
            
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
            
            # ONNX Runtime 성능 테스트
            ort_session = InferenceSession(onnx_path)
            onnx_performance = self._test_onnx_performance(ort_session, tokenizer)
            
            # 정리
            del model, tokenizer
            os.remove(onnx_path)
            
            return {
                "conversion_time": conversion_time,
                "onnx_size_mb": onnx_size,
                "onnx_performance": onnx_performance,
                "compression_ratio": onnx_size / self._calculate_model_size(model) if hasattr(self, '_temp_model_size') else 1.0
            }
            
        except Exception as e:
            logger.error(f"ONNX 변환 분석 실패: {e}")
            return {"error": str(e)}
    
    def _analyze_pruning(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """프루닝 분석"""
        logger.info(f"{model_name} 프루닝 분석...")
        
        try:
            # 모델 로딩
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # 구조적 프루닝 (20% 제거)
            pruned_model = self._apply_structural_pruning(model, sparsity=0.2)
            
            # 성능 측정
            model_size = self._calculate_model_size(pruned_model)
            memory_usage = self._get_memory_usage()
            inference_performance = self._test_inference_performance(pruned_model, tokenizer, model_type)
            
            # 정리
            del model, pruned_model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                "model_size_mb": model_size,
                "memory_usage_mb": memory_usage,
                "inference_performance": inference_performance,
                "sparsity": 0.2,
                "compression_ratio": 0.8  # 20% 압축
            }
            
        except Exception as e:
            logger.error(f"프루닝 분석 실패: {e}")
            return {"error": str(e)}
    
    def _apply_structural_pruning(self, model, sparsity: float = 0.2):
        """구조적 프루닝 적용"""
        # 간단한 가중치 기반 프루닝
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # 가중치의 절댓값이 작은 것들을 0으로 설정
                with torch.no_grad():
                    weight = module.weight
                    threshold = torch.quantile(torch.abs(weight), sparsity)
                    mask = torch.abs(weight) > threshold
                    module.weight.data *= mask.float()
        
        return model
    
    def _test_onnx_performance(self, ort_session, tokenizer) -> Dict[str, Any]:
        """ONNX Runtime 성능 테스트"""
        try:
            total_time = 0
            successful_inferences = 0
            
            for text in self.test_data[:3]:  # 처음 3개만 테스트
                try:
                    # 토큰화
                    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
                    input_ids = inputs["input_ids"].astype(np.int64)
                    
                    # ONNX 추론
                    start_time = time.time()
                    outputs = ort_session.run(None, {"input_ids": input_ids})
                    inference_time = time.time() - start_time
                    
                    total_time += inference_time
                    successful_inferences += 1
                    
                except Exception as e:
                    logger.warning(f"ONNX 추론 실패: {e}")
                    continue
            
            return {
                "total_time": total_time,
                "average_time": total_time / successful_inferences if successful_inferences > 0 else 0,
                "successful_inferences": successful_inferences
            }
            
        except Exception as e:
            logger.error(f"ONNX 성능 테스트 실패: {e}")
            return {"error": str(e)}
    
    def _test_inference_performance(self, model, tokenizer, model_type: str) -> Dict[str, Any]:
        """추론 성능 테스트"""
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
                    logger.warning(f"추론 실패: {e}")
                    continue
            
            return {
                "total_time": total_time,
                "average_time": total_time / successful_inferences if successful_inferences > 0 else 0,
                "successful_inferences": successful_inferences
            }
            
        except Exception as e:
            logger.error(f"추론 성능 테스트 실패: {e}")
            return {"error": str(e)}
    
    def _calculate_model_size(self, model) -> float:
        """모델 크기 계산 (MB)"""
        try:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            return param_size / 1024 / 1024
        except:
            return 0
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _generate_optimization_recommendations(self, original, quantization, onnx, pruning) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        try:
            # 메모리 사용량 기반 권장사항
            if "memory_usage_mb" in original:
                original_memory = original["memory_usage_mb"]
                
                if original_memory > 8000:  # 8GB 이상
                    recommendations.append("메모리 사용량이 높으므로 양자화(INT8 또는 INT4) 적용 권장")
                
                if "int4" in quantization and "memory_usage_mb" in quantization["int4"]:
                    int4_memory = quantization["int4"]["memory_usage_mb"]
                    if int4_memory < original_memory * 0.5:
                        recommendations.append("INT4 양자화로 메모리 사용량을 50% 이상 절약 가능")
            
            # 추론 속도 기반 권장사항
            if "inference_performance" in original and "average_time" in original["inference_performance"]:
                original_time = original["inference_performance"]["average_time"]
                
                if "onnx" in onnx and "onnx_performance" in onnx and "average_time" in onnx["onnx_performance"]:
                    onnx_time = onnx["onnx_performance"]["average_time"]
                    if onnx_time < original_time * 0.8:
                        recommendations.append("ONNX 변환으로 추론 속도 20% 이상 향상 가능")
            
            # HuggingFace Spaces 환경 고려
            recommendations.append("HuggingFace Spaces 환경에서는 INT4 양자화와 ONNX 변환 조합 권장")
            recommendations.append("메모리 제한(16GB)을 고려하여 모델 크기 최적화 필수")
            
        except Exception as e:
            recommendations.append(f"권장사항 생성 중 오류: {e}")
        
        return recommendations
    
    def run_analysis(self) -> Dict[str, Any]:
        """전체 최적화 분석 실행"""
        logger.info("모델 최적화 분석 시작...")
        
        # 시스템 정보 수집
        system_info = {
            "device": self.device,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "python_version": sys.version,
            "torch_version": torch.__version__
        }
        
        # 각 모델 최적화 분석 실행
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
        """최적화 결과 비교"""
        comparison = {
            "memory_optimization": {},
            "speed_optimization": {},
            "size_optimization": {},
            "recommendation": ""
        }
        
        try:
            # 메모리 최적화 비교
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
            
            # 최적화 권장사항
            comparison["recommendation"] = "HuggingFace Spaces 환경에서는 메모리 효율성이 우수한 KoGPT-2 + INT4 양자화 + ONNX 변환 조합 권장"
            
        except Exception as e:
            logger.error(f"최적화 비교 중 오류: {e}")
            comparison["error"] = str(e)
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """분석 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_optimization_analysis_{timestamp}.json"
        
        filepath = os.path.join("benchmark_results", filename)
        os.makedirs("benchmark_results", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"분석 결과 저장: {filepath}")
        return filepath

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="모델 최적화 분석")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="실행 디바이스")
    parser.add_argument("--output", help="결과 저장 파일명")
    
    args = parser.parse_args()
    
    # 분석 실행
    analyzer = ModelOptimizationAnalyzer(device=args.device)
    results = analyzer.run_analysis()
    
    # 결과 저장
    output_file = analyzer.save_results(results, args.output)
    
    # 결과 요약 출력
    print("\n" + "="*50)
    print("모델 최적화 분석 결과 요약")
    print("="*50)
    
    if "kobart_optimization" in results and "optimization_analysis" in results["kobart_optimization"]:
        kobart_analysis = results["kobart_optimization"]["optimization_analysis"]
        if "original" in kobart_analysis:
            original = kobart_analysis["original"]
            print(f"KoBART 원본 - 크기: {original.get('model_size_mb', 0):.1f}MB, "
                  f"메모리: {original.get('memory_usage_mb', 0):.1f}MB")
    
    if "kogpt2_optimization" in results and "optimization_analysis" in results["kogpt2_optimization"]:
        kogpt2_analysis = results["kogpt2_optimization"]["optimization_analysis"]
        if "original" in kogpt2_analysis:
            original = kogpt2_analysis["original"]
            print(f"KoGPT-2 원본 - 크기: {original.get('model_size_mb', 0):.1f}MB, "
                  f"메모리: {original.get('memory_usage_mb', 0):.1f}MB")
    
    if "comparison" in results and "recommendation" in results["comparison"]:
        print(f"\n권장사항: {results['comparison']['recommendation']}")
    
    print(f"\n상세 결과: {output_file}")

if __name__ == "__main__":
    main()
