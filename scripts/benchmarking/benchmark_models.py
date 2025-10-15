#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KoBART vs KoGPT-2 성능 비교 벤치마킹 스크립트
LawFirmAI 프로젝트 - TASK 1.2.1
"""

import os
import sys
import time
import psutil
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    pipeline
)
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelBenchmark:
    """AI 모델 성능 벤치마킹 클래스"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = {}
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> List[Dict[str, str]]:
        """법률 도메인 테스트 데이터 로드"""
        return [
            {
                "question": "계약서에서 주의해야 할 조항은 무엇인가요?",
                "context": "계약서 검토 시 중요한 사항들",
                "category": "contract"
            },
            {
                "question": "손해배상 청구권의 소멸시효는 몇 년인가요?",
                "context": "민법상 손해배상 관련 조항",
                "category": "civil_law"
            },
            {
                "question": "근로기준법상 휴게시간은 어떻게 규정되어 있나요?",
                "context": "근로기준법 휴게시간 관련 조항",
                "category": "labor_law"
            },
            {
                "question": "부동산 매매계약에서 중도금은 언제 지급해야 하나요?",
                "context": "부동산 매매계약 중도금 지급 시기",
                "category": "real_estate"
            },
            {
                "question": "이혼 시 재산분할은 어떻게 이루어지나요?",
                "context": "가족법상 이혼 재산분할",
                "category": "family_law"
            }
        ]
    
    def benchmark_kobart(self) -> Dict[str, Any]:
        """KoBART 모델 벤치마킹"""
        logger.info("KoBART 모델 벤치마킹 시작...")
        
        model_name = "skt/kobart-base-v1"
        results = {
            "model_name": model_name,
            "model_type": "seq2seq",
            "benchmark_time": datetime.now().isoformat()
        }
        
        try:
            # 모델 로딩 시간 측정
            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            model.to(self.device)
            loading_time = time.time() - start_time
            
            # 메모리 사용량 측정
            memory_usage = self._get_memory_usage()
            
            # 모델 정보 수집
            model_info = {
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "model_size_mb": self._get_model_size(model),
                "loading_time": loading_time,
                "memory_usage_mb": memory_usage
            }
            
            # 추론 성능 테스트
            inference_results = self._test_inference_kobart(model, tokenizer)
            
            results.update({
                "model_info": model_info,
                "inference_results": inference_results
            })
            
            # 모델 정리
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"KoBART 벤치마킹 실패: {e}")
            results["error"] = str(e)
            
        return results
    
    def benchmark_kogpt2(self) -> Dict[str, Any]:
        """KoGPT-2 모델 벤치마킹"""
        logger.info("KoGPT-2 모델 벤치마킹 시작...")
        
        model_name = "skt/kogpt2-base-v2"
        results = {
            "model_name": model_name,
            "model_type": "causal_lm",
            "benchmark_time": datetime.now().isoformat()
        }
        
        try:
            # 모델 로딩 시간 측정
            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            model.to(self.device)
            loading_time = time.time() - start_time
            
            # 메모리 사용량 측정
            memory_usage = self._get_memory_usage()
            
            # 모델 정보 수집
            model_info = {
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "model_size_mb": self._get_model_size(model),
                "loading_time": loading_time,
                "memory_usage_mb": memory_usage
            }
            
            # 추론 성능 테스트
            inference_results = self._test_inference_kogpt2(model, tokenizer)
            
            results.update({
                "model_info": model_info,
                "inference_results": inference_results
            })
            
            # 모델 정리
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"KoGPT-2 벤치마킹 실패: {e}")
            results["error"] = str(e)
            
        return results
    
    def _test_inference_kobart(self, model, tokenizer) -> Dict[str, Any]:
        """KoBART 추론 성능 테스트"""
        results = {
            "total_inference_time": 0,
            "average_inference_time": 0,
            "responses": [],
            "quality_scores": []
        }
        
        total_time = 0
        responses = []
        
        for i, test_case in enumerate(self.test_data):
            try:
                start_time = time.time()
                
                # 입력 전처리
                input_text = f"질문: {test_case['question']} 맥락: {test_case['context']}"
                inputs = tokenizer.encode(input_text, return_tensors="pt").to(self.device)
                
                # 추론 실행
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=512,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # 응답 디코딩
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                inference_time = time.time() - start_time
                
                total_time += inference_time
                responses.append({
                    "question": test_case['question'],
                    "response": response,
                    "inference_time": inference_time,
                    "category": test_case['category']
                })
                
                logger.info(f"KoBART 테스트 {i+1}/{len(self.test_data)} 완료")
                
            except Exception as e:
                logger.error(f"KoBART 추론 테스트 {i+1} 실패: {e}")
                continue
        
        results.update({
            "total_inference_time": total_time,
            "average_inference_time": total_time / len(self.test_data) if self.test_data else 0,
            "responses": responses
        })
        
        return results
    
    def _test_inference_kogpt2(self, model, tokenizer) -> Dict[str, Any]:
        """KoGPT-2 추론 성능 테스트"""
        results = {
            "total_inference_time": 0,
            "average_inference_time": 0,
            "responses": [],
            "quality_scores": []
        }
        
        total_time = 0
        responses = []
        
        for i, test_case in enumerate(self.test_data):
            try:
                start_time = time.time()
                
                # 입력 전처리 (KoGPT-2는 프롬프트 형식)
                prompt = f"질문: {test_case['question']}\n답변:"
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                # 추론 실행
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 200,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # 응답 디코딩
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 프롬프트 부분 제거
                response = response.replace(prompt, "").strip()
                
                inference_time = time.time() - start_time
                total_time += inference_time
                
                responses.append({
                    "question": test_case['question'],
                    "response": response,
                    "inference_time": inference_time,
                    "category": test_case['category']
                })
                
                logger.info(f"KoGPT-2 테스트 {i+1}/{len(self.test_data)} 완료")
                
            except Exception as e:
                logger.error(f"KoGPT-2 추론 테스트 {i+1} 실패: {e}")
                continue
        
        results.update({
            "total_inference_time": total_time,
            "average_inference_time": total_time / len(self.test_data) if self.test_data else 0,
            "responses": responses
        })
        
        return results
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _get_model_size(self, model) -> float:
        """모델 크기 반환 (MB)"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / 1024 / 1024
    
    def run_benchmark(self) -> Dict[str, Any]:
        """전체 벤치마킹 실행"""
        logger.info("모델 벤치마킹 시작...")
        
        # 시스템 정보 수집
        system_info = {
            "device": self.device,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "python_version": sys.version,
            "torch_version": torch.__version__
        }
        
        # 각 모델 벤치마킹 실행
        kobart_results = self.benchmark_kobart()
        kogpt2_results = self.benchmark_kogpt2()
        
        # 결과 종합
        benchmark_results = {
            "system_info": system_info,
            "kobart": kobart_results,
            "kogpt2": kogpt2_results,
            "comparison": self._compare_models(kobart_results, kogpt2_results)
        }
        
        return benchmark_results
    
    def _compare_models(self, kobart_results: Dict, kogpt2_results: Dict) -> Dict[str, Any]:
        """모델 성능 비교"""
        comparison = {
            "model_size_comparison": {},
            "memory_usage_comparison": {},
            "inference_speed_comparison": {},
            "recommendation": ""
        }
        
        try:
            # 모델 크기 비교
            if "model_info" in kobart_results and "model_info" in kogpt2_results:
                kobart_size = kobart_results["model_info"]["model_size_mb"]
                kogpt2_size = kogpt2_results["model_info"]["model_size_mb"]
                
                comparison["model_size_comparison"] = {
                    "kobart_mb": kobart_size,
                    "kogpt2_mb": kogpt2_size,
                    "size_ratio": kobart_size / kogpt2_size if kogpt2_size > 0 else 0
                }
            
            # 메모리 사용량 비교
            if "model_info" in kobart_results and "model_info" in kogpt2_results:
                kobart_memory = kobart_results["model_info"]["memory_usage_mb"]
                kogpt2_memory = kogpt2_results["model_info"]["memory_usage_mb"]
                
                comparison["memory_usage_comparison"] = {
                    "kobart_mb": kobart_memory,
                    "kogpt2_mb": kogpt2_memory,
                    "memory_ratio": kobart_memory / kogpt2_memory if kogpt2_memory > 0 else 0
                }
            
            # 추론 속도 비교
            if "inference_results" in kobart_results and "inference_results" in kogpt2_results:
                kobart_time = kobart_results["inference_results"]["average_inference_time"]
                kogpt2_time = kogpt2_results["inference_results"]["average_inference_time"]
                
                comparison["inference_speed_comparison"] = {
                    "kobart_seconds": kobart_time,
                    "kogpt2_seconds": kogpt2_time,
                    "speed_ratio": kobart_time / kogpt2_time if kogpt2_time > 0 else 0
                }
            
            # 권장사항 생성
            comparison["recommendation"] = self._generate_recommendation(comparison)
            
        except Exception as e:
            logger.error(f"모델 비교 중 오류: {e}")
            comparison["error"] = str(e)
        
        return comparison
    
    def _generate_recommendation(self, comparison: Dict) -> str:
        """모델 선택 권장사항 생성"""
        try:
            size_ratio = comparison.get("model_size_comparison", {}).get("size_ratio", 1)
            memory_ratio = comparison.get("memory_usage_comparison", {}).get("memory_ratio", 1)
            speed_ratio = comparison.get("inference_speed_comparison", {}).get("speed_ratio", 1)
            
            # HuggingFace Spaces 제약사항 고려 (16GB 메모리 제한)
            if memory_ratio > 1.5:  # KoBART가 메모리를 더 많이 사용
                return "KoGPT-2 권장: 메모리 효율성이 우수하여 HuggingFace Spaces 환경에 적합"
            elif speed_ratio > 1.2:  # KoBART가 더 느림
                return "KoGPT-2 권장: 추론 속도가 빠름"
            elif size_ratio > 1.3:  # KoBART가 더 큼
                return "KoGPT-2 권장: 모델 크기가 작아 배포에 유리"
            else:
                return "KoBART 권장: seq2seq 특성상 법률 질문-답변에 적합"
                
        except Exception as e:
            return f"권장사항 생성 실패: {e}"
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """벤치마킹 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_benchmark_results_{timestamp}.json"
        
        filepath = os.path.join("benchmark_results", filename)
        os.makedirs("benchmark_results", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"벤치마킹 결과 저장: {filepath}")
        return filepath

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI 모델 성능 벤치마킹")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="실행 디바이스")
    parser.add_argument("--output", help="결과 저장 파일명")
    
    args = parser.parse_args()
    
    # 벤치마킹 실행
    benchmark = ModelBenchmark(device=args.device)
    results = benchmark.run_benchmark()
    
    # 결과 저장
    output_file = benchmark.save_results(results, args.output)
    
    # 결과 요약 출력
    print("\n" + "="*50)
    print("벤치마킹 결과 요약")
    print("="*50)
    
    if "kobart" in results and "model_info" in results["kobart"]:
        kobart_info = results["kobart"]["model_info"]
        print(f"KoBART - 모델 크기: {kobart_info.get('model_size_mb', 0):.1f}MB, "
              f"메모리 사용량: {kobart_info.get('memory_usage_mb', 0):.1f}MB")
    
    if "kogpt2" in results and "model_info" in results["kogpt2"]:
        kogpt2_info = results["kogpt2"]["model_info"]
        print(f"KoGPT-2 - 모델 크기: {kogpt2_info.get('model_size_mb', 0):.1f}MB, "
              f"메모리 사용량: {kogpt2_info.get('memory_usage_mb', 0):.1f}MB")
    
    if "comparison" in results and "recommendation" in results["comparison"]:
        print(f"\n권장사항: {results['comparison']['recommendation']}")
    
    print(f"\n상세 결과: {output_file}")

if __name__ == "__main__":
    main()
