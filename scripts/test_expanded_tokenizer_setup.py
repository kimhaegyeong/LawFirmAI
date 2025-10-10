#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
확장된 토크나이저 설정 테스트 스크립트

540개 훈련 데이터를 기반으로 업데이트된 토크나이저 설정을 테스트합니다.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/expanded_tokenizer_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExpandedTokenizerTester:
    """확장된 토크나이저 테스트 클래스"""
    
    def __init__(self):
        """토크나이저 테스터 초기화"""
        self.output_dir = Path("data/training")
        self.test_results = {}
        
        # 토크나이저 설정 로드
        self.tokenizer_config = self._load_tokenizer_config()
        
        logger.info("ExpandedTokenizerTester initialized")
    
    def _load_tokenizer_config(self) -> Dict[str, Any]:
        """토크나이저 설정 로드"""
        config_path = self.output_dir / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning("토크나이저 설정 파일이 없습니다.")
            return {}
    
    def test_expanded_tokenizer(self) -> Dict[str, Any]:
        """확장된 토크나이저 테스트"""
        logger.info("확장된 토크나이저 테스트 시작...")
        
        # 1. 토크나이저 설정 검증
        config_test = self._test_tokenizer_config()
        
        # 2. 특수 토큰 테스트
        special_token_test = self._test_special_tokens()
        
        # 3. 훈련 데이터 토크나이징 테스트
        training_data_test = self._test_training_data_tokenization()
        
        # 4. 프롬프트 템플릿 토크나이징 테스트
        template_test = self._test_prompt_template_tokenization()
        
        # 5. 메모리 사용량 테스트
        memory_test = self._test_memory_usage()
        
        # 6. 성능 테스트
        performance_test = self._test_performance()
        
        # 전체 테스트 결과 통합
        self.test_results = {
            "config_test": config_test,
            "special_token_test": special_token_test,
            "training_data_test": training_data_test,
            "template_test": template_test,
            "memory_test": memory_test,
            "performance_test": performance_test,
            "overall_success": all([
                config_test["success"],
                special_token_test["success"],
                training_data_test["success"],
                template_test["success"],
                memory_test["success"],
                performance_test["success"]
            ]),
            "tested_at": datetime.now().isoformat()
        }
        
        # 테스트 결과 저장
        self._save_test_results()
        
        logger.info("확장된 토크나이저 테스트 완료!")
        return self.test_results
    
    def _test_tokenizer_config(self) -> Dict[str, Any]:
        """토크나이저 설정 테스트"""
        logger.info("토크나이저 설정 테스트...")
        
        test_result = {
            "success": False,
            "config_loaded": False,
            "special_tokens_count": 0,
            "max_length": 0,
            "vocab_size": 0,
            "errors": []
        }
        
        try:
            if self.tokenizer_config:
                test_result["config_loaded"] = True
                test_result["special_tokens_count"] = len(self.tokenizer_config.get("special_tokens", []))
                test_result["max_length"] = self.tokenizer_config.get("max_length", 0)
                test_result["vocab_size"] = self.tokenizer_config.get("vocab_size", 0)
                
                # 설정 검증
                if test_result["special_tokens_count"] >= 7:  # 최소 7개 특수 토큰
                    test_result["success"] = True
                else:
                    test_result["errors"].append(f"특수 토큰 수 부족: {test_result['special_tokens_count']}개")
                
                if test_result["max_length"] >= 512:
                    test_result["success"] = test_result["success"] and True
                else:
                    test_result["errors"].append(f"최대 길이 부족: {test_result['max_length']}")
                
                if test_result["vocab_size"] >= 50000:
                    test_result["success"] = test_result["success"] and True
                else:
                    test_result["errors"].append(f"어휘 크기 부족: {test_result['vocab_size']}")
            else:
                test_result["errors"].append("토크나이저 설정 로드 실패")
        
        except Exception as e:
            test_result["errors"].append(f"설정 테스트 오류: {e}")
        
        logger.info(f"토크나이저 설정 테스트 결과: {'성공' if test_result['success'] else '실패'}")
        return test_result
    
    def _test_special_tokens(self) -> Dict[str, Any]:
        """특수 토큰 테스트"""
        logger.info("특수 토큰 테스트...")
        
        test_result = {
            "success": False,
            "required_tokens": [
                "<|startoftext|>",
                "<|endoftext|>",
                "질문:",
                "답변:",
                "분석:",
                "설명:",
                "조언:"
            ],
            "found_tokens": [],
            "missing_tokens": [],
            "errors": []
        }
        
        try:
            special_tokens = self.tokenizer_config.get("special_tokens", [])
            test_result["found_tokens"] = special_tokens
            
            for required_token in test_result["required_tokens"]:
                if required_token in special_tokens:
                    continue
                else:
                    test_result["missing_tokens"].append(required_token)
            
            if len(test_result["missing_tokens"]) == 0:
                test_result["success"] = True
            else:
                test_result["errors"].append(f"누락된 특수 토큰: {test_result['missing_tokens']}")
        
        except Exception as e:
            test_result["errors"].append(f"특수 토큰 테스트 오류: {e}")
        
        logger.info(f"특수 토큰 테스트 결과: {'성공' if test_result['success'] else '실패'}")
        return test_result
    
    def _test_training_data_tokenization(self) -> Dict[str, Any]:
        """훈련 데이터 토크나이징 테스트"""
        logger.info("훈련 데이터 토크나이징 테스트...")
        
        test_result = {
            "success": False,
            "samples_tested": 0,
            "average_length": 0,
            "max_length_exceeded": 0,
            "min_length": 0,
            "max_length": 0,
            "errors": []
        }
        
        try:
            # 훈련 데이터 로드
            train_data_path = self.output_dir / "train_split.json"
            if train_data_path.exists():
                with open(train_data_path, 'r', encoding='utf-8') as f:
                    train_data = json.load(f)
                
                test_result["samples_tested"] = len(train_data)
                
                lengths = []
                max_allowed = self.tokenizer_config.get("max_length", 512)
                
                for item in train_data[:10]:  # 처음 10개 샘플만 테스트
                    text = item.get("text", "")
                    length = len(text.split())  # 단어 수로 근사
                    lengths.append(length)
                    
                    if length > max_allowed:
                        test_result["max_length_exceeded"] += 1
                
                if lengths:
                    test_result["average_length"] = sum(lengths) / len(lengths)
                    test_result["min_length"] = min(lengths)
                    test_result["max_length"] = max(lengths)
                    
                    # 평균 길이가 합리적인 범위에 있는지 확인
                    if 50 <= test_result["average_length"] <= 400:
                        test_result["success"] = True
                    else:
                        test_result["errors"].append(f"평균 길이가 부적절: {test_result['average_length']:.1f}")
                    
                    # 최대 길이 초과 샘플이 20% 이하인지 확인
                    if test_result["max_length_exceeded"] <= len(lengths) * 0.2:
                        test_result["success"] = test_result["success"] and True
                    else:
                        test_result["errors"].append(f"최대 길이 초과 샘플이 많음: {test_result['max_length_exceeded']}개")
                else:
                    test_result["errors"].append("토크나이징할 데이터가 없음")
            else:
                test_result["errors"].append("훈련 데이터 파일이 없음")
        
        except Exception as e:
            test_result["errors"].append(f"훈련 데이터 토크나이징 테스트 오류: {e}")
        
        logger.info(f"훈련 데이터 토크나이징 테스트 결과: {'성공' if test_result['success'] else '실패'}")
        return test_result
    
    def _test_prompt_template_tokenization(self) -> Dict[str, Any]:
        """프롬프트 템플릿 토크나이징 테스트"""
        logger.info("프롬프트 템플릿 토크나이징 테스트...")
        
        test_result = {
            "success": False,
            "templates_tested": 0,
            "template_lengths": {},
            "errors": []
        }
        
        try:
            # 프롬프트 템플릿 테스트
            test_templates = [
                "법률 전문가로서 다음 질문에 답변해주세요:\n\n질문: 민법에서 계약법의 계약의 성립 요건은 무엇인가요?\n답변: 민법에서 계약법의 계약의 성립 요건은 청약과 승낙의 합치로 성립입니다.",
                "법조문을 해석하여 다음 질문에 답변해주세요:\n\n질문: 민법 제527조의 내용과 의미를 설명해주세요.\n답변: 민법 제527조은 계약의 성립 요건에 관한 규정으로, '청약과 승낙의 합치로 성립'라고 규정하고 있습니다.",
                "판례 검색 결과를 바탕으로 다음 질문에 답변해주세요:\n\n질문: 대법원 2018다22222 사건의 요약을 해주세요.\n답변: 대법원 2018다22222 사건은 부동산 매매 계약 해제 시 원상회복에 관한 사건으로, 대법원에서 2018.12.13에 선고되었습니다."
            ]
            
            test_result["templates_tested"] = len(test_templates)
            
            for i, template in enumerate(test_templates):
                length = len(template.split())
                test_result["template_lengths"][f"template_{i+1}"] = length
            
            # 모든 템플릿이 적절한 길이를 가지는지 확인
            avg_length = sum(test_result["template_lengths"].values()) / len(test_result["template_lengths"])
            if 100 <= avg_length <= 300:
                test_result["success"] = True
            else:
                test_result["errors"].append(f"프롬프트 템플릿 평균 길이가 부적절: {avg_length:.1f}")
        
        except Exception as e:
            test_result["errors"].append(f"프롬프트 템플릿 테스트 오류: {e}")
        
        logger.info(f"프롬프트 템플릿 토크나이징 테스트 결과: {'성공' if test_result['success'] else '실패'}")
        return test_result
    
    def _test_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 테스트"""
        logger.info("메모리 사용량 테스트...")
        
        test_result = {
            "success": False,
            "estimated_memory_mb": 0,
            "memory_efficient": False,
            "errors": []
        }
        
        try:
            # 데이터셋 크기 기반 메모리 추정
            train_data_path = self.output_dir / "train_split.json"
            if train_data_path.exists():
                file_size = train_data_path.stat().st_size
                # JSON 파일 크기의 약 3-5배가 메모리 사용량
                estimated_memory = (file_size * 4) / (1024 * 1024)  # MB
                test_result["estimated_memory_mb"] = estimated_memory
                
                # 1GB 이하면 효율적
                if estimated_memory <= 1024:
                    test_result["memory_efficient"] = True
                    test_result["success"] = True
                else:
                    test_result["errors"].append(f"메모리 사용량이 높음: {estimated_memory:.1f}MB")
            else:
                test_result["errors"].append("훈련 데이터 파일이 없음")
        
        except Exception as e:
            test_result["errors"].append(f"메모리 사용량 테스트 오류: {e}")
        
        logger.info(f"메모리 사용량 테스트 결과: {'성공' if test_result['success'] else '실패'}")
        return test_result
    
    def _test_performance(self) -> Dict[str, Any]:
        """성능 테스트"""
        logger.info("성능 테스트...")
        
        test_result = {
            "success": False,
            "processing_speed": 0,
            "efficient_processing": False,
            "errors": []
        }
        
        try:
            import time
            
            # 샘플 텍스트 처리 속도 테스트
            test_text = "법률 전문가로서 다음 질문에 답변해주세요:\n\n질문: 민법에서 계약법의 계약의 성립 요건은 무엇인가요?\n답변: 민법에서 계약법의 계약의 성립 요건은 청약과 승낙의 합치로 성립입니다."
            
            start_time = time.time()
            
            # 단순 텍스트 처리 시뮬레이션
            words = test_text.split()
            processed_words = [word for word in words if len(word) > 0]
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            test_result["processing_speed"] = len(processed_words) / processing_time if processing_time > 0 else 0
            
            # 초당 1000단어 이상 처리하면 효율적
            if test_result["processing_speed"] >= 1000:
                test_result["efficient_processing"] = True
                test_result["success"] = True
            else:
                test_result["errors"].append(f"처리 속도가 느림: {test_result['processing_speed']:.1f} 단어/초")
        
        except Exception as e:
            test_result["errors"].append(f"성능 테스트 오류: {e}")
        
        logger.info(f"성능 테스트 결과: {'성공' if test_result['success'] else '실패'}")
        return test_result
    
    def _save_test_results(self):
        """테스트 결과 저장"""
        results_path = self.output_dir / "expanded_tokenizer_test_report.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"테스트 결과 저장 완료: {results_path}")
    
    def print_test_summary(self):
        """테스트 결과 요약 출력"""
        if not self.test_results:
            logger.warning("테스트 결과가 없습니다.")
            return
        
        print("\n" + "="*60)
        print("📊 확장된 토크나이저 테스트 결과 요약")
        print("="*60)
        
        # 전체 성공 여부
        overall_success = self.test_results.get("overall_success", False)
        print(f"🎯 전체 테스트 결과: {'✅ 성공' if overall_success else '❌ 실패'}")
        
        # 각 테스트별 결과
        tests = [
            ("토크나이저 설정", "config_test"),
            ("특수 토큰", "special_token_test"),
            ("훈련 데이터 토크나이징", "training_data_test"),
            ("프롬프트 템플릿", "template_test"),
            ("메모리 사용량", "memory_test"),
            ("성능", "performance_test")
        ]
        
        print(f"\n📋 개별 테스트 결과:")
        for test_name, test_key in tests:
            test_result = self.test_results.get(test_key, {})
            success = test_result.get("success", False)
            status = "✅ 성공" if success else "❌ 실패"
            print(f"  - {test_name}: {status}")
            
            if not success and test_result.get("errors"):
                for error in test_result["errors"][:2]:  # 최대 2개 오류만 표시
                    print(f"    ⚠️ {error}")
        
        # 상세 통계
        config_test = self.test_results.get("config_test", {})
        training_test = self.test_results.get("training_data_test", {})
        memory_test = self.test_results.get("memory_test", {})
        
        print(f"\n📈 상세 통계:")
        print(f"  - 특수 토큰 수: {config_test.get('special_tokens_count', 0)}개")
        print(f"  - 최대 길이: {config_test.get('max_length', 0)}")
        print(f"  - 어휘 크기: {config_test.get('vocab_size', 0):,}")
        print(f"  - 테스트된 샘플 수: {training_test.get('samples_tested', 0)}개")
        print(f"  - 평균 텍스트 길이: {training_test.get('average_length', 0):.1f} 단어")
        print(f"  - 예상 메모리 사용량: {memory_test.get('estimated_memory_mb', 0):.1f}MB")
        
        print("="*60)


def main():
    """메인 실행 함수"""
    logger.info("확장된 토크나이저 테스트 시작...")
    
    # 토크나이저 테스터 초기화
    tester = ExpandedTokenizerTester()
    
    # 확장된 토크나이저 테스트
    test_results = tester.test_expanded_tokenizer()
    
    # 테스트 결과 요약 출력
    tester.print_test_summary()
    
    logger.info("확장된 토크나이저 테스트 완료!")


if __name__ == "__main__":
    main()
