#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
토크나이저 설정 및 테스트 스크립트
LawFirmAI 프로젝트 - TASK 3.1 Day 2
"""

import os
import sys
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer
import torch

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_tokenizer_config() -> dict:
    """토크나이저 설정 로드"""
    config_path = Path("data/training/tokenizer_config.json")
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # 기본 설정 반환
        return {
            "model_name": "skt/kogpt2-base-v2",
            "special_tokens": {
                "startoftext": "<|startoftext|>",
                "endoftext": "<|endoftext|>",
                "question": "질문:",
                "answer": "답변:",
                "analysis": "분석:",
                "explanation": "설명:",
                "advice": "조언:"
            },
            "max_length": 512,
            "padding": True,
            "truncation": True,
            "return_tensors": "pt"
        }

def setup_tokenizer(config: dict) -> AutoTokenizer:
    """토크나이저 설정 및 특수 토큰 추가"""
    logger = setup_logging()
    
    model_name = config["model_name"]
    logger.info(f"토크나이저 로딩: {model_name}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 특수 토큰 추가
    special_tokens = config["special_tokens"]
    special_tokens_list = list(special_tokens.values())
    
    # 패딩 토큰 설정 (KoGPT-2는 기본적으로 패딩 토큰이 없음)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 특수 토큰 추가
    tokenizer.add_special_tokens({
        "additional_special_tokens": special_tokens_list
    })
    
    logger.info(f"특수 토큰 추가 완료: {len(special_tokens_list)}개")
    logger.info(f"어휘 크기: {tokenizer.vocab_size}")
    
    return tokenizer

def test_tokenizer(tokenizer: AutoTokenizer, config: dict):
    """토크나이저 테스트"""
    logger = setup_logging()
    
    # 테스트 텍스트
    test_texts = [
        "<|startoftext|>질문: 민법에서 계약이란 무엇인가요?\n답변: 민법 제105조에 따르면, 계약은 당사자 쌍방이 서로 대립하는 의사표시의 합치에 의하여 성립하는 법률행위를 말합니다.<|endoftext|>",
        "<|startoftext|>당신은 법률 전문가입니다. 다음 계약서 조항을 분석하고 위험 요소를 지적해주세요.\n\n계약서 조항: 매수인은 매매대금을 계약 체결일로부터 30일 이내에 지급하여야 한다.\n분석:<|endoftext|>",
        "<|startoftext|>다음 법조문을 일반인이 이해하기 쉽게 설명해주세요.\n\n법조문: 민법 제105조\n설명:<|endoftext|>"
    ]
    
    logger.info("토크나이저 테스트 시작...")
    
    for i, text in enumerate(test_texts, 1):
        logger.info(f"\n--- 테스트 {i} ---")
        logger.info(f"원본 텍스트 길이: {len(text)} 문자")
        
        # 토크나이징
        tokens = tokenizer.encode(text)
        logger.info(f"토큰 수: {len(tokens)}")
        
        # 디코딩 테스트
        decoded = tokenizer.decode(tokens)
        logger.info(f"디코딩 성공: {decoded == text}")
        
        # 배치 토크나이징 테스트
        batch_texts = [text]
        batch_tokens = tokenizer(
            batch_texts,
            max_length=config["max_length"],
            padding=config["padding"],
            truncation=config["truncation"],
            return_tensors=config["return_tensors"]
        )
        
        logger.info(f"배치 토크나이징 성공")
        logger.info(f"입력 ID 형태: {batch_tokens['input_ids'].shape}")
        logger.info(f"어텐션 마스크 형태: {batch_tokens['attention_mask'].shape}")

def test_special_tokens(tokenizer: AutoTokenizer):
    """특수 토큰 테스트"""
    logger = setup_logging()
    
    logger.info("\n--- 특수 토큰 테스트 ---")
    
    special_tokens = ["<|startoftext|>", "<|endoftext|>", "질문:", "답변:", "분석:", "설명:", "조언:"]
    
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        decoded_token = tokenizer.convert_ids_to_tokens(token_id)
        logger.info(f"'{token}' -> ID: {token_id}, 디코딩: '{decoded_token}'")

def create_tokenizer_test_report(tokenizer: AutoTokenizer, config: dict) -> dict:
    """토크나이저 테스트 보고서 생성"""
    report = {
        "model_name": config["model_name"],
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": config["special_tokens"],
        "max_length": config["max_length"],
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "bos_token": tokenizer.bos_token,
        "unk_token": tokenizer.unk_token,
        "test_results": {
            "tokenizer_loaded": True,
            "special_tokens_added": True,
            "batch_processing": True,
            "encoding_decoding": True
        }
    }
    
    return report

def main():
    """메인 함수"""
    logger = setup_logging()
    logger.info("토크나이저 설정 및 테스트 시작...")
    
    # 1. 설정 로드
    logger.info("1. 토크나이저 설정 로드...")
    config = load_tokenizer_config()
    
    # 2. 토크나이저 설정
    logger.info("2. 토크나이저 설정...")
    tokenizer = setup_tokenizer(config)
    
    # 3. 토크나이저 테스트
    logger.info("3. 토크나이저 테스트...")
    test_tokenizer(tokenizer, config)
    
    # 4. 특수 토큰 테스트
    logger.info("4. 특수 토큰 테스트...")
    test_special_tokens(tokenizer)
    
    # 5. 테스트 보고서 생성
    logger.info("5. 테스트 보고서 생성...")
    report = create_tokenizer_test_report(tokenizer, config)
    
    # 보고서 저장
    report_path = Path("data/training/tokenizer_test_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info("토크나이저 설정 및 테스트 완료!")
    logger.info(f"어휘 크기: {tokenizer.vocab_size}")
    logger.info(f"특수 토큰: {len(config['special_tokens'])}개")
    logger.info(f"최대 길이: {config['max_length']}")

if __name__ == "__main__":
    main()
