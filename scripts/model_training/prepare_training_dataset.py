#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 2: 데이터셋 준비 및 전처리 스크립트
LawFirmAI 프로젝트 - TASK 3.1 Day 2
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import random
import re

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

def create_sample_qa_dataset() -> List[Dict[str, Any]]:
    """샘플 Q&A 데이터셋 생성"""
    logger = setup_logging()
    
    sample_data = [
        {
            "id": "law_001",
            "question": "민법에서 계약이란 무엇인가요?",
            "answer": "민법 제105조에 따르면, 계약은 당사자 쌍방이 서로 대립하는 의사표시의 합치에 의하여 성립하는 법률행위를 말합니다.",
            "type": "law_definition",
            "source": "민법",
            "article": "제105조",
            "quality_score": 0.95,
            "confidence": 0.9
        },
        {
            "id": "law_002", 
            "question": "계약서 작성 시 주의해야 할 사항은 무엇인가요?",
            "answer": "계약서 작성 시에는 당사자의 표시, 목적물의 표시, 대금의 표시 등 계약의 필수 요소를 명확히 기재해야 합니다. 또한 계약의 성질에 따라 특별한 조항을 포함해야 할 수도 있습니다.",
            "type": "legal_advice",
            "source": "민법",
            "article": "제105조",
            "quality_score": 0.88,
            "confidence": 0.85
        },
        {
            "id": "precedent_001",
            "question": "부동산 매매계약에서 하자담보책임에 관한 판례는 어떻게 되나요?",
            "answer": "대법원 2018다12345 판결에 따르면, 매도인은 매매목적물의 하자에 대하여 담보책임을 지며, 하자로 인한 손해배상책임을 부담합니다. 다만 매수인이 하자를 알았거나 중대한 과실로 알지 못한 경우에는 예외입니다.",
            "type": "precedent_search",
            "source": "대법원 판례",
            "case_number": "2018다12345",
            "quality_score": 0.92,
            "confidence": 0.88
        },
        {
            "id": "law_003",
            "question": "채무불이행의 효과는 무엇인가요?",
            "answer": "채무불이행의 효과로는 강제이행, 손해배상, 계약해제 등이 있습니다. 민법 제390조에 따르면 채무자가 채무의 내용에 좇아 이행하지 아니한 때에는 채권자는 손해배상을 청구할 수 있습니다.",
            "type": "law_explanation",
            "source": "민법",
            "article": "제390조",
            "quality_score": 0.90,
            "confidence": 0.87
        },
        {
            "id": "precedent_002",
            "question": "근로계약서에 명시되지 않은 수당 지급 의무가 있나요?",
            "answer": "대법원 2019다67890 판결에 따르면, 근로계약서에 명시되지 않았더라도 상습적으로 지급되던 수당은 임금의 성질을 가지며, 사용자는 이를 지급할 의무가 있습니다. 다만 일회성 보상이나 특별한 경우의 수당은 예외입니다.",
            "type": "precedent_search",
            "source": "대법원 판례",
            "case_number": "2019다67890",
            "quality_score": 0.89,
            "confidence": 0.86
        },
        {
            "id": "law_004",
            "question": "법인의 권리능력과 행위능력에 대해 설명해주세요.",
            "answer": "법인은 법률에 의하여 권리능력을 가지며, 법인의 목적범위 내에서 권리와 의무의 주체가 됩니다. 법인의 행위능력은 이사나 대표자가 법인을 대표하여 행사합니다. 민법 제34조에 규정되어 있습니다.",
            "type": "law_explanation",
            "source": "민법",
            "article": "제34조",
            "quality_score": 0.93,
            "confidence": 0.89
        },
        {
            "id": "precedent_003",
            "question": "불법행위에서 과실상계의 요건은 무엇인가요?",
            "answer": "대법원 2020다11111 판결에 따르면, 불법행위에서 과실상계는 피해자에게 과실이 있고, 그 과실이 손해발생에 기여한 경우에 적용됩니다. 과실상계의 비율은 당사자의 과실 정도와 손해에 대한 기여도를 고려하여 결정합니다.",
            "type": "precedent_search",
            "source": "대법원 판례",
            "case_number": "2020다11111",
            "quality_score": 0.91,
            "confidence": 0.88
        },
        {
            "id": "law_005",
            "question": "소멸시효의 기간은 어떻게 되나요?",
            "answer": "민법 제162조에 따르면, 채권은 10년간 행사하지 아니하면 소멸시효가 완성됩니다. 다만 상사채권은 5년, 근로자의 임금채권은 3년의 단기소멸시효가 적용됩니다.",
            "type": "law_explanation",
            "source": "민법",
            "article": "제162조",
            "quality_score": 0.94,
            "confidence": 0.90
        },
        {
            "id": "precedent_004",
            "question": "건물명도소송에서 점유권의 성립요건은 무엇인가요?",
            "answer": "대법원 2021다22222 판결에 따르면, 점유권의 성립요건으로는 물건에 대한 사실상의 지배와 점유의 의사가 필요합니다. 건물의 경우 실제 거주하거나 사용하고 있다는 사실이 중요합니다.",
            "type": "precedent_search",
            "source": "대법원 판례",
            "case_number": "2021다22222",
            "quality_score": 0.87,
            "confidence": 0.84
        },
        {
            "id": "law_006",
            "question": "유언의 효력과 요건에 대해 설명해주세요.",
            "answer": "유언은 유언자의 사망 시에 효력이 발생하며, 유언자의 진정한 의사표시여야 합니다. 민법 제1060조에 따르면 유언은 법정된 방식에 따라 하여야 하며, 자필증서, 녹음, 공정증서 등의 방식이 있습니다.",
            "type": "law_explanation",
            "source": "민법",
            "article": "제1060조",
            "quality_score": 0.92,
            "confidence": 0.88
        }
    ]
    
    logger.info(f"샘플 Q&A 데이터셋 생성 완료: {len(sample_data)}개")
    return sample_data

def convert_to_kogpt2_format(qa_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Q&A 데이터셋을 KoGPT-2 입력 형식으로 변환"""
    logger = setup_logging()
    
    converted_data = []
    
    for item in qa_dataset:
        # KoGPT-2용 프롬프트 템플릿 적용
        prompt = f"<|startoftext|>질문: {item['question']}\n답변: {item['answer']}<|endoftext|>"
        
        converted_item = {
            "id": item["id"],
            "text": prompt,
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "source": item["source"],
            "quality_score": item["quality_score"],
            "confidence": item["confidence"],
            "metadata": {
                "article": item.get("article", ""),
                "case_number": item.get("case_number", ""),
                "converted_at": datetime.now().isoformat()
            }
        }
        
        converted_data.append(converted_item)
    
    logger.info(f"KoGPT-2 형식 변환 완료: {len(converted_data)}개")
    return converted_data

def split_dataset(data: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
    """데이터셋을 훈련/검증/테스트로 분할"""
    logger = setup_logging()
    
    # 데이터 셔플
    random.shuffle(data)
    
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    logger.info(f"데이터셋 분할 완료:")
    logger.info(f"  훈련 데이터: {len(train_data)}개 ({len(train_data)/total_size:.1%})")
    logger.info(f"  검증 데이터: {len(val_data)}개 ({len(val_data)/total_size:.1%})")
    logger.info(f"  테스트 데이터: {len(test_data)}개 ({len(test_data)/total_size:.1%})")
    
    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }

def create_prompt_templates() -> Dict[str, str]:
    """법률 특화 프롬프트 템플릿 생성"""
    templates = {
        "contract_analysis": """<|startoftext|>당신은 법률 전문가입니다. 다음 계약서 조항을 분석하고 위험 요소를 지적해주세요.

계약서 조항: {clause}
분석:<|endoftext|>""",
        
        "precedent_search": """<|startoftext|>다음 사건과 유사한 판례를 찾아주세요.

사건 개요: {case_summary}
유사 판례:<|endoftext|>""",
        
        "law_explanation": """<|startoftext|>다음 법조문을 일반인이 이해하기 쉽게 설명해주세요.

법조문: {law_article}
설명:<|endoftext|>""",
        
        "legal_advice": """<|startoftext|>다음 상황에서 법적 조언을 해주세요.

상황: {situation}
조언:<|endoftext|>""",
        
        "qa_format": """<|startoftext|>질문: {question}
답변: {answer}<|endoftext|>"""
    }
    
    return templates

def setup_tokenizer_config() -> Dict[str, Any]:
    """토크나이저 설정"""
    config = {
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
    
    return config

def save_datasets(split_data: Dict[str, List[Dict[str, Any]]], output_dir: str = "data/training"):
    """분할된 데이터셋 저장"""
    logger = setup_logging()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, data in split_data.items():
        filename = f"{split_name}_split.json"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{split_name} 데이터셋 저장 완료: {filepath} ({len(data)}개)")

def generate_statistics(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """데이터셋 통계 생성"""
    stats = {
        "total_samples": len(dataset),
        "type_distribution": {},
        "source_distribution": {},
        "quality_stats": {
            "average_score": 0.0,
            "min_score": 1.0,
            "max_score": 0.0
        },
        "confidence_stats": {
            "average_confidence": 0.0,
            "min_confidence": 1.0,
            "max_confidence": 0.0
        }
    }
    
    quality_scores = []
    confidence_scores = []
    
    for item in dataset:
        # 타입별 분포
        item_type = item["type"]
        stats["type_distribution"][item_type] = stats["type_distribution"].get(item_type, 0) + 1
        
        # 소스별 분포
        source = item["source"]
        stats["source_distribution"][source] = stats["source_distribution"].get(source, 0) + 1
        
        # 품질 점수
        quality_score = item["quality_score"]
        quality_scores.append(quality_score)
        stats["quality_stats"]["min_score"] = min(stats["quality_stats"]["min_score"], quality_score)
        stats["quality_stats"]["max_score"] = max(stats["quality_stats"]["max_score"], quality_score)
        
        # 신뢰도 점수
        confidence = item["confidence"]
        confidence_scores.append(confidence)
        stats["confidence_stats"]["min_confidence"] = min(stats["confidence_stats"]["min_confidence"], confidence)
        stats["confidence_stats"]["max_confidence"] = max(stats["confidence_stats"]["max_confidence"], confidence)
    
    # 평균 계산
    stats["quality_stats"]["average_score"] = sum(quality_scores) / len(quality_scores)
    stats["confidence_stats"]["average_confidence"] = sum(confidence_scores) / len(confidence_scores)
    
    return stats

def main():
    """메인 함수"""
    logger = setup_logging()
    logger.info("Day 2: 데이터셋 준비 및 전처리 시작...")
    
    # 1. 샘플 Q&A 데이터셋 생성
    logger.info("1. 샘플 Q&A 데이터셋 생성...")
    qa_dataset = create_sample_qa_dataset()
    
    # 2. KoGPT-2 형식으로 변환
    logger.info("2. KoGPT-2 형식으로 변환...")
    converted_data = convert_to_kogpt2_format(qa_dataset)
    
    # 3. 프롬프트 템플릿 생성
    logger.info("3. 프롬프트 템플릿 생성...")
    prompt_templates = create_prompt_templates()
    
    # 4. 토크나이저 설정
    logger.info("4. 토크나이저 설정...")
    tokenizer_config = setup_tokenizer_config()
    
    # 5. 데이터셋 분할
    logger.info("5. 데이터셋 분할 (8:1:1)...")
    split_data = split_dataset(converted_data)
    
    # 6. 데이터셋 저장
    logger.info("6. 데이터셋 저장...")
    save_datasets(split_data)
    
    # 7. 통계 생성
    logger.info("7. 통계 생성...")
    stats = generate_statistics(converted_data)
    
    # 통계 저장
    stats_path = Path("data/training/dataset_statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 프롬프트 템플릿 저장
    templates_path = Path("data/training/prompt_templates.json")
    with open(templates_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_templates, f, ensure_ascii=False, indent=2)
    
    # 토크나이저 설정 저장
    config_path = Path("data/training/tokenizer_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    
    logger.info("Day 2: 데이터셋 준비 및 전처리 완료!")
    logger.info(f"총 데이터: {len(converted_data)}개")
    logger.info(f"훈련 데이터: {len(split_data['train'])}개")
    logger.info(f"검증 데이터: {len(split_data['validation'])}개")
    logger.info(f"테스트 데이터: {len(split_data['test'])}개")

if __name__ == "__main__":
    main()
