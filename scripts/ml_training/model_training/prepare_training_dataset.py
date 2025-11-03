#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 2: ?°ì´?°ì…‹ ì¤€ë¹?ë°??„ì²˜ë¦??¤í¬ë¦½íŠ¸
LawFirmAI ?„ë¡œ?íŠ¸ - TASK 3.1 Day 2
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

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_logging():
    """ë¡œê¹… ?¤ì •"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_sample_qa_dataset() -> List[Dict[str, Any]]:
    """?˜í”Œ Q&A ?°ì´?°ì…‹ ?ì„±"""
    logger = setup_logging()
    
    sample_data = [
        {
            "id": "law_001",
            "question": "ë¯¼ë²•?ì„œ ê³„ì•½?´ë? ë¬´ì—‡?¸ê???",
            "answer": "ë¯¼ë²• ??05ì¡°ì— ?°ë¥´ë©? ê³„ì•½?€ ?¹ì‚¬???ë°©???œë¡œ ?€ë¦½í•˜???˜ì‚¬?œì‹œ???©ì¹˜???˜í•˜???±ë¦½?˜ëŠ” ë²•ë¥ ?‰ìœ„ë¥?ë§í•©?ˆë‹¤.",
            "type": "law_definition",
            "source": "ë¯¼ë²•",
            "article": "??05ì¡?,
            "quality_score": 0.95,
            "confidence": 0.9
        },
        {
            "id": "law_002", 
            "question": "ê³„ì•½???‘ì„± ??ì£¼ì˜?´ì•¼ ???¬í•­?€ ë¬´ì—‡?¸ê???",
            "answer": "ê³„ì•½???‘ì„± ?œì—???¹ì‚¬?ì˜ ?œì‹œ, ëª©ì ë¬¼ì˜ ?œì‹œ, ?€ê¸ˆì˜ ?œì‹œ ??ê³„ì•½???„ìˆ˜ ?”ì†Œë¥?ëª…í™•??ê¸°ì¬?´ì•¼ ?©ë‹ˆ?? ?í•œ ê³„ì•½???±ì§ˆ???°ë¼ ?¹ë³„??ì¡°í•­???¬í•¨?´ì•¼ ???˜ë„ ?ˆìŠµ?ˆë‹¤.",
            "type": "legal_advice",
            "source": "ë¯¼ë²•",
            "article": "??05ì¡?,
            "quality_score": 0.88,
            "confidence": 0.85
        },
        {
            "id": "precedent_001",
            "question": "ë¶€?™ì‚° ë§¤ë§¤ê³„ì•½?ì„œ ?˜ì?´ë³´ì±…ì„??ê´€???ë????´ë–»ê²??˜ë‚˜??",
            "answer": "?€ë²•ì› 2018??2345 ?ê²°???°ë¥´ë©? ë§¤ë„?¸ì? ë§¤ë§¤ëª©ì ë¬¼ì˜ ?˜ì???€?˜ì—¬ ?´ë³´ì±…ì„??ì§€ë©? ?˜ìë¡??¸í•œ ?í•´ë°°ìƒì±…ì„??ë¶€?´í•©?ˆë‹¤. ?¤ë§Œ ë§¤ìˆ˜?¸ì´ ?˜ìë¥??Œì•˜ê±°ë‚˜ ì¤‘ë???ê³¼ì‹¤ë¡??Œì? ëª»í•œ ê²½ìš°?ëŠ” ?ˆì™¸?…ë‹ˆ??",
            "type": "precedent_search",
            "source": "?€ë²•ì› ?ë?",
            "case_number": "2018??2345",
            "quality_score": 0.92,
            "confidence": 0.88
        },
        {
            "id": "law_003",
            "question": "ì±„ë¬´ë¶ˆì´?‰ì˜ ?¨ê³¼??ë¬´ì—‡?¸ê???",
            "answer": "ì±„ë¬´ë¶ˆì´?‰ì˜ ?¨ê³¼ë¡œëŠ” ê°•ì œ?´í–‰, ?í•´ë°°ìƒ, ê³„ì•½?´ì œ ?±ì´ ?ˆìŠµ?ˆë‹¤. ë¯¼ë²• ??90ì¡°ì— ?°ë¥´ë©?ì±„ë¬´?ê? ì±„ë¬´???´ìš©??ì¢‡ì•„ ?´í–‰?˜ì? ?„ë‹ˆ???Œì—??ì±„ê¶Œ?ëŠ” ?í•´ë°°ìƒ??ì²?µ¬?????ˆìŠµ?ˆë‹¤.",
            "type": "law_explanation",
            "source": "ë¯¼ë²•",
            "article": "??90ì¡?,
            "quality_score": 0.90,
            "confidence": 0.87
        },
        {
            "id": "precedent_002",
            "question": "ê·¼ë¡œê³„ì•½?œì— ëª…ì‹œ?˜ì? ?Šì? ?˜ë‹¹ ì§€ê¸??˜ë¬´ê°€ ?ˆë‚˜??",
            "answer": "?€ë²•ì› 2019??7890 ?ê²°???°ë¥´ë©? ê·¼ë¡œê³„ì•½?œì— ëª…ì‹œ?˜ì? ?Šì•˜?”ë¼???ìŠµ?ìœ¼ë¡?ì§€ê¸‰ë˜???˜ë‹¹?€ ?„ê¸ˆ???±ì§ˆ??ê°€ì§€ë©? ?¬ìš©?ëŠ” ?´ë? ì§€ê¸‰í•  ?˜ë¬´ê°€ ?ˆìŠµ?ˆë‹¤. ?¤ë§Œ ?¼íšŒ??ë³´ìƒ?´ë‚˜ ?¹ë³„??ê²½ìš°???˜ë‹¹?€ ?ˆì™¸?…ë‹ˆ??",
            "type": "precedent_search",
            "source": "?€ë²•ì› ?ë?",
            "case_number": "2019??7890",
            "quality_score": 0.89,
            "confidence": 0.86
        },
        {
            "id": "law_004",
            "question": "ë²•ì¸??ê¶Œë¦¬?¥ë ¥ê³??‰ìœ„?¥ë ¥???€???¤ëª…?´ì£¼?¸ìš”.",
            "answer": "ë²•ì¸?€ ë²•ë¥ ???˜í•˜??ê¶Œë¦¬?¥ë ¥??ê°€ì§€ë©? ë²•ì¸??ëª©ì ë²”ìœ„ ?´ì—??ê¶Œë¦¬?€ ?˜ë¬´??ì£¼ì²´ê°€ ?©ë‹ˆ?? ë²•ì¸???‰ìœ„?¥ë ¥?€ ?´ì‚¬???€?œìê°€ ë²•ì¸???€?œí•˜???‰ì‚¬?©ë‹ˆ?? ë¯¼ë²• ??4ì¡°ì— ê·œì •?˜ì–´ ?ˆìŠµ?ˆë‹¤.",
            "type": "law_explanation",
            "source": "ë¯¼ë²•",
            "article": "??4ì¡?,
            "quality_score": 0.93,
            "confidence": 0.89
        },
        {
            "id": "precedent_003",
            "question": "ë¶ˆë²•?‰ìœ„?ì„œ ê³¼ì‹¤?ê³„???”ê±´?€ ë¬´ì—‡?¸ê???",
            "answer": "?€ë²•ì› 2020??1111 ?ê²°???°ë¥´ë©? ë¶ˆë²•?‰ìœ„?ì„œ ê³¼ì‹¤?ê³„???¼í•´?ì—ê²?ê³¼ì‹¤???ˆê³ , ê·?ê³¼ì‹¤???í•´ë°œìƒ??ê¸°ì—¬??ê²½ìš°???ìš©?©ë‹ˆ?? ê³¼ì‹¤?ê³„??ë¹„ìœ¨?€ ?¹ì‚¬?ì˜ ê³¼ì‹¤ ?•ë„?€ ?í•´???€??ê¸°ì—¬?„ë? ê³ ë ¤?˜ì—¬ ê²°ì •?©ë‹ˆ??",
            "type": "precedent_search",
            "source": "?€ë²•ì› ?ë?",
            "case_number": "2020??1111",
            "quality_score": 0.91,
            "confidence": 0.88
        },
        {
            "id": "law_005",
            "question": "?Œë©¸?œíš¨??ê¸°ê°„?€ ?´ë–»ê²??˜ë‚˜??",
            "answer": "ë¯¼ë²• ??62ì¡°ì— ?°ë¥´ë©? ì±„ê¶Œ?€ 10?„ê°„ ?‰ì‚¬?˜ì? ?„ë‹ˆ?˜ë©´ ?Œë©¸?œíš¨ê°€ ?„ì„±?©ë‹ˆ?? ?¤ë§Œ ?ì‚¬ì±„ê¶Œ?€ 5?? ê·¼ë¡œ?ì˜ ?„ê¸ˆì±„ê¶Œ?€ 3?„ì˜ ?¨ê¸°?Œë©¸?œíš¨ê°€ ?ìš©?©ë‹ˆ??",
            "type": "law_explanation",
            "source": "ë¯¼ë²•",
            "article": "??62ì¡?,
            "quality_score": 0.94,
            "confidence": 0.90
        },
        {
            "id": "precedent_004",
            "question": "ê±´ë¬¼ëª…ë„?Œì†¡?ì„œ ?ìœ ê¶Œì˜ ?±ë¦½?”ê±´?€ ë¬´ì—‡?¸ê???",
            "answer": "?€ë²•ì› 2021??2222 ?ê²°???°ë¥´ë©? ?ìœ ê¶Œì˜ ?±ë¦½?”ê±´?¼ë¡œ??ë¬¼ê±´???€???¬ì‹¤?ì˜ ì§€ë°°ì? ?ìœ ???˜ì‚¬ê°€ ?„ìš”?©ë‹ˆ?? ê±´ë¬¼??ê²½ìš° ?¤ì œ ê±°ì£¼?˜ê±°???¬ìš©?˜ê³  ?ˆë‹¤???¬ì‹¤??ì¤‘ìš”?©ë‹ˆ??",
            "type": "precedent_search",
            "source": "?€ë²•ì› ?ë?",
            "case_number": "2021??2222",
            "quality_score": 0.87,
            "confidence": 0.84
        },
        {
            "id": "law_006",
            "question": "? ì–¸???¨ë ¥ê³??”ê±´???€???¤ëª…?´ì£¼?¸ìš”.",
            "answer": "? ì–¸?€ ? ì–¸?ì˜ ?¬ë§ ?œì— ?¨ë ¥??ë°œìƒ?˜ë©°, ? ì–¸?ì˜ ì§„ì •???˜ì‚¬?œì‹œ?¬ì•¼ ?©ë‹ˆ?? ë¯¼ë²• ??060ì¡°ì— ?°ë¥´ë©?? ì–¸?€ ë²•ì •??ë°©ì‹???°ë¼ ?˜ì—¬???˜ë©°, ?í•„ì¦ì„œ, ?¹ìŒ, ê³µì •ì¦ì„œ ?±ì˜ ë°©ì‹???ˆìŠµ?ˆë‹¤.",
            "type": "law_explanation",
            "source": "ë¯¼ë²•",
            "article": "??060ì¡?,
            "quality_score": 0.92,
            "confidence": 0.88
        }
    ]
    
    logger.info(f"?˜í”Œ Q&A ?°ì´?°ì…‹ ?ì„± ?„ë£Œ: {len(sample_data)}ê°?)
    return sample_data

def convert_to_kogpt2_format(qa_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Q&A ?°ì´?°ì…‹??KoGPT-2 ?…ë ¥ ?•ì‹?¼ë¡œ ë³€??""
    logger = setup_logging()
    
    converted_data = []
    
    for item in qa_dataset:
        # KoGPT-2???„ë¡¬?„íŠ¸ ?œí”Œë¦??ìš©
        prompt = f"<|startoftext|>ì§ˆë¬¸: {item['question']}\n?µë?: {item['answer']}<|endoftext|>"
        
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
    
    logger.info(f"KoGPT-2 ?•ì‹ ë³€???„ë£Œ: {len(converted_data)}ê°?)
    return converted_data

def split_dataset(data: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
    """?°ì´?°ì…‹???ˆë ¨/ê²€ì¦??ŒìŠ¤?¸ë¡œ ë¶„í• """
    logger = setup_logging()
    
    # ?°ì´???”í”Œ
    random.shuffle(data)
    
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    logger.info(f"?°ì´?°ì…‹ ë¶„í•  ?„ë£Œ:")
    logger.info(f"  ?ˆë ¨ ?°ì´?? {len(train_data)}ê°?({len(train_data)/total_size:.1%})")
    logger.info(f"  ê²€ì¦??°ì´?? {len(val_data)}ê°?({len(val_data)/total_size:.1%})")
    logger.info(f"  ?ŒìŠ¤???°ì´?? {len(test_data)}ê°?({len(test_data)/total_size:.1%})")
    
    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }

def create_prompt_templates() -> Dict[str, str]:
    """ë²•ë¥  ?¹í™” ?„ë¡¬?„íŠ¸ ?œí”Œë¦??ì„±"""
    templates = {
        "contract_analysis": """<|startoftext|>?¹ì‹ ?€ ë²•ë¥  ?„ë¬¸ê°€?…ë‹ˆ?? ?¤ìŒ ê³„ì•½??ì¡°í•­??ë¶„ì„?˜ê³  ?„í—˜ ?”ì†Œë¥?ì§€?í•´ì£¼ì„¸??

ê³„ì•½??ì¡°í•­: {clause}
ë¶„ì„:<|endoftext|>""",
        
        "precedent_search": """<|startoftext|>?¤ìŒ ?¬ê±´ê³?? ì‚¬???ë?ë¥?ì°¾ì•„ì£¼ì„¸??

?¬ê±´ ê°œìš”: {case_summary}
? ì‚¬ ?ë?:<|endoftext|>""",
        
        "law_explanation": """<|startoftext|>?¤ìŒ ë²•ì¡°ë¬¸ì„ ?¼ë°˜?¸ì´ ?´í•´?˜ê¸° ?½ê²Œ ?¤ëª…?´ì£¼?¸ìš”.

ë²•ì¡°ë¬? {law_article}
?¤ëª…:<|endoftext|>""",
        
        "legal_advice": """<|startoftext|>?¤ìŒ ?í™©?ì„œ ë²•ì  ì¡°ì–¸???´ì£¼?¸ìš”.

?í™©: {situation}
ì¡°ì–¸:<|endoftext|>""",
        
        "qa_format": """<|startoftext|>ì§ˆë¬¸: {question}
?µë?: {answer}<|endoftext|>"""
    }
    
    return templates

def setup_tokenizer_config() -> Dict[str, Any]:
    """? í¬?˜ì´?€ ?¤ì •"""
    config = {
        "model_name": "skt/kogpt2-base-v2",
        "special_tokens": {
            "startoftext": "<|startoftext|>",
            "endoftext": "<|endoftext|>",
            "question": "ì§ˆë¬¸:",
            "answer": "?µë?:",
            "analysis": "ë¶„ì„:",
            "explanation": "?¤ëª…:",
            "advice": "ì¡°ì–¸:"
        },
        "max_length": 512,
        "padding": True,
        "truncation": True,
        "return_tensors": "pt"
    }
    
    return config

def save_datasets(split_data: Dict[str, List[Dict[str, Any]]], output_dir: str = "data/training"):
    """ë¶„í• ???°ì´?°ì…‹ ?€??""
    logger = setup_logging()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, data in split_data.items():
        filename = f"{split_name}_split.json"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{split_name} ?°ì´?°ì…‹ ?€???„ë£Œ: {filepath} ({len(data)}ê°?")

def generate_statistics(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """?°ì´?°ì…‹ ?µê³„ ?ì„±"""
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
        # ?€?…ë³„ ë¶„í¬
        item_type = item["type"]
        stats["type_distribution"][item_type] = stats["type_distribution"].get(item_type, 0) + 1
        
        # ?ŒìŠ¤ë³?ë¶„í¬
        source = item["source"]
        stats["source_distribution"][source] = stats["source_distribution"].get(source, 0) + 1
        
        # ?ˆì§ˆ ?ìˆ˜
        quality_score = item["quality_score"]
        quality_scores.append(quality_score)
        stats["quality_stats"]["min_score"] = min(stats["quality_stats"]["min_score"], quality_score)
        stats["quality_stats"]["max_score"] = max(stats["quality_stats"]["max_score"], quality_score)
        
        # ? ë¢°???ìˆ˜
        confidence = item["confidence"]
        confidence_scores.append(confidence)
        stats["confidence_stats"]["min_confidence"] = min(stats["confidence_stats"]["min_confidence"], confidence)
        stats["confidence_stats"]["max_confidence"] = max(stats["confidence_stats"]["max_confidence"], confidence)
    
    # ?‰ê·  ê³„ì‚°
    stats["quality_stats"]["average_score"] = sum(quality_scores) / len(quality_scores)
    stats["confidence_stats"]["average_confidence"] = sum(confidence_scores) / len(confidence_scores)
    
    return stats

def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    logger = setup_logging()
    logger.info("Day 2: ?°ì´?°ì…‹ ì¤€ë¹?ë°??„ì²˜ë¦??œì‘...")
    
    # 1. ?˜í”Œ Q&A ?°ì´?°ì…‹ ?ì„±
    logger.info("1. ?˜í”Œ Q&A ?°ì´?°ì…‹ ?ì„±...")
    qa_dataset = create_sample_qa_dataset()
    
    # 2. KoGPT-2 ?•ì‹?¼ë¡œ ë³€??
    logger.info("2. KoGPT-2 ?•ì‹?¼ë¡œ ë³€??..")
    converted_data = convert_to_kogpt2_format(qa_dataset)
    
    # 3. ?„ë¡¬?„íŠ¸ ?œí”Œë¦??ì„±
    logger.info("3. ?„ë¡¬?„íŠ¸ ?œí”Œë¦??ì„±...")
    prompt_templates = create_prompt_templates()
    
    # 4. ? í¬?˜ì´?€ ?¤ì •
    logger.info("4. ? í¬?˜ì´?€ ?¤ì •...")
    tokenizer_config = setup_tokenizer_config()
    
    # 5. ?°ì´?°ì…‹ ë¶„í• 
    logger.info("5. ?°ì´?°ì…‹ ë¶„í•  (8:1:1)...")
    split_data = split_dataset(converted_data)
    
    # 6. ?°ì´?°ì…‹ ?€??
    logger.info("6. ?°ì´?°ì…‹ ?€??..")
    save_datasets(split_data)
    
    # 7. ?µê³„ ?ì„±
    logger.info("7. ?µê³„ ?ì„±...")
    stats = generate_statistics(converted_data)
    
    # ?µê³„ ?€??
    stats_path = Path("data/training/dataset_statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # ?„ë¡¬?„íŠ¸ ?œí”Œë¦??€??
    templates_path = Path("data/training/prompt_templates.json")
    with open(templates_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_templates, f, ensure_ascii=False, indent=2)
    
    # ? í¬?˜ì´?€ ?¤ì • ?€??
    config_path = Path("data/training/tokenizer_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    
    logger.info("Day 2: ?°ì´?°ì…‹ ì¤€ë¹?ë°??„ì²˜ë¦??„ë£Œ!")
    logger.info(f"ì´??°ì´?? {len(converted_data)}ê°?)
    logger.info(f"?ˆë ¨ ?°ì´?? {len(split_data['train'])}ê°?)
    logger.info(f"ê²€ì¦??°ì´?? {len(split_data['validation'])}ê°?)
    logger.info(f"?ŒìŠ¤???°ì´?? {len(split_data['test'])}ê°?)

if __name__ == "__main__":
    main()
