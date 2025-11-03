#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM ê¸°ë°˜ Q&A ?°ì´?°ì…‹ ?ì„± ?¤í–‰ ?¤í¬ë¦½íŠ¸

Ollama Qwen2.5:7b ëª¨ë¸???¬ìš©?˜ì—¬ ë²•ë¥  Q&A ?°ì´?°ì…‹???ì„±?©ë‹ˆ??
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.llm_qa_generator import LLMQAGenerator

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/generate_qa_with_llm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """ëª…ë ¹???¸ìˆ˜ ?Œì‹±"""
    parser = argparse.ArgumentParser(
        description="LLM ê¸°ë°˜ ë²•ë¥  Q&A ?°ì´?°ì…‹ ?ì„±",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
?¬ìš© ?ˆì‹œ:
  # ê¸°ë³¸ ?¤í–‰
  python scripts/generate_qa_with_llm.py

  # ?¹ì • ëª¨ë¸ê³??°ì´???€??ì§€??
  python scripts/generate_qa_with_llm.py --model qwen2.5:7b --data-type laws precedents

  # ì¶œë ¥ ?”ë ‰? ë¦¬?€ ëª©í‘œ ê°œìˆ˜ ì§€??
  python scripts/generate_qa_with_llm.py --output data/qa_dataset/llm_generated --target 3000

  # ?ŒìŠ¤??ëª¨ë“œ (?Œê·œëª??°ì´??
  python scripts/generate_qa_with_llm.py --dry-run --max-items 10
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='qwen2.5:7b',
        help='?¬ìš©??Ollama ëª¨ë¸ëª?(ê¸°ë³¸ê°? qwen2.5:7b)'
    )
    
    parser.add_argument(
        '--data-type',
        nargs='+',
        choices=['laws', 'precedents', 'constitutional_decisions', 'legal_interpretations'],
        default=['laws', 'precedents'],
        help='ì²˜ë¦¬???°ì´??? í˜• (ê¸°ë³¸ê°? laws precedents)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='?…ë ¥ ?°ì´???”ë ‰? ë¦¬ (ê¸°ë³¸ê°? data/processed)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/qa_dataset/llm_generated',
        help='ì¶œë ¥ ?”ë ‰? ë¦¬ (ê¸°ë³¸ê°? data/qa_dataset/llm_generated)'
    )
    
    parser.add_argument(
        '--target',
        type=int,
        default=3000,
        help='ëª©í‘œ Q&A ê°œìˆ˜ (ê¸°ë³¸ê°? 3000)'
    )
    
    parser.add_argument(
        '--max-items',
        type=int,
        default=100,
        help='?°ì´???€?…ë³„ ìµœë? ì²˜ë¦¬ ??ª© ??(ê¸°ë³¸ê°? 100)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM ?ì„± ?¨ë„ (0.0-1.0, ê¸°ë³¸ê°? 0.7)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1500,
        help='ìµœë? ? í° ??(ê¸°ë³¸ê°? 1500)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='ë°°ì¹˜ ì²˜ë¦¬ ?¬ê¸° (ê¸°ë³¸ê°? 10)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='?ŒìŠ¤??ëª¨ë“œ (?¤ì œ ?ì„±?˜ì? ?Šê³  ?¤ì •ë§??•ì¸)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='?ì„¸ ë¡œê·¸ ì¶œë ¥'
    )
    
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=0.6,
        help='?ˆì§ˆ ?ìˆ˜ ?„ê³„ê°?(ê¸°ë³¸ê°? 0.6)'
    )
    
    return parser.parse_args()


def validate_environment():
    """?˜ê²½ ê²€ì¦?""
    logger.info("?˜ê²½ ê²€ì¦?ì¤?..")
    
    # ?„ìš”???”ë ‰? ë¦¬ ?•ì¸
    data_dir = Path("data/processed")
    if not data_dir.exists():
        logger.error(f"?°ì´???”ë ‰? ë¦¬ê°€ ì¡´ì¬?˜ì? ?ŠìŠµ?ˆë‹¤: {data_dir}")
        return False
    
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.info("?˜ê²½ ê²€ì¦??„ë£Œ")
    return True


def test_ollama_connection(model: str) -> bool:
    """Ollama ?°ê²° ?ŒìŠ¤??""
    logger.info(f"Ollama ëª¨ë¸ '{model}' ?°ê²° ?ŒìŠ¤??ì¤?..")
    
    try:
        from source.utils.ollama_client import OllamaClient
        
        client = OllamaClient(model=model)
        success = client.test_connection()
        
        if success:
            logger.info("??Ollama ?°ê²° ?±ê³µ")
            return True
        else:
            logger.error("??Ollama ?°ê²° ?¤íŒ¨")
            return False
            
    except Exception as e:
        logger.error(f"??Ollama ?°ê²° ?ŒìŠ¤??ì¤??¤ë¥˜: {e}")
        return False


def generate_qa_dataset(args):
    """Q&A ?°ì´?°ì…‹ ?ì„±"""
    logger.info("LLM ê¸°ë°˜ Q&A ?°ì´?°ì…‹ ?ì„± ?œì‘...")
    
    try:
        # LLM Q&A ?ì„±ê¸?ì´ˆê¸°??
        generator = LLMQAGenerator(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # ?ˆì§ˆ ê¸°ì? ?…ë°?´íŠ¸
        generator.quality_criteria['min_quality_score'] = args.quality_threshold
        
        # ?°ì´?°ì…‹ ?ì„±
        success = generator.generate_dataset(
            data_dir=args.data_dir,
            output_dir=args.output,
            data_types=args.data_type,
            max_items_per_type=args.max_items
        )
        
        if success:
            logger.info("??Q&A ?°ì´?°ì…‹ ?ì„± ?„ë£Œ")
            
            # ê²°ê³¼ ?”ì•½ ì¶œë ¥
            print_summary(args.output)
            return True
        else:
            logger.error("??Q&A ?°ì´?°ì…‹ ?ì„± ?¤íŒ¨")
            return False
            
    except Exception as e:
        logger.error(f"Q&A ?°ì´?°ì…‹ ?ì„± ì¤??¤ë¥˜: {e}")
        return False


def print_summary(output_dir: str):
    """ê²°ê³¼ ?”ì•½ ì¶œë ¥"""
    try:
        import json
        
        stats_file = Path(output_dir) / "llm_qa_dataset_statistics.json"
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            print("\n" + "="*60)
            print("?“Š LLM ê¸°ë°˜ Q&A ?°ì´?°ì…‹ ?ì„± ê²°ê³¼")
            print("="*60)
            print(f"ì´?Q&A ???? {stats['total_pairs']:,}ê°?)
            print(f"ê³ í’ˆì§?Q&A: {stats['high_quality_pairs']:,}ê°?(?ˆì§ˆ?ìˆ˜ ??.8)")
            print(f"ì¤‘í’ˆì§?Q&A: {stats['medium_quality_pairs']:,}ê°?(?ˆì§ˆ?ìˆ˜ 0.6-0.8)")
            print(f"?€?ˆì§ˆ Q&A: {stats['low_quality_pairs']:,}ê°?(?ˆì§ˆ?ìˆ˜ <0.6)")
            print(f"?‰ê·  ?ˆì§ˆ ?ìˆ˜: {stats['average_quality_score']:.3f}")
            print(f"?¬ìš© ëª¨ë¸: {stats['model_used']}")
            print(f"?ì„± ?¨ë„: {stats['temperature']}")
            
            print("\n?“ˆ ?ŒìŠ¤ë³?ë¶„í¬:")
            for source, count in stats['source_distribution'].items():
                print(f"  - {source}: {count:,}ê°?)
            
            print("\n?“ˆ ?œì´?„ë³„ ë¶„í¬:")
            for difficulty, count in stats['difficulty_distribution'].items():
                print(f"  - {difficulty}: {count:,}ê°?)
            
            print("\n?“ˆ ì§ˆë¬¸ ? í˜•ë³?ë¶„í¬:")
            for q_type, count in stats['question_type_distribution'].items():
                print(f"  - {q_type}: {count:,}ê°?)
            
            print("\n?“ ?ì„±???Œì¼:")
            output_path = Path(output_dir)
            for file_path in output_path.glob("*.json"):
                print(f"  - {file_path.name}")
            
            print("="*60)
            
    except Exception as e:
        logger.error(f"ê²°ê³¼ ?”ì•½ ì¶œë ¥ ì¤??¤ë¥˜: {e}")


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    # ?¸ìˆ˜ ?Œì‹±
    args = parse_arguments()
    
    # ë¡œê·¸ ?ˆë²¨ ?¤ì •
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("LLM ê¸°ë°˜ Q&A ?°ì´?°ì…‹ ?ì„± ?¤í¬ë¦½íŠ¸ ?œì‘")
    logger.info(f"?¤ì •: {vars(args)}")
    
    # ?˜ê²½ ê²€ì¦?
    if not validate_environment():
        logger.error("?˜ê²½ ê²€ì¦??¤íŒ¨")
        return 1
    
    # Ollama ?°ê²° ?ŒìŠ¤??
    if not test_ollama_connection(args.model):
        logger.error("Ollama ?°ê²° ?¤íŒ¨")
        return 1
    
    # ?ŒìŠ¤??ëª¨ë“œ ?•ì¸
    if args.dry_run:
        logger.info("?” ?ŒìŠ¤??ëª¨ë“œ: ?¤ì œ ?ì„±?˜ì? ?ŠìŒ")
        logger.info("?¤ì • ?•ì¸ ?„ë£Œ. --dry-run ?µì…˜???œê±°?˜ê³  ?¤í–‰?˜ì„¸??")
        return 0
    
    # Q&A ?°ì´?°ì…‹ ?ì„±
    success = generate_qa_dataset(args)
    
    if success:
        logger.info("?‰ LLM ê¸°ë°˜ Q&A ?°ì´?°ì…‹ ?ì„± ?„ë£Œ!")
        return 0
    else:
        logger.error("??LLM ê¸°ë°˜ Q&A ?°ì´?°ì…‹ ?ì„± ?¤íŒ¨")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
