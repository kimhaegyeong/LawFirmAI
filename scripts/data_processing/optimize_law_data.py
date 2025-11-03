#!/usr/bin/env python3
"""
ë²•ë¥  ?°ì´??ìµœì ???¤í¬ë¦½íŠ¸
ê¸°ì¡´ ?„ì²˜ë¦¬ëœ ?°ì´?°ë? ?•ì¶•?˜ì—¬ ?©ëŸ‰??50-70% ì¤„ì…?ˆë‹¤.

ì£¼ìš” ìµœì ??
1. ì¤‘ë³µ ?„ë“œ ?œê±° (ë¶ˆí•„?”í•œ ?„ë“œ ?œê±°)
2. ë©”í??°ì´??ê°„ì†Œ??(ë¶ˆí•„?”í•œ ?„ë“œ ?œê±°)
3. ?ìŠ¤???•ì¶• (ë¶ˆí•„?”í•œ ê³µë°± ë°?ë°˜ë³µ ?ìŠ¤???œê±°)
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

def compress_legal_text(text: str) -> str:
    """ë²•ë¥  ?ìŠ¤???•ì¶•"""
    if not text:
        return ""
    
    # ë¶ˆí•„?”í•œ ê³µë°± ?œê±°
    text = re.sub(r'\s+', ' ', text)
    
    # ë°˜ë³µ?˜ëŠ” ë²•ë¥  ?©ì–´ ì¶•ì•½
    replacements = {
        '??ë²•ì— ?°ë¥´ë©?: '??ë²•ì— ?°ë¼',
        '?¤ìŒ ê°??¸ì˜ ?´ëŠ ?˜ë‚˜???´ë‹¹?˜ëŠ”': '?¤ìŒ???´ë‹¹?˜ëŠ”',
        '?¹ë³„?œì¥Â·ê´‘ì—­?œì¥Â·?¹ë³„?ì¹˜?œì¥Â·?„ì???: '?œÂ·ë„ì§€??,
        '?¹ë³„?ì¹˜?„ì???: '?¹ë³„?ì¹˜?„ì???,
        'ì¤‘ì•™?‰ì •ê¸°ê?????: 'ì¤‘ì•™?‰ì •ê¸°ê???,
        'ì§€ë°©ìì¹˜ë‹¨ì²´ì˜ ??: 'ì§€ë°©ìì¹˜ë‹¨ì²´ì¥',
        'êµ?? ?ëŠ” ì§€ë°©ìì¹˜ë‹¨ì²?: 'êµ??Â·ì§€ë°©ìì¹˜ë‹¨ì²?,
        '?´í•˜ "?œÂ·ë„ì§€?????œë‹¤': '?´í•˜ ?œÂ·ë„ì§€?¬ë¼ ??,
        '?´í•˜ "?¹ë??????œë‹¤': '?´í•˜ ?¹ë??œë¼ ??
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.strip()

def compress_law_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """ê°œë³„ ë²•ë¥  ?°ì´???•ì¶•"""
    # ?„ìˆ˜ ?„ë“œë§?? ì?
    compressed = {
        'law_id': data.get('law_id'),
        'law_name': data.get('law_name'),
        'law_type': data.get('law_type'),
        'category': data.get('category'),
        'promulgation_number': data.get('promulgation_number'),
        'promulgation_date': data.get('promulgation_date'),
        'enforcement_date': data.get('enforcement_date'),
        'amendment_type': data.get('amendment_type'),
        'ministry': data.get('ministry'),
        'articles': data.get('articles', [])
    }
    
    # articles ?´ë? ?ìŠ¤?¸ë„ ?•ì¶•
    for article in compressed['articles']:
        if 'article_content' in article:
            article['article_content'] = compress_legal_text(article['article_content'])
        
        # sub_articles???•ì¶•
        for sub_article in article.get('sub_articles', []):
            if 'content' in sub_article:
                sub_article['content'] = compress_legal_text(sub_article['content'])
    
    return compressed

def optimize_existing_data(input_dir: Path, output_dir: Path, backup: bool = True):
    """ê¸°ì¡´ ?°ì´??ìµœì ??""
    print(f"[START] ë²•ë¥  ?°ì´??ìµœì ???œì‘")
    print(f"   ?…ë ¥ ?”ë ‰? ë¦¬: {input_dir}")
    print(f"   ì¶œë ¥ ?”ë ‰? ë¦¬: {output_dir}")
    
    # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ë°±ì—… ?”ë ‰? ë¦¬ ?ì„±
    if backup:
        backup_dir = input_dir.parent / f"{input_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ë°±ì—… ?”ë ‰? ë¦¬: {backup_dir}")
    
    total_original_size = 0
    total_compressed_size = 0
    processed_files = 0
    
    # JSON ?Œì¼??ì²˜ë¦¬
    json_files = [f for f in input_dir.glob("*.json") if f.name != "processing_status.db"]
    
    print(f"   ì²˜ë¦¬???Œì¼ ?? {len(json_files)}")
    print()
    
    for json_file in json_files:
        try:
            # ?ë³¸ ?Œì¼ ?½ê¸°
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # ë°±ì—… ?ì„±
            if backup:
                backup_file = backup_dir / json_file.name
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            # ?•ì¶•???°ì´???ì„±
            compressed_data = compress_law_data(data)
            
            # ?•ì¶•???Œì¼ ?€??
            output_file = output_dir / json_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(compressed_data, f, ensure_ascii=False, separators=(',', ':'))
            
            # ?©ëŸ‰ ë¹„êµ
            original_size = json_file.stat().st_size
            compressed_size = output_file.stat().st_size
            
            total_original_size += original_size
            total_compressed_size += compressed_size
            processed_files += 1
            
            compression_ratio = (1 - compressed_size / original_size) * 100
            print(f"[OK] {json_file.name}: {original_size:,} -> {compressed_size:,} bytes ({compression_ratio:.1f}% ?•ì¶•)")
            
        except Exception as e:
            print(f"[ERROR] {json_file.name} ì²˜ë¦¬ ?¤íŒ¨: {e}")
    
    # processing_status.db ë³µì‚¬
    db_file = input_dir / "processing_status.db"
    if db_file.exists():
        import shutil
        shutil.copy2(db_file, output_dir / "processing_status.db")
        print(f"[COPY] processing_status.db ë³µì‚¬ ?„ë£Œ")
    
    # ê²°ê³¼ ?”ì•½
    print(f"\n[RESULT] ìµœì ??ê²°ê³¼:")
    print(f"   ì²˜ë¦¬???Œì¼ ?? {processed_files}")
    print(f"   ?ë³¸ ?©ëŸ‰: {total_original_size:,} bytes ({total_original_size/1024/1024:.1f} MB)")
    print(f"   ?•ì¶• ?©ëŸ‰: {total_compressed_size:,} bytes ({total_compressed_size/1024/1024:.1f} MB)")
    print(f"   ?•ì¶•ë¥? {(1 - total_compressed_size / total_original_size) * 100:.1f}%")
    print(f"   ?ˆì•½???©ëŸ‰: {(total_original_size - total_compressed_size)/1024/1024:.1f} MB")

def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ë²•ë¥  ?°ì´??ìµœì ??)
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='?…ë ¥ ?”ë ‰? ë¦¬ ê²½ë¡œ')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='ì¶œë ¥ ?”ë ‰? ë¦¬ ê²½ë¡œ')
    parser.add_argument('--no-backup', action='store_true',
                       help='ë°±ì—… ?ì„± ?ˆí•¨')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"???…ë ¥ ?”ë ‰? ë¦¬ê°€ ì¡´ì¬?˜ì? ?ŠìŠµ?ˆë‹¤: {input_dir}")
        sys.exit(1)
    
    optimize_existing_data(input_dir, output_dir, backup=not args.no_backup)

if __name__ == "__main__":
    main()
