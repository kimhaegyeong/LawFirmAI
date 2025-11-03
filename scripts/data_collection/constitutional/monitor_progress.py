#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ì§„í–‰ ?í™© ?¤ì‹œê°?ëª¨ë‹ˆ?°ë§ ?¤í¬ë¦½íŠ¸
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def monitor_progress():
    """?¤ì‹œê°?ì§„í–‰ ?í™© ëª¨ë‹ˆ?°ë§"""
    output_dir = Path("data/raw/constitutional_decisions")
    
    if not output_dir.exists():
        print("???˜ì§‘ ?”ë ‰? ë¦¬ê°€ ì¡´ì¬?˜ì? ?ŠìŠµ?ˆë‹¤.")
        return
    
    print("?” ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ì§„í–‰ ?í™© ?¤ì‹œê°?ëª¨ë‹ˆ?°ë§")
    print("=" * 60)
    print("Ctrl+Cë¥??ŒëŸ¬ ì¢…ë£Œ?˜ì„¸??")
    print("=" * 60)
    
    try:
        while True:
            # ?”ë©´ ?´ë¦¬??
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"???„ì¬ ?œê°„: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"?“ ?˜ì§‘ ?”ë ‰? ë¦¬: {output_dir}")
            print("=" * 60)
            
            # ì²´í¬?¬ì¸???Œì¼ ?•ì¸
            checkpoint_files = list(output_dir.glob("collection_checkpoint_*.json"))
            
            if checkpoint_files:
                # ê°€??ìµœê·¼ ì²´í¬?¬ì¸???Œì¼ ?½ê¸°
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                try:
                    with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    
                    stats = checkpoint_data.get('stats', {})
                    collected_count = stats.get('collected_count', 0)
                    target_count = stats.get('target_count', 1000)
                    keywords_processed = stats.get('keywords_processed', 0)
                    last_keyword = stats.get('last_keyword_processed', 'N/A')
                    api_requests = stats.get('api_requests_made', 0)
                    api_errors = stats.get('api_errors', 0)
                    status = stats.get('status', 'running')
                    
                    progress = (collected_count / target_count * 100) if target_count > 0 else 0
                    
                    print(f"?“Š ?˜ì§‘ ì§„í–‰ë¥? {progress:.1f}% ({collected_count:,}/{target_count:,}ê±?")
                    print(f"?” ì²˜ë¦¬???¤ì›Œ?? {keywords_processed}ê°?)
                    print(f"?“ ë§ˆì?ë§?ì²˜ë¦¬ ?¤ì›Œ?? {last_keyword}")
                    print(f"?Œ API ?”ì²­ ?? {api_requests:,}??)
                    print(f"??API ?¤ë¥˜ ?? {api_errors:,}??)
                    print(f"?“ˆ ?íƒœ: {status}")
                    
                    # ?ˆìƒ ?„ë£Œ ?œê°„ ê³„ì‚°
                    if collected_count > 0 and status == 'running':
                        remaining = target_count - collected_count
                        if remaining > 0:
                            # ê°„ë‹¨???ˆìƒ ?œê°„ ê³„ì‚° (API ?”ì²­ ??ê¸°ë°˜)
                            estimated_remaining_requests = remaining // 100 + 1
                            print(f"?±ï¸  ?ˆìƒ ?¨ì? API ?”ì²­: {estimated_remaining_requests:,}??)
                    
                    if status == 'completed':
                        print("?‰ ?˜ì§‘???„ë£Œ?˜ì—ˆ?µë‹ˆ??")
                        break
                    elif status == 'interrupted':
                        print("? ï¸ ?˜ì§‘??ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
                        break
                        
                except Exception as e:
                    print(f"??ì²´í¬?¬ì¸???Œì¼ ?½ê¸° ?¤ë¥˜: {e}")
            else:
                # ë°°ì¹˜ ?Œì¼ë¡?ì§„í–‰ ?í™© ì¶”ì •
                batch_files = list(output_dir.glob("batch_*.json"))
                if batch_files:
                    total_count = 0
                    for file_path in batch_files:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                count = data.get('metadata', {}).get('count', 0)
                                total_count += count
                        except:
                            pass
                    
                    print(f"?“ ?˜ì§‘??ë°°ì¹˜ ?Œì¼: {len(batch_files)}ê°?)
                    print(f"?“Š ì¶”ì • ?˜ì§‘ ê±´ìˆ˜: {total_count:,}ê±?)
                else:
                    print("???˜ì§‘ ?°ì´?°ê? ?†ìŠµ?ˆë‹¤.")
            
            print("=" * 60)
            print("5ì´????ˆë¡œê³ ì¹¨... (Ctrl+Cë¡?ì¢…ë£Œ)")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n?‘‹ ëª¨ë‹ˆ?°ë§??ì¢…ë£Œ?©ë‹ˆ??")
    except Exception as e:
        print(f"??ëª¨ë‹ˆ?°ë§ ì¤??¤ë¥˜ ë°œìƒ: {e}")

if __name__ == "__main__":
    monitor_progress()
