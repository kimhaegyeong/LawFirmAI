#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ë²•ë¥  ?˜ì§‘ ëª¨ë‹ˆ?°ë§ ?¤í¬ë¦½íŠ¸
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_collection():
    """?˜ì§‘ ì§„í–‰ ?í™© ëª¨ë‹ˆ?°ë§"""
    
    # ì²´í¬?¬ì¸???Œì¼ ê²½ë¡œ
    checkpoint_path = Path("data/checkpoints/laws_only/checkpoint.json")
    
    # ?°ì´???”ë ‰? ë¦¬ ê²½ë¡œ
    data_dir = Path("data/raw/assembly/law_only/20251016")
    
    print("=" * 60)
    print("LAW COLLECTION MONITOR")
    print("=" * 60)
    
    while True:
        try:
            # ì²´í¬?¬ì¸???•ë³´ ?½ê¸°
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                
                current_page = checkpoint.get('current_page', 0)
                collected_count = checkpoint.get('collected_count', 0)
                failed_count = checkpoint.get('failed_count', 0)
                target_count = checkpoint.get('target_count', 1895)
                timestamp = checkpoint.get('timestamp', 'Unknown')
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Collection Status:")
                print(f"  Current Page: {current_page}")
                print(f"  Collected: {collected_count}/{target_count} ({collected_count/target_count*100:.1f}%)")
                print(f"  Failed: {failed_count}")
                print(f"  Last Update: {timestamp}")
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] No checkpoint found")
            
            # ?°ì´???Œì¼ ê°œìˆ˜ ?•ì¸
            if data_dir.exists():
                files = list(data_dir.glob("*.json"))
                print(f"  Data Files: {len(files)}")
                
                if files:
                    # ìµœì‹  ?Œì¼ ?•ì¸
                    latest_file = max(files, key=lambda f: f.stat().st_mtime)
                    print(f"  Latest File: {latest_file.name}")
                    
                    # ìµœì‹  ?Œì¼??ë©”í??°ì´???•ì¸
                    try:
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            metadata = data.get('metadata', {})
                            page_number = metadata.get('page_number')
                            batch_count = metadata.get('count', 0)
                            print(f"  Latest Page Number: {page_number}")
                            print(f"  Latest Batch Count: {batch_count}")
                    except Exception as e:
                        print(f"  Error reading latest file: {e}")
            else:
                print(f"  Data directory not found: {data_dir}")
            
            # ì§„í–‰ë¥??œì‹œ
            if checkpoint_path.exists() and target_count > 0:
                progress = collected_count / target_count
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '?? * filled_length + '?? * (bar_length - filled_length)
                print(f"  Progress: [{bar}] {progress*100:.1f}%")
            
            print("-" * 60)
            
            # 10ì´??€ê¸?
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_collection()
