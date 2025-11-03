#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?˜ê²½ë³€???¤ì • ?„ìš°ë¯??¤í¬ë¦½íŠ¸
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """?˜ê²½ë³€???¤ì •"""
    print("?”§ LawFirmAI ?˜ê²½ë³€???¤ì •")
    print("=" * 50)
    
    # LAW_OPEN_API_OC ?¤ì •
    current_oc = os.getenv("LAW_OPEN_API_OC")
    if current_oc:
        print(f"??LAW_OPEN_API_OCê°€ ?´ë? ?¤ì •?˜ì–´ ?ˆìŠµ?ˆë‹¤: {current_oc}")
    else:
        print("??LAW_OPEN_API_OCê°€ ?¤ì •?˜ì? ?Šì•˜?µë‹ˆ??")
        print("\n?“ ?¤ì • ë°©ë²•:")
        print("1. PowerShell?ì„œ:")
        print("   $env:LAW_OPEN_API_OC='your_email@example.com'")
        print("\n2. CMD?ì„œ:")
        print("   set LAW_OPEN_API_OC=your_email@example.com")
        print("\n3. .env ?Œì¼ ?ì„±:")
        print("   LAW_OPEN_API_OC=your_email@example.com")
        
        # ?¬ìš©???…ë ¥ ë°›ê¸°
        email = input("\n?´ë©”??ì£¼ì†Œë¥??…ë ¥?˜ì„¸??(?ëŠ” Enterë¡?ê±´ë„ˆ?°ê¸°): ").strip()
        if email:
            os.environ["LAW_OPEN_API_OC"] = email
            print(f"???˜ê²½ë³€???¤ì • ?„ë£Œ: {email}")
        else:
            print("? ï¸ ?˜ê²½ë³€?˜ë? ?˜ë™?¼ë¡œ ?¤ì •?´ì£¼?¸ìš”.")
    
    # .env ?Œì¼ ?ì„±
    env_file = Path(".env")
    if not env_file.exists():
        print("\n?“„ .env ?Œì¼???ì„±?©ë‹ˆ??..")
        env_content = f"""# LawFirmAI ?˜ê²½ë³€???¤ì •
LAW_OPEN_API_OC={os.getenv("LAW_OPEN_API_OC", "your_email@example.com")}

# ?°ì´?°ë² ?´ìŠ¤ ?¤ì •
DATABASE_URL=sqlite:///./data/lawfirm.db

# ëª¨ë¸ ?¤ì •
MODEL_PATH=./models
MODEL_CACHE_DIR=./cache

# ë¡œê¹… ?¤ì •
LOG_LEVEL=INFO
LOG_DIR=./logs

# API ?¤ì •
API_HOST=0.0.0.0
API_PORT=8000
GRADIO_PORT=7860

# ?±ëŠ¥ ?¤ì •
MAX_WORKERS=4
BATCH_SIZE=50
TIMEOUT=60

# ë³´ì•ˆ ?¤ì •
SECRET_KEY=your_secret_key_here
API_KEY=your_api_key_here
"""
        try:
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            print("??.env ?Œì¼ ?ì„± ?„ë£Œ")
        except Exception as e:
            print(f"??.env ?Œì¼ ?ì„± ?¤íŒ¨: {e}")
    else:
        print("??.env ?Œì¼???´ë? ì¡´ì¬?©ë‹ˆ??")
    
    print("\n?‰ ?˜ê²½ë³€???¤ì • ?„ë£Œ!")
    print("?´ì œ ?ë? ?˜ì§‘ ?¤í¬ë¦½íŠ¸ë¥??¤í–‰?????ˆìŠµ?ˆë‹¤.")

if __name__ == "__main__":
    setup_environment()
