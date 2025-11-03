#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?ë? ?˜ì§‘ ê´€??ë¡œê¹… ? í‹¸ë¦¬í‹°
"""

import logging
import sys
import io
import os
from datetime import datetime
from pathlib import Path

# ?œê? ì¶œë ¥???„í•œ ?¸ì½”???¤ì •
if sys.platform == "win32":
    # Windows?ì„œ ?œê? ì¶œë ¥???„í•œ ?¤ì •
    try:
        import codecs
        # Windows ì½˜ì†”?ì„œ UTF-8 ì§€???•ì¸ ???¤ì •
        if hasattr(sys.stdout, 'detach'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        else:
            # detach()ê°€ ì§€?ë˜ì§€ ?ŠëŠ” ê²½ìš° ?¤ë¥¸ ë°©ë²• ?¬ìš©
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        # ?¸ì½”???¤ì • ?¤íŒ¨ ??ê¸°ë³¸ ?¤ì • ? ì?
        pass
    
    # ?˜ê²½ë³€???¸ì½”???¤ì •
    os.environ['PYTHONIOENCODING'] = 'utf-8'


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """ë¡œê¹… ?¤ì •??ì´ˆê¸°?”í•˜ê³?ë¡œê±°ë¥?ë°˜í™˜?©ë‹ˆ??"""
    
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # ë¡œê·¸ ?¬ë§· ?¤ì • (?´ëª¨ì§€ ?œê±°)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # ?Œì¼ ?¸ë“¤??(?¼ë³„ ë¡œí…Œ?´ì…˜, UTF-8 ?¸ì½”??
    log_file = log_dir / f"collect_precedents_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # ì½˜ì†” ?¸ë“¤???¤ì • (Windows ?œê? ì¶œë ¥ ê°œì„ )
    if sys.platform == "win32":
        try:
            # Windows ì½˜ì†” ?¸ì½”?©ì„ UTF-8ë¡??¤ì •
            import subprocess
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
            
            # Windows?ì„œ ?œê? ì¶œë ¥???„í•œ ?¹ë³„???¤íŠ¸ë¦??¸ë“¤??
            import io
            console_handler = logging.StreamHandler(
                io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            )
        except Exception:
            # ?¤íŒ¨ ??ê¸°ë³¸ ?¸ë“¤???¬ìš©
            console_handler = logging.StreamHandler(sys.stdout)
    else:
        # Linux/Mac?ì„œ??ê¸°ë³¸ ?¸ë“¤???¬ìš©
        console_handler = logging.StreamHandler(sys.stdout)
    
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # ë¡œê±° ?¤ì •
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # ì¤‘ë³µ ?¸ë“¤??ë°©ì?
    logger.propagate = False
    
    return logger
