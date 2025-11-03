#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?‰ì •?¬íŒë¡€ ?˜ì§‘??ë¡œê±° ?¤ì •
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    ?‰ì •?¬íŒë¡€ ?˜ì§‘??ë¡œê±° ?¤ì •
    
    Args:
        log_level: ë¡œê·¸ ?ˆë²¨ (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        ?¤ì •??ë¡œê±° ê°ì²´
    """
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # ë¡œê±° ?ì„±
    logger = logging.getLogger("administrative_appeal_collector")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # ê¸°ì¡´ ?¸ë“¤???œê±° (ì¤‘ë³µ ë°©ì?)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # ?¬ë§·???¤ì •
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # ?Œì¼ ?¸ë“¤???¤ì •
    log_file = log_dir / f"administrative_appeal_collection_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # ì½˜ì†” ?¸ë“¤???¤ì •
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Windows?ì„œ UTF-8 ?˜ê²½ ?¤ì •
    if sys.platform.startswith('win'):
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        # ì½˜ì†” ì½”ë“œ?˜ì´ì§€ë¥?UTF-8ë¡??¤ì •
        try:
            import subprocess
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
        except:
            pass
    
    return logger
