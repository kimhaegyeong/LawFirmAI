# -*- coding: utf-8 -*-
"""
Assembly Logger
Î°úÍπÖ ?§Ï†ï Î™®Îìà

Íµ¨Ï°∞?îÎêú Î°úÍπÖ???úÍ≥µ?©Îãà??
- ?åÏùº Î∞?ÏΩòÏÜî Ï∂úÎ†•
- JSON Íµ¨Ï°∞??Î°úÍ∑∏
- Î°úÍ∑∏ ?àÎ≤® ?§Ï†ï
- Î°úÍ∑∏ ?åÏ†Ñ
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON ?ïÌÉú??Î°úÍ∑∏ ?¨Îß∑??""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # ?àÏô∏ ?ïÎ≥¥ Ï∂îÍ?
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Ï∂îÍ? ?ÑÎìú Ï∂îÍ?
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Ïª¨Îü¨Í∞Ä ?ÅÏö©??ÏΩòÏÜî ?¨Îß∑??""
    
    # ANSI ?âÏÉÅ ÏΩîÎìú
    COLORS = {
        'DEBUG': '\033[36m',    # Ï≤?°ù??
        'INFO': '\033[32m',     # ?πÏÉâ
        'WARNING': '\033[33m',  # ?∏Î???
        'ERROR': '\033[31m',    # Îπ®Í∞Ñ??
        'CRITICAL': '\033[35m', # ?êÏ£º??
        'RESET': '\033[0m'      # Î¶¨ÏÖã
    }
    
    def format(self, record):
        # ?âÏÉÅ ?ÅÏö©
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # ?¨Îß∑ Î¨∏Ïûê?¥Ïóê ?âÏÉÅ Ï∂îÍ?
        colored_format = f"{color}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}"
        
        formatter = logging.Formatter(colored_format)
        return formatter.format(record)


def setup_logging(
    log_name: str = "assembly_collection",
    log_level: str = "INFO",
    log_dir: str = "logs",
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Î°úÍπÖ ?§Ï†ï
    
    Args:
        log_name: Î°úÍ±∞ ?¥Î¶Ñ
        log_level: Î°úÍ∑∏ ?àÎ≤® (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Î°úÍ∑∏ ?åÏùº ?Ä???îÎ†â?†Î¶¨
        console_output: ÏΩòÏÜî Ï∂úÎ†• ?¨Î?
        file_output: ?åÏùº Ï∂úÎ†• ?¨Î?
        json_format: JSON ?ïÌÉú Î°úÍ∑∏ ?¨Î?
        max_file_size: Î°úÍ∑∏ ?åÏùº ÏµúÎ? ?¨Í∏∞ (Î∞îÏù¥??
        backup_count: Î∞±ÏóÖ ?åÏùº Í∞úÏàò
    
    Returns:
        logging.Logger: ?§Ï†ï??Î°úÍ±∞
    """
    
    # Î°úÍ∑∏ ?îÎ†â?†Î¶¨ ?ùÏÑ±
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Î°úÍ±∞ ?ùÏÑ±
    logger = logging.getLogger(log_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Í∏∞Ï°¥ ?∏Îì§???úÍ±∞ (Ï§ëÎ≥µ Î∞©Ï?)
    logger.handlers.clear()
    
    # ÏΩòÏÜî ?∏Îì§??
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if json_format:
            console_formatter = JSONFormatter()
        else:
            console_formatter = ColoredFormatter()
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # ?åÏùº ?∏Îì§??
    if file_output:
        # ?ºÎ∞ò ?çÏä§??Î°úÍ∑∏ ?åÏùº
        log_file = log_path / f"{log_name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # JSON Î°úÍ∑∏ ?åÏùº (?†ÌÉù??
        if json_format:
            json_log_file = log_path / f"{log_name}_{datetime.now().strftime('%Y%m%d')}.json"
            json_handler = logging.handlers.RotatingFileHandler(
                json_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            json_handler.setLevel(getattr(logging, log_level.upper()))
            json_handler.setFormatter(JSONFormatter())
            logger.addHandler(json_handler)
    
    # Î°úÍ∑∏ ?§Ï†ï ?ÑÎ£å Î©îÏãúÏßÄ (Í∞ÑÎã®??
    print(f"??Logging configured: {log_name} ({log_level})")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Í∏∞Ï°¥ Î°úÍ±∞ Í∞Ä?∏Ïò§Í∏?
    
    Args:
        name: Î°úÍ±∞ ?¥Î¶Ñ
    
    Returns:
        logging.Logger: Î°úÍ±∞
    """
    return logging.getLogger(name)


def log_with_extra(logger: logging.Logger, level: str, message: str, **kwargs):
    """
    Ï∂îÍ? ?ÑÎìú?Ä ?®Íªò Î°úÍ∑∏ Í∏∞Î°ù
    
    Args:
        logger: Î°úÍ±∞
        level: Î°úÍ∑∏ ?àÎ≤®
        message: Î°úÍ∑∏ Î©îÏãúÏßÄ
        **kwargs: Ï∂îÍ? ?ÑÎìú
    """
    extra_fields = kwargs
    getattr(logger, level.lower())(message, extra={'extra_fields': extra_fields})


# ?∏Ïùò ?®Ïàò??
def log_progress(logger: logging.Logger, current: int, total: int, item_name: str = "items"):
    """ÏßÑÌñâÎ•?Î°úÍ∑∏"""
    percentage = (current / total) * 100 if total > 0 else 0
    logger.info(f"?ìä Progress: {current}/{total} {item_name} ({percentage:.1f}%)")


def log_memory_usage(logger: logging.Logger, memory_mb: float, limit_mb: float):
    """Î©îÎ™®Î¶??¨Ïö©??Î°úÍ∑∏"""
    percentage = (memory_mb / limit_mb) * 100 if limit_mb > 0 else 0
    status = "?†Ô∏è" if percentage > 80 else "??
    logger.info(f"{status} Memory: {memory_mb:.1f}MB / {limit_mb}MB ({percentage:.1f}%)")


def log_collection_stats(logger: logging.Logger, collected: int, failed: int, total: int):
    """?òÏßë ?µÍ≥Ñ Î°úÍ∑∏"""
    success_rate = (collected / total) * 100 if total > 0 else 0
    logger.info(f"?ìà Collection stats: {collected} collected, {failed} failed, {success_rate:.1f}% success rate")


def log_checkpoint_info(logger: logging.Logger, checkpoint_data: dict):
    """Ï≤¥ÌÅ¨?¨Ïù∏???ïÎ≥¥ Î°úÍ∑∏"""
    logger.info(f"?íæ Checkpoint info:")
    logger.info(f"   Data type: {checkpoint_data.get('data_type', 'unknown')}")
    logger.info(f"   Category: {checkpoint_data.get('category', 'None')}")
    logger.info(f"   Page: {checkpoint_data.get('current_page', 0)}/{checkpoint_data.get('total_pages', 0)}")
    logger.info(f"   Collected: {checkpoint_data.get('collected_count', 0)} items")
    logger.info(f"   Memory: {checkpoint_data.get('memory_usage_mb', 0):.1f}MB")
