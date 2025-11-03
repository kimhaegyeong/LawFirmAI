#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?„ìš© ë¡œê±°

?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ê³¼ì •?ì„œ ë°œìƒ?˜ëŠ” ë¡œê·¸ë¥?ì²´ê³„?ìœ¼ë¡?ê´€ë¦¬í•©?ˆë‹¤.
- ë¡œê·¸ ?ˆë²¨ë³?ë¶„ë¦¬ (INFO, WARNING, ERROR)
- ?Œì¼ ë°?ì½˜ì†” ì¶œë ¥ ì§€??
- ë¡œê·¸ ë¡œí…Œ?´ì…˜ ê¸°ëŠ¥
- ?±ëŠ¥ ëª¨ë‹ˆ?°ë§ ë¡œê·¸
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    ?Œì¬ê²°ì •ë¡€ ?˜ì§‘??ë¡œê±° ?¤ì •
    
    Args:
        log_level: ë¡œê·¸ ?ˆë²¨ (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: ë¡œê·¸ ?Œì¼ ê²½ë¡œ (None?´ë©´ ?ë™ ?ì„±)
        console_output: ì½˜ì†” ì¶œë ¥ ?¬ë?
    
    Returns:
        ?¤ì •??ë¡œê±° ê°ì²´
    """
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # ë¡œê·¸ ?Œì¼ëª??¤ì •
    if not log_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"constitutional_decision_collection_{timestamp}.log"
    else:
        log_file = Path(log_file)
    
    # ë¡œê±° ?ì„±
    logger = logging.getLogger('constitutional_decision_collector')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # ê¸°ì¡´ ?¸ë“¤???œê±° (ì¤‘ë³µ ë°©ì?)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # ë¡œê·¸ ?¬ë§· ?¤ì •
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # ?Œì¼ ?¸ë“¤??ì¶”ê?
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # ?Œì¼?ëŠ” ëª¨ë“  ?ˆë²¨ ë¡œê·¸
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # ì½˜ì†” ?¸ë“¤??ì¶”ê?
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Windows?ì„œ UTF-8 ?˜ê²½ ?¤ì •
    if sys.platform.startswith('win'):
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        try:
            import subprocess
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
        except:
            pass
    
    return logger


def log_collection_start(logger: logging.Logger, target_count: int, resume: bool = False):
    """?˜ì§‘ ?œì‘ ë¡œê·¸"""
    logger.info("=" * 80)
    logger.info("?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?œì‘")
    logger.info("=" * 80)
    logger.info(f"ëª©í‘œ ?˜ì§‘ ê±´ìˆ˜: {target_count:,}ê±?)
    logger.info(f"?¬ì‹œ??ëª¨ë“œ: {'?? if resume else '?„ë‹ˆ??}")
    logger.info(f"?œì‘ ?œê°„: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


def log_collection_progress(
    logger: logging.Logger,
    current_count: int,
    target_count: int,
    keyword: str,
    batch_count: int = 0
):
    """?˜ì§‘ ì§„í–‰ ?í™© ë¡œê·¸"""
    progress_percentage = (current_count / target_count * 100) if target_count > 0 else 0
    
    if batch_count > 0:
        logger.info(
            f"ì§„í–‰ë¥? {progress_percentage:.1f}% ({current_count:,}/{target_count:,}) "
            f"- ?¤ì›Œ?? '{keyword}' - ë°°ì¹˜: {batch_count}ê±?
        )
    else:
        logger.info(
            f"ì§„í–‰ë¥? {progress_percentage:.1f}% ({current_count:,}/{target_count:,}) "
            f"- ?¤ì›Œ?? '{keyword}'"
        )


def log_keyword_completion(
    logger: logging.Logger,
    keyword: str,
    collected_count: int,
    total_collected: int,
    api_requests: int,
    api_errors: int
):
    """?¤ì›Œ???„ë£Œ ë¡œê·¸"""
    logger.info(f"?¤ì›Œ??'{keyword}' ?„ë£Œ - ?˜ì§‘: {collected_count}ê±? ?„ì : {total_collected:,}ê±?)
    logger.info(f"API ?”ì²­: {api_requests:,}?? ?¤ë¥˜: {api_errors:,}??)


def log_batch_save(
    logger: logging.Logger,
    batch_count: int,
    category: str,
    file_path: str
):
    """ë°°ì¹˜ ?€??ë¡œê·¸"""
    logger.debug(f"ë°°ì¹˜ ?€???„ë£Œ - {category}: {batch_count}ê±?-> {file_path}")


def log_api_error(
    logger: logging.Logger,
    operation: str,
    error: Exception,
    retry_count: int = 0
):
    """API ?¤ë¥˜ ë¡œê·¸"""
    if retry_count > 0:
        logger.warning(f"API ?¤ë¥˜ ({operation}) - ?¬ì‹œ??{retry_count}?? {error}")
    else:
        logger.error(f"API ?¤ë¥˜ ({operation}): {error}")


def log_collection_completion(
    logger: logging.Logger,
    total_collected: int,
    target_count: int,
    start_time: datetime,
    end_time: datetime,
    api_stats: dict
):
    """?˜ì§‘ ?„ë£Œ ë¡œê·¸"""
    duration = end_time - start_time
    
    logger.info("=" * 80)
    logger.info("?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?„ë£Œ")
    logger.info("=" * 80)
    logger.info(f"?˜ì§‘ ê±´ìˆ˜: {total_collected:,}ê±?)
    logger.info(f"ëª©í‘œ ê±´ìˆ˜: {target_count:,}ê±?)
    logger.info(f"?¬ì„±ë¥? {(total_collected/target_count*100):.1f}%")
    logger.info(f"?Œìš” ?œê°„: {duration}")
    logger.info(f"?œì‘ ?œê°„: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"?„ë£Œ ?œê°„: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"API ?”ì²­ ?? {api_stats.get('total_requests', 0):,}??)
    logger.info(f"API ?¤ë¥˜ ?? {api_stats.get('error_count', 0):,}??)
    logger.info(f"?¨ì? ?”ì²­ ?? {api_stats.get('remaining_requests', 0):,}??)
    logger.info("=" * 80)


def log_collection_interruption(
    logger: logging.Logger,
    total_collected: int,
    reason: str = "?¬ìš©??ì¤‘ë‹¨"
):
    """?˜ì§‘ ì¤‘ë‹¨ ë¡œê·¸"""
    logger.warning("=" * 80)
    logger.warning("?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ì¤‘ë‹¨")
    logger.warning("=" * 80)
    logger.warning(f"ì¤‘ë‹¨ ?¬ìœ : {reason}")
    logger.warning(f"?˜ì§‘??ê±´ìˆ˜: {total_collected:,}ê±?)
    logger.warning(f"ì¤‘ë‹¨ ?œê°„: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.warning("?¬ì‹œ?‘í•˜?¤ë©´ ?™ì¼??ëª…ë ¹???¤ì‹œ ?¤í–‰?˜ì„¸??")
    logger.warning("=" * 80)


def log_error_with_traceback(
    logger: logging.Logger,
    error: Exception,
    context: str = ""
):
    """?ì„¸ ?¤ë¥˜ ë¡œê·¸ (?¤íƒ ?¸ë ˆ?´ìŠ¤ ?¬í•¨)"""
    if context:
        logger.error(f"?¤ë¥˜ ë°œìƒ ({context}): {error}")
    else:
        logger.error(f"?¤ë¥˜ ë°œìƒ: {error}")
    
    logger.error("?¤íƒ ?¸ë ˆ?´ìŠ¤:")
    logger.error(traceback.format_exc())


def log_performance_metrics(
    logger: logging.Logger,
    operation: str,
    duration: float,
    count: int = 0
):
    """?±ëŠ¥ ë©”íŠ¸ë¦?ë¡œê·¸"""
    if count > 0:
        rate = count / duration if duration > 0 else 0
        logger.info(f"?±ëŠ¥ ë©”íŠ¸ë¦?- {operation}: {duration:.2f}ì´? {count}ê±? {rate:.2f}ê±?ì´?)
    else:
        logger.info(f"?±ëŠ¥ ë©”íŠ¸ë¦?- {operation}: {duration:.2f}ì´?)


def log_memory_usage(logger: logging.Logger, context: str = ""):
    """ë©”ëª¨ë¦??¬ìš©??ë¡œê·¸"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        if context:
            logger.debug(f"ë©”ëª¨ë¦??¬ìš©??({context}): {memory_mb:.2f}MB")
        else:
            logger.debug(f"ë©”ëª¨ë¦??¬ìš©?? {memory_mb:.2f}MB")
    except ImportError:
        logger.debug("psutil???¤ì¹˜?˜ì? ?Šì•„ ë©”ëª¨ë¦??¬ìš©?‰ì„ ?•ì¸?????†ìŠµ?ˆë‹¤.")
    except Exception as e:
        logger.debug(f"ë©”ëª¨ë¦??¬ìš©???•ì¸ ?¤íŒ¨: {e}")


def log_checkpoint_save(logger: logging.Logger, checkpoint_file: str, data_count: int):
    """ì²´í¬?¬ì¸???€??ë¡œê·¸"""
    logger.debug(f"ì²´í¬?¬ì¸???€?? {checkpoint_file} ({data_count}ê±?")


def log_checkpoint_load(logger: logging.Logger, checkpoint_file: str, data_count: int):
    """ì²´í¬?¬ì¸??ë¡œë“œ ë¡œê·¸"""
    logger.info(f"ì²´í¬?¬ì¸??ë¡œë“œ: {checkpoint_file} ({data_count}ê±?")


def log_api_rate_limit(
    logger: logging.Logger,
    remaining_requests: int,
    reset_time: Optional[str] = None
):
    """API ?”ì²­ ?œí•œ ë¡œê·¸"""
    if remaining_requests < 100:
        logger.warning(f"API ?”ì²­ ?œë„ ë¶€ì¡? {remaining_requests}???¨ìŒ")
        if reset_time:
            logger.warning(f"?”ì²­ ?œë„ ë¦¬ì…‹ ?œê°„: {reset_time}")
    elif remaining_requests < 500:
        logger.info(f"API ?”ì²­ ?œë„: {remaining_requests}???¨ìŒ")


def log_duplicate_detection(
    logger: logging.Logger,
    decision_id: str,
    keyword: str
):
    """ì¤‘ë³µ ê°ì? ë¡œê·¸"""
    logger.debug(f"ì¤‘ë³µ ê°ì? - ê²°ì •ë¡€ ID: {decision_id}, ?¤ì›Œ?? '{keyword}'")


def log_decision_classification(
    logger: logging.Logger,
    decision_id: str,
    decision_type: str,
    confidence: float = 0.0
):
    """ê²°ì •? í˜• ë¶„ë¥˜ ë¡œê·¸"""
    if confidence > 0:
        logger.debug(f"ê²°ì •? í˜• ë¶„ë¥˜ - ID: {decision_id}, ? í˜•: {decision_type}, ? ë¢°?? {confidence:.2f}")
    else:
        logger.debug(f"ê²°ì •? í˜• ë¶„ë¥˜ - ID: {decision_id}, ? í˜•: {decision_type}")


def log_summary_generation(
    logger: logging.Logger,
    summary_file: str,
    total_decisions: int,
    decision_types: dict
):
    """?”ì•½ ?ì„± ë¡œê·¸"""
    logger.info(f"?˜ì§‘ ê²°ê³¼ ?”ì•½ ?ì„±: {summary_file}")
    logger.info(f"ì´?ê²°ì •ë¡€ ?? {total_decisions:,}ê±?)
    
    if decision_types:
        logger.info("ê²°ì •? í˜•ë³?ë¶„í¬:")
        for decision_type, count in sorted(decision_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {decision_type}: {count:,}ê±?)


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """ì§„í–‰ë¥?ë°??ì„±"""
    if total <= 0:
        return "[" + " " * width + "] 0.0%"
    
    percentage = current / total
    filled_width = int(width * percentage)
    bar = "?? * filled_width + "?? * (width - filled_width)
    return f"[{bar}] {percentage:.1f}%"


def log_progress_bar(
    logger: logging.Logger,
    current: int,
    total: int,
    prefix: str = "ì§„í–‰ë¥?,
    width: int = 50
):
    """ì§„í–‰ë¥?ë°?ë¡œê·¸"""
    progress_bar = create_progress_bar(current, total, width)
    logger.info(f"{prefix}: {progress_bar} ({current:,}/{total:,})")
