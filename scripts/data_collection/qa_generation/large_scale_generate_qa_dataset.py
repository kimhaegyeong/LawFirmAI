#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?€ê·œëª¨ Q&A ?°ì´?°ì…‹ ?ì„± ?¤í¬ë¦½íŠ¸

ëª©í‘œ: 3,000ê°??´ìƒ??Q&A ???ì„±
- ëª¨ë“  ?°ì´???ŒìŠ¤ ?œìš©
- ?¤ì–‘??ì§ˆë¬¸ ?¨í„´ ?ì„±
- ?ˆì§ˆ ê²€ì¦?ê°•í™”
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import random
import re

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/large_scale_qa_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ?€ê·œëª¨ Q&A ?ì„± ?œí”Œë¦?
LARGE_SCALE_QA_TEMPLATES = {
    'law_definition': [
        "{law_name}?´ë? ë¬´ì—‡?¸ê???",
        "{law_name}???•ì˜ë¥??¤ëª…?´ì£¼?¸ìš”.",
        "{law_name}??ëª©ì ?€ ë¬´ì—‡?¸ê???",
        "{law_name}???ìš© ë²”ìœ„???´ë–»ê²??˜ë‚˜??",
        "{law_name}???´ë–¤ ë²•ë¥ ?¸ê???",
        "{law_name}??ì£¼ìš” ?´ìš©?€ ë¬´ì—‡?¸ê???",
        "{law_name}??ê·œì •?˜ëŠ” ê²ƒì? ë¬´ì—‡?¸ê???",
        "{law_name}??ë²•ì  ?±ê²©?€ ë¬´ì—‡?¸ê???",
        "{law_name}???…ë²• ì·¨ì???ë¬´ì—‡?¸ê???",
        "{law_name}???µì‹¬ ê°œë…?€ ë¬´ì—‡?¸ê???",
        "{law_name}??ë³´í˜¸?˜ëŠ” ê²ƒì? ë¬´ì—‡?¸ê???",
        "{law_name}??ë²•ì  ê·¼ê±°??ë¬´ì—‡?¸ê???",
        "{law_name}???ìš© ?€?ì? ?„êµ¬?¸ê???",
        "{law_name}??ì£¼ìš” ?ì¹™?€ ë¬´ì—‡?¸ê???",
        "{law_name}??ë²•ì  ?¨ê³¼??ë¬´ì—‡?¸ê???"
    ],
    'law_article': [
        "{law_name} ??article}ì¡°ì˜ ?´ìš©???¤ëª…?´ì£¼?¸ìš”.",
        "{law_name} ??article}ì¡°ì—??ê·œì •?˜ëŠ” ?´ìš©?€ ë¬´ì—‡?¸ê???",
        "{law_name} ??article}ì¡°ì˜ ?”ê±´?€ ë¬´ì—‡?¸ê???",
        "{law_name} ??article}ì¡°ì˜ ?¨ê³¼??ë¬´ì—‡?¸ê???",
        "??article}ì¡°ëŠ” ë¬´ì—‡??ê·œì •?˜ê³  ?ˆë‚˜??",
        "??article}ì¡°ì˜ ?µì‹¬ ?´ìš©?€ ë¬´ì—‡?¸ê???",
        "??article}ì¡°ì—??ì¤‘ìš”??ë¶€ë¶„ì? ë¬´ì—‡?¸ê???",
        "??article}ì¡°ì˜ ë²•ì  ?˜ë???ë¬´ì—‡?¸ê???",
        "??article}ì¡°ì˜ ?ìš© ë²”ìœ„???´ë–»ê²??˜ë‚˜??",
        "??article}ì¡°ì˜ ?”êµ¬?¬í•­?€ ë¬´ì—‡?¸ê???",
        "??article}ì¡°ì˜ ê¸ˆì??¬í•­?€ ë¬´ì—‡?¸ê???",
        "??article}ì¡°ì˜ ?ˆìš©?¬í•­?€ ë¬´ì—‡?¸ê???",
        "??article}ì¡°ì˜ ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
        "??article}ì¡°ì˜ ê¸°ì??€ ë¬´ì—‡?¸ê???",
        "??article}ì¡°ì˜ ì¡°ê±´?€ ë¬´ì—‡?¸ê???"
    ],
    'law_article_title': [
        "{law_name} ??article}ì¡°ì˜ ?œëª©?€ ë¬´ì—‡?¸ê???",
        "??article}ì¡°ì˜ ?œëª©???Œë ¤ì£¼ì„¸??",
        "??article}ì¡°ëŠ” ?´ë–¤ ?´ìš©???¤ë£¨?˜ìš”?",
        "??article}ì¡°ì˜ ì£¼ì œ??ë¬´ì—‡?¸ê???",
        "??article}ì¡°ì˜ ëª…ì¹­?€ ë¬´ì—‡?¸ê???",
        "??article}ì¡°ì˜ ?œì œ??ë¬´ì—‡?¸ê???"
    ],
    'law_keyword': [
        "{keyword}???€??ë²•ì  ê·¼ê±°??ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?”ê±´?€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?¨ê³¼??ë¬´ì—‡?¸ê???",
        "{keyword}???€??ë²•ì  ?´ì„?€ ?´ë–»ê²??˜ë‚˜??",
        "{keyword}??ë²•ì ?¼ë¡œ ?´ë–»ê²??•ì˜?˜ë‚˜??",
        "{keyword}??ë²•ì  ?˜ë???ë¬´ì—‡?¸ê???",
        "{keyword}??ê´€??ë²•ë¥  ê·œì •?€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ì§€?„ëŠ” ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?±ê²©?€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?”êµ¬?¬í•­?€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
        "{keyword}??ë²•ì  ê¸°ì??€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ì¡°ê±´?€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ê¸ˆì???ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?ˆìš©?€ ë¬´ì—‡?¸ê???"
    ],
    'precedent_issue': [
        "{case_name} ?¬ê±´???ì ?€ ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´?ì„œ ?¤ë£¬ ë¬¸ì œ??ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´???µì‹¬ ?ì ???¤ëª…?´ì£¼?¸ìš”.",
        "{case_name} ?¬ê±´??ë²•ì  ?ì ?€ ë¬´ì—‡?¸ê???",
        "???¬ê±´??ì£¼ìš” ë¬¸ì œ?ì? ë¬´ì—‡?¸ê???",
        "ë²•ì›???ë‹¨?´ì•¼ ???ì ?€ ë¬´ì—‡?¸ê???",
        "?¬ê±´???µì‹¬?€ ë¬´ì—‡?¸ê???",
        "???¬ê±´??ë²•ì  ?ì ?€ ë¬´ì—‡?¸ê???",
        "?¬ê±´?ì„œ ?¤ë£¬ ?µì‹¬ ë¬¸ì œ??ë¬´ì—‡?¸ê???",
        "ë²•ì›???´ê²°?´ì•¼ ??ë¬¸ì œ??ë¬´ì—‡?¸ê???",
        "?¬ê±´??ì£¼ìš” ?ì ?€ ë¬´ì—‡?¸ê???",
        "???¬ê±´??ë²•ì  ë¬¸ì œ??ë¬´ì—‡?¸ê???",
        "?¬ê±´???µì‹¬ ë²•ì  ?ì ?€ ë¬´ì—‡?¸ê???",
        "???¬ê±´??ì£¼ìš” ë²•ì  ë¬¸ì œ??ë¬´ì—‡?¸ê???",
        "?¬ê±´?ì„œ ?œê¸°??ë¬¸ì œ??ë¬´ì—‡?¸ê???"
    ],
    'precedent_decision': [
        "{case_name} ?¬ê±´???ê²° ?´ìš©?€ ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´?ì„œ ë²•ì›???´ë¦° ê²°ë¡ ?€ ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´???ê²° ?”ì???ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´??ë²•ì› ?ë‹¨???¤ëª…?´ì£¼?¸ìš”.",
        "ë²•ì›???ë‹¨ ê·¼ê±°??ë¬´ì—‡?¸ê???",
        "?ê²°???µì‹¬ ?´ìš©?€ ë¬´ì—‡?¸ê???",
        "ë²•ì›???´ë¦° ê²°ë¡ ???”ì???ë¬´ì—‡?¸ê???",
        "???¬ê±´???ê²° ?´ìš©?€ ë¬´ì—‡?¸ê???",
        "ë²•ì›???ë‹¨ ?´ìš©???¤ëª…?´ì£¼?¸ìš”.",
        "?ê²°??ì£¼ìš” ?´ìš©?€ ë¬´ì—‡?¸ê???",
        "ë²•ì›???´ë¦° ?ë‹¨?€ ë¬´ì—‡?¸ê???",
        "???¬ê±´???ê²° ?”ì???ë¬´ì—‡?¸ê???",
        "ë²•ì›??ê²°ë¡ ?€ ë¬´ì—‡?¸ê???",
        "?ê²°??ë²•ì  ?˜ë???ë¬´ì—‡?¸ê???",
        "ë²•ì›???ë‹¨ ê·¼ê±°??ë¬´ì—‡?¸ê???"
    ],
    'precedent_court': [
        "{case_name} ?¬ê±´???´ë‹¹??ë²•ì›?€ ?´ë””?¸ê???",
        "???¬ê±´??ì²˜ë¦¬??ë²•ì›?€ ë¬´ì—‡?¸ê???",
        "?ê²°???´ë¦° ë²•ì›?€ ?´ë””?¸ê???",
        "?¬ê±´???´ë‹¹??ë²•ì›???´ë¦„?€ ë¬´ì—‡?¸ê???",
        "???¬ê±´???´ë‹¹ ë²•ì›?€ ?´ë””?¸ê???",
        "?ê²°???´ë¦° ë²•ì›???´ë¦„?€ ë¬´ì—‡?¸ê???",
        "?¬ê±´??ì²˜ë¦¬??ë²•ì›?€ ?´ë””?¸ê???",
        "???¬ê±´??ë²•ì›?€ ë¬´ì—‡?¸ê???",
        "?ê²°???´ë¦° ë²•ì›?€ ë¬´ì—‡?¸ê???",
        "?¬ê±´???´ë‹¹??ë²•ì›?€ ë¬´ì—‡?¸ê???"
    ],
    'precedent_date': [
        "{case_name} ?¬ê±´???ê²°?¼ì? ?¸ì œ?¸ê???",
        "???¬ê±´???ê²° ? ì§œ???¸ì œ?¸ê???",
        "?ê²°???´ë ¤ì§?? ì§œ???¸ì œ?¸ê???",
        "?¬ê±´???ê²°?¼ì„ ?Œë ¤ì£¼ì„¸??",
        "???¬ê±´???ê²°??? ì§œ???¸ì œ?¸ê???",
        "?ê²°?¼ì? ?¸ì œ?¸ê???",
        "?¬ê±´???ê²° ?œê¸°???¸ì œ?¸ê???",
        "???¬ê±´???ê²° ?œì ?€ ?¸ì œ?¸ê???"
    ],
    'precedent_case_number': [
        "{case_name} ?¬ê±´???¬ê±´ë²ˆí˜¸??ë¬´ì—‡?¸ê???",
        "???¬ê±´???¬ê±´ë²ˆí˜¸ë¥??Œë ¤ì£¼ì„¸??",
        "?¬ê±´ë²ˆí˜¸??ë¬´ì—‡?¸ê???",
        "???¬ê±´??ë²ˆí˜¸??ë¬´ì—‡?¸ê???",
        "?¬ê±´??ë²ˆí˜¸ë¥??Œë ¤ì£¼ì„¸??",
        "???¬ê±´???¬ê±´ë²ˆí˜¸??ë¬´ì—‡?¸ê???",
        "?¬ê±´??ê³ ìœ ë²ˆí˜¸??ë¬´ì—‡?¸ê???",
        "???¬ê±´???ë³„ë²ˆí˜¸??ë¬´ì—‡?¸ê???"
    ],
    'constitutional_issue': [
        "{case_name} ?¬ê±´???Œë²•???ì ?€ ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´?ì„œ ?¤ë£¬ ê¸°ë³¸ê¶?ë¬¸ì œ??ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´???Œë²•?¬íŒ???ë‹¨ ?€?ì? ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´???Œë²•???˜ë???ë¬´ì—‡?¸ê???",
        "?Œë²•?¬íŒ?Œê? ?ë‹¨???ì ?€ ë¬´ì—‡?¸ê???",
        "ê¸°ë³¸ê¶?ì¹¨í•´ ?¬ë?ê°€ ë¬¸ì œ??ê²ƒì? ë¬´ì—‡?¸ê???",
        "???¬ê±´???Œë²•???ì ?€ ë¬´ì—‡?¸ê???",
        "?Œë²•?¬íŒ?Œê? ?¤ë£¬ ë¬¸ì œ??ë¬´ì—‡?¸ê???",
        "?¬ê±´???Œë²•???˜ë???ë¬´ì—‡?¸ê???",
        "???¬ê±´??ê¸°ë³¸ê¶?ë¬¸ì œ??ë¬´ì—‡?¸ê???",
        "?Œë²•?¬íŒ?Œê? ?ë‹¨???€?ì? ë¬´ì—‡?¸ê???",
        "?¬ê±´???Œë²•???ì ?€ ë¬´ì—‡?¸ê???",
        "???¬ê±´???Œë²•??ë¬¸ì œ??ë¬´ì—‡?¸ê???",
        "?Œë²•?¬íŒ?Œê? ?´ê²°??ë¬¸ì œ??ë¬´ì—‡?¸ê???",
        "?¬ê±´???Œë²•???µì‹¬?€ ë¬´ì—‡?¸ê???"
    ],
    'constitutional_decision': [
        "{case_name} ?¬ê±´???Œë²•?¬íŒ??ê²°ì •?€ ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´?ì„œ ?Œë²•?¬íŒ?Œê? ?´ë¦° ê²°ë¡ ?€ ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´???Œë²•?¬íŒ???ë‹¨???¤ëª…?´ì£¼?¸ìš”.",
        "{case_name} ?¬ê±´???Œë²•???ë‹¨?€ ë¬´ì—‡?¸ê???",
        "?Œë²•?¬íŒ?Œì˜ ê²°ì • ?”ì???ë¬´ì—‡?¸ê???",
        "?Œë²•?¬íŒ?Œê? ?´ë¦° ê²°ë¡ ?€ ë¬´ì—‡?¸ê???",
        "???¬ê±´???Œë²•?¬íŒ??ê²°ì •?€ ë¬´ì—‡?¸ê???",
        "?Œë²•?¬íŒ?Œì˜ ?ë‹¨ ?´ìš©?€ ë¬´ì—‡?¸ê???",
        "?¬ê±´???Œë²•?¬íŒ??ê²°ì •???¤ëª…?´ì£¼?¸ìš”.",
        "?Œë²•?¬íŒ?Œê? ?´ë¦° ?ë‹¨?€ ë¬´ì—‡?¸ê???",
        "???¬ê±´???Œë²•??ê²°ì •?€ ë¬´ì—‡?¸ê???",
        "?Œë²•?¬íŒ?Œì˜ ê²°ë¡ ?€ ë¬´ì—‡?¸ê???",
        "?¬ê±´???Œë²•???ë‹¨?€ ë¬´ì—‡?¸ê???",
        "?Œë²•?¬íŒ?Œì˜ ê²°ì • ?´ìš©?€ ë¬´ì—‡?¸ê???",
        "???¬ê±´???Œë²•?¬íŒ???ë‹¨?€ ë¬´ì—‡?¸ê???"
    ],
    'interpretation_question': [
        "{topic}???€??ë²•ë ¹?´ì„?€ ?´ë–»ê²??˜ë‚˜??",
        "{topic}??ë²•ì  ?´ì„ ê¸°ì??€ ë¬´ì—‡?¸ê???",
        "{topic}???€??ì¤‘ì•™ë¶€ì²˜ì˜ ?´ì„?€ ë¬´ì—‡?¸ê???",
        "{topic}??ë²•ë ¹ ?ìš© ê¸°ì??€ ë¬´ì—‡?¸ê???",
        "{topic}???€??ê³µì‹ ?´ì„?€ ë¬´ì—‡?¸ê???",
        "{topic}??ë²•ì  ?˜ë???ë¬´ì—‡?¸ê???",
        "{topic}???€??ë²•ì  ?´ì„?€ ë¬´ì—‡?¸ê???",
        "{topic}???´ì„ ê¸°ì??€ ë¬´ì—‡?¸ê???",
        "{topic}???€???•ë? ?´ì„?€ ë¬´ì—‡?¸ê???",
        "{topic}??ë²•ë ¹ ?´ì„?€ ?´ë–»ê²??˜ë‚˜??",
        "{topic}???€??ê³µì‹???´ì„?€ ë¬´ì—‡?¸ê???",
        "{topic}??ë²•ì  ?´ì„?€ ë¬´ì—‡?¸ê???",
        "{topic}???€??ë²•ë ¹???´ì„?€ ë¬´ì—‡?¸ê???",
        "{topic}???´ì„ ?´ìš©?€ ë¬´ì—‡?¸ê???",
        "{topic}???€??ë²•ì  ?´ì„ ?´ìš©?€ ë¬´ì—‡?¸ê???"
    ],
    'general_legal': [
        "{keyword}???€??ë²•ì  ê·¼ê±°??ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?”ê±´?€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?¨ê³¼??ë¬´ì—‡?¸ê???",
        "{keyword}???€??ë²•ì  ?´ì„?€ ?´ë–»ê²??˜ë‚˜??",
        "{keyword}??ë²•ì ?¼ë¡œ ?´ë–»ê²??•ì˜?˜ë‚˜??",
        "{keyword}??ë²•ì  ?˜ë???ë¬´ì—‡?¸ê???",
        "{keyword}??ê´€??ë²•ë¥  ê·œì •?€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ì§€?„ëŠ” ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?±ê²©?€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?”êµ¬?¬í•­?€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
        "{keyword}??ë²•ì  ê¸°ì??€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ì¡°ê±´?€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ê¸ˆì???ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?ˆìš©?€ ë¬´ì—‡?¸ê???"
    ]
}

# ?€ê·œëª¨ ?µë? ?ì„± ?œí”Œë¦?
LARGE_SCALE_ANSWER_TEMPLATES = {
    'law_definition': [
        "{law_name}?€ {definition}??ëª©ì ?¼ë¡œ ?˜ëŠ” ë²•ë¥ ?…ë‹ˆ??",
        "{law_name}??{definition}??ê´€???¬í•­??ê·œì •??ë²•ë¥ ?…ë‹ˆ??",
        "{law_name}??ëª©ì ?€ {definition}?…ë‹ˆ??",
        "{law_name}?€ {definition}??ê·œì •?˜ëŠ” ë²•ë¥ ?…ë‹ˆ??",
        "{law_name}??{definition}???€??ë²•ì  ê·¼ê±°ë¥??œê³µ?©ë‹ˆ??",
        "{law_name}???µì‹¬?€ {definition}?…ë‹ˆ??",
        "{law_name}?€ {definition}??ë³´ì¥?˜ëŠ” ë²•ë¥ ?…ë‹ˆ??",
        "{law_name}??{definition}??ê´€??ë²•ì  ê·œì •?…ë‹ˆ??",
        "{law_name}??ê¸°ë³¸ ?ì¹™?€ {definition}?…ë‹ˆ??",
        "{law_name}?€ {definition}???¤í˜„?˜ëŠ” ë²•ë¥ ?…ë‹ˆ??"
    ],
    'law_article': [
        "{law_name} ??article}ì¡°ì— ?°ë¥´ë©? {content}?…ë‹ˆ??",
        "??article}ì¡°ì—?œëŠ” {content}?¼ê³  ê·œì •?˜ê³  ?ˆìŠµ?ˆë‹¤.",
        "{law_name} ??article}ì¡°ì˜ ?´ìš©?€ {content}?…ë‹ˆ??",
        "??article}ì¡°ì— ê·œì •???´ìš©?€ {content}?…ë‹ˆ??",
        "??article}ì¡°ëŠ” {content}ë¥?ëª…ì‹œ?˜ê³  ?ˆìŠµ?ˆë‹¤.",
        "??article}ì¡°ì—??ì¤‘ìš”??ê²ƒì? {content}?…ë‹ˆ??",
        "??article}ì¡°ì— ?°ë¥´ë©?{content}?…ë‹ˆ??",
        "??article}ì¡°ì—??ê·œì •?˜ëŠ” ë°”ëŠ” {content}?…ë‹ˆ??",
        "??article}ì¡°ì˜ ?µì‹¬ ?´ìš©?€ {content}?…ë‹ˆ??",
        "??article}ì¡°ì—??ëª…ì‹œ?˜ëŠ” ê²ƒì? {content}?…ë‹ˆ??"
    ],
    'law_article_title': [
        "{law_name} ??article}ì¡°ì˜ ?œëª©?€ '{title}'?…ë‹ˆ??",
        "??article}ì¡°ì˜ ?œëª©?€ '{title}'?…ë‹ˆ??",
        "??article}ì¡°ëŠ” '{title}'??ê´€???´ìš©?…ë‹ˆ??",
        "??article}ì¡°ì˜ ì£¼ì œ??'{title}'?…ë‹ˆ??",
        "??article}ì¡°ì˜ ëª…ì¹­?€ '{title}'?…ë‹ˆ??",
        "??article}ì¡°ì˜ ?œì œ??'{title}'?…ë‹ˆ??"
    ],
    'law_keyword': [
        "{law_name}???°ë¥´ë©?{keyword}??{content}?…ë‹ˆ??",
        "{keyword}???€??ë²•ì  ?•ì˜??{content}?…ë‹ˆ??",
        "{keyword}??ë²•ì  ?˜ë???{content}?…ë‹ˆ??",
        "{keyword}??{content}ë¡?ê·œì •?˜ì–´ ?ˆìŠµ?ˆë‹¤.",
        "{keyword}??ê´€??ë²•ë¥  ê·œì •?€ {content}?…ë‹ˆ??",
        "{keyword}??ë²•ì  ê°œë…?€ {content}?…ë‹ˆ??",
        "{keyword}??{content}ë¡??•ì˜?©ë‹ˆ??",
        "{keyword}??ë²•ì  ?´ìš©?€ {content}?…ë‹ˆ??",
        "{keyword}???€??ë²•ì  ê·œì •?€ {content}?…ë‹ˆ??",
        "{keyword}??ë²•ì  ?•ì˜??{content}?…ë‹ˆ??"
    ],
    'precedent_issue': [
        "{case_name} ?¬ê±´???ì ?€ {issue}?…ë‹ˆ??",
        "???¬ê±´?ì„œ ?¤ë£¬ ë¬¸ì œ??{issue}?…ë‹ˆ??",
        "ë²•ì›???ë‹¨???ì ?€ {issue}?…ë‹ˆ??",
        "?¬ê±´???µì‹¬ ?ì ?€ {issue}?…ë‹ˆ??",
        "???¬ê±´??ì£¼ìš” ë¬¸ì œ??{issue}?…ë‹ˆ??",
        "ë²•ì  ?ì ?€ {issue}?…ë‹ˆ??",
        "?¬ê±´???µì‹¬?€ {issue}?…ë‹ˆ??",
        "???¬ê±´??ë²•ì  ë¬¸ì œ??{issue}?…ë‹ˆ??",
        "?¬ê±´?ì„œ ?œê¸°??ë¬¸ì œ??{issue}?…ë‹ˆ??",
        "ë²•ì›???´ê²°?´ì•¼ ??ë¬¸ì œ??{issue}?…ë‹ˆ??"
    ],
    'precedent_decision': [
        "{case_name} ?¬ê±´?ì„œ ë²•ì›?€ {decision}?¼ê³  ?ë‹¨?ˆìŠµ?ˆë‹¤.",
        "ë²•ì›???ê²° ?´ìš©?€ {decision}?…ë‹ˆ??",
        "???¬ê±´???ê²° ?”ì???{decision}?…ë‹ˆ??",
        "ë²•ì›???´ë¦° ê²°ë¡ ?€ {decision}?…ë‹ˆ??",
        "?ê²°???µì‹¬?€ {decision}?…ë‹ˆ??",
        "ë²•ì›???ë‹¨?€ {decision}?…ë‹ˆ??",
        "???¬ê±´???ê²° ?´ìš©?€ {decision}?…ë‹ˆ??",
        "ë²•ì›???´ë¦° ?ë‹¨?€ {decision}?…ë‹ˆ??",
        "?¬ê±´???ê²° ?”ì???{decision}?…ë‹ˆ??",
        "ë²•ì›??ê²°ë¡ ?€ {decision}?…ë‹ˆ??"
    ],
    'precedent_court': [
        "{case_name} ?¬ê±´???´ë‹¹??ë²•ì›?€ {court}?…ë‹ˆ??",
        "???¬ê±´??ì²˜ë¦¬??ë²•ì›?€ {court}?…ë‹ˆ??",
        "?ê²°???´ë¦° ë²•ì›?€ {court}?…ë‹ˆ??",
        "?¬ê±´???´ë‹¹??ë²•ì›?€ {court}?…ë‹ˆ??",
        "???¬ê±´???´ë‹¹ ë²•ì›?€ {court}?…ë‹ˆ??",
        "?ê²°???´ë¦° ë²•ì›?€ {court}?…ë‹ˆ??",
        "?¬ê±´??ì²˜ë¦¬??ë²•ì›?€ {court}?…ë‹ˆ??",
        "???¬ê±´??ë²•ì›?€ {court}?…ë‹ˆ??",
        "?ê²°???´ë¦° ë²•ì›?€ {court}?…ë‹ˆ??",
        "?¬ê±´???´ë‹¹??ë²•ì›?€ {court}?…ë‹ˆ??"
    ],
    'precedent_date': [
        "{case_name} ?¬ê±´???ê²°?¼ì? {date}?…ë‹ˆ??",
        "???¬ê±´???ê²° ? ì§œ??{date}?…ë‹ˆ??",
        "?ê²°???´ë ¤ì§?? ì§œ??{date}?…ë‹ˆ??",
        "?¬ê±´???ê²°?¼ì? {date}?…ë‹ˆ??",
        "???¬ê±´???ê²°??? ì§œ??{date}?…ë‹ˆ??",
        "?ê²°?¼ì? {date}?…ë‹ˆ??",
        "?¬ê±´???ê²° ?œê¸°??{date}?…ë‹ˆ??",
        "???¬ê±´???ê²° ?œì ?€ {date}?…ë‹ˆ??"
    ],
    'precedent_case_number': [
        "{case_name} ?¬ê±´???¬ê±´ë²ˆí˜¸??{case_number}?…ë‹ˆ??",
        "???¬ê±´???¬ê±´ë²ˆí˜¸??{case_number}?…ë‹ˆ??",
        "?¬ê±´ë²ˆí˜¸??{case_number}?…ë‹ˆ??",
        "???¬ê±´??ë²ˆí˜¸??{case_number}?…ë‹ˆ??",
        "?¬ê±´??ë²ˆí˜¸??{case_number}?…ë‹ˆ??",
        "???¬ê±´???¬ê±´ë²ˆí˜¸??{case_number}?…ë‹ˆ??",
        "?¬ê±´??ê³ ìœ ë²ˆí˜¸??{case_number}?…ë‹ˆ??",
        "???¬ê±´???ë³„ë²ˆí˜¸??{case_number}?…ë‹ˆ??"
    ],
    'constitutional_issue': [
        "{case_name} ?¬ê±´???Œë²•???ì ?€ {issue}?…ë‹ˆ??",
        "???¬ê±´?ì„œ ?¤ë£¬ ê¸°ë³¸ê¶?ë¬¸ì œ??{issue}?…ë‹ˆ??",
        "?Œë²•?¬íŒ?Œê? ?ë‹¨???€?ì? {issue}?…ë‹ˆ??",
        "?¬ê±´???Œë²•???˜ë???{issue}?…ë‹ˆ??",
        "?Œë²•???ì ?€ {issue}?…ë‹ˆ??",
        "ê¸°ë³¸ê¶?ë¬¸ì œ??{issue}?…ë‹ˆ??",
        "???¬ê±´???Œë²•???ì ?€ {issue}?…ë‹ˆ??",
        "?Œë²•?¬íŒ?Œê? ?¤ë£¬ ë¬¸ì œ??{issue}?…ë‹ˆ??",
        "?¬ê±´???Œë²•???˜ë???{issue}?…ë‹ˆ??",
        "???¬ê±´??ê¸°ë³¸ê¶?ë¬¸ì œ??{issue}?…ë‹ˆ??"
    ],
    'constitutional_decision': [
        "{case_name} ?¬ê±´?ì„œ ?Œë²•?¬íŒ?ŒëŠ” {decision}?¼ê³  ê²°ì •?ˆìŠµ?ˆë‹¤.",
        "?Œë²•?¬íŒ?Œì˜ ê²°ì • ?´ìš©?€ {decision}?…ë‹ˆ??",
        "???¬ê±´???Œë²•?¬íŒ???ë‹¨?€ {decision}?…ë‹ˆ??",
        "?Œë²•?¬íŒ?Œê? ?´ë¦° ê²°ë¡ ?€ {decision}?…ë‹ˆ??",
        "?Œë²•?¬íŒ?Œì˜ ê²°ì • ?”ì???{decision}?…ë‹ˆ??",
        "?Œë²•???ë‹¨?€ {decision}?…ë‹ˆ??",
        "???¬ê±´???Œë²•?¬íŒ??ê²°ì •?€ {decision}?…ë‹ˆ??",
        "?Œë²•?¬íŒ?Œì˜ ?ë‹¨ ?´ìš©?€ {decision}?…ë‹ˆ??",
        "?¬ê±´???Œë²•?¬íŒ??ê²°ì •?€ {decision}?…ë‹ˆ??",
        "?Œë²•?¬íŒ?Œê? ?´ë¦° ?ë‹¨?€ {decision}?…ë‹ˆ??"
    ],
    'interpretation_question': [
        "{topic}???€??ë²•ë ¹?´ì„?€ {interpretation}?…ë‹ˆ??",
        "{topic}??ë²•ì  ?´ì„ ê¸°ì??€ {interpretation}?…ë‹ˆ??",
        "ì¤‘ì•™ë¶€ì²˜ì˜ ?´ì„???°ë¥´ë©?{interpretation}?…ë‹ˆ??",
        "{topic}??ë²•ë ¹ ?ìš© ê¸°ì??€ {interpretation}?…ë‹ˆ??",
        "ê³µì‹ ?´ì„?€ {interpretation}?…ë‹ˆ??",
        "{topic}??ë²•ì  ?˜ë???{interpretation}?…ë‹ˆ??",
        "{topic}???€??ë²•ì  ?´ì„?€ {interpretation}?…ë‹ˆ??",
        "{topic}???´ì„ ê¸°ì??€ {interpretation}?…ë‹ˆ??",
        "{topic}???€???•ë? ?´ì„?€ {interpretation}?…ë‹ˆ??",
        "{topic}??ë²•ë ¹ ?´ì„?€ {interpretation}?…ë‹ˆ??"
    ],
    'general_legal': [
        "{keyword}???€??ë²•ì  ê·¼ê±°??{basis}?…ë‹ˆ??",
        "{keyword}??ë²•ì  ?”ê±´?€ {requirement}?…ë‹ˆ??",
        "{keyword}??ë²•ì  ?¨ê³¼??{effect}?…ë‹ˆ??",
        "{keyword}???€??ë²•ì  ?´ì„?€ {interpretation}?…ë‹ˆ??",
        "{keyword}??{definition}ë¡??•ì˜?©ë‹ˆ??",
        "{keyword}??ë²•ì  ?˜ë???{meaning}?…ë‹ˆ??",
        "{keyword}??ë²•ì  ?±ê²©?€ {nature}?…ë‹ˆ??",
        "{keyword}??ë²•ì  ì§€?„ëŠ” {status}?…ë‹ˆ??",
        "{keyword}??ë²•ì  ?”êµ¬?¬í•­?€ {requirement}?…ë‹ˆ??",
        "{keyword}??ë²•ì  ?ˆì°¨??{procedure}?…ë‹ˆ??"
    ]
}


class LargeScaleQADatasetGenerator:
    """?€ê·œëª¨ Q&A ?°ì´?°ì…‹ ?ì„± ?´ë˜??""
    
    def __init__(self):
        self.qa_pairs = []
        self.logger = logging.getLogger(__name__)
        
    def generate_law_qa_pairs(self, law_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ë²•ë ¹ ?°ì´?°ì—??Q&A ???ì„± (?€ê·œëª¨ ë²„ì „)"""
        qa_pairs = []
        
        try:
            law_name = law_data.get('law_name', '')
            articles = law_data.get('articles', [])
            cleaned_content = law_data.get('cleaned_content', '')
            
            if not law_name:
                return qa_pairs
            
            # 1. ë²•ë ¹ ?•ì˜ ê´€??Q&A (??ë§ì? ?¨í„´)
            if cleaned_content:
                definition = self._extract_law_definition(cleaned_content)
                if definition:
                    for template in LARGE_SCALE_QA_TEMPLATES['law_definition'][:8]:  # ì²˜ìŒ 8ê°??¬ìš©
                        question = template.format(law_name=law_name)
                        answer = random.choice(LARGE_SCALE_ANSWER_TEMPLATES['law_definition']).format(
                            law_name=law_name, definition=definition
                        )
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'source': 'law_definition',
                            'law_name': law_name,
                            'confidence': 0.9,
                            'difficulty': 'easy'
                        })
            
            # 2. ì¡°ë¬¸ë³?Q&A (??ë§ì? ?¨í„´)
            for article in articles[:15]:  # ì²˜ìŒ 15ê°?ì¡°ë¬¸ ?¬ìš©
                article_number = article.get('article_number', '')
                content = article.get('content', '')
                title = article.get('title', '')
                
                if article_number and content:
                    # ì¡°ë¬¸ ?´ìš© Q&A (??ë§ì? ?¨í„´)
                    for template in LARGE_SCALE_QA_TEMPLATES['law_article'][:6]:
                        question = template.format(law_name=law_name, article=article_number)
                        answer = random.choice(LARGE_SCALE_ANSWER_TEMPLATES['law_article']).format(
                            law_name=law_name, article=article_number, content=content[:200] + "..."
                        )
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'source': 'law_article',
                            'law_name': law_name,
                            'article_number': article_number,
                            'confidence': 0.8,
                            'difficulty': 'medium'
                        })
                
                # ì¡°ë¬¸ ?œëª© Q&A
                if title:
                    for template in LARGE_SCALE_QA_TEMPLATES['law_article_title']:
                        question = template.format(law_name=law_name, article=article_number)
                        answer = random.choice(LARGE_SCALE_ANSWER_TEMPLATES['law_article_title']).format(
                            law_name=law_name, article=article_number, title=title
                        )
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'source': 'law_article_title',
                            'law_name': law_name,
                            'article_number': article_number,
                            'confidence': 0.95,
                            'difficulty': 'easy'
                        })
            
            # 3. ?¤ì›Œ??ê¸°ë°˜ Q&A (??ë§ì? ?¤ì›Œ??
            entities = law_data.get('entities', {})
            keywords = entities.get('keywords', [])
            for keyword in keywords[:15]:  # ?ìœ„ 15ê°??¤ì›Œ???¬ìš©
                for template in LARGE_SCALE_QA_TEMPLATES['law_keyword'][:5]:
                    question = template.format(keyword=keyword)
                    answer = self._generate_keyword_answer(keyword, law_name, cleaned_content)
                    if answer:
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'source': 'keyword_based',
                            'law_name': law_name,
                            'keyword': keyword,
                            'confidence': 0.7,
                            'difficulty': 'medium'
                        })
            
            # 4. ë²•ë ¹ëª?ê¸°ë°˜ ?¼ë°˜ Q&A
            if law_name:
                for template in LARGE_SCALE_QA_TEMPLATES['general_legal'][:5]:
                    question = template.format(keyword=law_name)
                    answer = f"{law_name}?€ {law_name}??ê´€???¬í•­??ê·œì •??ë²•ë¥ ?…ë‹ˆ??"
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': 'law_name_based',
                        'law_name': law_name,
                        'confidence': 0.6,
                        'difficulty': 'easy'
                    })
            
        except Exception as e:
            self.logger.error(f"Error generating law QA pairs: {e}")
        
        return qa_pairs
    
    def generate_precedent_qa_pairs(self, precedent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """?ë? ?°ì´?°ì—??Q&A ???ì„± (?€ê·œëª¨ ë²„ì „)"""
        qa_pairs = []
        
        try:
            case_name = precedent_data.get('case_name', '')
            issue = precedent_data.get('issue', '')
            reasoning = precedent_data.get('reasoning', '')
            conclusion = precedent_data.get('conclusion', '')
            court = precedent_data.get('court', '')
            date = precedent_data.get('date', '')
            case_number = precedent_data.get('case_number', '')
            
            if not case_name:
                return qa_pairs
            
            # 1. ?ì  ê´€??Q&A (??ë§ì? ?¨í„´)
            if issue:
                for template in LARGE_SCALE_QA_TEMPLATES['precedent_issue'][:8]:
                    question = template.format(case_name=case_name)
                    answer = random.choice(LARGE_SCALE_ANSWER_TEMPLATES['precedent_issue']).format(
                        case_name=case_name, issue=issue
                    )
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': 'precedent_issue',
                        'case_name': case_name,
                        'court': court,
                        'confidence': 0.9,
                        'difficulty': 'medium'
                    })
            
            # 2. ?ê²° ?´ìš© Q&A (??ë§ì? ?¨í„´)
            if reasoning:
                for template in LARGE_SCALE_QA_TEMPLATES['precedent_decision'][:6]:
                    question = template.format(case_name=case_name)
                    answer = random.choice(LARGE_SCALE_ANSWER_TEMPLATES['precedent_decision']).format(
                        case_name=case_name, decision=reasoning[:200] + "..."
                    )
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': 'precedent_decision',
                        'case_name': case_name,
                        'court': court,
                        'confidence': 0.8,
                        'difficulty': 'hard'
                    })
            
            # 3. ë²•ì› ?•ë³´ Q&A
            if court:
                for template in LARGE_SCALE_QA_TEMPLATES['precedent_court']:
                    question = template.format(case_name=case_name)
                    answer = random.choice(LARGE_SCALE_ANSWER_TEMPLATES['precedent_court']).format(
                        case_name=case_name, court=court
                    )
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': 'precedent_court',
                        'case_name': case_name,
                        'court': court,
                        'confidence': 0.95,
                        'difficulty': 'easy'
                    })
            
            # 4. ?ê²°??Q&A
            if date:
                for template in LARGE_SCALE_QA_TEMPLATES['precedent_date']:
                    question = template.format(case_name=case_name)
                    answer = random.choice(LARGE_SCALE_ANSWER_TEMPLATES['precedent_date']).format(
                        case_name=case_name, date=date
                    )
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': 'precedent_date',
                        'case_name': case_name,
                        'court': court,
                        'date': date,
                        'confidence': 0.95,
                        'difficulty': 'easy'
                    })
            
            # 5. ?¬ê±´ë²ˆí˜¸ Q&A
            if case_number:
                for template in LARGE_SCALE_QA_TEMPLATES['precedent_case_number']:
                    question = template.format(case_name=case_name)
                    answer = random.choice(LARGE_SCALE_ANSWER_TEMPLATES['precedent_case_number']).format(
                        case_name=case_name, case_number=case_number
                    )
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': 'precedent_case_number',
                        'case_name': case_name,
                        'court': court,
                        'case_number': case_number,
                        'confidence': 0.95,
                        'difficulty': 'easy'
                    })
            
            # 6. ê²°ë¡  Q&A
            if conclusion:
                question = f"{case_name} ?¬ê±´??ê²°ë¡ ?€ ë¬´ì—‡?¸ê???"
                answer = f"{case_name} ?¬ê±´?ì„œ {conclusion}?¼ê³  ?ë‹¨?ˆìŠµ?ˆë‹¤."
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'source': 'precedent_conclusion',
                    'case_name': case_name,
                    'court': court,
                    'confidence': 0.95,
                    'difficulty': 'easy'
                })
            
        except Exception as e:
            self.logger.error(f"Error generating precedent QA pairs: {e}")
        
        return qa_pairs
    
    def generate_constitutional_qa_pairs(self, decision_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """?Œì¬ê²°ì •ë¡€ ?°ì´?°ì—??Q&A ???ì„± (?€ê·œëª¨ ë²„ì „)"""
        qa_pairs = []
        
        try:
            case_name = decision_data.get('case_name', '')
            issue = decision_data.get('issue', '')
            reasoning = decision_data.get('reasoning', '')
            conclusion = decision_data.get('conclusion', '')
            decision_type = decision_data.get('decision_type', '')
            
            if not case_name:
                return qa_pairs
            
            # 1. ?Œë²•???ì  Q&A (??ë§ì? ?¨í„´)
            if issue:
                for template in LARGE_SCALE_QA_TEMPLATES['constitutional_issue'][:8]:
                    question = template.format(case_name=case_name)
                    answer = random.choice(LARGE_SCALE_ANSWER_TEMPLATES['constitutional_issue']).format(
                        case_name=case_name, issue=issue
                    )
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': 'constitutional_issue',
                        'case_name': case_name,
                        'decision_type': decision_type,
                        'confidence': 0.9,
                        'difficulty': 'hard'
                    })
            
            # 2. ?Œë²•?¬íŒ??ê²°ì • Q&A (??ë§ì? ?¨í„´)
            if reasoning:
                for template in LARGE_SCALE_QA_TEMPLATES['constitutional_decision'][:6]:
                    question = template.format(case_name=case_name)
                    answer = random.choice(LARGE_SCALE_ANSWER_TEMPLATES['constitutional_decision']).format(
                        case_name=case_name, decision=reasoning[:200] + "..."
                    )
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': 'constitutional_decision',
                        'case_name': case_name,
                        'decision_type': decision_type,
                        'confidence': 0.8,
                        'difficulty': 'hard'
                    })
            
        except Exception as e:
            self.logger.error(f"Error generating constitutional QA pairs: {e}")
        
        return qa_pairs
    
    def generate_interpretation_qa_pairs(self, interpretation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ë²•ë ¹?´ì„ë¡€ ?°ì´?°ì—??Q&A ???ì„± (?€ê·œëª¨ ë²„ì „)"""
        qa_pairs = []
        
        try:
            case_name = interpretation_data.get('case_name', '')
            issue = interpretation_data.get('issue', '')
            reasoning = interpretation_data.get('reasoning', '')
            topic = interpretation_data.get('topic', '')
            ministry = interpretation_data.get('ministry', '')
            
            if not topic:
                return qa_pairs
            
            # 1. ?´ì„ ì£¼ì œ Q&A (??ë§ì? ?¨í„´)
            if issue:
                for template in LARGE_SCALE_QA_TEMPLATES['interpretation_question'][:8]:
                    question = template.format(topic=topic)
                    answer = random.choice(LARGE_SCALE_ANSWER_TEMPLATES['interpretation_question']).format(
                        topic=topic, interpretation=issue[:200] + "..."
                    )
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': 'interpretation_question',
                        'topic': topic,
                        'ministry': ministry,
                        'confidence': 0.8,
                        'difficulty': 'medium'
                    })
            
            # 2. êµ¬ì²´???´ì„ Q&A
            if case_name and reasoning:
                question = f"{topic}???€??{ministry}???´ì„?€ ë¬´ì—‡?¸ê???"
                answer = f"{ministry}???´ì„???°ë¥´ë©?{reasoning[:200]}...?…ë‹ˆ??"
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'source': 'interpretation_detail',
                    'topic': topic,
                    'ministry': ministry,
                    'case_name': case_name,
                    'confidence': 0.7,
                    'difficulty': 'medium'
                })
            
        except Exception as e:
            self.logger.error(f"Error generating interpretation QA pairs: {e}")
        
        return qa_pairs
    
    def _extract_law_definition(self, content: str) -> str:
        """ë²•ë ¹ ?•ì˜ ì¶”ì¶œ"""
        sentences = content.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 20:
                return first_sentence
        return ""
    
    def _generate_keyword_answer(self, keyword: str, law_name: str, content: str) -> str:
        """?¤ì›Œ??ê¸°ë°˜ ?µë? ?ì„±"""
        sentences = content.split('.')
        for sentence in sentences:
            if keyword in sentence and len(sentence) > 20:
                return f"{law_name}???°ë¥´ë©?{sentence.strip()}?…ë‹ˆ??"
        return ""
    
    def calculate_quality_score(self, qa_pair: Dict[str, Any]) -> float:
        """Q&A ?ì˜ ?ˆì§ˆ ?ìˆ˜ ê³„ì‚° (?€ê·œëª¨ ë²„ì „)"""
        score = 0.0
        
        # ê¸°ë³¸ ?ìˆ˜
        score += 0.15
        
        # ì§ˆë¬¸ ê¸¸ì´ ?ìˆ˜
        question_length = len(qa_pair.get('question', ''))
        if 10 <= question_length <= 100:
            score += 0.25
        elif 100 < question_length <= 200:
            score += 0.15
        
        # ?µë? ê¸¸ì´ ?ìˆ˜
        answer_length = len(qa_pair.get('answer', ''))
        if 20 <= answer_length <= 500:
            score += 0.3
        elif 500 < answer_length <= 1000:
            score += 0.2
        
        # ? ë¢°???ìˆ˜
        confidence = qa_pair.get('confidence', 0.5)
        score += confidence * 0.3
        
        return min(score, 1.0)
    
    def generate_dataset(self, data_dir: str = "data/processed", output_dir: str = "data/qa_dataset") -> bool:
        """?„ì²´ Q&A ?°ì´?°ì…‹ ?ì„± (?€ê·œëª¨ ë²„ì „)"""
        try:
            data_path = Path(data_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("?€ê·œëª¨ Q&A ?°ì´?°ì…‹ ?ì„± ?œì‘...")
            
            # ê°??°ì´???€?…ë³„ ì²˜ë¦¬
            data_types = ['laws', 'precedents', 'constitutional_decisions', 'legal_interpretations']
            
            for data_type in data_types:
                self.logger.info(f"{data_type} ?°ì´??ì²˜ë¦¬ ì¤?..")
                
                data_files = list(data_path.glob(f"{data_type}/*.json"))
                processed_count = 0
                
                for file_path in data_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # ?°ì´?°ê? ë°°ì—´??ê²½ìš° ê°???ª©ë³„ë¡œ ì²˜ë¦¬
                        if isinstance(data, list):
                            for item in data:
                                if not isinstance(item, dict):
                                    continue
                                
                                # ?°ì´???€?…ì— ?°ë¥¸ Q&A ?ì„±
                                if data_type == 'laws':
                                    qa_pairs = self.generate_law_qa_pairs(item)
                                elif data_type == 'precedents':
                                    qa_pairs = self.generate_precedent_qa_pairs(item)
                                elif data_type == 'constitutional_decisions':
                                    qa_pairs = self.generate_constitutional_qa_pairs(item)
                                elif data_type == 'legal_interpretations':
                                    qa_pairs = self.generate_interpretation_qa_pairs(item)
                                else:
                                    continue
                                
                                # ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°
                                for qa_pair in qa_pairs:
                                    qa_pair['quality_score'] = self.calculate_quality_score(qa_pair)
                                    qa_pair['generated_at'] = datetime.now().isoformat()
                                
                                self.qa_pairs.extend(qa_pairs)
                                processed_count += 1
                                
                                # ì§„í–‰ ?í™© ë¡œê¹…
                                if processed_count % 100 == 0:
                                    self.logger.info(f"{data_type}: {processed_count}ê°???ª© ì²˜ë¦¬ ?„ë£Œ, ?„ì¬ Q&A: {len(self.qa_pairs)}ê°?)
                        else:
                            # ?¨ì¼ ê°ì²´??ê²½ìš°
                            if data_type == 'laws':
                                qa_pairs = self.generate_law_qa_pairs(data)
                            elif data_type == 'precedents':
                                qa_pairs = self.generate_precedent_qa_pairs(data)
                            elif data_type == 'constitutional_decisions':
                                qa_pairs = self.generate_constitutional_qa_pairs(data)
                            elif data_type == 'legal_interpretations':
                                qa_pairs = self.generate_interpretation_qa_pairs(data)
                            else:
                                continue
                            
                            # ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°
                            for qa_pair in qa_pairs:
                                qa_pair['quality_score'] = self.calculate_quality_score(qa_pair)
                                qa_pair['generated_at'] = datetime.now().isoformat()
                            
                            self.qa_pairs.extend(qa_pairs)
                            processed_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
                        continue
                
                self.logger.info(f"{data_type} ì²˜ë¦¬ ?„ë£Œ: {processed_count}ê°???ª©, ì´?Q&A: {len(self.qa_pairs)}ê°?)
            
            # ?ˆì§ˆ ?ìˆ˜ë³??•ë ¬
            self.qa_pairs.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # ?°ì´?°ì…‹ ?€??
            self._save_dataset(output_path)
            
            # ?µê³„ ?ì„±
            self._generate_statistics(output_path)
            
            self.logger.info(f"?€ê·œëª¨ Q&A ?°ì´?°ì…‹ ?ì„± ?„ë£Œ: {len(self.qa_pairs)}ê°???)
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating dataset: {e}")
            return False
    
    def _save_dataset(self, output_path: Path):
        """?°ì´?°ì…‹ ?€??""
        # ?„ì²´ ?°ì´?°ì…‹ ?€??
        with open(output_path / "large_scale_qa_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=2)
        
        # ?ˆì§ˆë³?ë¶„í•  ?€??
        high_quality = [qa for qa in self.qa_pairs if qa.get('quality_score', 0) >= 0.8]
        medium_quality = [qa for qa in self.qa_pairs if 0.6 <= qa.get('quality_score', 0) < 0.8]
        low_quality = [qa for qa in self.qa_pairs if qa.get('quality_score', 0) < 0.6]
        
        with open(output_path / "large_scale_qa_dataset_high_quality.json", 'w', encoding='utf-8') as f:
            json.dump(high_quality, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "large_scale_qa_dataset_medium_quality.json", 'w', encoding='utf-8') as f:
            json.dump(medium_quality, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "large_scale_qa_dataset_low_quality.json", 'w', encoding='utf-8') as f:
            json.dump(low_quality, f, ensure_ascii=False, indent=2)
    
    def _generate_statistics(self, output_path: Path):
        """?µê³„ ?•ë³´ ?ì„±"""
        stats = {
            'total_pairs': len(self.qa_pairs),
            'high_quality_pairs': len([qa for qa in self.qa_pairs if qa.get('quality_score', 0) >= 0.8]),
            'medium_quality_pairs': len([qa for qa in self.qa_pairs if 0.6 <= qa.get('quality_score', 0) < 0.8]),
            'low_quality_pairs': len([qa for qa in self.qa_pairs if qa.get('quality_score', 0) < 0.6]),
            'average_quality_score': sum(qa.get('quality_score', 0) for qa in self.qa_pairs) / len(self.qa_pairs) if self.qa_pairs else 0,
            'source_distribution': {},
            'difficulty_distribution': {},
            'generated_at': datetime.now().isoformat()
        }
        
        # ?ŒìŠ¤ë³?ë¶„í¬
        for qa in self.qa_pairs:
            source = qa.get('source', 'unknown')
            stats['source_distribution'][source] = stats['source_distribution'].get(source, 0) + 1
        
        # ?œì´?„ë³„ ë¶„í¬
        for qa in self.qa_pairs:
            difficulty = qa.get('difficulty', 'unknown')
            stats['difficulty_distribution'][difficulty] = stats['difficulty_distribution'].get(difficulty, 0) + 1
        
        with open(output_path / "large_scale_qa_dataset_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # ?€ê·œëª¨ Q&A ?°ì´?°ì…‹ ?ì„±
    generator = LargeScaleQADatasetGenerator()
    success = generator.generate_dataset()
    
    if success:
        logger.info("?€ê·œëª¨ Q&A ?°ì´?°ì…‹ ?ì„±???„ë£Œ?˜ì—ˆ?µë‹ˆ??")
    else:
        logger.error("?€ê·œëª¨ Q&A ?°ì´?°ì…‹ ?ì„±???¤íŒ¨?ˆìŠµ?ˆë‹¤.")


if __name__ == "__main__":
    main()
