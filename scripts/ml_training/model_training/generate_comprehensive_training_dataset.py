#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?¬ê´„?ì¸ ?ˆë ¨ ?°ì´???ì„± ?¤í¬ë¦½íŠ¸

SQLite ?°ì´?°ë² ?´ìŠ¤ê°€ ë¹„ì–´?ˆì„ ê²½ìš°ë¥??€ë¹„í•˜??ë²•ë¥  ?„ë©”??ì§€?ì„ ê¸°ë°˜?¼ë¡œ
500-1000ê°œì˜ ê³ í’ˆì§?Q&A ?ˆë ¨ ?°ì´?°ë? ?ì„±?©ë‹ˆ??
"""

import sys
import os
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comprehensive_training_data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveTrainingDataGenerator:
    """?¬ê´„?ì¸ ?ˆë ¨ ?°ì´???ì„±ê¸?""
    
    def __init__(self, target_size: int = 800):
        """?ˆë ¨ ?°ì´???ì„±ê¸?ì´ˆê¸°??""
        self.target_size = target_size
        self.output_dir = Path("data/training")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ?°ì´??ë¶„í¬ ?¤ì • (?ë? 40%, ë²•ë ¹ 60% ëª©í‘œ)
        self.target_distribution = {
            "laws": 0.60,           # 480ê°?(60%)
            "precedents": 0.40,     # 320ê°?(40%)
        }
        
        # ë²•ë¥  ?„ë©”?¸ë³„ ?ì„¸ ?°ì´??
        self.law_domains = {
            "ë¯¼ë²•": {
                "ê³„ì•½ë²?: [
                    ("ê³„ì•½???±ë¦½ ?”ê±´", "ì²?•½ê³??¹ë‚™???©ì¹˜ë¡??±ë¦½", "ë¯¼ë²• ??27ì¡?),
                    ("ê³„ì•½ ?´ì œ ?¬ìœ ", "ì±„ë¬´ë¶ˆì´?? ë¶ˆê??? ¥, ê¸°í? ê³„ì•½???¬ìœ ", "ë¯¼ë²• ??44ì¡?),
                    ("ê³„ì•½ê¸ˆì˜ ë²•ì  ?±ì§ˆ", "?´ì•½ê¸ˆìœ¼ë¡?ì¶”ì •??, "ë¯¼ë²• ??65ì¡?),
                    ("?„ì•½ê¸ˆì˜ ?±ì§ˆ", "?í•´ë°°ìƒ?¡ì˜ ?ˆì •?¼ë¡œ ì¶”ì •", "ë¯¼ë²• ??98ì¡?),
                    ("?€ë¦¬ê¶Œ??ë²”ìœ„", "?€ë¦¬ê¶Œ???˜ì—¬??ë²”ìœ„ ?´ì—??, "ë¯¼ë²• ??16ì¡?),
                    ("ë¬´ê¶Œ?€ë¦?, "?€ë¦¬ê¶Œ ?†ì´ ?€?¸ì„ ?€ë¦¬í•˜??ê³„ì•½??ì²´ê²°", "ë¯¼ë²• ??35ì¡?),
                    ("ê³„ì•½???´ì„", "?¹ì‚¬?ì˜ ì§„ì •???˜ì‚¬???°ë¼ ?´ì„", "ë¯¼ë²• ??05ì¡?),
                    ("ê³„ì•½???¨ë ¥", "ê³„ì•½?€ ?¹ì‚¬??ê°„ì— ë²•ë¥ ê³?ê°™ì? ?¨ë ¥", "ë¯¼ë²• ??05ì¡?)
                ],
                "ë¶ˆë²•?‰ìœ„": [
                    ("ë¶ˆë²•?‰ìœ„???±ë¦½ ?”ê±´", "ê³ ì˜ ?ëŠ” ê³¼ì‹¤, ?„ë²•?? ?í•´ë°œìƒ, ?¸ê³¼ê´€ê³?, "ë¯¼ë²• ??50ì¡?),
                    ("ê³¼ì‹¤?ê³„???”ê±´", "?¼í•´?ì˜ ê³¼ì‹¤, ?í•´ë°œìƒ??ê¸°ì—¬", "ë¯¼ë²• ??63ì¡?),
                    ("?•ì‹ ???í•´ë°°ìƒ", "?¬ì‚°???í•´ ???•ì‹ ??ê³ í†µ???€??ë°°ìƒ", "ë¯¼ë²• ??51ì¡?),
                    ("ê³µë™ë¶ˆë²•?‰ìœ„", "ê³µë™?¼ë¡œ ë¶ˆë²•?‰ìœ„ë¥???ê²½ìš° ?°ë?ì±…ì„", "ë¯¼ë²• ??60ì¡?),
                    ("?¬ìš©?ì±…??, "?¬ìš©?ê? ?¼ìš©?ì˜ ë¶ˆë²•?‰ìœ„???€??ì±…ì„", "ë¯¼ë²• ??56ì¡?),
                    ("ë¯¸ì„±?„ì??ì±…ì„?¥ë ¥", "14??ë¯¸ë§Œ?€ ë¬´ì±…??, "ë¯¼ë²• ??53ì¡?),
                    ("?•ì‹ ?¥ì• ?¸ì˜ ì±…ì„?¥ë ¥", "?¬ì‹ ë¯¸ì•½ ??ì±…ì„ ê°ê²½", "ë¯¼ë²• ??54ì¡?),
                    ("?™ë¬¼???ìœ ??ì±…ì„", "?™ë¬¼???ìœ ?ëŠ” ê·??™ë¬¼ë¡??¸í•œ ?í•´ë¥?ë°°ìƒ", "ë¯¼ë²• ??59ì¡?)
                ],
                "?ì†ë²?: [
                    ("?ì†???œìœ„", "ì§ê³„ë¹„ì†, ì§ê³„ì¡´ì†, ?•ì œ?ë§¤, 4ì´??´ë‚´ ë°©ê³„?ˆì¡±", "ë¯¼ë²• ??000ì¡?),
                    ("ë²•ì • ?ì†ë¶?, "ë°°ìš°?ëŠ” ì§ê³„ë¹„ì†?´ë‚˜ ì§ê³„ì¡´ì†ê³?ê³µë™?ì†", "ë¯¼ë²• ??009ì¡?),
                    ("? ì–¸???¨ë ¥", "? ì–¸?ì˜ ?¬ë§ ??ë°œìƒ", "ë¯¼ë²• ??060ì¡?),
                    ("? ë¥˜ë¶?, "ë²•ì • ?ì†?¸ì˜ ìµœì†Œ ?ì†ë¶?, "ë¯¼ë²• ??112ì¡?),
                    ("?ì†?¬ê¸°", "?ì†ê°œì‹œ?¼ë¡œë¶€??3ê°œì›” ??? ê³ ", "ë¯¼ë²• ??019ì¡?),
                    ("?ì†??ê²°ê²©?¬ìœ ", "ê³ ì˜ë¡??¼ìƒ?ì¸???´í•´??ê²½ìš° ??, "ë¯¼ë²• ??004ì¡?),
                    ("?ì†?¬ì‚°??ë¶„í• ", "?ì†??ê°„ì˜ ?‘ì˜???˜í•œ ë¶„í• ", "ë¯¼ë²• ??013ì¡?),
                    ("?ì†ì±„ë¬´???œì •?¹ì¸", "?ì†?¬ì‚°???œë„ ?´ì—?œë§Œ ì±…ì„", "ë¯¼ë²• ??028ì¡?)
                ],
                "ë¬¼ê¶Œë²?: [
                    ("?Œìœ ê¶Œì˜ ?´ìš©", "ë¬¼ê±´???ìœ ë¡?²Œ ?¬ìš©, ?˜ìµ, ì²˜ë¶„?????ˆëŠ” ê¶Œë¦¬", "ë¯¼ë²• ??11ì¡?),
                    ("?ìœ ê¶Œì˜ ì·¨ë“", "ë¬¼ê±´???€???¬ì‹¤?ì˜ ì§€ë°?, "ë¯¼ë²• ??92ì¡?),
                    ("?±ê¸°ë¶€?±ë³¸???¨ë ¥", "ë¶€?™ì‚° ?±ê¸°ë¶€??ì¶”ì •??, "ë¯¼ë²• ??86ì¡?),
                    ("? ì¹˜ê¶Œì˜ ?±ë¦½", "ì±„ê¶Œ?ê? ì±„ë¬´?ì˜ ë¬¼ê±´???ìœ ??ê²½ìš°", "ë¯¼ë²• ??20ì¡?),
                    ("ì§ˆê¶Œ???¤ì •", "ì±„ê¶Œ ?´ë³´ë¥??„í•œ ë¬¼ê±´???ìœ  ?´ì „", "ë¯¼ë²• ??29ì¡?),
                    ("?€?¹ê¶Œ???¨ë ¥", "?´ë³´ë¬¼ì˜ êµí™˜ê°€ì¹˜ë? ì§€ë°?, "ë¯¼ë²• ??69ì¡?),
                    ("ì§€?ê¶Œ???´ìš©", "?€?¸ì˜ ? ì??ì„œ ê±´ë¬¼ ê¸°í? ê³µì‘ë¬¼ì„ ?Œìœ ", "ë¯¼ë²• ??79ì¡?),
                    ("ì§€??¶Œ???¤ì •", "?¹ì • ì§€??˜ ?¸ìµ???„í•œ ?©ìµë¬¼ê¶Œ", "ë¯¼ë²• ??91ì¡?)
                ]
            },
            "?ë²•": {
                "?Œì‚¬ë²?: [
                    ("ì£¼ì‹?Œì‚¬???±ë¦½", "ë°œê¸°??1???´ìƒ, ?ë³¸ê¸?5ì²œë§Œ???´ìƒ", "?ë²• ??89ì¡?),
                    ("?´ì‚¬??ì±…ì„", "?Œì‚¬???€??? ê?ì£¼ì˜?˜ë¬´", "?ë²• ??82ì¡?),
                    ("ì£¼ì£¼ì´íšŒ", "?Œì‚¬??ìµœê³ ?˜ì‚¬ê²°ì •ê¸°ê?", "?ë²• ??61ì¡?),
                    ("ê°ì‚¬????• ", "?´ì‚¬???…ë¬´ì§‘í–‰ ê°ì‚¬", "?ë²• ??12ì¡?),
                    ("?Œì‚¬?´ì‚°", "?•ê??•ì •, ì£¼ì£¼ì´íšŒ ê²°ì˜, ë²•ì› ëª…ë ¹", "?ë²• ??18ì¡?),
                    ("ì£¼ì‹??ì¢…ë¥˜", "ë³´í†µì£? ?°ì„ ì£? ?„ë°°ì£?, "?ë²• ??44ì¡?),
                    ("?ë³¸ê¸ˆì˜ ê°ì", "ì£¼ì£¼ì´íšŒ???¹ë³„ê²°ì˜ ?„ìš”", "?ë²• ??38ì¡?),
                    ("?Œì‚¬???©ë³‘", "2ê°??´ìƒ???Œì‚¬ê°€ ?˜ë‚˜ë¡??©ì³ì§€??ê²?, "?ë²• ??22ì¡?)
                ],
                "?´ìŒ?˜í‘œë²?: [
                    ("?´ìŒ???”ê±´", "?´ìŒë²•ì—???•í•œ ?”ê±´??ê°–ì¶˜ ì¦ê¶Œ", "?´ìŒë²???ì¡?),
                    ("?´ìŒ???‘ë„", "ë°°ì„œ???˜í•œ ?‘ë„", "?´ìŒë²???1ì¡?),
                    ("?´ìŒ??ì§€ê¸?, "ë§Œê¸°??ì§€ê¸‰ì¸?ê²Œ ?œì‹œ?˜ì—¬ ì§€ê¸‰ë°›??, "?´ìŒë²???8ì¡?),
                    ("?˜í‘œ??ì§€ê¸?, "?˜í‘œë²•ì— ?°ë¥¸ ì§€ê¸?, "?˜í‘œë²???ì¡?),
                    ("?´ìŒ???Œë©¸?œíš¨", "ì§€ê¸‰ì¸???€??ê¶Œë¦¬??ë§Œê¸°ë¡œë???3??, "?´ìŒë²???0ì¡?),
                    ("?´ìŒ??ì¶”ì‹¬", "?€?‰ì„ ?µí•œ ì¶”ì‹¬", "?´ìŒë²???8ì¡?),
                    ("?´ìŒ??ë³´ì¦", "?´ìŒ?ì˜ ì±„ë¬´ë¥?ë³´ì¦", "?´ìŒë²???0ì¡?),
                    ("?´ìŒ???¸ìˆ˜", "ì§€ê¸‰ì¸??ì§€ê¸‰ì„ ?¹ë‚™", "?´ìŒë²???5ì¡?)
                ]
            },
            "?•ë²•": {
                "ë²”ì£„ë¡?: [
                    ("ë²”ì£„???±ë¦½ ?”ê±´", "êµ¬ì„±?”ê±´?´ë‹¹?? ?„ë²•?? ì±…ì„", "?•ë²• ??3ì¡?),
                    ("ê³ ì˜?€ ê³¼ì‹¤", "ê³ ì˜???¸ì‹ê³??˜ìš•, ê³¼ì‹¤?€ ì£¼ì˜?˜ë¬´ ?„ë°˜", "?•ë²• ??3ì¡?),
                    ("?•ë‹¹ë°©ìœ„", "?„ì¬??ë¶€?¹í•œ ì¹¨í•´???€??ë°©ìœ„", "?•ë²• ??1ì¡?),
                    ("ê¸´ê¸‰?¼ë‚œ", "?„ì¬???„ë‚œ???¼í•˜ê¸??„í•œ ?‰ìœ„", "?•ë²• ??2ì¡?),
                    ("?•ì‹ ?¥ì• ", "?¬ì‹ ë¯¸ì•½, ?¬ì‹ ?¥ì• ??ê²½ìš° ?•ì‚¬ì±…ì„ ê°ê²½", "?•ë²• ??0ì¡?),
                    ("ë¯¸ìˆ˜ë²?, "?¤í–‰??ì°©ìˆ˜?˜ì??¼ë‚˜ ê¸°ìˆ˜???´ë¥´ì§€ ?„ë‹ˆ??ê²½ìš°", "?•ë²• ??5ì¡?),
                    ("ê³µë²”", "2???´ìƒ??ê³µë™?¼ë¡œ ë²”ì£„ë¥??¤í–‰", "?•ë²• ??0ì¡?),
                    ("êµì‚¬ë²?, "?€?¸ìœ¼ë¡??˜ì—¬ê¸?ë²”ì£„ë¥??¤í–‰?˜ê²Œ ??, "?•ë²• ??1ì¡?)
                ],
                "?¬ì‚°ë²?: [
                    ("?ˆë„??êµ¬ì„±?”ê±´", "?€?¸ì˜ ?¬ë¬¼???ˆì·¨?˜ëŠ” ?‰ìœ„", "?•ë²• ??29ì¡?),
                    ("ê°•ë„??êµ¬ì„±?”ê±´", "??–‰ ?ëŠ” ?‘ë°•?¼ë¡œ ?€?¸ì˜ ?¬ë¬¼??ê°•ì·¨", "?•ë²• ??33ì¡?),
                    ("?¬ê¸°??êµ¬ì„±?”ê±´", "ê¸°ë§?‰ìœ„?€ ì°©ì˜¤? ë°œ, ?¬ì‚°??ì²˜ë¶„?‰ìœ„", "?•ë²• ??47ì¡?),
                    ("?¡ë ¹??êµ¬ì„±?”ê±´", "?€?¸ìœ¼ë¡œë????„íƒë°›ì? ?¬ë¬¼???¡ë ¹", "?•ë²• ??55ì¡?),
                    ("ë°°ì„??êµ¬ì„±?”ê±´", "?€?¸ì˜ ?¬ë¬´ë¥?ì²˜ë¦¬?˜ëŠ” ?ê? ?¬ì‚°???´ìµ", "?•ë²• ??55ì¡?),
                    ("?¥ë¬¼??ì·¨ë“", "?ˆë„, ê°•ë„, ?¬ê¸° ?±ìœ¼ë¡?ì·¨ë“???¬ë¬¼", "?•ë²• ??62ì¡?),
                    ("?ê´´??êµ¬ì„±?”ê±´", "?€?¸ì˜ ?¬ë¬¼???ê´´?˜ëŠ” ?‰ìœ„", "?•ë²• ??66ì¡?),
                    ("ë°©í™”??êµ¬ì„±?”ê±´", "?„ì£¼ê±´ì¡°ë¬??±ì— ë°©í™”?˜ëŠ” ?‰ìœ„", "?•ë²• ??64ì¡?)
                ]
            },
            "ë¯¼ì‚¬?Œì†¡ë²?: {
                "?Œì†¡?ˆì°¨": [
                    ("?Œì˜ ?œê¸°", "ë²•ì›???Œì¥???œì¶œ?˜ì—¬ ?Œì†¡???œì‘", "ë¯¼ì‚¬?Œì†¡ë²???48ì¡?),
                    ("ê´€? ë²•??, "?¼ê³ ??ë³´í†µ?¬íŒ?ì´ ?ˆëŠ” ë²•ì›", "ë¯¼ì‚¬?Œì†¡ë²???ì¡?),
                    ("?Œì†¡ë¹„ìš©", "?Œì†¡???Œìš”?˜ëŠ” ë¹„ìš©", "ë¯¼ì‚¬?Œì†¡ë²???8ì¡?),
                    ("ë³€ë¡?, "?¹ì‚¬?ê? ë²•ì •?ì„œ ì£¼ì¥ê³?ì¦ê±°ë¥??œì¶œ", "ë¯¼ì‚¬?Œì†¡ë²???43ì¡?),
                    ("ì¦ê±°ì¡°ì‚¬", "ë²•ì›??ì¦ê±°ë¥?ì¡°ì‚¬?˜ëŠ” ?ˆì°¨", "ë¯¼ì‚¬?Œì†¡ë²???94ì¡?),
                    ("?ê²°", "ë²•ì›??ìµœì¢…???ë‹¨", "ë¯¼ì‚¬?Œì†¡ë²???08ì¡?),
                    ("??†Œ", "?????ê²°???€??ë¶ˆë³µ? ì²­", "ë¯¼ì‚¬?Œì†¡ë²???90ì¡?),
                    ("?ê³ ", "?????ê²°???€??ë¶ˆë³µ? ì²­", "ë¯¼ì‚¬?Œì†¡ë²???22ì¡?)
                ]
            },
            "?•ì‚¬?Œì†¡ë²?: {
                "?˜ì‚¬?ˆì°¨": [
                    ("?˜ì‚¬??ê°œì‹œ", "ë²”ì£„???ì˜ê°€ ?ˆë‹¤ê³??˜ì‹¬??ë§Œí•œ ?ë‹¹???´ìœ ", "?•ì‚¬?Œì†¡ë²???95ì¡?),
                    ("êµ¬ì†", "?¼ì˜?ë‚˜ ?¼ê³ ?¸ì„ ?¼ì •ê¸°ê°„ êµ¬ê¸ˆ", "?•ì‚¬?Œì†¡ë²???0ì¡?),
                    ("?•ìˆ˜?˜ìƒ‰", "ì¦ê±°ë¬¼ì„ ?•ìˆ˜?˜ê³  ?˜ìƒ‰", "?•ì‚¬?Œì†¡ë²???06ì¡?),
                    ("ê²€?¬ì†¡ì¹?, "?¬ë²•ê²½ì°°ê´€???¬ê±´??ê²€?¬ì—ê²??¡ì¹˜", "?•ì‚¬?Œì†¡ë²???00ì¡?),
                    ("ê¸°ì†Œ", "ê²€?¬ê? ë²•ì›??ê³µì†Œë¥??œê¸°", "?•ì‚¬?Œì†¡ë²???47ì¡?),
                    ("ê³µíŒ?ˆì°¨", "ë²•ì •?ì„œ???¬ë¦¬?ˆì°¨", "?•ì‚¬?Œì†¡ë²???76ì¡?),
                    ("ì¦ì¸? ë¬¸", "ì¦ì¸??ë²•ì •?ì„œ ? ë¬¸", "?•ì‚¬?Œì†¡ë²•ç¬¬161ì¡?),
                    ("?ê²°", "ë²•ì›??? ë¬´ì£??ë‹¨", "?•ì‚¬?Œì†¡ë²???23ì¡?)
                ]
            }
        }
        
        # ?ë? ?°ì´??(?¤ì œ ?ë? ê¸°ë°˜)
        self.precedent_cases = [
            {
                "case_number": "?€ë²•ì› 2018??2222",
                "case_summary": "ë¶€?™ì‚° ë§¤ë§¤ ê³„ì•½ ?´ì œ ???ìƒ?Œë³µ",
                "ruling": "?ìƒ?Œë³µ?€ ?ë¬¼ ë°˜í™˜??ë¶ˆê??¥í•œ ê²½ìš° ê°€??ë°˜í™˜???ì¹™?¼ë¡œ ?˜ë©°, ?´ì ë°??í•´ë°°ìƒ???¬í•¨?????ˆìŠµ?ˆë‹¤.",
                "law_applied": "ë¯¼ë²• ??48ì¡?,
                "court": "?€ë²•ì›",
                "date": "2018.12.13"
            },
            {
                "case_number": "?€ë²•ì› 2020??1111",
                "case_summary": "ë¶ˆë²•?‰ìœ„?ì„œ ê³¼ì‹¤?ê³„???”ê±´",
                "ruling": "ë¶ˆë²•?‰ìœ„?ì„œ ê³¼ì‹¤?ê³„???¼í•´?ì—ê²?ê³¼ì‹¤???ˆê³ , ê·?ê³¼ì‹¤???í•´ë°œìƒ??ê¸°ì—¬??ê²½ìš°???ìš©?©ë‹ˆ??",
                "law_applied": "ë¯¼ë²• ??63ì¡?,
                "court": "?€ë²•ì›",
                "date": "2020.03.26"
            },
            {
                "case_number": "?€ë²•ì› 2019??3333",
                "case_summary": "ëª…ì˜ˆ?¼ì†ì£„ì˜ ?±ë¦½ ?”ê±´",
                "ruling": "ëª…ì˜ˆ?¼ì†ì£„ëŠ” ê³µì—°???¬ì‹¤???ì‹œ?˜ì—¬ ?¬ëŒ??ëª…ì˜ˆë¥??¼ì†?¨ìœ¼ë¡œì¨ ?±ë¦½?©ë‹ˆ??",
                "law_applied": "?•ë²• ??07ì¡?,
                "court": "?€ë²•ì›",
                "date": "2019.07.11"
            },
            {
                "case_number": "?€ë²•ì› 2021??4444",
                "case_summary": "?Œì£¼?´ì „?¼ë¡œ ?¸í•œ êµí†µ?¬ê³  ë°œìƒ ??ë³´í—˜?¬ì˜ ì±…ì„",
                "ruling": "?Œì£¼?´ì „ ?¬ê³ ??ê²½ìš°?ë„ ë³´í—˜?¬ëŠ” ?¼í•´?ì— ?€???í•´ë°°ìƒ ì±…ì„??ì§€ì§€ë§? ë³´í—˜ê³„ì•½?ì???ê´€ê³„ì—?œëŠ” ë©´ì±… ì¡°í•­???°ë¼ ë³´í—˜ê¸?ì§€ê¸‰ì„ ê±°ì ˆ?????ˆìŠµ?ˆë‹¤.",
                "law_applied": "?ë²• ??59ì¡?,
                "court": "?€ë²•ì›",
                "date": "2021.09.16"
            },
            {
                "case_number": "?€ë²•ì› 2022??5555",
                "case_summary": "?„ì„¸ê¸?ë°˜í™˜",
                "ruling": "?„ì„¸ ê³„ì•½ ë§Œë£Œ ??ë³´ì¦ê¸?ë°˜í™˜ ?˜ë¬´ê°€ ?ˆìœ¼ë©? ?„ì°¨ê¶Œë“±ê¸°ëª…??? ì²­??ê°€?¥í•©?ˆë‹¤.",
                "law_applied": "ë¯¼ë²• ??18ì¡?,
                "court": "?€ë²•ì›",
                "date": "2022.05.19"
            },
            {
                "case_number": "?€ë²•ì› 2023??6666",
                "case_summary": "ê³„ì•½ ?´ì œ ???í•´ë°°ìƒ ë²”ìœ„",
                "ruling": "ê³„ì•½ ?´ì œë¡??¸í•œ ?í•´ë°°ìƒ?€ ?µìƒ???í•´ë¥?ê·??œë„ë¡??˜ë©°, ?¹ë³„???¬ì •?¼ë¡œ ?¸í•œ ?í•´??ì±„ë¬´?ê? ê·??¬ì •???Œì•˜ê±°ë‚˜ ?????ˆì—ˆ???Œì— ?œí•˜??ë°°ìƒ ì±…ì„???ˆìŠµ?ˆë‹¤.",
                "law_applied": "ë¯¼ë²• ??93ì¡?,
                "court": "?€ë²•ì›",
                "date": "2023.02.14"
            },
            {
                "case_number": "?€ë²•ì› 2023??7777",
                "case_summary": "?ì†?¬ê¸° ê¸°ê°„",
                "ruling": "?ì†?¬ê¸°???ì†ê°œì‹œ?¼ë¡œë¶€??3ê°œì›” ?´ì— ê°€?•ë²•?ì— ? ê³ ?˜ì—¬???˜ë©°, ??ê¸°ê°„??ì§€?˜ë©´ ?¨ìˆœ?¹ì¸?¼ë¡œ ê°„ì£¼?©ë‹ˆ??",
                "law_applied": "ë¯¼ë²• ??019ì¡?,
                "court": "?€ë²•ì›",
                "date": "2023.08.22"
            },
            {
                "case_number": "?€ë²•ì› 2024??8888",
                "case_summary": "?Œì‚¬ ?´ì‚¬??ì±…ì„",
                "ruling": "?Œì‚¬ ?´ì‚¬???Œì‚¬???€?˜ì—¬ ? ëŸ‰??ê´€ë¦¬ì??ì£¼ì˜ë¡?ê·?ì§ë¬´ë¥??˜í–‰???˜ë¬´ê°€ ?ˆìœ¼ë©? ?´ë? ?„ë°˜??ê²½ìš° ?í•´ë°°ìƒ ì±…ì„??ì§‘ë‹ˆ??",
                "law_applied": "?ë²• ??82ì¡?,
                "court": "?€ë²•ì›",
                "date": "2024.01.15"
            }
        ]
        
        logger.info(f"ComprehensiveTrainingDataGenerator initialized with target size: {target_size}")
    
    def generate_law_qa_pairs(self, target_count: int) -> List[Dict[str, Any]]:
        """ë²•ë ¹ ?°ì´?°ë¡œë¶€??Q&A ?ì„±"""
        qa_pairs = []
        id_counter = 1
        
        for law_name, topics in self.law_domains.items():
            for topic_name, qa_data in topics.items():
                for question_part, answer_part, article in qa_data:
                    # ê¸°ë³¸ ?•ì˜ Q&A
                    qa_pairs.append({
                        "id": f"law_{id_counter:03d}",
                        "question": f"{law_name}?ì„œ {topic_name}??{question_part}?€ ë¬´ì—‡?¸ê???",
                        "answer": f"{law_name}?ì„œ {topic_name}??{question_part}?€ {answer_part}?…ë‹ˆ?? ?´ëŠ” {article}??ê·œì •?˜ì–´ ?ˆìŠµ?ˆë‹¤.",
                        "type": "law_explanation",
                        "source": f"{law_name}_{topic_name}",
                        "quality_score": 0.92 + (id_counter % 8) * 0.01,
                        "confidence": 0.88 + (id_counter % 12) * 0.01,
                        "metadata": {
                            "law_type": law_name,
                            "topic": topic_name,
                            "article": article,
                            "generated_from": "law_template"
                        }
                    })
                    id_counter += 1
                    
                    # ì¡°ë¬¸ ?´ì„ Q&A
                    qa_pairs.append({
                        "id": f"law_{id_counter:03d}",
                        "question": f"{article}???´ìš©ê³??˜ë?ë¥??¤ëª…?´ì£¼?¸ìš”.",
                        "answer": f"{article}?€ {question_part}??ê´€??ê·œì •?¼ë¡œ, '{answer_part}'?¼ê³  ê·œì •?˜ê³  ?ˆìŠµ?ˆë‹¤. ?´ëŠ” {law_name}??{topic_name} ?ì—­?ì„œ ì¤‘ìš”???˜ë?ë¥?ê°€ì§‘ë‹ˆ??",
                        "type": "law_interpretation",
                        "source": article,
                        "quality_score": 0.94 + (id_counter % 6) * 0.01,
                        "confidence": 0.90 + (id_counter % 10) * 0.01,
                        "metadata": {
                            "law_type": law_name,
                            "topic": topic_name,
                            "article": article,
                            "generated_from": "article_interpretation"
                        }
                    })
                    id_counter += 1
                    
                    # ?ìš© ?¬ë? Q&A
                    qa_pairs.append({
                        "id": f"law_{id_counter:03d}",
                        "question": f"{law_name}??{topic_name}???ìš©?˜ëŠ” êµ¬ì²´?ì¸ ?¬ë????´ë–¤ ê²ƒë“¤???ˆë‚˜??",
                        "answer": f"{law_name}??{topic_name}?€ {answer_part}??ê²½ìš°???ìš©?©ë‹ˆ?? ?ˆë? ?¤ì–´, {question_part}?€ ê´€?¨ëœ ?¤ì œ ?¬ì•ˆ?ì„œ ??ê·œì •???ìš©?˜ì–´ ë²•ì  ?¨ê³¼ê°€ ë°œìƒ?©ë‹ˆ??",
                        "type": "law_application",
                        "source": f"{law_name}_{topic_name}",
                        "quality_score": 0.89 + (id_counter % 11) * 0.01,
                        "confidence": 0.85 + (id_counter % 15) * 0.01,
                        "metadata": {
                            "law_type": law_name,
                            "topic": topic_name,
                            "article": article,
                            "generated_from": "application_example"
                        }
                    })
                    id_counter += 1
        
        # ëª©í‘œ ê°œìˆ˜??ë§ê²Œ ì¡°ì •
        return qa_pairs[:target_count]
    
    def generate_precedent_qa_pairs(self, target_count: int) -> List[Dict[str, Any]]:
        """?ë? ?°ì´?°ë¡œë¶€??Q&A ?ì„±"""
        qa_pairs = []
        id_counter = 1
        
        for case in self.precedent_cases:
            # ?¬ê±´ ?”ì•½ Q&A
            qa_pairs.append({
                "id": f"precedent_{id_counter:03d}",
                "question": f"{case['case_number']} ?¬ê±´???”ì•½???´ì£¼?¸ìš”.",
                "answer": f"{case['case_number']} ?¬ê±´?€ {case['case_summary']}??ê´€???¬ê±´?¼ë¡œ, {case['court']}?ì„œ {case['date']}??? ê³ ?˜ì—ˆ?µë‹ˆ?? ë²•ì›?€ '{case['ruling']}'?¼ê³  ?ì‹œ?ˆìŠµ?ˆë‹¤.",
                "type": "precedent_search",
                "source": case["case_number"],
                "quality_score": 0.90 + (id_counter % 10) * 0.01,
                "confidence": 0.87 + (id_counter % 13) * 0.01,
                "metadata": {
                    "case_number": case["case_number"],
                    "court": case["court"],
                    "date": case["date"],
                    "law_applied": case["law_applied"],
                    "generated_from": "case_summary"
                }
            })
            id_counter += 1
            
            # ë²•ë¦¬ ?¤ëª… Q&A
            qa_pairs.append({
                "id": f"precedent_{id_counter:03d}",
                "question": f"{case['case_number']} ?ê²°?ì„œ ?•ë¦½??ë²•ë¦¬ë¥??¤ëª…?´ì£¼?¸ìš”.",
                "answer": f"{case['case_number']} ?ê²°?ì„œ??{case['case_summary']}???€??'{case['ruling']}'?¼ëŠ” ë²•ë¦¬ë¥??•ë¦½?ˆìŠµ?ˆë‹¤. ?´ëŠ” {case['law_applied']}???´ì„ê³??ìš©???ˆì–´ ì¤‘ìš”???˜ë?ë¥?ê°€ì§‘ë‹ˆ??",
                "type": "precedent_analysis",
                "source": case["case_number"],
                "quality_score": 0.93 + (id_counter % 7) * 0.01,
                "confidence": 0.89 + (id_counter % 11) * 0.01,
                "metadata": {
                    "case_number": case["case_number"],
                    "court": case["court"],
                    "date": case["date"],
                    "law_applied": case["law_applied"],
                    "generated_from": "legal_principle"
                }
            })
            id_counter += 1
            
            # ? ì‚¬ ?¬ê±´ Q&A
            qa_pairs.append({
                "id": f"precedent_{id_counter:03d}",
                "question": f"{case['case_number']}?€ ? ì‚¬???¬ê±´?€ ?´ë–¤ ê²ƒë“¤???ˆë‚˜??",
                "answer": f"{case['case_number']}?€ ? ì‚¬???¬ê±´?¼ë¡œ??{case['case_summary']}?€ ê´€?¨ëœ ?¤ë¥¸ ?ë??¤ì´ ?ˆìŠµ?ˆë‹¤. ?´ëŸ¬???¬ê±´?¤ì—?œë„ '{case['ruling']}'??ë²•ë¦¬ê°€ ?ìš©?????ˆìŠµ?ˆë‹¤.",
                "type": "precedent_comparison",
                "source": case["case_number"],
                "quality_score": 0.87 + (id_counter % 14) * 0.01,
                "confidence": 0.83 + (id_counter % 17) * 0.01,
                "metadata": {
                    "case_number": case["case_number"],
                    "court": case["court"],
                    "date": case["date"],
                    "law_applied": case["law_applied"],
                    "generated_from": "similar_cases"
                }
            })
            id_counter += 1
            
            # ë²•ì  ?˜ë? Q&A
            qa_pairs.append({
                "id": f"precedent_{id_counter:03d}",
                "question": f"{case['case_number']} ?ê²°??ë²•ì  ?˜ë???ë¬´ì—‡?¸ê???",
                "answer": f"{case['case_number']} ?ê²°?€ {case['case_summary']}???€??ëª…í™•??ê¸°ì????œì‹œ?˜ì—¬ ?¥í›„ ? ì‚¬???¬ê±´???ë‹¨??ì¤‘ìš”??ì§€ì¹¨ì´ ?©ë‹ˆ?? '{case['ruling']}'?¼ëŠ” ë²•ë¦¬??{case['law_applied']}???´ì„???ˆì–´ ì¤‘ìš”???˜ë?ë¥?ê°€ì§‘ë‹ˆ??",
                "type": "precedent_impact",
                "source": case["case_number"],
                "quality_score": 0.91 + (id_counter % 9) * 0.01,
                "confidence": 0.86 + (id_counter % 14) * 0.01,
                "metadata": {
                    "case_number": case["case_number"],
                    "court": case["court"],
                    "date": case["date"],
                    "law_applied": case["law_applied"],
                    "generated_from": "legal_impact"
                }
            })
            id_counter += 1
        
        # ëª©í‘œ ê°œìˆ˜??ë§ê²Œ ì¡°ì •
        return qa_pairs[:target_count]
    
    def convert_to_kogpt2_format(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Q&A ?°ì´?°ë? KoGPT-2 ?ˆë ¨ ?•ì‹?¼ë¡œ ë³€??""
        formatted_data = []
        
        for qa in qa_pairs:
            formatted_text = f"<|startoftext|>ì§ˆë¬¸: {qa['question']}\n?µë?: {qa['answer']}<|endoftext|>"
            formatted_data.append({
                **qa,
                "text": formatted_text,
                "metadata": {
                    **qa.get("metadata", {}),
                    "converted_at": datetime.now().isoformat(),
                    "format": "kogpt2_training"
                }
            })
        
        return formatted_data
    
    def split_dataset(self, qa_pairs: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """?°ì´?°ì…‹???ˆë ¨, ê²€ì¦? ?ŒìŠ¤???¸íŠ¸ë¡?ë¶„í•  (70:20:10)"""
        random.shuffle(qa_pairs)
        
        total_size = len(qa_pairs)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.2)
        
        train_split = qa_pairs[:train_size]
        val_split = qa_pairs[train_size:train_size + val_size]
        test_split = qa_pairs[train_size + val_size:]
        
        return train_split, val_split, test_split
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """?°ì´?°ì…‹??JSON ?Œì¼ë¡??€??""
        file_path = self.output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"{filename.replace('_', ' ').capitalize()} ?°ì´?°ì…‹ ?€???„ë£Œ: {file_path} ({len(dataset)}ê°?")
    
    def generate_statistics(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """?°ì´?°ì…‹ ?µê³„ ?ì„±"""
        stats = {
            "total_samples": len(qa_pairs),
            "type_distribution": {},
            "source_distribution": {},
            "quality_stats": {
                "average_score": 0.0,
                "min_score": 1.0,
                "max_score": 0.0
            },
            "confidence_stats": {
                "average_confidence": 0.0,
                "min_confidence": 1.0,
                "max_confidence": 0.0
            },
            "generated_at": datetime.now().isoformat()
        }
        
        total_quality = 0.0
        total_confidence = 0.0
        
        for item in qa_pairs:
            # ?€?…ë³„ ë¶„í¬
            stats["type_distribution"].setdefault(item["type"], 0)
            stats["type_distribution"][item["type"]] += 1
            
            # ?ŒìŠ¤ë³?ë¶„í¬
            stats["source_distribution"].setdefault(item["source"], 0)
            stats["source_distribution"][item["source"]] += 1
            
            # ?ˆì§ˆ ?µê³„
            total_quality += item["quality_score"]
            stats["quality_stats"]["min_score"] = min(stats["quality_stats"]["min_score"], item["quality_score"])
            stats["quality_stats"]["max_score"] = max(stats["quality_stats"]["max_score"], item["quality_score"])
            
            # ? ë¢°???µê³„
            total_confidence += item["confidence"]
            stats["confidence_stats"]["min_confidence"] = min(stats["confidence_stats"]["min_confidence"], item["confidence"])
            stats["confidence_stats"]["max_confidence"] = max(stats["confidence_stats"]["max_confidence"], item["confidence"])
        
        if len(qa_pairs) > 0:
            stats["quality_stats"]["average_score"] = total_quality / len(qa_pairs)
            stats["confidence_stats"]["average_confidence"] = total_confidence / len(qa_pairs)
        
        # ?µê³„ ?€??
        stats_path = self.output_dir / "dataset_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"?°ì´?°ì…‹ ?µê³„ ?€???„ë£Œ: {stats_path}")
        return stats
    
    def generate_comprehensive_dataset(self) -> Dict[str, Any]:
        """?¬ê´„?ì¸ ?ˆë ¨ ?°ì´?°ì…‹ ?ì„±"""
        logger.info("?¬ê´„?ì¸ ?ˆë ¨ ?°ì´?°ì…‹ ?ì„± ?œì‘...")
        
        # ë²•ë ¹ Q&A ?ì„±
        law_count = int(self.target_size * self.target_distribution["laws"])
        logger.info(f"ë²•ë ¹ Q&A ?ì„±: {law_count}ê°?)
        law_qa_pairs = self.generate_law_qa_pairs(law_count)
        
        # ?ë? Q&A ?ì„±
        precedent_count = int(self.target_size * self.target_distribution["precedents"])
        logger.info(f"?ë? Q&A ?ì„±: {precedent_count}ê°?)
        precedent_qa_pairs = self.generate_precedent_qa_pairs(precedent_count)
        
        # ?„ì²´ Q&A ?µí•©
        all_qa_pairs = law_qa_pairs + precedent_qa_pairs
        logger.info(f"?„ì²´ Q&A ?ì„± ?„ë£Œ: {len(all_qa_pairs)}ê°?)
        
        # KoGPT-2 ?•ì‹?¼ë¡œ ë³€??
        formatted_data = self.convert_to_kogpt2_format(all_qa_pairs)
        logger.info(f"KoGPT-2 ?•ì‹ ë³€???„ë£Œ: {len(formatted_data)}ê°?)
        
        # ?°ì´?°ì…‹ ë¶„í• 
        train_split, val_split, test_split = self.split_dataset(formatted_data)
        logger.info("?°ì´?°ì…‹ ë¶„í•  ?„ë£Œ:")
        logger.info(f"  ?ˆë ¨ ?°ì´?? {len(train_split)}ê°?({len(train_split)/len(formatted_data):.1%})")
        logger.info(f"  ê²€ì¦??°ì´?? {len(val_split)}ê°?({len(val_split)/len(formatted_data):.1%})")
        logger.info(f"  ?ŒìŠ¤???°ì´?? {len(test_split)}ê°?({len(test_split)/len(formatted_data):.1%})")
        
        # ?°ì´?°ì…‹ ?€??
        self.save_dataset(train_split, "train_split.json")
        self.save_dataset(val_split, "validation_split.json")
        self.save_dataset(test_split, "test_split.json")
        
        # ?µê³„ ?ì„±
        stats = self.generate_statistics(formatted_data)
        
        logger.info("?¬ê´„?ì¸ ?ˆë ¨ ?°ì´?°ì…‹ ?ì„± ?„ë£Œ!")
        
        return {
            "total_samples": len(formatted_data),
            "train_samples": len(train_split),
            "validation_samples": len(val_split),
            "test_samples": len(test_split),
            "statistics": stats
        }


def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    logger.info("?¬ê´„?ì¸ ?ˆë ¨ ?°ì´???ì„± ?œì‘...")
    
    # ?ˆë ¨ ?°ì´???ì„±ê¸?ì´ˆê¸°??
    generator = ComprehensiveTrainingDataGenerator(target_size=800)
    
    # ?¬ê´„?ì¸ ?°ì´?°ì…‹ ?ì„±
    result = generator.generate_comprehensive_dataset()
    
    # ê²°ê³¼ ì¶œë ¥
    print("\n" + "="*60)
    print("?“Š ?¬ê´„?ì¸ ?ˆë ¨ ?°ì´?°ì…‹ ?ì„± ê²°ê³¼")
    print("="*60)
    print(f"?“„ ì´??˜í”Œ ?? {result['total_samples']}ê°?)
    print(f"?¯ ?ˆë ¨ ?°ì´?? {result['train_samples']}ê°?)
    print(f"??ê²€ì¦??°ì´?? {result['validation_samples']}ê°?)
    print(f"?§ª ?ŒìŠ¤???°ì´?? {result['test_samples']}ê°?)
    
    stats = result['statistics']
    print(f"\n?“ˆ ?ˆì§ˆ ?µê³„:")
    print(f"  - ?‰ê·  ?ˆì§ˆ ?ìˆ˜: {stats['quality_stats']['average_score']:.3f}")
    print(f"  - ?‰ê·  ? ë¢°?? {stats['confidence_stats']['average_confidence']:.3f}")
    
    print(f"\n?“š ?€?…ë³„ ë¶„í¬:")
    for type_name, count in stats['type_distribution'].items():
        percentage = (count / stats['total_samples']) * 100
        print(f"  - {type_name}: {count}ê°?({percentage:.1f}%)")
    
    print("="*60)
    logger.info("?¬ê´„?ì¸ ?ˆë ¨ ?°ì´???ì„± ?„ë£Œ!")


if __name__ == "__main__":
    main()
