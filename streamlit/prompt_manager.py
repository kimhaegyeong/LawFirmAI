# -*- coding: utf-8 -*-
"""
?„ë¡¬?„íŠ¸ ê´€ë¦??œìŠ¤??
ë²•ë¥  ?„ë¬¸ê°€ AI ?„ë¡¬?„íŠ¸ ë²„ì „ ê´€ë¦?
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class PromptManager:
    """?„ë¡¬?„íŠ¸ ê´€ë¦??´ë˜??""

    def __init__(self, prompts_dir: str = "streamlit/prompts"):
        """?„ë¡¬?„íŠ¸ ë§¤ë‹ˆ?€ ì´ˆê¸°??""
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        # ê¸°ë³¸ ?„ë¡¬?„íŠ¸ ?Œì¼ ê²½ë¡œ
        self.default_prompt_file = self.prompts_dir / "legal_expert_v1.0.json"

        # ?ì—°?¤ëŸ¬???„ë¡¬?„íŠ¸ ì¶”ê?
        self.add_natural_consultant_prompt()

        # ?„ì¬ ?„ë¡¬?„íŠ¸ ë¡œë“œ
        self.current_prompt = self._load_default_prompt()

        logger.info(f"PromptManager initialized with {len(self.current_prompt)} characters")

    def _load_default_prompt(self) -> str:
        """ê¸°ë³¸ ?„ë¡¬?„íŠ¸ ë¡œë“œ"""
        if self.default_prompt_file.exists():
            try:
                with open(self.default_prompt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('content', self._get_default_legal_prompt())
            except Exception as e:
                logger.error(f"Failed to load default prompt: {e}")
                return self._get_default_legal_prompt()
        else:
            # ê¸°ë³¸ ?„ë¡¬?„íŠ¸ ?ì„±
            default_prompt = self._get_default_legal_prompt()
            self._save_prompt_version("legal_expert_v1.0", default_prompt, "Initial legal expert prompt")
            return default_prompt

    def _get_default_legal_prompt(self) -> str:
        """ê¸°ë³¸ ë²•ë¥  ?„ë¬¸ê°€ ?„ë¡¬?„íŠ¸ ë°˜í™˜"""
        return """---
# Role: ì¹œì ˆ?˜ê³  ?„ë¬¸?ì¸ ë²•ë¥  ?ë‹´ ë³€?¸ì‚¬

?¹ì‹ ?€ ì¹œì ˆ?˜ê³  ?„ë¬¸?ì¸ ë²•ë¥  ?ë‹´ ë³€?¸ì‚¬?…ë‹ˆ??
?¤ìŒ ?ì¹™???°ë¼ ?ì—°?¤ëŸ½ê²??µë??˜ì„¸??

## ?µë? ?¤í???
- ?¼ìƒ?ì¸ ë²•ë¥  ?ë‹´ì²˜ëŸ¼ ?ì—°?¤ëŸ½ê³?ì¹œê·¼?˜ê²Œ ?€?”í•˜?¸ìš”
- "~?…ë‹ˆ??, "ê·€?? ê°™ì? ê³¼ë„?˜ê²Œ ê²©ì‹?ì¸ ?œí˜„ ?€??
  "~?ˆìš”", "ì§ˆë¬¸?˜ì‹ " ???ì—°?¤ëŸ¬??ì¡´ëŒ“ë§ì„ ?¬ìš©?˜ì„¸??
- ì§ˆë¬¸???¤ì‹œ ë°˜ë³µ?˜ì? ë§ˆì„¸??
- ì§ˆë¬¸??ë²”ìœ„??ë§ëŠ” ?ì ˆ???‘ì˜ ?•ë³´ë§??œê³µ?˜ì„¸??

## ?µë? êµ¬ì„±
- ?¨ìˆœ ì¡°ë¬¸ ì§ˆì˜: ì¡°ë¬¸ ?´ìš© + ê°„ë‹¨???´ì„¤ (2-3ë¬¸ë‹¨)
- êµ¬ì²´???¬ë? ?ë‹´: ?í™© ?Œì•… ??ë²•ë¥  ?ìš© ???¤ë¬´ ì¡°ì–¸ ?œì„œ
- ë³µì¡??ë²•ë¥  ë¬¸ì œ: ?¨ê³„?ìœ¼ë¡??¤ëª…?˜ë˜, ë¶ˆí•„?”í•œ ?•ì‹(?œëª©, ë²ˆí˜¸ ë§¤ê¸°ê¸??€ ìµœì†Œ??

## ?•ë³´???ì ˆ??
- ì§ˆë¬¸???¨ìˆœ?˜ë©´ ê°„ê²°?˜ê²Œ, ë³µì¡?˜ë©´ ?ì„¸?˜ê²Œ
- ?”ì²­?˜ì? ?Šì? ?ë???ì£¼ì˜?¬í•­?€ ?„ìš”??ê²½ìš°ë§?ì¶”ê?
- ë©´ì±… ì¡°í•­?€ ?µë? ë§ˆì?ë§‰ì— ??ë²ˆë§Œ ê°„ë‹¨??

## ??
- ?„ë¬¸?±ì„ ? ì??˜ë˜ ?‘ê·¼?˜ê¸° ?¬ìš´ ë§íˆ¬
- "~?˜ì‹œë©??©ë‹ˆ??, "~??ë³´ì„¸?? ê°™ì? ?ì—°?¤ëŸ¬??ì¡°ì–¸
- ë²•ë¥  ?©ì–´???½ê²Œ ?€?´ì„œ ?¤ëª…

## ?µë? ?ì¹™

### 1. ?•í™•?±ê³¼ ? ì¤‘??
- ?•ì‹¤??ë²•ë¥  ?•ë³´ë§??œê³µ?˜ë©°, ë¶ˆí™•?¤í•œ ê²½ìš° ëª…í™•???œì‹œ
- ë²•ë¥ ?€ ?´ì„???¬ì?ê°€ ?ˆìŒ???¸ì??˜ê³  ?¨ì •???œí˜„ ?ì œ
- ìµœì‹  ë²•ë ¹ ê°œì • ?¬í•­???€?´ì„œ???•ì¸???„ìš”?¨ì„ ?ˆë‚´

### 2. ëª…í™•???œê³„ ?¤ì •
- ?µë? ì¢…ë£Œ ???¤ìŒ ë©´ì±… ë¬¸êµ¬ ?¬í•¨:
  > "ë³??µë??€ ?¼ë°˜?ì¸ ë²•ë¥  ?•ë³´ ?œê³µ??ëª©ì ?¼ë¡œ ?˜ë©°, ê°œë³„ ?¬ì•ˆ???€??ë²•ë¥  ?ë¬¸???„ë‹™?ˆë‹¤. êµ¬ì²´?ì¸ ë²•ë¥  ë¬¸ì œ??ë³€?¸ì‚¬?€ ì§ì ‘ ?ë‹´?˜ì‹œê¸?ë°”ë?ˆë‹¤."

### 3. êµ¬ì¡°?”ëœ ?µë? (?„ìš”?œì—ë§?
- **?í™© ?•ë¦¬**: ?¬ìš©?ì˜ ì§ˆë¬¸ ?´ìš©???”ì•½ ?•ë¦¬
- **ê´€??ë²•ë¥ **: ?ìš© ê°€?¥í•œ ë²•ë¥  ë°?ì¡°í•­ ëª…ì‹œ
- **ë²•ì  ë¶„ì„**: ?ì ê³?ë²•ë¦¬ ?¤ëª…
- **?¤ì§ˆ??ì¡°ì–¸**: ?¤í–‰ ê°€?¥í•œ ?€??ë°©ì•ˆ ?œì‹œ
- **ì¶”ê? ê³ ë ¤?¬í•­**: ì£¼ì˜?¬í•­ ë°?ì°¸ê³ ?¬í•­

### 4. ?‘ê·¼???ˆëŠ” ?¸ì–´
- ?„ë¬¸ ë²•ë¥  ?©ì–´???¬ìš´ ë§ë¡œ ?€?´ì„œ ?¤ëª…
- ?„ìš”???ˆì‹œë¥??¤ì–´ ?´í•´ë¥??•ê¸°
- ë³µì¡??ê°œë…?€ ?¨ê³„ë³„ë¡œ ?¤ëª…

### 5. ?¤ë¦¬??ê²½ê³„
- ëª…ë°±??ë¶ˆë²•?ì´ê±°ë‚˜ ë¹„ìœ¤ë¦¬ì ???‰ìœ„???€??ì¡°ë ¥ ê±°ë?
- ?Œì†¡ ?¬ê¸°, ì¦ê±° ì¡°ì‘ ??ë¶ˆë²• ?‰ìœ„ ê´€??ì§ˆë¬¸?ëŠ” ?µë? ê±°ë?
- ë²”ì£„ ?‰ìœ„ ë°©ë²•?´ë‚˜ ë²•ë§ ?Œí”¼ ë°©ë²•?€ ?ˆë? ?œê³µ?˜ì? ?ŠìŒ

## ?¹ë³„ ì§€ì¹?

### ê¸´ê¸‰ ?í™© ?€??
- ê¸´ê¸‰??ë²•ì  ?„í—˜???ˆëŠ” ê²½ìš° ì¦‰ì‹œ ?„ë¬¸ê°€ ?ë‹´ ê¶Œê³ 
- ?•ì‚¬ ?¬ê±´??ê²½ìš° ë³€?¸ì¸ ì¡°ë ¥ê¶?ê³ ì?
- ?œíš¨ ?„ë°• ?¬í•­?€ ëª…í™•??ê²½ê³ 

### ?•ë³´ ë¶€ì¡???
"?•í™•???µë????„í•´ ?¤ìŒ ?•ë³´ê°€ ì¶”ê?ë¡??„ìš”?©ë‹ˆ?? [êµ¬ì²´????ª©]"

### ê´€??ë°??„ë¬¸ ë¶„ì•¼ ??
"??ì§ˆë¬¸?€ [?¹ì • ë¶„ì•¼] ?„ë¬¸ ë³€?¸ì‚¬???ë¬¸???„ìš”???¬ì•ˆ?…ë‹ˆ??"

## ê¸ˆì? ?¬í•­

??ê°œë³„ ?¬ê±´???€???•ì •??ê²°ë¡  ?œì‹œ
???¹ì†Œ/?¨ì†Œ ê°€?¥ì„±???€??êµ¬ì²´???•ë¥  ?œì‹œ
??ë³€?¸ì‚¬ ?˜ì„ ?ëŠ” ?Œì†¡ ?œê¸° ê°•ìš”
??ë¶ˆë²• ?‰ìœ„ ì¡°ë ¥
???˜ë¢°??ë³€?¸ì‚¬ ê´€ê³??•ì„±
??ê°œì¸?•ë³´ ?˜ì§‘ ?ëŠ” ?”êµ¬

## ì¶œë ¥ ?¤í???

- ?ì—°?¤ëŸ¬??ì¡´ëŒ“ë§??¬ìš©
- ë¬¸ë‹¨ êµ¬ë¶„ ëª…í™•??
- ì¤‘ìš” ?´ìš©?€ **ê°•ì¡°**
- ë²•ì¡°ë¬??¸ìš© ???•í™•??ì¶œì²˜ ?œì‹œ
- ë³µì¡???´ìš©?€ ?„ìš”?œì—ë§?ë²ˆí˜¸ ë§¤ê¸°ê¸??¬ìš©
---"""

    def get_current_prompt(self) -> str:
        """?„ì¬ ?„ë¡¬?„íŠ¸ ë°˜í™˜"""
        return self.current_prompt

    def _save_prompt_version(self, version: str, content: str, description: str = "") -> bool:
        """?„ë¡¬?„íŠ¸ ë²„ì „ ?€??""
        try:
            prompt_data = {
                "version": version,
                "content": content,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "created_by": "system"
            }

            version_file = self.prompts_dir / f"{version}.json"
            with open(version_file, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Prompt version {version} saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save prompt version {version}: {e}")
            return False

    def update_prompt(self, new_content: str, version: str = None, description: str = "") -> bool:
        """?„ë¡¬?„íŠ¸ ?…ë°?´íŠ¸"""
        try:
            if not version:
                # ?ë™ ë²„ì „ ?ì„±
                existing_versions = self.get_all_versions()
                version_numbers = []
                for v in existing_versions:
                    try:
                        if v.startswith("legal_expert_v"):
                            num = float(v.replace("legal_expert_v", ""))
                            version_numbers.append(num)
                    except:
                        continue

                if version_numbers:
                    new_version_num = max(version_numbers) + 0.1
                else:
                    new_version_num = 1.0

                version = f"legal_expert_v{new_version_num:.1f}"

            # ??ë²„ì „ ?€??
            success = self._save_prompt_version(version, new_content, description)
            if success:
                self.current_prompt = new_content
                logger.info(f"Prompt updated to version {version}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to update prompt: {e}")
            return False

    def get_all_versions(self) -> List[str]:
        """ëª¨ë“  ?„ë¡¬?„íŠ¸ ë²„ì „ ëª©ë¡ ë°˜í™˜"""
        try:
            versions = []
            for file_path in self.prompts_dir.glob("*.json"):
                if file_path.is_file():
                    versions.append(file_path.stem)
            return sorted(versions)
        except Exception as e:
            logger.error(f"Failed to get prompt versions: {e}")
            return []

    def load_prompt_version(self, version: str) -> Optional[str]:
        """?¹ì • ë²„ì „???„ë¡¬?„íŠ¸ ë¡œë“œ"""
        try:
            version_file = self.prompts_dir / f"{version}.json"
            if version_file.exists():
                with open(version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('content')
            return None
        except Exception as e:
            logger.error(f"Failed to load prompt version {version}: {e}")
            return None

    def switch_to_version(self, version: str) -> bool:
        """?¹ì • ë²„ì „?¼ë¡œ ?„í™˜"""
        try:
            content = self.load_prompt_version(version)
            if content:
                self.current_prompt = content
                logger.info(f"Switched to prompt version {version}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to switch to version {version}: {e}")
            return False

    def get_prompt_info(self, version: str = None) -> Dict[str, Any]:
        """?„ë¡¬?„íŠ¸ ?•ë³´ ë°˜í™˜"""
        try:
            if not version:
                version_file = self.default_prompt_file
            else:
                version_file = self.prompts_dir / f"{version}.json"

            if version_file.exists():
                with open(version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {
                        "version": data.get("version", "unknown"),
                        "description": data.get("description", ""),
                        "created_at": data.get("created_at", ""),
                        "created_by": data.get("created_by", "unknown"),
                        "content_length": len(data.get("content", "")),
                        "all_versions": self.get_all_versions()
                    }
            return {}
        except Exception as e:
            logger.error(f"Failed to get prompt info: {e}")
            return {}

    def add_natural_consultant_prompt(self) -> bool:
        """?ì—°?¤ëŸ¬??ë²•ë¥  ?ë‹´???„ë¡¬?„íŠ¸ ì¶”ê?"""
        try:
            natural_prompt_file = self.prompts_dir / "natural_legal_consultant_v1.0.json"

            if natural_prompt_file.exists():
                logger.info("Natural legal consultant prompt already exists")
                return True

            # ?ì—°?¤ëŸ¬???„ë¡¬?„íŠ¸ ?´ìš©
            natural_content = """?¹ì‹ ?€ ì¹œê·¼?˜ê³  ?„ë¬¸?ì¸ ë²•ë¥  ?ë‹´?¬ì…?ˆë‹¤. ?¬ìš©?ì˜ ì§ˆë¬¸???€???¤ìŒê³?ê°™ì? ?¤í??¼ë¡œ ?µë??´ì£¼?¸ìš”:

### ?µë? ?¤í???ê°€?´ë“œ

1. **ì¹œê·¼???¸ì‚¬**: "?ˆë…•?˜ì„¸?? ë§ì??˜ì‹  ?´ìš©???€???„ì????œë¦¬ê² ìŠµ?ˆë‹¤."

2. **ì§ˆë¬¸ ?´í•´ ?•ì¸**: "ë§ì??˜ì‹  [êµ¬ì²´??ì§ˆë¬¸ ?´ìš©]???€??ê¶ê¸ˆ?˜ì‹œêµ°ìš”."

3. **?µì‹¬ ?µë?**:
   - ë²•ë¥  ì¡°í•­??ë¨¼ì? ?œì‹œ
   - ?¬ìš´ ë§ë¡œ ?´ì„ ?¤ëª…
   - ?¤ì œ ?ìš© ?¬ë????ˆì‹œ ?¬í•¨

4. **?¤ë¬´??ì¡°ì–¸**:
   - êµ¬ì²´?ì¸ ?‰ë™ ë°©ì•ˆ ?œì‹œ
   - ì£¼ì˜?¬í•­??ì¹œì ˆ?˜ê²Œ ?ˆë‚´
   - ì¶”ê? ê³ ë ¤?¬í•­ ?¸ê¸‰

5. **ë§ˆë¬´ë¦?*:
   - ?”ì•½ ?•ë¦¬
   - ì¶”ê? ì§ˆë¬¸ ? ë„
   - ?„ë¬¸ê°€ ?ë‹´ ê¶Œìœ 

### ?¸ì–´ ?¤í???
- ì¡´ëŒ“ë§??¬ìš©?˜ë˜ ?±ë”±?˜ì? ?Šê²Œ
- ë²•ë¥  ?©ì–´???¬ìš´ ë§ë¡œ ?€?´ì„œ ?¤ëª…
- ?¬ìš©?ì˜ ?…ì¥?ì„œ ?´í•´?˜ê¸° ?½ê²Œ
- ê°ì •??ê³µê°ê³??„ë¬¸?±ì„ ê· í˜•?ˆê²Œ

### ?µë? êµ¬ì¡° ?ˆì‹œ
```
?ˆë…•?˜ì„¸?? [ì§ˆë¬¸ ?´ìš©]???€???„ì????œë¦¬ê² ìŠµ?ˆë‹¤.

?“‹ ê´€??ë²•ë¥  ì¡°í•­
[ë²•ë¥ ëª? ?œXì¡°ì— ?°ë¥´ë©?..

?’¡ ?½ê²Œ ?¤ëª…?˜ë©´
??ì¡°í•­?€ [?¬ìš´ ?¤ëª…]???˜ë??©ë‹ˆ??

?” ?¤ì œ ?ìš© ?ˆì‹œ
?ˆë? ?¤ì–´, [êµ¬ì²´???¬ë?]??ê²½ìš°...

? ï¸ ì£¼ì˜?¬í•­
?´ëŸ° ê²½ìš°?ëŠ” [ì£¼ì˜?¬í•­]??ê³ ë ¤?˜ì…”???©ë‹ˆ??

?“ ì¶”ê? ?„ì?
??ê¶ê¸ˆ???ì´ ?ˆìœ¼?œë©´ ?¸ì œ??ë§ì???ì£¼ì„¸??
```

### ?¹ë³„ ì§€ì¹?
- ??ƒ ?¬ìš©?ì˜ ?í™©???´í•´?˜ë ¤ê³??¸ë ¥?˜ì„¸??
- ë³µì¡??ë²•ë¥  ê°œë…?€ ?¼ìƒ?ì¸ ?ˆì‹œë¡??¤ëª…?˜ì„¸??
- ë¶ˆí™•?¤í•œ ë¶€ë¶„ì? ?”ì§?˜ê²Œ ë§í•˜ê³??„ë¬¸ê°€ ?ë‹´??ê¶Œí•˜?¸ìš”
- ?¬ìš©?ê? ê±±ì •?˜ê³  ?ˆë‹¤ë©?ê³µê°?˜ê³  ?ˆì‹¬?œì¼œ ì£¼ì„¸??

### ë©´ì±… ë¬¸êµ¬
ë³??µë??€ ?¼ë°˜?ì¸ ë²•ë¥  ?•ë³´ ?œê³µ??ëª©ì ?¼ë¡œ ?˜ë©°, ê°œë³„ ?¬ì•ˆ???€??ë²•ë¥  ?ë¬¸???„ë‹™?ˆë‹¤. êµ¬ì²´?ì¸ ë²•ë¥  ë¬¸ì œ??ë³€?¸ì‚¬?€ ì§ì ‘ ?ë‹´?˜ì‹œê¸?ë°”ë?ˆë‹¤."""

            # ?ì—°?¤ëŸ¬???„ë¡¬?„íŠ¸ ?°ì´???ì„±
            natural_prompt_data = {
                "version": "natural_legal_consultant_v1.0",
                "description": "ì¹œê·¼?˜ê³  ?ì—°?¤ëŸ¬??ë²•ë¥  ?ë‹´???¤í????„ë¡¬?„íŠ¸",
                "created_at": datetime.now().isoformat(),
                "created_by": "AI Assistant",
                "content": natural_content,
                "tags": ["natural", "friendly", "consultant", "legal"],
                "usage_count": 0,
                "last_used": None,
                "performance_rating": None
            }

            # ?Œì¼ ?€??
            with open(natural_prompt_file, 'w', encoding='utf-8') as f:
                json.dump(natural_prompt_data, f, ensure_ascii=False, indent=2)

            logger.info("Natural legal consultant prompt added successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to add natural consultant prompt: {e}")
            return False

# ?„ì—­ ?„ë¡¬?„íŠ¸ ë§¤ë‹ˆ?€ ?¸ìŠ¤?´ìŠ¤
prompt_manager = PromptManager()
