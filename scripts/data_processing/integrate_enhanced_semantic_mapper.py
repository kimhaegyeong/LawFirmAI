#!/usr/bin/env python3
"""
ê¸°ì¡´ keyword_mapper.py??SemanticKeywordMapperë¥??•ì¥??ë²„ì „?¼ë¡œ êµì²´?˜ëŠ” ?¤í¬ë¦½íŠ¸
"""

import os
import shutil
from datetime import datetime

def backup_original_file():
    """?ë³¸ ?Œì¼ ë°±ì—…"""
    original_file = "source/services/langgraph/keyword_mapper.py"
    backup_file = f"source/services/langgraph/keyword_mapper_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    if os.path.exists(original_file):
        shutil.copy2(original_file, backup_file)
        print(f"?ë³¸ ?Œì¼ ë°±ì—… ?„ë£Œ: {backup_file}")
        return True
    return False

def integrate_enhanced_semantic_mapper():
    """?¥ìƒ??SemanticKeywordMapper ?µí•©"""
    try:
        # ë°±ì—… ?ì„±
        if not backup_original_file():
            print("?ë³¸ ?Œì¼??ì°¾ì„ ???†ìŠµ?ˆë‹¤.")
            return False
        
        # ê¸°ì¡´ ?Œì¼ ?½ê¸°
        with open("source/services/langgraph/keyword_mapper.py", 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # ?¥ìƒ??SemanticKeywordMapper ?´ë˜???½ê¸°
        with open("source/services/langgraph/enhanced_semantic_relations.py", 'r', encoding='utf-8') as f:
            enhanced_content = f.read()
        
        # EnhancedSemanticKeywordMapper ?´ë˜??ì¶”ì¶œ
        start_marker = "class EnhancedSemanticKeywordMapper:"
        end_marker = "# ?¬ìš© ?ˆì‹œ"
        
        start_idx = enhanced_content.find(start_marker)
        end_idx = enhanced_content.find(end_marker)
        
        if start_idx == -1 or end_idx == -1:
            print("?¥ìƒ???´ë˜?¤ë? ì°¾ì„ ???†ìŠµ?ˆë‹¤.")
            return False
        
        enhanced_class = enhanced_content[start_idx:end_idx].strip()
        
        # ê¸°ì¡´ SemanticKeywordMapper ?´ë˜??êµì²´
        old_start = original_content.find("class SemanticKeywordMapper:")
        if old_start == -1:
            print("ê¸°ì¡´ SemanticKeywordMapper ?´ë˜?¤ë? ì°¾ì„ ???†ìŠµ?ˆë‹¤.")
            return False
        
        # ê¸°ì¡´ ?´ë˜?¤ì˜ ??ì°¾ê¸°
        old_end = original_content.find("class EnhancedKeywordMapper:", old_start)
        if old_end == -1:
            print("ê¸°ì¡´ ?´ë˜?¤ì˜ ?ì„ ì°¾ì„ ???†ìŠµ?ˆë‹¤.")
            return False
        
        # ?ˆë¡œ???´ìš© ?ì„±
        new_content = (
            original_content[:old_start] +
            enhanced_class + "\n\n" +
            original_content[old_end:]
        )
        
        # ?Œì¼ ?€??
        with open("source/services/langgraph/keyword_mapper.py", 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("?¥ìƒ??SemanticKeywordMapper ?µí•© ?„ë£Œ")
        return True
        
    except Exception as e:
        print(f"?µí•© ì¤??¤ë¥˜ ë°œìƒ: {e}")
        return False

if __name__ == "__main__":
    print("SemanticKeywordMapper ?•ì¥ ?µí•© ?œì‘")
    
    if integrate_enhanced_semantic_mapper():
        print("?µí•© ?„ë£Œ!")
    else:
        print("?µí•© ?¤íŒ¨!")
