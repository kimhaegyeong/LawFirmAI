# -*- coding: utf-8 -*-
"""
?„ë¡¬?„íŠ¸ ë¹Œë” ëª¨ë“ˆ
ë¦¬íŒ©? ë§: legal_workflow_enhanced.py?ì„œ ?„ë¡¬?„íŠ¸ ë¹Œë” ë©”ì„œ??ë¶„ë¦¬
"""

from typing import Any, Dict, List, Optional


class QueryBuilder:
    """ì¿¼ë¦¬ ê´€???„ë¡¬?„íŠ¸ ë¹Œë”"""

    @staticmethod
    def build_semantic_query(query: str, expanded_terms: List[str]) -> str:
        """?˜ë???ê²€?‰ìš© ì¿¼ë¦¬ ?ì„±"""
        # ?µì‹¬ ?¤ì›Œ??3-5ê°?? íƒ
        key_terms = expanded_terms[:5] if expanded_terms else []
        if key_terms:
            return f"{query} {' '.join(key_terms)}"
        return query

    @staticmethod
    def build_keyword_queries(
        query: str,
        expanded_terms: List[str],
        query_type: str
    ) -> List[str]:
        """?¤ì›Œ??ê²€?‰ìš© ì¿¼ë¦¬ ë¦¬ìŠ¤???ì„±"""
        queries = []

        # ?ë³¸ ì¿¼ë¦¬
        queries.append(query)

        # ì§ˆë¬¸ ? í˜•ë³??¹í™” ì¿¼ë¦¬
        if query_type == "precedent_search":
            # ?ë? ê²€?? "?ë?", "?¬ê±´", "?€ë²•ì›" ??ì¶”ê?
            queries.append(f"{query} ?ë?")
            queries.append(f"{query} ?¬ê±´")
        elif query_type == "law_inquiry":
            # ë²•ë ¹ ì¡°ë¬¸ ê²€?? "ë²•ë¥ ", "ì¡°í•­", "ì¡°ë¬¸" ??ì¶”ê?
            queries.append(f"{query} ë²•ë¥  ì¡°í•­")
            queries.append(f"{query} ë²•ë ¹")
        elif query_type == "legal_advice":
            # ë²•ë¥  ì¡°ì–¸: "ì¡°ì–¸", "?´ì„", "ê¶Œë¦¬" ??ì¶”ê?
            queries.append(f"{query} ì¡°ì–¸")
            queries.append(f"{query} ?´ì„")

        # ?•ì¥???¤ì›Œ??ì¡°í•© (ìµœë? 3ê°?
        if expanded_terms and len(expanded_terms) >= 3:
            queries.append(" ".join(expanded_terms[:3]))

        return queries[:5]  # ìµœë? 5ê°?ì¿¼ë¦¬

    @staticmethod
    def build_conversation_context_dict(context) -> Optional[Dict[str, Any]]:
        """ConversationContextë¥??•ì…”?ˆë¦¬ë¡?ë³€??""
        try:
            if not context:
                return None

            return {
                "session_id": context.session_id if hasattr(context, 'session_id') else "",
                "turn_count": len(context.turns) if hasattr(context, 'turns') else 0,
                "entities": {
                    entity_type: list(entity_set)
                    for entity_type, entity_set in (context.entities or {}).items()
                } if hasattr(context, 'entities') and context.entities else {},
                "topic_stack": list(context.topic_stack) if hasattr(context, 'topic_stack') else [],
                "recent_topics": list(context.topic_stack[-3:]) if hasattr(context, 'topic_stack') and context.topic_stack else []
            }
        except Exception as e:
            return None


class PromptBuilder:
    """?¼ë°˜ ?„ë¡¬?„íŠ¸ ë¹Œë”"""

    @staticmethod
    def build_query_enhancement_prompt_base(
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str,
        format_field_info: callable,
        format_query_guide: callable
    ) -> str:
        """
        ì¿¼ë¦¬ ê°•í™”ë¥??„í•œ LLM ?„ë¡¬?„íŠ¸ ?ì„± (ê¸°ë³¸ ë²„ì „)

        Args:
            query: ?ë³¸ ì¿¼ë¦¬
            query_type: ì§ˆë¬¸ ? í˜•
            extracted_keywords: ì¶”ì¶œ???¤ì›Œ??ëª©ë¡
            legal_field: ë²•ë¥  ë¶„ì•¼
            format_field_info: ë²•ë¥  ë¶„ì•¼ ?•ë³´ ?¬ë§· ?¨ìˆ˜
            format_query_guide: ì§ˆë¬¸ ? í˜•ë³?ê°€?´ë“œ ?¬ë§· ?¨ìˆ˜

        Returns:
            ?„ë¡¬?„íŠ¸ ë¬¸ì??
        """
        # ?…ë ¥ ?°ì´??ê²€ì¦?ë°??•ê·œ??
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        if not query_type or not isinstance(query_type, str):
            query_type = "general_question"

        if not isinstance(extracted_keywords, list):
            extracted_keywords = []

        if not isinstance(legal_field, str):
            legal_field = ""

        legal_field_text = legal_field.strip() if legal_field else "ë¯¸ì???

        # ?¤ì›Œ???ì„¸ ?•ë³´ êµ¬ì„±
        keywords_info = ""
        if extracted_keywords and len(extracted_keywords) > 0:
            keywords_list = ", ".join(extracted_keywords[:10])
            keywords_info = f"""
### ì¶”ì¶œ???¤ì›Œ??
{keywords_list}
**ì´?{len(extracted_keywords)}ê°?*"""
        else:
            keywords_info = """
### ì¶”ì¶œ???¤ì›Œ??
(?†ìŒ)"""

        # ì§ˆë¬¸ ? í˜•ë³?ê°€?´ë“œ ?•ë³´ ê°€?¸ì˜¤ê¸?(format_query_guide ?¬ìš©)
        query_guide = format_query_guide(query_type) if format_query_guide else {}
        field_info = format_field_info(legal_field) if format_field_info else {}

        # ?°ì´?°ë² ?´ìŠ¤ êµ¬ì¡° ?•ë³´
        database_info = """
## ?“Š ê²€???€???°ì´?°ë² ?´ìŠ¤ êµ¬ì¡°

### ì£¼ìš” ?Œì´ë¸?ë°??„ë“œ

**ë²•ë ¹ ?°ì´??(statutes, statute_articles)**
- ë²•ë ¹ëª?(statutes.name), ?½ì¹­ (statutes.abbrv)
- ì¡°ë¬¸ë²ˆí˜¸ (statute_articles.article_no), ì¡°í•­ ë²ˆí˜¸ (clause_no, item_no)
- ì¡°ë¬¸ ?´ìš© (statute_articles.text), ?œëª© (statute_articles.heading)
- ?œí–‰??(statutes.effective_date), ê³µí¬??(statutes.proclamation_date)

**?ë? ?°ì´??(cases, case_paragraphs)**
- ?¬ê±´ë²ˆí˜¸ (cases.case_number, ?•ì‹: YYYY???˜XXXXX)
- ë²•ì›ëª?(cases.court: ?€ë²•ì›, ê³ ë“±ë²•ì›, ì§€ë°©ë²•????
- ?¬ê±´ëª?(cases.casenames)
- ? ê³ ??(cases.announce_date)
- ?ë? ë³¸ë¬¸ (case_paragraphs.text)

**?¬ê²°ë¡€ ?°ì´??(decisions, decision_paragraphs)**
- ê¸°ê? (decisions.org)
- ë¬¸ì„œ ID (decisions.doc_id)
- ê²°ì •??(decisions.decision_date)
- ?¬ê²° ?´ìš© (decision_paragraphs.text)

**? ê¶Œ?´ì„ ?°ì´??(interpretations, interpretation_paragraphs)**
- ê¸°ê? (interpretations.org)
- ë¬¸ì„œ ID (interpretations.doc_id)
- ?œëª© (interpretations.title)
- ?‘ë‹µ??(interpretations.response_date)
- ?´ì„ ?´ìš© (interpretation_paragraphs.text)

### ê²€??ë°©ì‹
- **ë²¡í„° ê²€??*: ?˜ë? ê¸°ë°˜ ? ì‚¬??ê²€??(ë²•ë¥  ì¡°ë¬¸, ?ë? ë³¸ë¬¸ ?„ì²´ ?ìŠ¤??
- **?¤ì›Œ??ê²€??*: FTS5 ê¸°ë°˜ ?¤ì›Œ??ë§¤ì¹­ (ë²•ë ¹ëª? ì¡°ë¬¸ë²ˆí˜¸, ?¬ê±´ë²ˆí˜¸ ??
- **?˜ì´ë¸Œë¦¬??ê²€??*: ë²¡í„° + ?¤ì›Œ??ê²°ê³¼ ë³‘í•© ë°??¬ë­??
"""

        prompt = f"""?¹ì‹ ?€ ë²•ë¥  ê²€??ì¿¼ë¦¬ ìµœì ???„ë¬¸ê°€?…ë‹ˆ?? ì£¼ì–´ì§?ê²€??ì¿¼ë¦¬ë¥?ë²•ë¥  ?°ì´?°ë² ?´ìŠ¤ ê²€?‰ì— ìµœì ?”í•˜?„ë¡ ê°œì„ ?´ì£¼?¸ìš”.

## ?¯ ?‘ì—… ëª©í‘œ

ì£¼ì–´ì§?ì§ˆë¬¸???€???¤ìŒ???˜í–‰?˜ì„¸??
1. **ê²€???•í™•???¥ìƒ**: ë²•ë¥  ?°ì´?°ë² ?´ìŠ¤?ì„œ ê´€??ë¬¸ì„œë¥????•í™•?˜ê²Œ ì°¾ì„ ???ˆë„ë¡??¤ì›Œ??ìµœì ??
2. **ê²€??ë²”ìœ„ ?•ì¥**: ?™ì˜?? ê´€???©ì–´, ?ìœ„ ê°œë…??ì¶”ê??˜ì—¬ ê²€???„ë½ ë°©ì?
3. **ê²€???¨ìœ¨??ì¦ë?**: ë²¡í„° ê²€?‰ê³¼ ?¤ì›Œ??ê²€??ëª¨ë‘???¨ê³¼?ì¸ ì¿¼ë¦¬ ?ì„±
4. **ë²•ë¥  ?„ë¬¸??ë°˜ì˜**: ë²•ë¥  ë¶„ì•¼ ?¹ì„±ê³?ì§ˆë¬¸ ? í˜•??ë§ëŠ” ?„ë¬¸ ?©ì–´ ?œìš©

{database_info}

## ?“‹ ?…ë ¥ ?•ë³´ (?ì„¸)

### ê¸°ë³¸ ?•ë³´
**?ë³¸ ì¿¼ë¦¬**: "{query}"
**ì§ˆë¬¸ ? í˜•**: {query_type} ({query_guide.get('description', '?¼ë°˜ ê²€??)})
**ë²•ë¥  ë¶„ì•¼**: {legal_field_text}

{keywords_info}

### ì§ˆë¬¸ ? í˜•ë³?ê²€???„ëµ
**?„ì¬ ì§ˆë¬¸ ? í˜•**: {query_guide.get('description', '?¼ë°˜ ê²€??)}

**ê²€??ì´ˆì **: {query_guide.get('search_focus', 'ê´€??ë²•ë ¹, ?ë?, ë²•ë¥  ?©ì–´')}

**ê²€???„ëµ**: {query_guide.get('search_strategy', '?µì‹¬ ?¤ì›Œ??ì¤‘ì‹¬ ê²€??)}

**?°ì´?°ë² ?´ìŠ¤ ?„ë“œ**: {query_guide.get('database_fields', '?„ì²´ ?°ì´?°ë² ?´ìŠ¤')}

**ì¶”ì²œ ?¤ì›Œ??*: {', '.join(query_guide.get('keyword_suggestions', [])[:8])}

### ë²•ë¥  ë¶„ì•¼ë³??•ë³´
{format_field_info(legal_field) if format_field_info else '?†ìŒ'}

## ?” ì¿¼ë¦¬ ìµœì ??ì§€ì¹?

### 1. ?˜ë? ë³´ì¡´
- ?ë³¸ ì¿¼ë¦¬???µì‹¬ ?˜ë„?€ ëª©ì ??ë°˜ë“œ??? ì??˜ì„¸??
- ?¬ìš©?ê? ì°¾ê³ ???˜ëŠ” ë²•ë¥  ?•ë³´??ë³¸ì§ˆ???Œì•…?˜ì„¸??

### 2. ë²•ë¥  ?©ì–´ ?•ì¥
- **?™ì˜??ì¶”ê?**: ë²•ë¥  ?©ì–´???¤ì–‘???œí˜„ ì¶”ê? (?? "ê³„ì•½" ??"ê³„ì•½??, "ê³„ì•½ê´€ê³?)
- **?ìœ„/?˜ìœ„ ê°œë…**: ?¼ë°˜ ê°œë…ê³?êµ¬ì²´??ê°œë… ëª¨ë‘ ?¬í•¨ (?? "?í•´ë°°ìƒ" ??"ë¶ˆë²•?‰ìœ„ ?í•´ë°°ìƒ", "ê³„ì•½ ?„ë°˜ ?í•´ë°°ìƒ")
- **ë²•ë¥  ?©ì–´ ?•ê·œ??*: ë²•ë¥ ?ì„œ ?¬ìš©?˜ëŠ” ê³µì‹ ?©ì–´ë¡?ë³€??(?? "?´í˜¼" ??"?¼ì¸?´ì†Œ")

### 3. ê²€??ìµœì ??
- **ë²¡í„° ê²€??ìµœì ??*: ?˜ë??ìœ¼ë¡?? ì‚¬??ë¬¸ì„œë¥?ì°¾ê¸° ?„í•œ ?µì‹¬ ê°œë… ?¤ì›Œ???¬í•¨
- **?¤ì›Œ??ê²€??ìµœì ??*: ë²•ë ¹ëª? ì¡°ë¬¸ë²ˆí˜¸, ?¬ê±´ë²ˆí˜¸ ???•í™•??ë§¤ì¹­ ê°€?¥í•œ ?©ì–´ ?¬í•¨
- **?˜ì´ë¸Œë¦¬??ê²€??*: ??ê²€??ë°©ì‹ ëª¨ë‘???¨ê³¼?ì¸ ê· í˜• ?¡íŒ ì¿¼ë¦¬ ?ì„±

### 4. ì§ˆë¬¸ ? í˜•ë³??¹í™”
- **?ë? ê²€??*: ?¬ê±´ë²ˆí˜¸ ?¨í„´, ë²•ì›ëª? ?ì‹œ?¬í•­ ê´€???¤ì›Œ??ì¶”ê?
- **ë²•ë ¹ ì¡°íšŒ**: ë²•ë ¹ëª? ì¡°ë¬¸ë²ˆí˜¸, ì¡°í•­???µì‹¬ ë²•ë¦¬ ?©ì–´ ?¬í•¨
- **ë²•ë¥  ì¡°ì–¸**: ë¬¸ì œ ?í™©???µì‹¬ ë²•ë¥  ê°œë… + ê´€??ì¡°ë¬¸ + ? ì‚¬ ?ë? ?¨í„´ ì¡°í•©

### 5. ê°„ê²°??? ì?
- ?µì‹¬ ?¤ì›Œ?œëŠ” ë°˜ë“œ??? ì?
- ê²€?‰ì— ë¶ˆí•„?”í•œ ?˜ì‹?´ë‚˜ ì¤‘ë³µ ?œí˜„ ?œê±°
- ìµœë? 50???´ë‚´ë¡?ê°„ê²°?˜ê²Œ ? ì?

## ?“¤ ì¶œë ¥ ?•ì‹

?¤ìŒ JSON ?•ì‹?¼ë¡œ ?‘ë‹µ?˜ì„¸??(?¤ëª… ?†ì´ JSONë§?ì¶œë ¥):

```json
{{
    "optimized_query": "ìµœì ?”ëœ ê²€??ì¿¼ë¦¬ (50???´ë‚´)",
    "expanded_keywords": ["?¤ì›Œ??", "?¤ì›Œ??", "?¤ì›Œ??", ...],
    "keyword_variants": ["ë³€??ì¿¼ë¦¬1", "ë³€??ì¿¼ë¦¬2", ...],
    "legal_terms": ["ë²•ë¥  ?©ì–´1", "ë²•ë¥  ?©ì–´2", ...],
    "reasoning": "ìµœì ???¬ìœ  ë°?ê²€???„ëµ ?¤ëª… (?œêµ­??"
}}
```

## ? ï¸ ì£¼ì˜?¬í•­

1. **?ë³¸ ì¿¼ë¦¬ ?˜ë? ë³´ì¡´**: ìµœì ??ê³¼ì •?ì„œ ?¬ìš©?ì˜ ?ë˜ ?˜ë„ë¥??œê³¡?˜ì? ë§ˆì„¸??
2. **ë²•ë¥  ?„ë¬¸??*: ë²•ë¥  ?©ì–´ë¥??•í™•?˜ê²Œ ?¬ìš©?˜ê³ , ë²•ë¥  ?°ì´?°ë² ?´ìŠ¤ êµ¬ì¡°ë¥?ê³ ë ¤?˜ì„¸??
3. **ê²€???¨ìœ¨??*: ë²¡í„° ê²€?‰ê³¼ ?¤ì›Œ??ê²€??ëª¨ë‘???¨ê³¼?ì¸ ì¿¼ë¦¬ë¥??ì„±?˜ì„¸??
4. **ê°„ê²°??*: ë¶ˆí•„?”í•œ ?¨ì–´ë¥??œê±°?˜ê³  ?µì‹¬ ?¤ì›Œ?œë§Œ ?¬í•¨?˜ì„¸??
"""

        return prompt
