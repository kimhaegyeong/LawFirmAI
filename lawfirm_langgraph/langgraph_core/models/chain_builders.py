# -*- coding: utf-8 -*-
"""
?„ë¡¬?„íŠ¸ ì²´ì¸ ë¹Œë” ëª¨ë“ˆ
ë¦¬íŒ©? ë§: legal_workflow_enhanced.py?ì„œ ?„ë¡¬?„íŠ¸ ì²´ì¸ ë¹Œë” ?¨ìˆ˜ ë¶„ë¦¬
"""

from typing import Any, Dict, Optional


class DirectAnswerChainBuilder:
    """ì§ì ‘ ?µë? ?ì„± ì²´ì¸ ë¹Œë”"""

    @staticmethod
    def build_query_type_analysis_prompt(query: str) -> str:
        """ì§ˆë¬¸ ? í˜• ë¶„ì„ ?„ë¡¬?„íŠ¸ ?ì„±"""
        return f"""?¤ìŒ ì§ˆë¬¸??? í˜•??ë¶„ì„?´ì£¼?¸ìš”.

ì§ˆë¬¸: {query}

?¤ìŒ ? í˜• ì¤??˜ë‚˜ë¥?? íƒ?˜ì„¸??
- greeting (?¸ì‚¬ë§?: "?ˆë…•?˜ì„¸??, "ê³ ë§ˆ?Œìš”", "ê°ì‚¬?©ë‹ˆ?? ??
- term_definition (?©ì–´ ?•ì˜): ë²•ë¥  ?©ì–´??ê°œë…???•ì˜ë¥?ë¬»ëŠ” ì§ˆë¬¸
- simple_question (ê°„ë‹¨??ì§ˆë¬¸): ?¼ë°˜ ë²•ë¥  ?ì‹?¼ë¡œ ?µë? ê°€?¥í•œ ê°„ë‹¨??ì§ˆë¬¸

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "query_type": "greeting" | "term_definition" | "simple_question",
    "confidence": 0.0-1.0,
    "reasoning": "?ë‹¨ ê·¼ê±° (?œêµ­??"
}}
"""

    @staticmethod
    def build_prompt_generation_prompt(query: str, query_type: str) -> str:
        """?ì ˆ???„ë¡¬?„íŠ¸ ?ì„±"""
        if query_type == "greeting":
            return f"""?¬ìš©?ì˜ ?¸ì‚¬??ì¹œì ˆ?˜ê²Œ ?‘ë‹µ?˜ì„¸??

{query}

ê°„ë‹¨?˜ê³  ì¹œì ˆ?˜ê²Œ ?‘ë‹µ?´ì£¼?¸ìš”. (1-2ë¬¸ì¥)"""
        elif query_type == "term_definition":
            return f"""?¤ìŒ ë²•ë¥  ?©ì–´???€??ê°„ë‹¨ëª…ë£Œ?˜ê²Œ ?•ì˜ë¥??œê³µ?˜ì„¸??

?©ì–´: {query}

?¤ìŒ ?•ì‹???°ë¼ì£¼ì„¸??
1. ?©ì–´???•ì˜ (1-2ë¬¸ì¥)
2. ê°„ë‹¨???¤ëª… (1ë¬¸ì¥)
ì´?2-3ë¬¸ì¥?¼ë¡œ ê°„ê²°?˜ê²Œ ?‘ì„±?´ì£¼?¸ìš”."""
        else:
            # simple_question
            return f"""?¤ìŒ ë²•ë¥  ì§ˆë¬¸??ê°„ë‹¨ëª…ë£Œ?˜ê²Œ ?µí•˜?¸ìš”:

ì§ˆë¬¸: {query}

ë²•ë¥  ?©ì–´??ê°œë…???€???•ì˜??ê°„ë‹¨???¤ëª…???œê³µ?˜ì„¸?? ê²€???†ì´ ?¼ë°˜?ì¸ ë²•ë¥  ì§€?ìœ¼ë¡??µë??˜ì„¸?? (2-4ë¬¸ì¥)"""

    @staticmethod
    def build_initial_answer_prompt(prev_output: Any) -> str:
        """ì´ˆê¸° ?µë? ?ì„± ?„ë¡¬?„íŠ¸"""
        if isinstance(prev_output, str):
            return prev_output
        elif isinstance(prev_output, dict):
            return prev_output.get("prompt", "")
        return ""

    @staticmethod
    def build_quality_validation_prompt(query: str, answer: str) -> str:
        """?µë? ?ˆì§ˆ ê²€ì¦??„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ?µë????ˆì§ˆ??ê²€ì¦í•´ì£¼ì„¸??

ì§ˆë¬¸: {query}
?µë?: {answer[:500]}

?¤ìŒ ê¸°ì??¼ë¡œ ê²€ì¦í•˜?¸ìš”:
1. **?ì ˆ??ê¸¸ì´**: ?ˆë¬´ ì§§ì???ê¸¸ì????ŠìŒ (10-500??
2. **ì§ˆë¬¸???€??ì§ì ‘?ì¸ ?µë?**: ì§ˆë¬¸??ë§ëŠ” ?µë??¸ê??
3. **ëª…í™•??*: ?µë???ëª…í™•?˜ê³  ?´í•´?˜ê¸° ?¬ìš´ê°€?
4. **?„ì„±??*: ?µë????„ì „?œê??

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "is_valid": true | false,
    "quality_score": 0.0-1.0,
    "issues": ["ë¬¸ì œ??", "ë¬¸ì œ??"],
    "needs_improvement": true | false
}}
"""

    @staticmethod
    def build_answer_improvement_prompt(query: str, original_answer: str, issues: list) -> str:
        """?µë? ê°œì„  ?„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ?µë???ê°œì„ ?´ì£¼?¸ìš”.

ì§ˆë¬¸: {query}
?ë³¸ ?µë?: {original_answer}
ë¬¸ì œ?? {', '.join(issues) if issues else "?†ìŒ"}

?¤ìŒ ë¬¸ì œ?ì„ ?´ê²°?˜ì—¬ ê°œì„ ???µë????‘ì„±?´ì£¼?¸ìš”:
{chr(10).join([f"- {issue}" for issue in issues[:3]]) if issues else "?†ìŒ"}
"""


class ClassificationChainBuilder:
    """ì§ˆë¬¸ ë¶„ë¥˜ ì²´ì¸ ë¹Œë”"""

    @staticmethod
    def build_question_type_prompt(query: str) -> str:
        """ì§ˆë¬¸ ? í˜• ë¶„ë¥˜ ?„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ë²•ë¥  ì§ˆë¬¸??ì§ˆë¬¸ ? í˜•?¼ë¡œ ë¶„ë¥˜?´ì£¼?¸ìš”.

ì§ˆë¬¸: {query}

ë¶„ë¥˜ ê°€?¥í•œ ? í˜•:
1. precedent_search - ?ë?, ?¬ê±´, ë²•ì› ?ê²°, ?ì‹œ?¬í•­ ê´€??
2. law_inquiry - ë²•ë¥  ì¡°ë¬¸, ë²•ë ¹, ê·œì •???´ìš©??ë¬»ëŠ” ì§ˆë¬¸
3. legal_advice - ë²•ë¥  ì¡°ì–¸, ?´ì„, ê¶Œë¦¬ êµ¬ì œ ë°©ë²•??ë¬»ëŠ” ì§ˆë¬¸
4. procedure_guide - ë²•ì  ?ˆì°¨, ?Œì†¡ ë°©ë²•, ?€??ë°©ë²•??ë¬»ëŠ” ì§ˆë¬¸
5. term_explanation - ë²•ë¥  ?©ì–´???•ì˜???˜ë?ë¥?ë¬»ëŠ” ì§ˆë¬¸
6. general_question - ë²”ìš©?ì¸ ë²•ë¥  ì§ˆë¬¸

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "query_type": "precedent_search" | "law_inquiry" | "legal_advice" | "procedure_guide" | "term_explanation" | "general_question",
    "confidence": 0.0-1.0,
    "reasoning": "?ë‹¨ ê·¼ê±°"
}}
"""

    @staticmethod
    def build_legal_field_prompt(query: str, query_type: str) -> str:
        """ë²•ë¥  ë¶„ì•¼ ì¶”ì¶œ ?„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ì§ˆë¬¸?ì„œ ê´€??ë²•ë¥  ë¶„ì•¼ë¥?ì¶”ì¶œ?´ì£¼?¸ìš”.

ì§ˆë¬¸: {query}
ì§ˆë¬¸ ? í˜•: {query_type}

ê°€?¥í•œ ë²•ë¥  ë¶„ì•¼:
- civil (ë¯¼ì‚¬ë²?: ê³„ì•½, ?í•´ë°°ìƒ, ì±„ê¶Œì±„ë¬´ ??
- criminal (?•ì‚¬ë²?: ?•ì‚¬ë²”ì£„, ì²˜ë²Œ, ?•ëŸ‰ ??
- administrative (?‰ì •ë²?: ?‰ì •ì²˜ë¶„, ?‰ì •?Œì†¡ ??
- intellectual_property (ì§€?ì¬?°ê¶Œë²?: ?¹í—ˆ, ?í‘œ, ?€?‘ê¶Œ ??

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "legal_field": "civil" | "criminal" | "administrative" | "intellectual_property",
    "confidence": 0.0-1.0,
    "reasoning": "?ë‹¨ ê·¼ê±°"
}}
"""

    @staticmethod
    def build_complexity_prompt(query: str, query_type: str, legal_field: str) -> str:
        """ë³µì¡???‰ê? ?„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ì§ˆë¬¸??ë³µì¡?„ë? ?‰ê??´ì£¼?¸ìš”.

ì§ˆë¬¸: {query}
ì§ˆë¬¸ ? í˜•: {query_type}
ë²•ë¥  ë¶„ì•¼: {legal_field}

ë³µì¡??ê¸°ì?:
- simple (?¨ìˆœ): ê°„ë‹¨???©ì–´ ?•ì˜???¼ë°˜ ë²•ë¥  ?ì‹ ì§ˆë¬¸
- moderate (ë³´í†µ): ?¼ë°˜?ì¸ ë²•ë¥  ì§ˆë¬¸, ê²€?‰ì´ ?„ìš”??ê²½ìš°
- complex (ë³µì¡): ?¬ëŸ¬ ë²•ë¥  ì¡°í•­?´ë‚˜ ?ë? ë¹„êµ, ë³µì¡???¬ë? ë¶„ì„ ??

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "complexity": "simple" | "moderate" | "complex",
    "needs_search": true | false,
    "reasoning": "?ë‹¨ ê·¼ê±°"
}}
"""

    @staticmethod
    def build_search_necessity_prompt(query: str, query_type: str, complexity: str) -> str:
        """ê²€???„ìš”???‰ê? ?„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ì§ˆë¬¸??ê²€?‰ì´ ?„ìš”?œì? ?‰ê??´ì£¼?¸ìš”.

ì§ˆë¬¸: {query}
ì§ˆë¬¸ ? í˜•: {query_type}
ë³µì¡?? {complexity}

ê²€?‰ì´ ?„ìš”??ê²½ìš°:
- ?ë???ë²•ë ¹ ì¡°ë¬¸ ?¸ìš©???„ìš”??ê²½ìš°
- ìµœì‹  ë²•ë¥  ?•ë³´ê°€ ?„ìš”??ê²½ìš°
- êµ¬ì²´?ì¸ ë²•ë¥  ?¬ë????ë?ê°€ ?„ìš”??ê²½ìš°

ê²€?‰ì´ ë¶ˆí•„?”í•œ ê²½ìš°:
- ê°„ë‹¨??ë²•ë¥  ?©ì–´ ?•ì˜
- ?¼ë°˜?ì¸ ë²•ë¥  ?ì‹
- ë³µì¡?„ê? simple??ê²½ìš°

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "needs_search": true | false,
    "search_type": "semantic" | "keyword" | "hybrid",
    "reasoning": "?ë‹¨ ê·¼ê±°"
}}
"""


class QueryEnhancementChainBuilder:
    """ì¿¼ë¦¬ ê°•í™” ì²´ì¸ ë¹Œë”"""

    @staticmethod
    def build_query_analysis_prompt(query: str, query_type: str, legal_field: str) -> str:
        """ì¿¼ë¦¬ ë¶„ì„ ?„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ë²•ë¥  ê²€??ì¿¼ë¦¬ë¥?ë¶„ì„?˜ê³  ?µì‹¬ ?¤ì›Œ?œë? ì¶”ì¶œ?´ì£¼?¸ìš”.

?ë³¸ ì¿¼ë¦¬: {query}
ì§ˆë¬¸ ? í˜•: {query_type}
ë²•ë¥  ë¶„ì•¼: {legal_field}

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "core_keywords": ["?¤ì›Œ??", "?¤ì›Œ??", "?¤ì›Œ??"],
    "query_intent": "ê²€???˜ë„ ?¤ëª…",
    "key_concepts": ["?µì‹¬ ê°œë…1", "?µì‹¬ ê°œë…2"]
}}
"""

    @staticmethod
    def build_keyword_expansion_prompt(query: str, query_analysis: Dict[str, Any]) -> str:
        """?¤ì›Œ???•ì¥ ?„ë¡¬?„íŠ¸"""
        core_keywords = query_analysis.get("core_keywords", [])
        key_concepts = query_analysis.get("key_concepts", [])

        return f"""?¤ìŒ ì¿¼ë¦¬???¤ì›Œ?œë? ?•ì¥?˜ê³  ë³€?•ì„ ?ì„±?´ì£¼?¸ìš”.

?ë³¸ ì¿¼ë¦¬: {query}
?µì‹¬ ?¤ì›Œ?? {', '.join(core_keywords)}
?µì‹¬ ê°œë…: {', '.join(key_concepts)}

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "expanded_keywords": ["?•ì¥???¤ì›Œ??", "?•ì¥???¤ì›Œ??"],
    "keyword_variants": ["ë³€???¤ì›Œ??", "ë³€???¤ì›Œ??"],
    "synonyms": ["?™ì˜??", "?™ì˜??"]
}}
"""

    @staticmethod
    def build_query_optimization_prompt(
        query: str,
        query_analysis: Dict[str, Any],
        keyword_expansion: Dict[str, Any]
    ) -> str:
        """ì¿¼ë¦¬ ìµœì ???„ë¡¬?„íŠ¸"""
        expanded_keywords = keyword_expansion.get("expanded_keywords", [])
        keyword_variants = keyword_expansion.get("keyword_variants", [])

        return f"""?¤ìŒ ì¿¼ë¦¬ë¥?ë²•ë¥  ê²€?‰ì— ìµœì ?”í•´ì£¼ì„¸??

?ë³¸ ì¿¼ë¦¬: {query}
?•ì¥???¤ì›Œ?? {', '.join(expanded_keywords)}
ë³€???¤ì›Œ?? {', '.join(keyword_variants)}

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "optimized_query": "ìµœì ?”ëœ ê²€??ì¿¼ë¦¬",
    "semantic_query": "?˜ë???ê²€?‰ìš© ì¿¼ë¦¬",
    "keyword_queries": ["?¤ì›Œ??ê²€?‰ìš© ì¿¼ë¦¬1", "?¤ì›Œ??ê²€?‰ìš© ì¿¼ë¦¬2"],
    "reasoning": "ìµœì ???¬ìœ "
}}
"""

    @staticmethod
    def build_query_validation_prompt(query: str, optimized_query: Dict[str, Any]) -> str:
        """ì¿¼ë¦¬ ê²€ì¦??„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ìµœì ?”ëœ ì¿¼ë¦¬ë¥?ê²€ì¦í•´ì£¼ì„¸??

?ë³¸ ì¿¼ë¦¬: {query}
ìµœì ?”ëœ ì¿¼ë¦¬: {optimized_query.get('optimized_query', '')}

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "is_valid": true | false,
    "quality_score": 0.0-1.0,
    "issues": ["ë¬¸ì œ??", "ë¬¸ì œ??"],
    "recommendations": ["ê¶Œê³ ?¬í•­1", "ê¶Œê³ ?¬í•­2"]
}}
"""


class AnswerGenerationChainBuilder:
    """?µë? ?ì„± ì²´ì¸ ë¹Œë”"""

    @staticmethod
    def build_initial_answer_prompt(optimized_prompt: str) -> str:
        """ì´ˆê¸° ?µë? ?ì„± ?„ë¡¬?„íŠ¸"""
        return optimized_prompt

    @staticmethod
    def build_validation_prompt(answer: str) -> str:
        """?µë? ê²€ì¦??„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ê¸°ì??¼ë¡œ ?µë???ê²€ì¦í•˜?¸ìš”:

1. **ê¸¸ì´**: ìµœì†Œ 50???´ìƒ
2. **?´ìš© ?„ì„±??*: ì§ˆë¬¸???€??ì§ì ‘?ì¸ ?µë? ?¬í•¨
3. **ë²•ì  ê·¼ê±°**: ê´€??ë²•ë ¹, ì¡°í•­, ?ë? ?¸ìš© ?¬ë?
4. **êµ¬ì¡°**: ëª…í™•???¹ì…˜ê³??¼ë¦¬???ë¦„
5. **?¼ê???*: ?µë? ?„ì²´???¼ë¦¬???¼ê???

?µë?:
{answer[:2000]}

?¤ìŒ ?•ì‹?¼ë¡œ ê²€ì¦?ê²°ê³¼ë¥??œê³µ?˜ì„¸??
{{
    "is_valid": true/false,
    "quality_score": 0.0-1.0,
    "issues": [
        "ë¬¸ì œ??1",
        "ë¬¸ì œ??2"
    ],
    "strengths": [
        "ê°•ì  1",
        "ê°•ì  2"
    ],
    "recommendations": [
        "ê°œì„  ê¶Œê³  1",
        "ê°œì„  ê¶Œê³  2"
    ]
}}
"""

    @staticmethod
    def build_improvement_instructions_prompt(
        original_answer: str,
        validation_result: Dict[str, Any]
    ) -> str:
        """ê°œì„  ì§€???ì„± ?„ë¡¬?„íŠ¸"""
        issues = validation_result.get("issues", [])
        recommendations = validation_result.get("recommendations", [])
        quality_score = validation_result.get("quality_score", 1.0)

        return f"""?¤ìŒ ?µë???ê²€ì¦?ê²°ê³¼ë¥?ë°”íƒ•?¼ë¡œ ê°œì„  ì§€?œë? ?‘ì„±?˜ì„¸??

**?ë³¸ ?µë?**:
{original_answer[:1500]}

**ê²€ì¦?ê²°ê³¼**:
- ?ˆì§ˆ ?ìˆ˜: {quality_score:.2f}/1.0
- ë¬¸ì œ?? {', '.join(issues[:5]) if issues else '?†ìŒ'}
- ê¶Œê³ ?¬í•­: {', '.join(recommendations[:5]) if recommendations else '?†ìŒ'}

**ê°œì„  ì§€???‘ì„± ?”ì²­**:
??ê²€ì¦?ê²°ê³¼ë¥?ë°”íƒ•?¼ë¡œ ?µë???ê°œì„ ?˜ê¸° ?„í•œ êµ¬ì²´?ì¸ ì§€?œì‚¬??„ ?‘ì„±?˜ì„¸??

?¤ìŒ ?•ì‹?¼ë¡œ ?œê³µ?˜ì„¸??
{{
    "needs_improvement": true,
    "improvement_instructions": [
        "ê°œì„  ì§€??1: êµ¬ì²´?ìœ¼ë¡??´ë–¤ ë¶€ë¶„ì„ ?´ë–»ê²?ê°œì„ ? ì?",
        "ê°œì„  ì§€??2: ..."
    ],
    "preserve_content": [
        "ë³´ì¡´???´ìš© 1",
        "ë³´ì¡´???´ìš© 2"
    ],
    "focus_areas": [
        "ì¤‘ì  ê°œì„  ?ì—­ 1",
        "ì¤‘ì  ê°œì„  ?ì—­ 2"
    ]
}}
"""

    @staticmethod
    def build_improved_answer_prompt(
        original_prompt: str,
        improvement_instructions: Dict[str, Any]
    ) -> str:
        """ê°œì„ ???µë? ?ì„± ?„ë¡¬?„íŠ¸"""
        improvement_text = "\n".join(improvement_instructions.get("improvement_instructions", []))
        preserve_content = "\n".join(improvement_instructions.get("preserve_content", []))

        return f"""{original_prompt}

---

## ?”§ ê°œì„  ?”ì²­

???„ë¡¬?„íŠ¸ë¡??ì„±???µë????¤ìŒ ì§€?œì‚¬??— ?°ë¼ ê°œì„ ?˜ì„¸??

**ê°œì„  ì§€?œì‚¬??*:
{improvement_text}

**ë³´ì¡´???´ìš©** (ë°˜ë“œ???¬í•¨):
{preserve_content if preserve_content else "?ë³¸ ?µë???ëª¨ë“  ë²•ì  ?•ë³´?€ ê·¼ê±°"}

**ì¤‘ì  ê°œì„  ?ì—­**:
{', '.join(improvement_instructions.get("focus_areas", []))}

??ì§€?œì‚¬??— ?°ë¼ ?µë???ê°œì„ ?˜ë˜, ?ë³¸ ?µë???ë²•ì  ê·¼ê±°?€ ?•ë³´??ë°˜ë“œ??ë³´ì¡´?˜ì„¸??
"""

    @staticmethod
    def build_final_validation_prompt(answer: str) -> str:
        """ìµœì¢… ê²€ì¦??„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ?µë???ìµœì¢… ê²€ì¦í•˜?¸ìš”.

?µë?:
{answer[:2000]}

?¤ìŒ ê¸°ì??¼ë¡œ ìµœì¢… ê²€ì¦í•˜?¸ìš”:
1. **?„ì„±??*: ?µë????„ì „?œê??
2. **?•í™•??*: ë²•ì  ?•ë³´ê°€ ?•í™•?œê??
3. **ëª…í™•??*: ?µë???ëª…í™•?œê??
4. **êµ¬ì¡°**: ?¼ë¦¬??êµ¬ì¡°ê°€ ?ˆëŠ”ê°€?

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?˜ì„¸??
{{
    "is_valid": true/false,
    "final_score": 0.0-1.0,
    "ready_for_user": true/false
}}
"""


class DocumentAnalysisChainBuilder:
    """ë¬¸ì„œ ë¶„ì„ ì²´ì¸ ë¹Œë”"""

    @staticmethod
    def build_document_type_verification_prompt(text: str, detected_type: str) -> str:
        """ë¬¸ì„œ ? í˜• ?•ì¸ ?„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ë¬¸ì„œ??? í˜•???•ì¸?˜ê³  ê²€ì¦í•´ì£¼ì„¸??

ë¬¸ì„œ ?´ìš© (?¼ë?):
{text[:2000]}

?¤ì›Œ??ê¸°ë°˜ ê°ì? ê²°ê³¼: {detected_type}

?¤ìŒ ë¬¸ì„œ ? í˜• ì¤??˜ë‚˜ë¡??•ì¸?´ì£¼?¸ìš”:
- contract (ê³„ì•½??: ê³„ì•½?? ê°??? ê³„ì•½ ì¡°ê±´ ??
- complaint (ê³ ì†Œ??: ê³ ì†Œ?? ?¼ê³ ?Œì¸, ê³ ì†Œ????
- agreement (?©ì˜??: ?©ì˜?? ?©ì˜, ?ë°© ?©ì˜ ??
- power_of_attorney (?„ì„??: ?„ì„?? ?„ì„?? ?˜ì„????
- general_legal_document (?¼ë°˜ ë²•ë¥  ë¬¸ì„œ): ?„ì— ?´ë‹¹?˜ì? ?ŠëŠ” ê²½ìš°

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "document_type": "contract" | "complaint" | "agreement" | "power_of_attorney" | "general_legal_document",
    "confidence": 0.0-1.0,
    "reasoning": "?ë‹¨ ê·¼ê±° (?œêµ­??"
}}
"""

    @staticmethod
    def build_clause_extraction_prompt(text: str, document_type: str) -> str:
        """ì£¼ìš” ì¡°í•­ ì¶”ì¶œ ?„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ë¬¸ì„œ?ì„œ ì£¼ìš” ì¡°í•­??ì¶”ì¶œ?´ì£¼?¸ìš”.

ë¬¸ì„œ ?´ìš©:
{text[:3000]}

ë¬¸ì„œ ? í˜•: {document_type}

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "key_clauses": [
        {{
            "clause_number": "ì¡°í•­ ë²ˆí˜¸",
            "title": "ì¡°í•­ ?œëª©",
            "content": "ì¡°í•­ ?´ìš©",
            "importance": "high" | "medium" | "low"
        }}
    ],
    "total_clauses": ?«ì
}}
"""

    @staticmethod
    def build_issue_identification_prompt(
        text: str,
        document_type: str,
        key_clauses: list
    ) -> str:
        """ë¬¸ì œ???ë³„ ?„ë¡¬?„íŠ¸"""
        clauses_summary = "\n".join([
            f"- {c.get('title', 'N/A')}: {c.get('content', '')[:100]}"
            for c in key_clauses[:5]
        ])

        return f"""?¤ìŒ ë¬¸ì„œ?ì„œ ? ì¬??ë¬¸ì œ?ì„ ?ë³„?´ì£¼?¸ìš”.

ë¬¸ì„œ ?´ìš© (?¼ë?):
{text[:2000]}

ë¬¸ì„œ ? í˜•: {document_type}

ì£¼ìš” ì¡°í•­:
{clauses_summary}

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "issues": [
        {{
            "severity": "high" | "medium" | "low",
            "type": "missing_clause" | "vague_term" | "unclear_provision" | "potential_risk",
            "description": "ë¬¸ì œ???¤ëª…",
            "location": "ì¡°í•­ ë²ˆí˜¸ ?ëŠ” ?„ì¹˜",
            "recommendation": "ê°œì„  ê¶Œê³ "
        }}
    ],
    "total_issues": ?«ì
}}
"""

    @staticmethod
    def build_summary_generation_prompt(
        text: str,
        document_type: str,
        key_clauses: list,
        issues: list
    ) -> str:
        """?”ì•½ ?ì„± ?„ë¡¬?„íŠ¸"""
        return f"""?¤ìŒ ë¬¸ì„œë¥??”ì•½?´ì£¼?¸ìš”.

ë¬¸ì„œ ? í˜•: {document_type}
ì£¼ìš” ì¡°í•­ ?? {len(key_clauses)}
ë°œê²¬??ë¬¸ì œ???? {len(issues)}

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "summary": "ë¬¸ì„œ ?„ì²´ ?”ì•½ (3-5ë¬¸ì¥)",
    "key_points": ["?µì‹¬ ?¬ì¸??", "?µì‹¬ ?¬ì¸??", "?µì‹¬ ?¬ì¸??"],
    "main_clauses": ["ì£¼ìš” ì¡°í•­ ?”ì•½1", "ì£¼ìš” ì¡°í•­ ?”ì•½2"],
    "critical_issues": ["ì¤‘ìš”??ë¬¸ì œ??", "ì¤‘ìš”??ë¬¸ì œ??"]
}}
"""

    @staticmethod
    def build_improvement_recommendations_prompt(
        document_type: str,
        issues: list
    ) -> str:
        """ê°œì„  ê¶Œê³  ?ì„± ?„ë¡¬?„íŠ¸"""
        issues_summary = "\n".join([
            f"- [{i.get('severity', 'unknown')}] {i.get('description', 'N/A')}: {i.get('recommendation', 'N/A')}"
            for i in issues[:5]
        ])

        return f"""?¤ìŒ ë¬¸ì„œ??ë¬¸ì œ?ì„ ë°”íƒ•?¼ë¡œ ê°œì„  ê¶Œê³ ë¥??‘ì„±?´ì£¼?¸ìš”.

ë¬¸ì„œ ? í˜•: {document_type}

ë°œê²¬??ë¬¸ì œ??
{issues_summary}

?¤ìŒ ?•ì‹?¼ë¡œ ?‘ë‹µ?´ì£¼?¸ìš”:
{{
    "recommendations": [
        {{
            "priority": "high" | "medium" | "low",
            "description": "ê°œì„  ê¶Œê³  ?¤ëª…",
            "action_items": ["êµ¬ì²´???‰ë™ 1", "êµ¬ì²´???‰ë™ 2"]
        }}
    ],
    "overall_assessment": "?„ì²´ ?‰ê?"
}}
"""
