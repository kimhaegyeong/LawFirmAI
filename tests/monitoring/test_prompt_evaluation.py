"""
?„ë¡¬?„íŠ¸ ?‰ê? ë°?ê°œì„  ?ŒìŠ¤??
generate_answer_enhanced?ì„œ ?ì„±???„ë¡¬?„íŠ¸ë¥??‰ê??˜ê³  ê°œì„ ?ì„ ì°¾ìŠµ?ˆë‹¤.
"""
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def evaluate_prompt(prompt_text: str) -> Dict[str, Any]:
    """?„ë¡¬?„íŠ¸ë¥??‰ê??˜ê³  ê°œì„ ?ì„ ì°¾ìŠµ?ˆë‹¤."""
    issues = []
    suggestions = []

    # 1. ì¤‘ë³µ ë¬¸ì„œ ?¹ì…˜ ?•ì¸
    doc_section_patterns = [
        r"##\s*ê²€?‰ëœ ë²•ë¥  ë¬¸ì„œ",
        r"##\s*?œê³µ??ë²•ë¥  ë¬¸ì„œ",
        r"##\s*ê²€?‰ëœ ?ë? ë¬¸ì„œ",
        r"##\s*ê²€?‰ëœ ë²•ë¥  ë¬¸ì„œ ë°??•ë³´",
        r"##\s*ê²€?‰ëœ ë²•ë¥  ë¬¸ì„œ ë°??ë?"
    ]

    found_sections = []
    for pattern in doc_section_patterns:
        matches = re.findall(pattern, prompt_text, re.IGNORECASE)
        if matches:
            found_sections.extend(matches)

    if len(found_sections) > 1:
        unique_sections = set(found_sections)
        if len(unique_sections) > 1:
            issues.append({
                "type": "ì¤‘ë³µ ë¬¸ì„œ ?¹ì…˜",
                "severity": "high",
                "description": f"ë¬¸ì„œ ?¹ì…˜??{len(unique_sections)}ë²??˜í???,
                "sections": list(unique_sections)
            })
            suggestions.append("ë¬¸ì„œ ?¹ì…˜????ë²ˆë§Œ ?¬í•¨?˜ë„ë¡??˜ì • ?„ìš”")

    # 2. ì§€ì¹?ë¬¸êµ¬ ì¤‘ë³µ ?•ì¸
    instruction_phrases = [
        "ë°˜ë“œ??ì°¸ê³ ?˜ì—¬ ?µë??˜ì„¸??,
        "ë°˜ë“œ????ë¬¸ì„œ?¤ì„ ì°¸ê³ ",
        "ë°˜ë“œ???œìš©",
        "ìµœì†Œ 2ê°??´ìƒ ?¸ìš©",
        "?ˆë? ê¸ˆì?"
    ]

    phrase_counts = {}
    for phrase in instruction_phrases:
        count = len(re.findall(re.escape(phrase), prompt_text, re.IGNORECASE))
        if count > 1:
            phrase_counts[phrase] = count

    if phrase_counts:
        issues.append({
            "type": "ì§€ì¹?ë¬¸êµ¬ ì¤‘ë³µ",
            "severity": "medium",
            "description": "ê°™ì? ì§€ì¹¨ì´ ?¬ëŸ¬ ë²?ë°˜ë³µ??,
            "phrases": phrase_counts
        })
        suggestions.append("ì§€ì¹?ë¬¸êµ¬ë¥??µí•©?˜ì—¬ ??ë²ˆë§Œ ?œì‹œ")

    # 3. ë¬¸ì„œ ëª©ë¡ ì¤‘ë³µ ?•ì¸
    # "ë¬¸ì„œ 1:", "ë¬¸ì„œ 2:" ê°™ì? ?¨í„´ ì°¾ê¸°
    doc_number_pattern = r"ë¬¸ì„œ\s*\d+\s*:"
    doc_numbers = re.findall(doc_number_pattern, prompt_text)

    if len(doc_numbers) > len(set(doc_numbers)):
        issues.append({
            "type": "ë¬¸ì„œ ë²ˆí˜¸ ì¤‘ë³µ",
            "severity": "high",
            "description": "ê°™ì? ë¬¸ì„œ ë²ˆí˜¸ê°€ ?¬ëŸ¬ ë²??˜í???,
            "count": len(doc_numbers)
        })
        suggestions.append("ë¬¸ì„œ ëª©ë¡ ì¤‘ë³µ ?œê±° ?„ìš”")

    # 4. ?„ë¡¬?„íŠ¸ ê¸¸ì´ ?•ì¸
    prompt_length = len(prompt_text)
    token_estimate = prompt_length // 3  # ?€?µì ??? í° ??ì¶”ì •

    if prompt_length > 8000:
        issues.append({
            "type": "?„ë¡¬?„íŠ¸ ê¸¸ì´",
            "severity": "medium",
            "description": f"?„ë¡¬?„íŠ¸ê°€ ?ˆë¬´ ê¹ë‹ˆ??({prompt_length}?? ??{token_estimate} ? í°)",
            "length": prompt_length,
            "estimated_tokens": token_estimate
        })
        suggestions.append("?„ë¡¬?„íŠ¸ ê¸¸ì´ ìµœì ???„ìš” (ì¤‘ë³µ ?œê±°, ë¶ˆí•„?”í•œ ?¹ì…˜ ?œê±°)")

    # 5. ë¬¸ì„œ ?†ìŒ ë©”ì‹œì§€ ?¤ë¥˜ ?•ì¸
    if "?„ì¬ ê´€??ë²•ë¥  ë¬¸ì„œë¥?ì°¾ì? ëª»í–ˆ?µë‹ˆ?? in prompt_text:
        # ë¬¸ì„œ ?¹ì…˜???ˆëŠ”ì§€ ?•ì¸
        if "ê²€?‰ëœ ë²•ë¥  ë¬¸ì„œ" in prompt_text or "## ?”" in prompt_text:
            issues.append({
                "type": "ë¡œì§ ?¤ë¥˜",
                "severity": "critical",
                "description": "ë¬¸ì„œê°€ ?ˆëŠ”?°ë„ 'ë¬¸ì„œë¥?ì°¾ì? ëª»í–ˆ?µë‹ˆ?? ë©”ì‹œì§€ê°€ ?œì‹œ??
            })
            suggestions.append("final_instruction_section ë¡œì§ ?˜ì • ?„ìš”")

    # 6. ?„ìˆ˜ ì¤€???¬í•­ ?¹ì…˜ ?•ì¸
    if "## ? ï¸ ?„ìˆ˜ ì¤€???¬í•­" in prompt_text or "## ? ï¸ ?µì‹¬ ì§€ì¹? in prompt_text:
        section_match = re.search(r"##\s*? ï¸\s*(?„ìˆ˜ ì¤€???¬í•­|?µì‹¬ ì§€ì¹?\s*\n\n(.*?)(?=\n##|\Z)", prompt_text, re.DOTALL)
        if section_match:
            section_content = section_match.group(2).strip()
            if len(section_content) < 50:
                issues.append({
                    "type": "ë¹??„ìˆ˜ ì¤€???¬í•­",
                    "severity": "medium",
                    "description": "?„ìˆ˜ ì¤€???¬í•­ ?¹ì…˜??ê±°ì˜ ë¹„ì–´?ˆìŒ",
                    "content_length": len(section_content)
                })
                suggestions.append("?„ìˆ˜ ì¤€???¬í•­ ?¹ì…˜???ì ˆ???´ìš© ì¶”ê? ?„ìš”")

    return {
        "issues": issues,
        "suggestions": suggestions,
        "metrics": {
            "prompt_length": prompt_length,
            "estimated_tokens": token_estimate,
            "document_sections": len(found_sections),
            "total_issues": len(issues),
            "critical_issues": len([i for i in issues if i.get("severity") == "critical"]),
            "high_issues": len([i for i in issues if i.get("severity") == "high"]),
            "medium_issues": len([i for i in issues if i.get("severity") == "medium"])
        }
    }

def find_latest_prompt_file() -> str:
    """ê°€??ìµœê·¼ ?„ë¡¬?„íŠ¸ ?Œì¼ ì°¾ê¸°"""
    debug_dir = Path("debug/prompts")
    if not debug_dir.exists():
        return None

    prompt_files = list(debug_dir.glob("prompt_*.txt"))
    if not prompt_files:
        return None

    # ?˜ì • ?œê°„ ê¸°ì??¼ë¡œ ?•ë ¬
    prompt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(prompt_files[0])

def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    print("=" * 80)
    print("?„ë¡¬?„íŠ¸ ?‰ê? ë°?ê°œì„  ?ŒìŠ¤??)
    print("=" * 80)

    # ìµœì‹  ?„ë¡¬?„íŠ¸ ?Œì¼ ì°¾ê¸°
    prompt_file = find_latest_prompt_file()

    if not prompt_file or not os.path.exists(prompt_file):
        print("???„ë¡¬?„íŠ¸ ?Œì¼??ì°¾ì„ ???†ìŠµ?ˆë‹¤.")
        print("   ë¨¼ì? LangGraph ?ŒìŠ¤?¸ë? ?¤í–‰?˜ì—¬ ?„ë¡¬?„íŠ¸ë¥??ì„±?˜ì„¸??")
        return

    print(f"\n?“„ ?„ë¡¬?„íŠ¸ ?Œì¼: {prompt_file}")

    # ?„ë¡¬?„íŠ¸ ?½ê¸°
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    print(f"?„ë¡¬?„íŠ¸ ê¸¸ì´: {len(prompt_text):,}??n")

    # ?„ë¡¬?„íŠ¸ ?‰ê?
    evaluation = evaluate_prompt(prompt_text)

    # ê²°ê³¼ ì¶œë ¥
    print("=" * 80)
    print("?“Š ?‰ê? ê²°ê³¼")
    print("=" * 80)

    metrics = evaluation["metrics"]
    print(f"\në©”íŠ¸ë¦?")
    print(f"  - ?„ë¡¬?„íŠ¸ ê¸¸ì´: {metrics['prompt_length']:,}??)
    print(f"  - ?ˆìƒ ? í° ?? {metrics['estimated_tokens']:,}")
    print(f"  - ë¬¸ì„œ ?¹ì…˜ ?? {metrics['document_sections']}")
    print(f"  - ì´??´ìŠˆ ?? {metrics['total_issues']}")
    print(f"  - ?¬ê° ?´ìŠˆ: {metrics['critical_issues']}")
    print(f"  - ?’ì? ?°ì„ ?œìœ„ ?´ìŠˆ: {metrics['high_issues']}")
    print(f"  - ì¤‘ê°„ ?°ì„ ?œìœ„ ?´ìŠˆ: {metrics['medium_issues']}")

    # ?´ìŠˆ ì¶œë ¥
    if evaluation["issues"]:
        print(f"\n? ï¸ ë°œê²¬???´ìŠˆ ({len(evaluation['issues'])}ê°?:")
        for idx, issue in enumerate(evaluation["issues"], 1):
            severity_icon = {
                "critical": "?”´",
                "high": "?Ÿ ",
                "medium": "?Ÿ¡",
                "low": "?”µ"
            }.get(issue["severity"], "??)

            print(f"\n{idx}. {severity_icon} [{issue['severity'].upper()}] {issue['type']}")
            print(f"   ?¤ëª…: {issue['description']}")
            if "details" in issue:
                print(f"   ?ì„¸: {issue['details']}")

    # ê°œì„  ?œì•ˆ ì¶œë ¥
    if evaluation["suggestions"]:
        print(f"\n?’¡ ê°œì„  ?œì•ˆ ({len(evaluation['suggestions'])}ê°?:")
        for idx, suggestion in enumerate(evaluation["suggestions"], 1):
            print(f"  {idx}. {suggestion}")

    if not evaluation["issues"]:
        print("\n???„ë¡¬?„íŠ¸???¬ê°??ë¬¸ì œê°€ ?†ìŠµ?ˆë‹¤!")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
