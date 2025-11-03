# -*- coding: utf-8 -*-
"""
generate_and_validate_answer?ì„œ ê²€??ê²°ê³¼ê°€ ?„ë¡¬?„íŠ¸ ?‘ì„±???¬ìš©?˜ëŠ”ì§€ ê²€??
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def analyze_search_results_in_prompt():
    """generate_and_validate_answer?ì„œ ê²€??ê²°ê³¼ ?¬ìš© ?¬ë? ë¶„ì„"""
    print("=" * 80)
    print("generate_and_validate_answer ê²€??ê²°ê³¼ ?¬ìš© ?¬ë? ê²€??)
    print("=" * 80)

    try:
        from source.agents.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow

        # legal_workflow_enhanced.py ?Œì¼ ?½ê¸°
        workflow_file = project_root / "core" / "agents" / "legal_workflow_enhanced.py"
        with open(workflow_file, "r", encoding="utf-8") as f:
            content = f.read()

        print("\n?“‹ ê²€????ª©:")
        print("-" * 80)

        # 1. generate_and_validate_answer ë©”ì„œ??êµ¬ì¡° ?•ì¸
        print("\n1ï¸âƒ£ generate_and_validate_answer ë©”ì„œ??êµ¬ì¡°:")
        if "def generate_and_validate_answer" in content:
            print("   ??generate_and_validate_answer ë©”ì„œ??ì¡´ì¬")
            if "generate_answer_enhanced" in content.split("def generate_and_validate_answer")[1].split("\n    def ")[0]:
                print("   ??generate_answer_enhanced ë©”ì„œ?œë? ?¸ì¶œ??)

        # 2. generate_answer_enhanced?ì„œ retrieved_docs ?¬ìš© ?•ì¸
        print("\n2ï¸âƒ£ generate_answer_enhanced?ì„œ retrieved_docs ?¬ìš©:")
        generate_answer_section = content.split("def generate_answer_enhanced")[1].split("\n    def ")[0]

        checks = [
            ("retrieved_docs = self._get_state_value", "state?ì„œ retrieved_docs ê°€?¸ì˜¤ê¸?),
            ("context_dict", "context_dict ?ì„±"),
            ("structured_documents", "structured_documents ?¬í•¨"),
            ("retrieved_docs", "retrieved_docs ì°¸ì¡°"),
            ("unified_prompt_manager.get_optimized_prompt", "unified_prompt_manager??context_dict ?„ë‹¬"),
            ("SEARCH RESULTS INJECTION", "ê²€??ê²°ê³¼ ê°•ì œ ì£¼ì… ë¡œì§"),
            ("SEARCH RESULTS ENFORCED", "ê²€??ê²°ê³¼ ê°•ì œ ë³´ê°• ë¡œì§"),
        ]

        for check_str, description in checks:
            if check_str in generate_answer_section:
                print(f"   ??{description}")
            else:
                print(f"   ? ï¸ {description} - ?•ì¸ ?„ìš”")

        # 3. context_dict??ê²€??ê²°ê³¼ ?¬í•¨ ?¬ë? ?•ì¸
        print("\n3ï¸âƒ£ context_dict??ê²€??ê²°ê³¼ ?¬í•¨:")
        context_dict_checks = [
            ("structured_documents", "structured_documents ?„ë“œ"),
            ("legal_references", "legal_references ?„ë“œ"),
            ("document_count", "document_count ?„ë“œ"),
            ("docs_included", "docs_included ?„ë“œ"),
        ]

        for check_str, description in context_dict_checks:
            count = generate_answer_section.count(check_str)
            if count > 0:
                print(f"   ??{description} (?¬ìš© {count}??")
            else:
                print(f"   ? ï¸ {description} - ?¬ìš© ????)

        # 4. retrieved_docs ??structured_documents ë³€??ë¡œì§ ?•ì¸
        print("\n4ï¸âƒ£ retrieved_docs ??structured_documents ë³€??ë¡œì§:")
        if "normalized_documents" in generate_answer_section:
            print("   ??normalized_documents ë³€??ë¡œì§ ì¡´ì¬")
            if "SEARCH RESULTS INJECTION" in generate_answer_section:
                print("   ??ê²€??ê²°ê³¼ ê°•ì œ ì£¼ì… ë¡œì§ ì¡´ì¬")

        # 5. ?„ë¡¬?„íŠ¸ ê²€ì¦?ë¡œì§ ?•ì¸
        print("\n5ï¸âƒ£ ?„ë¡¬?„íŠ¸??ê²€??ê²°ê³¼ ?¬í•¨ ?¬ë? ê²€ì¦?")
        validation_checks = [
            ("PROMPT VALIDATION", "?„ë¡¬?„íŠ¸ ê²€ì¦?ë¡œì§"),
            ("has_documents_section", "ë¬¸ì„œ ?¹ì…˜ ?•ì¸"),
            ("ê²€?‰ëœ ë²•ë¥  ë¬¸ì„œ", "ë¬¸ì„œ ?¹ì…˜ ?¤ì›Œ???•ì¸"),
        ]

        for check_str, description in validation_checks:
            if check_str in generate_answer_section:
                print(f"   ??{description}")
            else:
                print(f"   ? ï¸ {description} - ?•ì¸ ?„ìš”")

        # 6. unified_prompt_manager?ì„œ structured_documents ?¬ìš© ?•ì¸
        print("\n6ï¸âƒ£ unified_prompt_manager?ì„œ structured_documents ?¬ìš©:")
        try:
            from source.services.unified_prompt_manager import UnifiedPromptManager
            prompt_manager_file = project_root / "source" / "services" / "unified_prompt_manager.py"
            with open(prompt_manager_file, "r", encoding="utf-8") as f:
                prompt_manager_content = f.read()

            if "structured_documents" in prompt_manager_content:
                print("   ??structured_documents ?¬ìš©")
                if "prompt_optimized_text" in prompt_manager_content:
                    print("   ??prompt_optimized_text ?¬ìš©")
                    if "_optimize_context" in prompt_manager_content:
                        optimize_section = prompt_manager_content.split("def _optimize_context")[1].split("\n    def ")[0]
                        if "structured_documents" in optimize_section:
                            print("   ??_optimize_context?ì„œ structured_documents ì²˜ë¦¬")
                        else:
                            print("   ? ï¸ _optimize_context?ì„œ structured_documents ë¯¸ì‚¬??ê°€??)
        except Exception as e:
            print(f"   ? ï¸ unified_prompt_manager ?•ì¸ ì¤??¤ë¥˜: {e}")

        # 7. ìµœì¢… ê²°ë¡ 
        print("\n" + "=" * 80)
        print("?“Š ìµœì¢… ê²°ë¡ ")
        print("=" * 80)
        print("""
ê²€??ê²°ê³¼ê°€ ?„ë¡¬?„íŠ¸ ?‘ì„±???¬ìš©?˜ëŠ” ê²½ë¡œ:

1. generate_and_validate_answer (1087ë²??¼ì¸)
   ?”â?> generate_answer_enhanced ?¸ì¶œ (1111ë²??¼ì¸)

2. generate_answer_enhanced (5219ë²??¼ì¸)
   ?œâ?> retrieved_docs ê°€?¸ì˜¤ê¸?(5250ë²??¼ì¸)
   ?œâ?> context_dict ?ì„± (5386-5395ë²??¼ì¸)
   ??  ?œâ? structured_documents ?¬í•¨
   ??  ?œâ? legal_references ?¬í•¨
   ??  ?”â? document_count, docs_included ?¬í•¨
   ?œâ?> retrieved_docs ??structured_documents ë³€??(5533-5620ë²??¼ì¸)
   ??  ?”â? ê²€??ê²°ê³¼ê°€ ?†ìœ¼ë©?ê°•ì œë¡?ë³€?˜í•˜???¬í•¨
   ?œâ?> unified_prompt_manager.get_optimized_prompt ?¸ì¶œ (5638ë²??¼ì¸)
   ??  ?”â? context_dict ?„ë‹¬ (structured_documents ?¬í•¨)
   ?”â?> ?„ë¡¬?„íŠ¸ ê²€ì¦?(5670-5716ë²??¼ì¸)
       ?”â? ë¬¸ì„œ ?¹ì…˜ ?¬í•¨ ?¬ë? ?•ì¸

3. unified_prompt_manager.get_optimized_prompt
   ?”â?> _optimize_context ë©”ì„œ?œì—??structured_documents ì²˜ë¦¬
       ?”â? prompt_optimized_textê°€ ?ˆì–´??structured_documents ê°•ì œ ?¬í•¨ (443-447ë²??¼ì¸)

??ê²°ë¡ : ê²€?‰ëœ ê²°ê³¼(retrieved_docs)???„ë¡¬?„íŠ¸ ?‘ì„±???¬ìš©?©ë‹ˆ??
   - retrieved_docs ??structured_documents ë³€??
   - context_dict???¬í•¨
   - unified_prompt_manager???„ë‹¬
   - ìµœì¢… ?„ë¡¬?„íŠ¸??ë¬¸ì„œ ?¹ì…˜?¼ë¡œ ?¬í•¨
        """)

        # 8. ? ì¬??ë¬¸ì œ???•ì¸
        print("\n" + "=" * 80)
        print("? ï¸ ? ì¬??ë¬¸ì œ??)
        print("=" * 80)

        warnings = []

        # retrieved_docsê°€ ?†ì„ ??ì²˜ë¦¬
        if "retrieved_docs is empty" in generate_answer_section:
            print("   ??retrieved_docsê°€ ?†ì„ ??ê²½ê³  ë¡œê¹… ì¡´ì¬")
        else:
            warnings.append("retrieved_docsê°€ ?†ì„ ??ê²½ê³  ë¡œê¹… ?†ìŒ")

        # context_dict ê²€ì¦?
        if "CONTEXT VALIDATION" in generate_answer_section:
            print("   ??context_dict ê²€ì¦?ë¡œì§ ì¡´ì¬")
        else:
            warnings.append("context_dict ê²€ì¦?ë¡œì§ ?†ìŒ")

        # ?„ë¡¬?„íŠ¸??ë¬¸ì„œ ?¬í•¨ ?¬ë? ê²€ì¦?
        if "PROMPT VALIDATION ERROR" in generate_answer_section:
            print("   ???„ë¡¬?„íŠ¸ ê²€ì¦??ëŸ¬ ì²˜ë¦¬ ì¡´ì¬")
        else:
            warnings.append("?„ë¡¬?„íŠ¸ ê²€ì¦??ëŸ¬ ì²˜ë¦¬ ?†ìŒ")

        if warnings:
            print("\n   ? ï¸ ë°œê²¬??ë¬¸ì œ:")
            for warning in warnings:
                print(f"      - {warning}")
        else:
            print("   ??ë°œê²¬??ë¬¸ì œ ?†ìŒ")

        return True

    except Exception as e:
        print(f"\n??ë¶„ì„ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = analyze_search_results_in_prompt()
    sys.exit(0 if success else 1)
