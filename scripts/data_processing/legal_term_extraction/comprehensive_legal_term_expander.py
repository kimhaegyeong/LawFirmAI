# -*- coding: utf-8 -*-
"""
?„ì²´ ë²•ë¥  ?©ì–´ ?•ì¥ ë°??€???œìŠ¤??
ë°°ì¹˜ ì²˜ë¦¬, ?ˆì§ˆ ê´€ë¦? ?°ì´?°ë² ?´ìŠ¤ ?€??ê¸°ëŠ¥ ?¬í•¨
"""

import os
import sys
import json
import logging
import sqlite3
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ?œê? ì¶œë ¥???„í•œ ?˜ê²½ ?¤ì •
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['GRPC_PYTHON_LOG_VERBOSITY'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Google API ê´€??ê²½ê³  ë¹„í™œ?±í™”
import warnings
warnings.filterwarnings('ignore')

# ?˜ê²½ë³€??ë¡œë“œ
env_path = r"D:\project\LawFirmAI\LawFirmAI\.env"
load_dotenv(env_path)

logger = logging.getLogger(__name__)

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class LegalTermExpansion(BaseModel):
    """ë²•ë¥  ?©ì–´ ?•ì¥ ê²°ê³¼ ëª¨ë¸"""
    synonyms: List[str] = Field(description="?™ì˜??ëª©ë¡")
    related_terms: List[str] = Field(description="ê´€???©ì–´ ëª©ë¡")
    precedent_keywords: List[str] = Field(description="?ë? ?¤ì›Œ??ëª©ë¡")
    confidence: float = Field(description="? ë¢°???ìˆ˜ (0.0-1.0)")


class LegalTermBatchExpander:
    """ë²•ë¥  ?©ì–´ ë°°ì¹˜ ?•ì¥ê¸?""
    
    def __init__(self, 
                 model_name: str = "gemini-2.0-flash-exp",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 batch_size: int = 10,
                 delay_between_batches: float = 2.0):
        """ë°°ì¹˜ ?•ì¥ê¸?ì´ˆê¸°??""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.CRITICAL)
        
        # ?˜ê²½ë³€?˜ì—??API ???•ì¸
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key or api_key == "your_google_api_key_here":
            raise ValueError("GOOGLE_API_KEYê°€ ?¤ì •?˜ì? ?Šì•˜?µë‹ˆ?? .env ?Œì¼??? íš¨??Google API ?¤ë? ?¤ì •?˜ì„¸??")
        
        if not api_key.startswith("AIza"):
            raise ValueError("? íš¨?˜ì? ?Šì? Google API ???•ì‹?…ë‹ˆ?? Google API ?¤ëŠ” 'AIza'ë¡??œì‘?´ì•¼ ?©ë‹ˆ??")
        
        # Gemini ëª¨ë¸ ì´ˆê¸°??
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        # ì¶œë ¥ ?Œì„œ ì´ˆê¸°??
        self.output_parser = JsonOutputParser(pydantic_object=LegalTermExpansion)
        
        # ?„ë¡¬?„íŠ¸ ?œí”Œë¦??¤ì •
        self.prompt_template = self._create_prompt_template()
        
        # ë²•ë¥  ?„ë©”?¸ë³„ ?„ë¡¬?„íŠ¸ ë¡œë“œ
        self.domain_prompts = self._load_domain_prompts()
        
        # ë°°ì¹˜ ì²˜ë¦¬ ?¤ì •
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        
        self.logger.info(f"LegalTermBatchExpander ì´ˆê¸°???„ë£Œ: {model_name}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """?„ë¡¬?„íŠ¸ ?œí”Œë¦??ì„±"""
        template = """
?¹ì‹ ?€ ?œêµ­ ë²•ë¥  ?„ë¬¸ê°€?…ë‹ˆ?? ì£¼ì–´ì§?ë²•ë¥  ?©ì–´???€???•í™•?˜ê³  ?„ë¬¸?ì¸ ?©ì–´ ?•ì¥???˜í–‰?´ì£¼?¸ìš”.

ë²•ë¥  ?„ë©”?? {domain}
ê¸°ë³¸ ?©ì–´: {base_term}

{domain_context}

?¤ìŒ ?•ì‹?¼ë¡œ JSON ?‘ë‹µ???ì„±?´ì£¼?¸ìš”:
{format_instructions}

?”êµ¬?¬í•­:
1. ?™ì˜?´ëŠ” ?˜ë?ê°€ ?™ì¼?˜ê±°??ë§¤ìš° ? ì‚¬???©ì–´ë§??¬í•¨
2. ê´€???©ì–´??ë²•ë¥ ?ìœ¼ë¡??°ê???ê°œë…??
3. ?ë? ?¤ì›Œ?œëŠ” ?¤ì œ ?ë??ì„œ ?¬ìš©?˜ëŠ” ?¤ì›Œ??
4. ? ë¢°?„ëŠ” ?ì„±???©ì–´?¤ì˜ ?•í™•?±ì„ ?‰ê? (0.0-1.0)
5. ëª¨ë“  ?©ì–´???œêµ­?´ë¡œ ?‘ì„±
6. ê°?ì¹´í…Œê³ ë¦¬??ìµœë? 5ê°??©ì–´ ?ì„±
"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _load_domain_prompts(self) -> Dict[str, str]:
        """ë²•ë¥  ?„ë©”?¸ë³„ ?„ë¡¬?„íŠ¸ ë¡œë“œ"""
        domain_prompts = {
            "ë¯¼ì‚¬ë²?: """
ë¯¼ì‚¬ë²??„ë©”???¹í™” ì§€ì¹?
- ê³„ì•½, ë¶ˆë²•?‰ìœ„, ?Œìœ ê¶? ?ì†, ê°€ì¡±ê?ê³???ë¯¼ì‚¬ ë¶„ìŸ ê´€???©ì–´
- ?í•´ë°°ìƒ, ê³„ì•½?´ì?, ?Œìœ ê¶Œì´?? ?ì†ë¶„í•  ??êµ¬ì²´??ë²•ë¥  ?‰ìœ„
- ë¯¼ë²• ì¡°ë¬¸ê³??°ê????„ë¬¸ ?©ì–´ ?¬ìš©
""",
            "?•ì‚¬ë²?: """
?•ì‚¬ë²??„ë©”???¹í™” ì§€ì¹?
- ë²”ì£„ ? í˜•, ì²˜ë²Œ, ?Œì†¡?ˆì°¨, ê´€?¨ì¸ë¬????•ì‚¬ ?¬ê±´ ê´€???©ì–´
- ?´ì¸, ?ˆë„, ?¬ê¸°, ê°•ë„, ê°•ê°„ ??êµ¬ì²´??ë²”ì£„ ? í˜•
- ?•ë²• ì¡°ë¬¸ê³??°ê????„ë¬¸ ?©ì–´ ?¬ìš©
""",
            "ê°€ì¡±ë²•": """
ê°€ì¡±ë²• ?„ë©”???¹í™” ì§€ì¹?
- ?¼ì¸ê´€ê³? ì¹œìê´€ê³? ?¬ì‚°ê´€ê³???ê°€ì¡?ê´€??ë²•ë¥  ?©ì–´
- ?´í˜¼, ?‘ìœ¡ê¶? ì¹œê¶Œ, ?ì† ??êµ¬ì²´??ê°€ì¡±ë²• ?¬ì•ˆ
- ê°€ì¡±ë²• ì¡°ë¬¸ê³??°ê????„ë¬¸ ?©ì–´ ?¬ìš©
""",
            "?ì‚¬ë²?: """
?ì‚¬ë²??„ë©”???¹í™” ì§€ì¹?
- ?Œì‚¬ë²? ?í–‰?? ?´ìŒ?˜í‘œ ???ì—… ê´€??ë²•ë¥  ?©ì–´
- ì£¼ì‹?Œì‚¬, ? í•œ?Œì‚¬, ?í–‰?? ?´ìŒ, ?˜í‘œ ??êµ¬ì²´???ì‚¬ë²?ê°œë…
- ?ë²• ì¡°ë¬¸ê³??°ê????„ë¬¸ ?©ì–´ ?¬ìš©
""",
            "?‰ì •ë²?: """
?‰ì •ë²??„ë©”???¹í™” ì§€ì¹?
- ?‰ì •?‰ìœ„, ?‰ì •?ˆì°¨, ?‰ì •?Œì†¡ ???‰ì • ê´€??ë²•ë¥  ?©ì–´
- ?‰ì •ì²˜ë¶„, ?‰ì •ì§€?? ?‰ì •?¬íŒ ??êµ¬ì²´???‰ì •ë²?ê°œë…
- ?‰ì •ë²?ì¡°ë¬¸ê³??°ê????„ë¬¸ ?©ì–´ ?¬ìš©
""",
            "?¸ë™ë²?: """
?¸ë™ë²??„ë©”???¹í™” ì§€ì¹?
- ê·¼ë¡œê³„ì•½, ?„ê¸ˆ, ê·¼ë¡œ?œê°„, ?´ê? ??ê·¼ë¡œ ê´€??ë²•ë¥  ?©ì–´
- ?´ê³ , ë¶€?¹í•´ê³? ?¤ì—…ê¸‰ì—¬ ??êµ¬ì²´???¸ë™ë²?ê°œë…
- ê·¼ë¡œê¸°ì?ë²? ?¸ë™ì¡°í•©ë²???ê´€??ë²•ë ¹ ?©ì–´ ?¬ìš©
""",
            "ê¸°í?": """
?¼ë°˜ ë²•ë¥  ?„ë©”??ì§€ì¹?
- ?¤ì–‘??ë²•ë¥  ë¶„ì•¼??ê³µí†µ?ìœ¼ë¡??¬ìš©?˜ëŠ” ?©ì–´
- ë²•ë¥  ?¼ë°˜ë¡? ë²•ì›, ê²€?? ë³€?¸ì‚¬ ??ë²•ë¥  ?œë„ ê´€???©ì–´
- ?Œë²•, êµ? œë²???ê¸°í? ë²•ë¥  ë¶„ì•¼ ?©ì–´
"""
        }
        
        return domain_prompts
    
    def expand_term(self, base_term: str, domain: str = "ë¯¼ì‚¬ë²?) -> Dict[str, Any]:
        """?¨ì¼ ?©ì–´ ?•ì¥"""
        try:
            # ?„ë©”??ì»¨í…?¤íŠ¸ ê°€?¸ì˜¤ê¸?
            domain_context = self.domain_prompts.get(domain, self.domain_prompts["ê¸°í?"])
            
            # ?„ë¡¬?„íŠ¸ ?ì„±
            prompt = self.prompt_template.format_messages(
                domain=domain,
                base_term=base_term,
                domain_context=domain_context,
                format_instructions=self.output_parser.get_format_instructions()
            )
            
            # LLM ?¸ì¶œ
            response = self.model.invoke(prompt)
            
            # ?‘ë‹µ ?Œì‹±
            parsed_response = self.output_parser.parse(response.content)
            
            # ê²°ê³¼ ê²€ì¦?ë°??•ì œ
            validated_result = self._validate_and_refine_result(parsed_response, base_term)
            
            return validated_result
            
        except Exception as e:
            print(f"?©ì–´ ?•ì¥ ì¤??¤ë¥˜ ë°œìƒ: {e}")
            return {
                "synonyms": [],
                "related_terms": [],
                "precedent_keywords": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _validate_and_refine_result(self, result: Dict[str, Any], base_term: str) -> Dict[str, Any]:
        """ê²°ê³¼ ê²€ì¦?ë°??•ì œ"""
        try:
            validated_result = {
                "synonyms": [],
                "related_terms": [],
                "precedent_keywords": [],
                "confidence": 0.0,
                "expanded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # ?™ì˜??ê²€ì¦?
            if "synonyms" in result and isinstance(result["synonyms"], list):
                validated_synonyms = []
                for synonym in result["synonyms"]:
                    if self._is_valid_legal_term(synonym) and synonym != base_term:
                        validated_synonyms.append(synonym.strip())
                validated_result["synonyms"] = validated_synonyms[:5]  # ìµœë? 5ê°?
            
            # ê´€???©ì–´ ê²€ì¦?
            if "related_terms" in result and isinstance(result["related_terms"], list):
                validated_related = []
                for term in result["related_terms"]:
                    if self._is_valid_legal_term(term) and term != base_term:
                        validated_related.append(term.strip())
                validated_result["related_terms"] = validated_related[:5]  # ìµœë? 5ê°?
            
            # ?ë? ?¤ì›Œ??ê²€ì¦?
            if "precedent_keywords" in result and isinstance(result["precedent_keywords"], list):
                validated_keywords = []
                for keyword in result["precedent_keywords"]:
                    if self._is_valid_legal_term(keyword) and keyword != base_term:
                        validated_keywords.append(keyword.strip())
                validated_result["precedent_keywords"] = validated_keywords[:5]  # ìµœë? 5ê°?
            
            # ? ë¢°??ê³„ì‚°
            confidence = result.get("confidence", 0.0)
            if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
                validated_result["confidence"] = float(confidence)
            else:
                # ?ë™ ? ë¢°??ê³„ì‚°
                validated_result["confidence"] = self._calculate_confidence(validated_result, base_term)
            
            return validated_result
            
        except Exception as e:
            self.logger.error(f"ê²°ê³¼ ê²€ì¦?ì¤??¤ë¥˜ ë°œìƒ: {e}")
            return {
                "synonyms": [],
                "related_terms": [],
                "precedent_keywords": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _is_valid_legal_term(self, term: str) -> bool:
        """ë²•ë¥  ?©ì–´ ? íš¨??ê²€ì¦?""
        if not term or not isinstance(term, str):
            return False
        
        term = term.strip()
        
        # ê¸°ë³¸ ê¸¸ì´ ê²€ì¦?
        if len(term) < 2 or len(term) > 20:
            return False
        
        # ?œê? ?¬í•¨ ê²€ì¦?
        if not re.search(r'[ê°€-??', term):
            return False
        
        # ?¹ìˆ˜ë¬¸ì ?œí•œ
        if re.search(r'[^\wê°€-??s]', term):
            return False
        
        # ë²•ë¥  ?„ë©”???¤ì›Œ??ê²€ì¦?
        legal_keywords = [
            'ë²?, 'ê¶?, 'ì±…ì„', '?í•´', 'ê³„ì•½', '?Œì†¡', 'ì²˜ë²Œ', '?œì¬',
            'ë°°ìƒ', 'ë³´ìƒ', 'ì²?µ¬', '?œê¸°', '?´ì?', '?„ë°˜', 'ì¹¨í•´',
            '?ˆì°¨', '? ì²­', '?¬ë¦¬', '?ê²°', '??†Œ', '?ê³ ',
            'ë²•ì›', 'ê²€??, 'ë³€?¸ì‚¬', '?¼ê³ ??, '?ê³ ', '?¼ê³ '
        ]
        
        # ë²•ë¥  ?¤ì›Œ?œê? ?¬í•¨?˜ì–´ ?ˆê±°??ë²•ë¥  ê´€???©ì–´?¸ì? ?•ì¸
        has_legal_keyword = any(keyword in term for keyword in legal_keywords)
        is_legal_concept = re.search(r'[ê°€-??{2,6}(?:ë²?ê¶?ì±…ì„|?í•´|ê³„ì•½|?Œì†¡)', term)
        
        return has_legal_keyword or is_legal_concept or len(term) >= 3
    
    def _calculate_confidence(self, result: Dict[str, Any], base_term: str) -> float:
        """? ë¢°???ë™ ê³„ì‚°"""
        try:
            total_terms = len(result.get("synonyms", [])) + len(result.get("related_terms", [])) + len(result.get("precedent_keywords", []))
            
            if total_terms == 0:
                return 0.0
            
            # ê¸°ë³¸ ?ìˆ˜
            base_score = 0.5
            
            # ?©ì–´ ?˜ì— ?°ë¥¸ ?ìˆ˜
            term_count_score = min(total_terms / 15, 0.3)  # ìµœë? 0.3??
            
            # ?©ì–´ ?ˆì§ˆ ?ìˆ˜
            quality_score = 0.0
            for category in ["synonyms", "related_terms", "precedent_keywords"]:
                for term in result.get(category, []):
                    if self._is_valid_legal_term(term):
                        quality_score += 0.1
            
            quality_score = min(quality_score, 0.2)  # ìµœë? 0.2??
            
            final_score = base_score + term_count_score + quality_score
            return min(final_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"? ë¢°??ê³„ì‚° ì¤??¤ë¥˜ ë°œìƒ: {e}")
            return 0.5
    
    def expand_all_terms(self, terms: List[str], domain: str = "ë¯¼ì‚¬ë²?) -> Dict[str, Any]:
        """?„ì²´ ?©ì–´ë¥?ë°°ì¹˜ë¡??•ì¥"""
        try:
            self.logger.info(f"?„ì²´ ?©ì–´ ?•ì¥ ?œì‘: {len(terms)}ê°??©ì–´ ({domain})")
            
            results = {}
            successful_expansions = 0
            failed_expansions = 0
            
            # ë°°ì¹˜ ?¨ìœ„ë¡?ì²˜ë¦¬
            for i in range(0, len(terms), self.batch_size):
                batch_terms = terms[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (len(terms) + self.batch_size - 1) // self.batch_size
                
                self.logger.info(f"ë°°ì¹˜ {batch_num}/{total_batches} ì²˜ë¦¬ ì¤? {len(batch_terms)}ê°??©ì–´")
                
                # ë°°ì¹˜ ?´ì—???œì°¨ ì²˜ë¦¬
                for j, term in enumerate(batch_terms):
                    try:
                        self.logger.info(f"?©ì–´ ?•ì¥: {term} ({i+j+1}/{len(terms)})")
                        
                        expansion_result = self.expand_term(term, domain)
                        results[term] = expansion_result
                        
                        if "error" not in expansion_result:
                            successful_expansions += 1
                        else:
                            failed_expansions += 1
                        
                        # ì§„í–‰ë¥?ë¡œê¹…
                        progress = (i + j + 1) / len(terms) * 100
                        self.logger.info(f"ì§„í–‰ë¥? {progress:.1f}% ({i+j+1}/{len(terms)})")
                        
                    except Exception as e:
                        self.logger.error(f"?©ì–´ '{term}' ?•ì¥ ì¤??¤ë¥˜: {e}")
                        results[term] = {
                            "synonyms": [],
                            "related_terms": [],
                            "precedent_keywords": [],
                            "confidence": 0.0,
                            "error": str(e)
                        }
                        failed_expansions += 1
                
                # ë°°ì¹˜ ê°?ì§€??
                if i + self.batch_size < len(terms):
                    self.logger.info(f"ë°°ì¹˜ ê°?ì§€?? {self.delay_between_batches}ì´?)
                    time.sleep(self.delay_between_batches)
                
                # API ? ë‹¹??ê´€ë¦¬ë? ?„í•œ ì¶”ê? ì§€??
                if i + self.batch_size < len(terms):
                    self.logger.info("API ? ë‹¹??ê´€ë¦¬ë? ?„í•œ ì¶”ê? 2ì´?ì§€??..")
                    time.sleep(2)
            
            # ?„ì²´ ê²°ê³¼ ?”ì•½
            batch_summary = {
                "total_terms": len(terms),
                "successful_expansions": successful_expansions,
                "failed_expansions": failed_expansions,
                "success_rate": successful_expansions / len(terms) if terms else 0,
                "domain": domain,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": results
            }
            
            self.logger.info(f"?„ì²´ ?•ì¥ ?„ë£Œ: {successful_expansions}/{len(terms)} ?±ê³µ ({batch_summary['success_rate']:.1%})")
            
            return batch_summary
            
        except Exception as e:
            self.logger.error(f"?„ì²´ ?•ì¥ ì¤??¤ë¥˜ ë°œìƒ: {e}")
            return {
                "total_terms": len(terms),
                "successful_expansions": 0,
                "failed_expansions": len(terms),
                "success_rate": 0.0,
                "error": str(e),
                "results": {}
            }
    
    def save_progress(self, results: Dict[str, Any], checkpoint_file: str):
        """ì§„í–‰ ?í™© ?€??""
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"ì§„í–‰ ?í™© ?€???„ë£Œ: {checkpoint_file}")
        except Exception as e:
            self.logger.error(f"ì§„í–‰ ?í™© ?€??ì¤??¤ë¥˜: {e}")
    
    def resume_from_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """ì¤‘ë‹¨???‘ì—… ?¬ê°œ"""
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                self.logger.info(f"ì²´í¬?¬ì¸?¸ì—??ë³µêµ¬: {checkpoint_file}")
                return results
            else:
                self.logger.info(f"ì²´í¬?¬ì¸???Œì¼??ì¡´ì¬?˜ì? ?ŠìŒ: {checkpoint_file}")
                return {}
        except Exception as e:
            self.logger.error(f"ì²´í¬?¬ì¸??ë³µêµ¬ ì¤??¤ë¥˜: {e}")
            return {}


class QualityValidator:
    """?©ì–´ ?•ì¥ ?ˆì§ˆ ê²€ì¦ê¸°"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.CRITICAL)
    
    def validate_expansion_quality(self, original_term: str, expansion_result: Dict) -> float:
        """?•ì¥ ê²°ê³¼ ?ˆì§ˆ ê²€ì¦?""
        try:
            quality_score = 0.0
            
            # ê¸°ë³¸ ?ìˆ˜
            base_score = 0.3
            
            # ?©ì–´ ???ìˆ˜
            total_terms = len(expansion_result.get("synonyms", [])) + \
                         len(expansion_result.get("related_terms", [])) + \
                         len(expansion_result.get("precedent_keywords", []))
            
            term_count_score = min(total_terms / 15, 0.3)  # ìµœë? 0.3??
            
            # ?©ì–´ ?ˆì§ˆ ?ìˆ˜
            quality_score = 0.0
            for category in ["synonyms", "related_terms", "precedent_keywords"]:
                for term in expansion_result.get(category, []):
                    if self._is_high_quality_term(term):
                        quality_score += 0.1
            
            quality_score = min(quality_score, 0.4)  # ìµœë? 0.4??
            
            final_score = base_score + term_count_score + quality_score
            return min(final_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"?ˆì§ˆ ê²€ì¦?ì¤??¤ë¥˜: {e}")
            return 0.5
    
    def _is_high_quality_term(self, term: str) -> bool:
        """ê³ í’ˆì§??©ì–´ ?ë‹¨"""
        if not term or len(term) < 2:
            return False
        
        # ë²•ë¥  ?„ë¬¸ ?©ì–´ ?¤ì›Œ??
        legal_keywords = [
            'ë²?, 'ê¶?, 'ì±…ì„', '?í•´', 'ê³„ì•½', '?Œì†¡', 'ì²˜ë²Œ', '?œì¬',
            'ë°°ìƒ', 'ë³´ìƒ', 'ì²?µ¬', '?œê¸°', '?´ì?', '?„ë°˜', 'ì¹¨í•´',
            '?ˆì°¨', '? ì²­', '?¬ë¦¬', '?ê²°', '??†Œ', '?ê³ '
        ]
        
        return any(keyword in term for keyword in legal_keywords)
    
    def filter_low_quality_terms(self, results: Dict[str, Any], threshold: float = 0.7) -> Dict[str, Any]:
        """?€?ˆì§ˆ ?©ì–´ ?„í„°ë§?""
        try:
            filtered_results = {}
            
            for term, expansion_result in results.items():
                quality_score = self.validate_expansion_quality(term, expansion_result)
                
                if quality_score >= threshold:
                    expansion_result["quality_score"] = quality_score
                    filtered_results[term] = expansion_result
                else:
                    self.logger.info(f"?€?ˆì§ˆ ?©ì–´ ?œì™¸: {term} (?ˆì§ˆ?ìˆ˜: {quality_score:.2f})")
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"?ˆì§ˆ ?„í„°ë§?ì¤??¤ë¥˜: {e}")
            return results
    
    def generate_quality_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """?ˆì§ˆ ë³´ê³ ???ì„±"""
        try:
            total_terms = len(results)
            quality_scores = []
            
            for term, expansion_result in results.items():
                quality_score = self.validate_expansion_quality(term, expansion_result)
                quality_scores.append(quality_score)
            
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                min_quality = min(quality_scores)
                max_quality = max(quality_scores)
            else:
                avg_quality = min_quality = max_quality = 0.0
            
            report = {
                "total_terms": total_terms,
                "average_quality": avg_quality,
                "min_quality": min_quality,
                "max_quality": max_quality,
                "high_quality_terms": len([s for s in quality_scores if s >= 0.8]),
                "medium_quality_terms": len([s for s in quality_scores if 0.6 <= s < 0.8]),
                "low_quality_terms": len([s for s in quality_scores if s < 0.6]),
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"?ˆì§ˆ ë³´ê³ ???ì„± ì¤??¤ë¥˜: {e}")
            return {}


class LegalTermDatabase:
    """ë²•ë¥  ?©ì–´ ?°ì´?°ë² ?´ìŠ¤ ê´€ë¦¬ì"""
    
    def __init__(self, db_path: str = "data/legal_terms.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.CRITICAL)
        self._create_tables()
    
    def _create_tables(self):
        """?°ì´?°ë² ?´ìŠ¤ ?Œì´ë¸??ì„±"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # ?©ì–´ ?Œì´ë¸?
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS legal_terms (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        term TEXT UNIQUE NOT NULL,
                        domain TEXT NOT NULL,
                        synonyms TEXT,
                        related_terms TEXT,
                        precedent_keywords TEXT,
                        confidence REAL,
                        quality_score REAL,
                        expanded_at TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # ?•ì¥ ?´ë ¥ ?Œì´ë¸?
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS expansion_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        term TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        expansion_result TEXT,
                        success BOOLEAN,
                        error_message TEXT,
                        processed_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info("?°ì´?°ë² ?´ìŠ¤ ?Œì´ë¸??ì„± ?„ë£Œ")
                
        except Exception as e:
            self.logger.error(f"?°ì´?°ë² ?´ìŠ¤ ?Œì´ë¸??ì„± ì¤??¤ë¥˜: {e}")
    
    def save_expanded_terms(self, terms: Dict[str, Any], domain: str = "ë¯¼ì‚¬ë²?):
        """?•ì¥???©ì–´ ?€??""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for term, expansion_result in terms.items():
                    # ê¸°ì¡´ ?©ì–´ ?•ì¸
                    cursor.execute("SELECT id FROM legal_terms WHERE term = ?", (term,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # ê¸°ì¡´ ?©ì–´ ?…ë°?´íŠ¸
                        cursor.execute('''
                            UPDATE legal_terms SET
                                synonyms = ?, related_terms = ?, precedent_keywords = ?,
                                confidence = ?, quality_score = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE term = ?
                        ''', (
                            json.dumps(expansion_result.get("synonyms", []), ensure_ascii=False),
                            json.dumps(expansion_result.get("related_terms", []), ensure_ascii=False),
                            json.dumps(expansion_result.get("precedent_keywords", []), ensure_ascii=False),
                            expansion_result.get("confidence", 0.0),
                            expansion_result.get("quality_score", 0.0),
                            term
                        ))
                    else:
                        # ???©ì–´ ?½ì…
                        cursor.execute('''
                            INSERT INTO legal_terms 
                            (term, domain, synonyms, related_terms, precedent_keywords, 
                             confidence, quality_score, expanded_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            term,
                            domain,
                            json.dumps(expansion_result.get("synonyms", []), ensure_ascii=False),
                            json.dumps(expansion_result.get("related_terms", []), ensure_ascii=False),
                            json.dumps(expansion_result.get("precedent_keywords", []), ensure_ascii=False),
                            expansion_result.get("confidence", 0.0),
                            expansion_result.get("quality_score", 0.0),
                            expansion_result.get("expanded_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        ))
                
                conn.commit()
                self.logger.info(f"?©ì–´ ?€???„ë£Œ: {len(terms)}ê°?)
                
        except Exception as e:
            self.logger.error(f"?©ì–´ ?€??ì¤??¤ë¥˜: {e}")
    
    def get_terms_by_domain(self, domain: str) -> List[Dict]:
        """?„ë©”?¸ë³„ ?©ì–´ ì¡°íšŒ"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM legal_terms WHERE domain = ?", (domain,))
                rows = cursor.fetchall()
                
                terms = []
                for row in rows:
                    term_data = {
                        "term": row["term"],
                        "domain": row["domain"],
                        "synonyms": json.loads(row["synonyms"]) if row["synonyms"] else [],
                        "related_terms": json.loads(row["related_terms"]) if row["related_terms"] else [],
                        "precedent_keywords": json.loads(row["precedent_keywords"]) if row["precedent_keywords"] else [],
                        "confidence": row["confidence"],
                        "quality_score": row["quality_score"],
                        "expanded_at": row["expanded_at"]
                    }
                    terms.append(term_data)
                
                return terms
                
        except Exception as e:
            self.logger.error(f"?©ì–´ ì¡°íšŒ ì¤??¤ë¥˜: {e}")
            return []
    
    def get_all_terms(self) -> List[Dict]:
        """ëª¨ë“  ?©ì–´ ì¡°íšŒ"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM legal_terms ORDER BY domain, term")
                rows = cursor.fetchall()
                
                terms = []
                for row in rows:
                    term_data = {
                        "term": row["term"],
                        "domain": row["domain"],
                        "synonyms": json.loads(row["synonyms"]) if row["synonyms"] else [],
                        "related_terms": json.loads(row["related_terms"]) if row["related_terms"] else [],
                        "precedent_keywords": json.loads(row["precedent_keywords"]) if row["precedent_keywords"] else [],
                        "confidence": row["confidence"],
                        "quality_score": row["quality_score"],
                        "expanded_at": row["expanded_at"]
                    }
                    terms.append(term_data)
                
                return terms
                
        except Exception as e:
            self.logger.error(f"?„ì²´ ?©ì–´ ì¡°íšŒ ì¤??¤ë¥˜: {e}")
            return []
    
    def update_term_quality(self, term: str, quality_score: float):
        """?©ì–´ ?ˆì§ˆ ?ìˆ˜ ?…ë°?´íŠ¸"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE legal_terms SET quality_score = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE term = ?
                ''', (quality_score, term))
                
                conn.commit()
                self.logger.info(f"?©ì–´ ?ˆì§ˆ ?ìˆ˜ ?…ë°?´íŠ¸: {term} -> {quality_score}")
                
        except Exception as e:
            self.logger.error(f"?ˆì§ˆ ?ìˆ˜ ?…ë°?´íŠ¸ ì¤??¤ë¥˜: {e}")
    
    def export_to_json(self, output_file: str):
        """JSON ?Œì¼ë¡??´ë³´?´ê¸°"""
        try:
            terms = self.get_all_terms()
            
            export_data = {
                "metadata": {
                    "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_terms": len(terms),
                    "domains": list(set(term["domain"] for term in terms))
                },
                "dictionary": {}
            }
            
            for term_data in terms:
                export_data["dictionary"][term_data["term"]] = {
                    "synonyms": term_data["synonyms"],
                    "related_terms": term_data["related_terms"],
                    "precedent_keywords": term_data["precedent_keywords"],
                    "confidence": term_data["confidence"],
                    "quality_score": term_data["quality_score"],
                    "domain": term_data["domain"],
                    "expanded_at": term_data["expanded_at"]
                }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"JSON ?´ë³´?´ê¸° ?„ë£Œ: {output_file}")
            
        except Exception as e:
            self.logger.error(f"JSON ?´ë³´?´ê¸° ì¤??¤ë¥˜: {e}")


class ProgressMonitor:
    """ì§„í–‰ ?í™© ëª¨ë‹ˆ?°ë§"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.CRITICAL)
        self.total_terms = 0
        self.processed_terms = 0
        self.successful_expansions = 0
        self.failed_expansions = 0
        self.start_time = None
    
    def start_monitoring(self, total_terms: int):
        """ëª¨ë‹ˆ?°ë§ ?œì‘"""
        self.total_terms = total_terms
        self.processed_terms = 0
        self.successful_expansions = 0
        self.failed_expansions = 0
        self.start_time = datetime.now()
        self.logger.info(f"ëª¨ë‹ˆ?°ë§ ?œì‘: {total_terms}ê°??©ì–´")
    
    def update_progress(self, batch_results: Dict[str, Any]):
        """ì§„í–‰ ?í™© ?…ë°?´íŠ¸"""
        try:
            if "results" in batch_results:
                batch_size = len(batch_results["results"])
                self.processed_terms += batch_size
                self.successful_expansions += batch_results.get("successful_expansions", 0)
                self.failed_expansions += batch_results.get("failed_expansions", 0)
                
                progress = self.processed_terms / self.total_terms * 100
                self.logger.info(f"ì§„í–‰ë¥? {progress:.1f}% ({self.processed_terms}/{self.total_terms})")
                
        except Exception as e:
            self.logger.error(f"ì§„í–‰ ?í™© ?…ë°?´íŠ¸ ì¤??¤ë¥˜: {e}")
    
    def generate_progress_report(self) -> Dict[str, Any]:
        """ì§„í–‰ ?í™© ë³´ê³ ???ì„±"""
        try:
            if self.start_time:
                elapsed_time = datetime.now() - self.start_time
                elapsed_seconds = elapsed_time.total_seconds()
                
                if self.processed_terms > 0:
                    avg_time_per_term = elapsed_seconds / self.processed_terms
                    remaining_terms = self.total_terms - self.processed_terms
                    estimated_remaining_time = remaining_terms * avg_time_per_term
                else:
                    avg_time_per_term = 0
                    estimated_remaining_time = 0
            else:
                elapsed_seconds = 0
                avg_time_per_term = 0
                estimated_remaining_time = 0
            
            report = {
                "total_terms": self.total_terms,
                "processed_terms": self.processed_terms,
                "successful_expansions": self.successful_expansions,
                "failed_expansions": self.failed_expansions,
                "success_rate": self.successful_expansions / self.processed_terms if self.processed_terms > 0 else 0,
                "progress_percentage": self.processed_terms / self.total_terms * 100 if self.total_terms > 0 else 0,
                "elapsed_time_seconds": elapsed_seconds,
                "average_time_per_term": avg_time_per_term,
                "estimated_remaining_time": estimated_remaining_time,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"ì§„í–‰ ?í™© ë³´ê³ ???ì„± ì¤??¤ë¥˜: {e}")
            return {}


def safe_print(text: str):
    """?ˆì „???œê? ì¶œë ¥ ?¨ìˆ˜"""
    try:
        # ?Œì¼ë¡?ì¶œë ¥?˜ì—¬ ?œê? ë¬¸ì œ ?´ê²°
        with open('legal_term_expansion_output.txt', 'a', encoding='utf-8') as f:
            f.write(text + '\n')
        
        # ì½˜ì†” ì¶œë ¥?€ ASCIIë¡?ë³€?˜í•˜??ê¹¨ì§ ë°©ì?
        try:
            ascii_text = text.encode('ascii', 'ignore').decode('ascii')
            if ascii_text.strip():
                print(ascii_text)
        except:
            print("[?œê? ì¶œë ¥ - legal_term_expansion_output.txt ?Œì¼ ì°¸ì¡°]")
    except Exception:
        # ê¸°í? ?¤ë¥˜ ???ë³¸ ì¶œë ¥
        print(text)


def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    # ë¡œê¹… ë¹„í™œ?±í™”
    logging.getLogger().setLevel(logging.CRITICAL)
    
    safe_print("?„ì²´ ë²•ë¥  ?©ì–´ ?•ì¥ ë°??€???œìŠ¤??)
    safe_print("=" * 50)
    
    try:
        # ?œìŠ¤??ì´ˆê¸°??
        expander = LegalTermBatchExpander()
        validator = QualityValidator()
        database = LegalTermDatabase()
        monitor = ProgressMonitor()
        
        # ?ŒìŠ¤?¸ìš© ë²•ë¥  ?©ì–´ ëª©ë¡ (?¤ì œë¡œëŠ” ??ë§ì? ?©ì–´ ?¬ìš©)
        test_terms = [
            # ë¯¼ì‚¬ë²??„ë©”??
            "?í•´ë°°ìƒ", "ê³„ì•½", "?Œìœ ê¶?, "?„ë?ì°?, "ë¶ˆë²•?‰ìœ„", "?ì†", "?´í˜¼", "êµí†µ?¬ê³ ", "ê·¼ë¡œ",
            "ë¬¼ê¶Œ", "ì±„ê¶Œ", "ê°€ì¡±ê?ê³?, "ì¹œì¡±", "?‘ìœ¡ê¶?, "ì¹œê¶Œ", "?„ìë£?, "?¬ì‚°ë¶„í• ",
            
            # ?•ì‚¬ë²??„ë©”??
            "?´ì¸", "?ˆë„", "?¬ê¸°", "ê°•ë„", "ê°•ê°„", "??–‰", "?í•´", "ëª…ì˜ˆ?¼ì†", "ëª¨ë…",
            "?„ì£¼", "ì¦ê±°?¸ë©¸", "?„ì¦", "ë¬´ê³ ", "ê³µê°ˆ", "?¡ë ¹", "ë°°ì„", "?Œë¬¼",
            
            # ?ì‚¬ë²??„ë©”??
            "ì£¼ì‹?Œì‚¬", "? í•œ?Œì‚¬", "?©ëª…?Œì‚¬", "?©ì?Œì‚¬", "?í–‰??, "?´ìŒ", "?˜í‘œ", "?´ìƒ",
            "ë³´í—˜", "?´ì†¡", "?„ì„", "?€ë¦?, "ì¤‘ê°œ", "?„ê¸‰", "?„ì¹˜", "?¬ìš©?€ì°?,
            
            # ?‰ì •ë²??„ë©”??
            "?‰ì •ì²˜ë¶„", "?‰ì •ì§€??, "?‰ì •?¬íŒ", "?‰ì •?Œì†¡", "?ˆê?", "?¸ê?", "?¹ì¸", "ë©´í—ˆ",
            "?±ë¡", "? ê³ ", "? ì²­", "ê³ ì?", "ê³µê³ ", "ê³µì‹œ", "ì¡°ì‚¬", "ê²€??,
            
            # ?¸ë™ë²??„ë©”??
            "ê·¼ë¡œê³„ì•½", "?„ê¸ˆ", "ê·¼ë¡œ?œê°„", "?´ê?", "?´ê³ ", "ë¶€?¹í•´ê³?, "?¤ì—…ê¸‰ì—¬", "?°ì—…?¬í•´",
            "?¸ë™ì¡°í•©", "?¨ì²´êµì„­", "?¨ì²´?‘ì•½", "?Œì—…", "ë¡œí¬?„ì›ƒ", "ë¶„ìŸì¡°ì •", "ì¤‘ì¬", "ì¡°ì •"
        ]
        
        safe_print(f"?•ì¥???©ì–´ ?? {len(test_terms)}ê°?)
        safe_print(f"?„ë©”?¸ë³„ ë¶„ë¥˜:")
        safe_print(f"  - ë¯¼ì‚¬ë²? 18ê°?)
        safe_print(f"  - ?•ì‚¬ë²? 18ê°?)
        safe_print(f"  - ?ì‚¬ë²? 16ê°?)
        safe_print(f"  - ?‰ì •ë²? 16ê°?)
        safe_print(f"  - ?¸ë™ë²? 16ê°?)
        safe_print("-" * 30)
        
        # ëª¨ë‹ˆ?°ë§ ?œì‘
        monitor.start_monitoring(len(test_terms))
        
        # ?„ë©”?¸ë³„ ?•ì¥
        domains = {
            "ë¯¼ì‚¬ë²?: test_terms[:18],
            "?•ì‚¬ë²?: test_terms[18:36],
            "?ì‚¬ë²?: test_terms[36:52],
            "?‰ì •ë²?: test_terms[52:68],
            "?¸ë™ë²?: test_terms[68:84]
        }
        
        all_results = {}
        
        for domain, terms in domains.items():
            safe_print(f"\n{domain} ?„ë©”???•ì¥ ?œì‘: {len(terms)}ê°??©ì–´")
            safe_print("-" * 30)
            
            # ?©ì–´ ?•ì¥
            domain_results = expander.expand_all_terms(terms, domain)
            
            # ?ˆì§ˆ ê²€ì¦?
            validated_results = validator.filter_low_quality_terms(domain_results["results"], threshold=0.6)
            
            # ?°ì´?°ë² ?´ìŠ¤ ?€??
            database.save_expanded_terms(validated_results, domain)
            
            # ê²°ê³¼ ?µí•©
            all_results.update(validated_results)
            
            # ì§„í–‰ ?í™© ?…ë°?´íŠ¸
            monitor.update_progress(domain_results)
            
            safe_print(f"{domain} ?„ë©”???•ì¥ ?„ë£Œ:")
            safe_print(f"  ?±ê³µ: {domain_results['successful_expansions']}ê°?)
            safe_print(f"  ?¤íŒ¨: {domain_results['failed_expansions']}ê°?)
            safe_print(f"  ?±ê³µë¥? {domain_results['success_rate']:.1%}")
        
        # ?„ì²´ ?ˆì§ˆ ë³´ê³ ???ì„±
        quality_report = validator.generate_quality_report(all_results)
        
        # ì§„í–‰ ?í™© ë³´ê³ ???ì„±
        progress_report = monitor.generate_progress_report()
        
        # JSON ?Œì¼ë¡??´ë³´?´ê¸°
        database.export_to_json("data/comprehensive_legal_term_dictionary.json")
        
        # ìµœì¢… ê²°ê³¼ ì¶œë ¥
        safe_print(f"\n?„ì²´ ë²•ë¥  ?©ì–´ ?•ì¥ ?„ë£Œ!")
        safe_print("=" * 50)
        safe_print(f"ì´?ì²˜ë¦¬ ?©ì–´: {progress_report['total_terms']}ê°?)
        safe_print(f"?±ê³µ: {progress_report['successful_expansions']}ê°?)
        safe_print(f"?¤íŒ¨: {progress_report['failed_expansions']}ê°?)
        safe_print(f"?±ê³µë¥? {progress_report['success_rate']:.1%}")
        safe_print(f"?‰ê·  ?ˆì§ˆ: {quality_report.get('average_quality', 0):.2f}")
        safe_print(f"ì²˜ë¦¬ ?œê°„: {progress_report['elapsed_time_seconds']:.1f}ì´?)
        safe_print(f"?€???„ì¹˜: data/comprehensive_legal_term_dictionary.json")
        
    except ValueError as e:
        safe_print(f"ì´ˆê¸°???¤ë¥˜: {e}")
        safe_print("?´ê²° ë°©ë²•: .env ?Œì¼??? íš¨??GOOGLE_API_KEYë¥??¤ì •?˜ì„¸??")
    except Exception as e:
        safe_print(f"?¤í–‰ ì¤??¤ë¥˜ ë°œìƒ: {e}")


if __name__ == "__main__":
    main()
