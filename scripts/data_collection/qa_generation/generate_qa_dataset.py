#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q&A ?°ì´?°ì…‹ ?ì„± ?¤í¬ë¦½íŠ¸

?˜ì§‘??ë²•ë¥  ?°ì´?°ë? ê¸°ë°˜?¼ë¡œ Q&A ?°ì´?°ì…‹???ì„±?©ë‹ˆ??
- ?ë™ Q&A ?ì„± ?Œì´?„ë¼??
- ë²•ë¥  ?„ë¬¸ê°€ ê²€? ìš© ?°ì´??ì¤€ë¹?
- ?ˆì§ˆ ?ìˆ˜ ë§¤ê¸°ê¸?
- ?°ì´?°ì…‹ ìµœì¢… ê²€ì¦?
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

from source.data.data_processor import LegalDataProcessor

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/generate_qa_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Q&A ?ì„± ?œí”Œë¦?
QA_TEMPLATES = {
    'law_definition': [
        "{law_name}?´ë? ë¬´ì—‡?¸ê???",
        "{law_name}???•ì˜ë¥??¤ëª…?´ì£¼?¸ìš”.",
        "{law_name}??ëª©ì ?€ ë¬´ì—‡?¸ê???",
        "{law_name}???ìš© ë²”ìœ„???´ë–»ê²??˜ë‚˜??"
    ],
    'law_article': [
        "{law_name} ??article}ì¡°ì˜ ?´ìš©???¤ëª…?´ì£¼?¸ìš”.",
        "{law_name} ??article}ì¡°ì—??ê·œì •?˜ëŠ” ?´ìš©?€ ë¬´ì—‡?¸ê???",
        "{law_name} ??article}ì¡°ì˜ ?”ê±´?€ ë¬´ì—‡?¸ê???",
        "{law_name} ??article}ì¡°ì˜ ?¨ê³¼??ë¬´ì—‡?¸ê???"
    ],
    'precedent_issue': [
        "{case_name} ?¬ê±´???ì ?€ ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´?ì„œ ?¤ë£¬ ë¬¸ì œ??ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´???µì‹¬ ?ì ???¤ëª…?´ì£¼?¸ìš”.",
        "{case_name} ?¬ê±´??ë²•ì  ?ì ?€ ë¬´ì—‡?¸ê???"
    ],
    'precedent_decision': [
        "{case_name} ?¬ê±´???ê²° ?´ìš©?€ ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´?ì„œ ë²•ì›???´ë¦° ê²°ë¡ ?€ ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´???ê²° ?”ì???ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´??ë²•ì› ?ë‹¨???¤ëª…?´ì£¼?¸ìš”."
    ],
    'constitutional_issue': [
        "{case_name} ?¬ê±´???Œë²•???ì ?€ ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´?ì„œ ?¤ë£¬ ê¸°ë³¸ê¶?ë¬¸ì œ??ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´???Œë²•?¬íŒ???ë‹¨ ?€?ì? ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´???Œë²•???˜ë???ë¬´ì—‡?¸ê???"
    ],
    'constitutional_decision': [
        "{case_name} ?¬ê±´???Œë²•?¬íŒ??ê²°ì •?€ ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´?ì„œ ?Œë²•?¬íŒ?Œê? ?´ë¦° ê²°ë¡ ?€ ë¬´ì—‡?¸ê???",
        "{case_name} ?¬ê±´???Œë²•?¬íŒ???ë‹¨???¤ëª…?´ì£¼?¸ìš”.",
        "{case_name} ?¬ê±´???Œë²•???ë‹¨?€ ë¬´ì—‡?¸ê???"
    ],
    'interpretation_question': [
        "{topic}???€??ë²•ë ¹?´ì„?€ ?´ë–»ê²??˜ë‚˜??",
        "{topic}??ë²•ì  ?´ì„ ê¸°ì??€ ë¬´ì—‡?¸ê???",
        "{topic}???€??ì¤‘ì•™ë¶€ì²˜ì˜ ?´ì„?€ ë¬´ì—‡?¸ê???",
        "{topic}??ë²•ë ¹ ?ìš© ê¸°ì??€ ë¬´ì—‡?¸ê???"
    ],
    'general_legal': [
        "{keyword}???€??ë²•ì  ê·¼ê±°??ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?”ê±´?€ ë¬´ì—‡?¸ê???",
        "{keyword}??ë²•ì  ?¨ê³¼??ë¬´ì—‡?¸ê???",
        "{keyword}???€??ë²•ì  ?´ì„?€ ?´ë–»ê²??˜ë‚˜??"
    ]
}

# ?µë? ?ì„± ?œí”Œë¦?
ANSWER_TEMPLATES = {
    'law_definition': [
        "{law_name}?€ {definition}??ëª©ì ?¼ë¡œ ?˜ëŠ” ë²•ë¥ ?…ë‹ˆ??",
        "{law_name}??{definition}??ê´€???¬í•­??ê·œì •??ë²•ë¥ ?…ë‹ˆ??",
        "{law_name}??ëª©ì ?€ {definition}?…ë‹ˆ??",
        "{law_name}?€ {definition}??ê·œì •?˜ëŠ” ë²•ë¥ ?…ë‹ˆ??"
    ],
    'law_article': [
        "{law_name} ??article}ì¡°ì— ?°ë¥´ë©? {content}?…ë‹ˆ??",
        "??article}ì¡°ì—?œëŠ” {content}?¼ê³  ê·œì •?˜ê³  ?ˆìŠµ?ˆë‹¤.",
        "{law_name} ??article}ì¡°ì˜ ?´ìš©?€ {content}?…ë‹ˆ??",
        "??article}ì¡°ì— ê·œì •???´ìš©?€ {content}?…ë‹ˆ??"
    ],
    'precedent_issue': [
        "{case_name} ?¬ê±´???ì ?€ {issue}?…ë‹ˆ??",
        "???¬ê±´?ì„œ ?¤ë£¬ ë¬¸ì œ??{issue}?…ë‹ˆ??",
        "ë²•ì›???ë‹¨???ì ?€ {issue}?…ë‹ˆ??",
        "?¬ê±´???µì‹¬ ?ì ?€ {issue}?…ë‹ˆ??"
    ],
    'precedent_decision': [
        "{case_name} ?¬ê±´?ì„œ ë²•ì›?€ {decision}?¼ê³  ?ë‹¨?ˆìŠµ?ˆë‹¤.",
        "ë²•ì›???ê²° ?´ìš©?€ {decision}?…ë‹ˆ??",
        "???¬ê±´???ê²° ?”ì???{decision}?…ë‹ˆ??",
        "ë²•ì›???´ë¦° ê²°ë¡ ?€ {decision}?…ë‹ˆ??"
    ],
    'constitutional_issue': [
        "{case_name} ?¬ê±´???Œë²•???ì ?€ {issue}?…ë‹ˆ??",
        "???¬ê±´?ì„œ ?¤ë£¬ ê¸°ë³¸ê¶?ë¬¸ì œ??{issue}?…ë‹ˆ??",
        "?Œë²•?¬íŒ?Œê? ?ë‹¨???€?ì? {issue}?…ë‹ˆ??",
        "?¬ê±´???Œë²•???˜ë???{issue}?…ë‹ˆ??"
    ],
    'constitutional_decision': [
        "{case_name} ?¬ê±´?ì„œ ?Œë²•?¬íŒ?ŒëŠ” {decision}?¼ê³  ê²°ì •?ˆìŠµ?ˆë‹¤.",
        "?Œë²•?¬íŒ?Œì˜ ê²°ì • ?´ìš©?€ {decision}?…ë‹ˆ??",
        "???¬ê±´???Œë²•?¬íŒ???ë‹¨?€ {decision}?…ë‹ˆ??",
        "?Œë²•?¬íŒ?Œê? ?´ë¦° ê²°ë¡ ?€ {decision}?…ë‹ˆ??"
    ],
    'interpretation_question': [
        "{topic}???€??ë²•ë ¹?´ì„?€ {interpretation}?…ë‹ˆ??",
        "{topic}??ë²•ì  ?´ì„ ê¸°ì??€ {interpretation}?…ë‹ˆ??",
        "ì¤‘ì•™ë¶€ì²˜ì˜ ?´ì„???°ë¥´ë©?{interpretation}?…ë‹ˆ??",
        "{topic}??ë²•ë ¹ ?ìš© ê¸°ì??€ {interpretation}?…ë‹ˆ??"
    ],
    'general_legal': [
        "{keyword}???€??ë²•ì  ê·¼ê±°??{basis}?…ë‹ˆ??",
        "{keyword}??ë²•ì  ?”ê±´?€ {requirement}?…ë‹ˆ??",
        "{keyword}??ë²•ì  ?¨ê³¼??{effect}?…ë‹ˆ??",
        "{keyword}???€??ë²•ì  ?´ì„?€ {interpretation}?…ë‹ˆ??"
    ]
}


class QADatasetGenerator:
    """Q&A ?°ì´?°ì…‹ ?ì„± ?´ë˜??""
    
    def __init__(self):
        self.processor = LegalDataProcessor()
        self.qa_pairs = []
        self.logger = logging.getLogger(__name__)
        
    def generate_law_qa_pairs(self, law_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ë²•ë ¹ ?°ì´?°ì—??Q&A ???ì„±"""
        qa_pairs = []
        
        try:
            law_name = law_data.get('law_name', '')
            articles = law_data.get('articles', [])
            cleaned_content = law_data.get('cleaned_content', '')
            
            # 1. ë²•ë ¹ ?•ì˜ ê´€??Q&A
            if law_name and cleaned_content:
                # ë²•ë ¹ ?•ì˜ ì¶”ì¶œ (ì²?ë²ˆì§¸ ë¬¸ì¥ ?ëŠ” ì²?ë²ˆì§¸ ì¡°ë¬¸)
                definition = self._extract_law_definition(cleaned_content)
                if definition:
                    question = random.choice(QA_TEMPLATES['law_definition']).format(law_name=law_name)
                    answer = random.choice(ANSWER_TEMPLATES['law_definition']).format(
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
            
            # 2. ì¡°ë¬¸ë³?Q&A
            for article in articles:
                article_number = article.get('article_number', '')
                content = article.get('content', '')
                title = article.get('title', '')
                
                if article_number and content:
                    # ì¡°ë¬¸ ?´ìš© Q&A
                    question = random.choice(QA_TEMPLATES['law_article']).format(
                        law_name=law_name, article=article_number
                    )
                    answer = random.choice(ANSWER_TEMPLATES['law_article']).format(
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
                        question = f"{law_name} ??article_number}ì¡°ì˜ ?œëª©?€ ë¬´ì—‡?¸ê???"
                        answer = f"{law_name} ??article_number}ì¡°ì˜ ?œëª©?€ '{title}'?…ë‹ˆ??"
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'source': 'law_article_title',
                            'law_name': law_name,
                            'article_number': article_number,
                            'confidence': 0.95,
                            'difficulty': 'easy'
                        })
            
            # 3. ?¤ì›Œ??ê¸°ë°˜ Q&A
            entities = law_data.get('entities', {})
            keywords = entities.get('keywords', [])
            for keyword in keywords[:5]:  # ?ìœ„ 5ê°??¤ì›Œ?œë§Œ ?¬ìš©
                question = random.choice(QA_TEMPLATES['general_legal']).format(keyword=keyword)
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
            
        except Exception as e:
            self.logger.error(f"Error generating law QA pairs: {e}")
        
        return qa_pairs
    
    def generate_precedent_qa_pairs(self, precedent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """?ë? ?°ì´?°ì—??Q&A ???ì„±"""
        qa_pairs = []
        
        try:
            case_name = precedent_data.get('case_name', '')
            issue = precedent_data.get('issue', '')
            reasoning = precedent_data.get('reasoning', '')
            conclusion = precedent_data.get('conclusion', '')
            court = precedent_data.get('court', '')
            
            # 1. ?ì  ê´€??Q&A
            if case_name and issue:
                question = random.choice(QA_TEMPLATES['precedent_issue']).format(case_name=case_name)
                answer = random.choice(ANSWER_TEMPLATES['precedent_issue']).format(
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
            
            # 2. ?ê²° ?´ìš© Q&A
            if case_name and reasoning:
                question = random.choice(QA_TEMPLATES['precedent_decision']).format(case_name=case_name)
                answer = random.choice(ANSWER_TEMPLATES['precedent_decision']).format(
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
            
            # 3. ê²°ë¡  Q&A
            if case_name and conclusion:
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
        """?Œì¬ê²°ì •ë¡€ ?°ì´?°ì—??Q&A ???ì„±"""
        qa_pairs = []
        
        try:
            case_name = decision_data.get('case_name', '')
            issue = decision_data.get('issue', '')
            reasoning = decision_data.get('reasoning', '')
            conclusion = decision_data.get('conclusion', '')
            decision_type = decision_data.get('decision_type', '')
            
            # 1. ?Œë²•???ì  Q&A
            if case_name and issue:
                question = random.choice(QA_TEMPLATES['constitutional_issue']).format(case_name=case_name)
                answer = random.choice(ANSWER_TEMPLATES['constitutional_issue']).format(
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
            
            # 2. ?Œë²•?¬íŒ??ê²°ì • Q&A
            if case_name and reasoning:
                question = random.choice(QA_TEMPLATES['constitutional_decision']).format(case_name=case_name)
                answer = random.choice(ANSWER_TEMPLATES['constitutional_decision']).format(
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
        """ë²•ë ¹?´ì„ë¡€ ?°ì´?°ì—??Q&A ???ì„±"""
        qa_pairs = []
        
        try:
            case_name = interpretation_data.get('case_name', '')
            issue = interpretation_data.get('issue', '')
            reasoning = interpretation_data.get('reasoning', '')
            topic = interpretation_data.get('topic', '')
            ministry = interpretation_data.get('ministry', '')
            
            # 1. ?´ì„ ì£¼ì œ Q&A
            if topic and issue:
                question = random.choice(QA_TEMPLATES['interpretation_question']).format(topic=topic)
                answer = random.choice(ANSWER_TEMPLATES['interpretation_question']).format(
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
        # ì²?ë²ˆì§¸ ë¬¸ì¥?ì„œ ?•ì˜ ì¶”ì¶œ
        sentences = content.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 20:  # ì¶©ë¶„??ê¸¸ì´??ë¬¸ì¥ë§?
                return first_sentence
        return ""
    
    def _generate_keyword_answer(self, keyword: str, law_name: str, content: str) -> str:
        """?¤ì›Œ??ê¸°ë°˜ ?µë? ?ì„±"""
        # ?¤ì›Œ?œê? ?¬í•¨??ë¬¸ì¥ ì°¾ê¸°
        sentences = content.split('.')
        for sentence in sentences:
            if keyword in sentence and len(sentence) > 20:
                return f"{law_name}???°ë¥´ë©?{sentence.strip()}?…ë‹ˆ??"
        return ""
    
    def calculate_quality_score(self, qa_pair: Dict[str, Any]) -> float:
        """Q&A ?ì˜ ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°"""
        score = 0.0
        
        # ê¸°ë³¸ ?ìˆ˜
        score += 0.3
        
        # ì§ˆë¬¸ ê¸¸ì´ ?ìˆ˜
        question_length = len(qa_pair.get('question', ''))
        if 10 <= question_length <= 100:
            score += 0.2
        elif 100 < question_length <= 200:
            score += 0.1
        
        # ?µë? ê¸¸ì´ ?ìˆ˜
        answer_length = len(qa_pair.get('answer', ''))
        if 20 <= answer_length <= 500:
            score += 0.3
        elif 500 < answer_length <= 1000:
            score += 0.2
        
        # ? ë¢°???ìˆ˜
        confidence = qa_pair.get('confidence', 0.5)
        score += confidence * 0.2
        
        return min(score, 1.0)
    
    def generate_dataset(self, data_dir: str = "data/processed", output_dir: str = "data/qa_dataset") -> bool:
        """?„ì²´ Q&A ?°ì´?°ì…‹ ?ì„±"""
        try:
            data_path = Path(data_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Q&A ?°ì´?°ì…‹ ?ì„± ?œì‘...")
            
            # ê°??°ì´???€?…ë³„ ì²˜ë¦¬
            data_types = ['laws', 'precedents', 'constitutional_decisions', 'legal_interpretations']
            
            for data_type in data_types:
                self.logger.info(f"{data_type} ?°ì´??ì²˜ë¦¬ ì¤?..")
                
                data_files = list(data_path.glob(f"{data_type}/*.json"))
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
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
                        continue
            
            # ?ˆì§ˆ ?ìˆ˜ë³??•ë ¬
            self.qa_pairs.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # ?°ì´?°ì…‹ ?€??
            self._save_dataset(output_path)
            
            # ?µê³„ ?ì„±
            self._generate_statistics(output_path)
            
            self.logger.info(f"Q&A ?°ì´?°ì…‹ ?ì„± ?„ë£Œ: {len(self.qa_pairs)}ê°???)
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating dataset: {e}")
            return False
    
    def _save_dataset(self, output_path: Path):
        """?°ì´?°ì…‹ ?€??""
        # ?„ì²´ ?°ì´?°ì…‹ ?€??
        with open(output_path / "qa_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=2)
        
        # ?ˆì§ˆë³?ë¶„í•  ?€??
        high_quality = [qa for qa in self.qa_pairs if qa.get('quality_score', 0) >= 0.8]
        medium_quality = [qa for qa in self.qa_pairs if 0.6 <= qa.get('quality_score', 0) < 0.8]
        low_quality = [qa for qa in self.qa_pairs if qa.get('quality_score', 0) < 0.6]
        
        with open(output_path / "qa_dataset_high_quality.json", 'w', encoding='utf-8') as f:
            json.dump(high_quality, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "qa_dataset_medium_quality.json", 'w', encoding='utf-8') as f:
            json.dump(medium_quality, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "qa_dataset_low_quality.json", 'w', encoding='utf-8') as f:
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
        
        with open(output_path / "qa_dataset_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Q&A ?°ì´?°ì…‹ ?ì„±
    generator = QADatasetGenerator()
    success = generator.generate_dataset()
    
    if success:
        logger.info("Q&A ?°ì´?°ì…‹ ?ì„±???„ë£Œ?˜ì—ˆ?µë‹ˆ??")
    else:
        logger.error("Q&A ?°ì´?°ì…‹ ?ì„±???¤íŒ¨?ˆìŠµ?ˆë‹¤.")


if __name__ == "__main__":
    main()
