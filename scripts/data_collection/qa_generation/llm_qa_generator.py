#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM ê¸°ë°˜ Q&A ?°ì´?°ì…‹ ?ì„±ê¸?

Ollama Qwen2.5:7b ëª¨ë¸???¬ìš©?˜ì—¬ ?¤ì–‘?˜ê³  ?ì—°?¤ëŸ¬??ë²•ë¥  Q&Aë¥??ì„±
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

from source.utils.ollama_client import OllamaClient
from source.utils.qa_quality_validator import QAQualityValidator

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/llm_qa_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LLMQAGenerator:
    """LLM ê¸°ë°˜ Q&A ?ì„±ê¸?""
    
    def __init__(
        self, 
        model: str = "qwen2.5:7b",
        temperature: float = 0.7,
        max_tokens: int = 1500
    ):
        """
        LLM Q&A ?ì„±ê¸?ì´ˆê¸°??
        
        Args:
            model: ?¬ìš©??Ollama ëª¨ë¸ëª?
            temperature: ?ì„± ?¨ë„
            max_tokens: ìµœë? ? í° ??
        """
        self.ollama_client = OllamaClient(model=model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.qa_pairs = []
        self.logger = logging.getLogger(__name__)
        
        # ?ˆì§ˆ ê²€ì¦ê¸° ì´ˆê¸°??
        self.quality_validator = QAQualityValidator()
        
        # ì§ˆë¬¸ ? í˜• ?•ì˜
        self.question_types = [
            "ê°œë… ?¤ëª…", "?¤ì œ ?ìš©", "?”ê±´/?¨ê³¼", 
            "ë¹„êµ/ì°¨ì´", "?ˆì°¨", "?ˆì‹œ", "ì£¼ì˜?¬í•­",
            "ë²•ì  ê·¼ê±°", "?¤ë¬´ ?ìš©", "?ˆì™¸ ?¬í•­"
        ]
        
        # ?ˆì§ˆ ?„í„°ë§?ê¸°ì?
        self.quality_criteria = {
            "min_question_length": 10,
            "max_question_length": 200,
            "min_answer_length": 20,
            "max_answer_length": 1000,
            "min_quality_score": 0.6
        }
    
    def generate_law_qa_pairs(self, law_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ë²•ë ¹ ?°ì´?°ì—??LLM ê¸°ë°˜ Q&A ???ì„±"""
        qa_pairs = []
        
        try:
            law_name = law_data.get('law_name', '')
            articles = law_data.get('articles', [])
            cleaned_content = law_data.get('cleaned_content', '')
            
            if not law_name:
                return qa_pairs
            
            self.logger.info(f"ë²•ë ¹ '{law_name}' ì²˜ë¦¬ ì¤?.. (ì¡°ë¬¸ {len(articles)}ê°?")
            
            # ë²•ë ¹ ?„ì²´ ?•ì˜ Q&A ?ì„±
            if cleaned_content:
                definition_qa = self._generate_law_definition_qa(law_name, cleaned_content)
                qa_pairs.extend(definition_qa)
            
            # ê°?ì¡°ë¬¸ë³?Q&A ?ì„±
            for article in articles[:10]:  # ì²˜ìŒ 10ê°?ì¡°ë¬¸ë§?ì²˜ë¦¬
                article_qa = self._generate_article_qa(law_name, article)
                qa_pairs.extend(article_qa)
            
            # ?ˆì§ˆ ê²€ì¦?ë°??„í„°ë§?
            filtered_qa = self._filter_qa_pairs(qa_pairs)
            
            self.logger.info(f"ë²•ë ¹ '{law_name}'?ì„œ {len(filtered_qa)}ê°?Q&A ?ì„±")
            return filtered_qa
            
        except Exception as e:
            self.logger.error(f"ë²•ë ¹ Q&A ?ì„± ì¤??¤ë¥˜: {e}")
            return []
    
    def generate_precedent_qa_pairs(self, precedent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """?ë? ?°ì´?°ì—??LLM ê¸°ë°˜ Q&A ???ì„±"""
        qa_pairs = []
        
        try:
            case_name = precedent_data.get('case_name', '')
            issue = precedent_data.get('issue', '')
            reasoning = precedent_data.get('reasoning', '')
            conclusion = precedent_data.get('conclusion', '')
            court = precedent_data.get('court', '')
            
            if not case_name:
                return qa_pairs
            
            self.logger.info(f"?ë? '{case_name}' ì²˜ë¦¬ ì¤?..")
            
            # ?ë? ì»¨í…?¤íŠ¸ êµ¬ì„±
            context = f"""
            ?¬ê±´ëª? {case_name}
            ë²•ì›: {court}
            ?ì : {issue}
            ?ê²° ?”ì?: {reasoning[:500] if reasoning else ''}
            ê²°ë¡ : {conclusion[:300] if conclusion else ''}
            """
            
            # ?¤ì–‘??ê´€?ì—??Q&A ?ì„±
            precedent_qa = self._generate_precedent_qa(context, case_name)
            qa_pairs.extend(precedent_qa)
            
            # ?ˆì§ˆ ê²€ì¦?ë°??„í„°ë§?
            filtered_qa = self._filter_qa_pairs(qa_pairs)
            
            self.logger.info(f"?ë? '{case_name}'?ì„œ {len(filtered_qa)}ê°?Q&A ?ì„±")
            return filtered_qa
            
        except Exception as e:
            self.logger.error(f"?ë? Q&A ?ì„± ì¤??¤ë¥˜: {e}")
            return []
    
    def generate_constitutional_qa_pairs(self, decision_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """?Œì¬ê²°ì •ë¡€ ?°ì´?°ì—??LLM ê¸°ë°˜ Q&A ???ì„±"""
        qa_pairs = []
        
        try:
            case_name = decision_data.get('case_name', '')
            issue = decision_data.get('issue', '')
            reasoning = decision_data.get('reasoning', '')
            decision_type = decision_data.get('decision_type', '')
            
            if not case_name:
                return qa_pairs
            
            self.logger.info(f"?Œì¬ê²°ì •ë¡€ '{case_name}' ì²˜ë¦¬ ì¤?..")
            
            # ?Œì¬ê²°ì •ë¡€ ì»¨í…?¤íŠ¸ êµ¬ì„±
            context = f"""
            ?¬ê±´ëª? {case_name}
            ê²°ì • ? í˜•: {decision_type}
            ?Œë²•???ì : {issue}
            ?Œë²•?¬íŒ???ë‹¨: {reasoning[:500] if reasoning else ''}
            """
            
            # ?Œë²•??ê´€?ì—??Q&A ?ì„±
            constitutional_qa = self._generate_constitutional_qa(context, case_name)
            qa_pairs.extend(constitutional_qa)
            
            # ?ˆì§ˆ ê²€ì¦?ë°??„í„°ë§?
            filtered_qa = self._filter_qa_pairs(qa_pairs)
            
            self.logger.info(f"?Œì¬ê²°ì •ë¡€ '{case_name}'?ì„œ {len(filtered_qa)}ê°?Q&A ?ì„±")
            return filtered_qa
            
        except Exception as e:
            self.logger.error(f"?Œì¬ê²°ì •ë¡€ Q&A ?ì„± ì¤??¤ë¥˜: {e}")
            return []
    
    def _generate_law_definition_qa(self, law_name: str, content: str) -> List[Dict[str, Any]]:
        """ë²•ë ¹ ?•ì˜ ê¸°ë°˜ Q&A ?ì„±"""
        context = f"""
        ë²•ë ¹ëª? {law_name}
        ë²•ë ¹ ?´ìš©: {content[:800]}
        """
        
        qa_pairs = self.ollama_client.generate_qa_pairs(
            context=context,
            qa_count=3,
            question_types=["ê°œë… ?¤ëª…", "ëª©ì ", "?ìš© ë²”ìœ„"],
            temperature=self.temperature
        )
        
        # ë©”í??°ì´??ì¶”ê?
        for qa in qa_pairs:
            qa.update({
                'source': 'law_definition_llm',
                'law_name': law_name,
                'confidence': 0.9,
                'difficulty': 'easy',
                'generated_at': datetime.now().isoformat()
            })
        
        return qa_pairs
    
    def _generate_article_qa(self, law_name: str, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ì¡°ë¬¸ ê¸°ë°˜ Q&A ?ì„±"""
        article_number = article.get('article_number', '')
        content = article.get('content', '')
        title = article.get('title', '')
        
        if not content and not title:
            return []
        
        context = f"""
        ë²•ë ¹ëª? {law_name}
        ì¡°ë¬¸ ë²ˆí˜¸: ??article_number}ì¡?
        ì¡°ë¬¸ ?œëª©: {title}
        ì¡°ë¬¸ ?´ìš©: {content[:600]}
        """
        
        qa_pairs = self.ollama_client.generate_qa_pairs(
            context=context,
            qa_count=2,
            question_types=["?¤ì œ ?ìš©", "?”ê±´/?¨ê³¼", "?ˆì°¨", "ì£¼ì˜?¬í•­"],
            temperature=self.temperature
        )
        
        # ë©”í??°ì´??ì¶”ê?
        for qa in qa_pairs:
            qa.update({
                'source': 'law_article_llm',
                'law_name': law_name,
                'article_number': article_number,
                'confidence': 0.8,
                'difficulty': 'medium',
                'generated_at': datetime.now().isoformat()
            })
        
        return qa_pairs
    
    def _generate_precedent_qa(self, context: str, case_name: str) -> List[Dict[str, Any]]:
        """?ë? ê¸°ë°˜ Q&A ?ì„±"""
        qa_pairs = self.ollama_client.generate_qa_pairs(
            context=context,
            qa_count=3,
            question_types=["?¤ë¬´ ?ìš©", "?œì‚¬??, "?ˆë°© ì¡°ì¹˜", "? ì‚¬ ?¬ë?"],
            temperature=self.temperature
        )
        
        # ë©”í??°ì´??ì¶”ê?
        for qa in qa_pairs:
            qa.update({
                'source': 'precedent_llm',
                'case_name': case_name,
                'confidence': 0.8,
                'difficulty': 'hard',
                'generated_at': datetime.now().isoformat()
            })
        
        return qa_pairs
    
    def _generate_constitutional_qa(self, context: str, case_name: str) -> List[Dict[str, Any]]:
        """?Œì¬ê²°ì •ë¡€ ê¸°ë°˜ Q&A ?ì„±"""
        qa_pairs = self.ollama_client.generate_qa_pairs(
            context=context,
            qa_count=2,
            question_types=["?Œë²•???˜ë?", "ê¸°ë³¸ê¶?, "ë²•ì  ?¨ê³¼"],
            temperature=self.temperature
        )
        
        # ë©”í??°ì´??ì¶”ê?
        for qa in qa_pairs:
            qa.update({
                'source': 'constitutional_llm',
                'case_name': case_name,
                'confidence': 0.8,
                'difficulty': 'hard',
                'generated_at': datetime.now().isoformat()
            })
        
        return qa_pairs
    
    def _filter_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Q&A ???ˆì§ˆ ?„í„°ë§?(ê°œì„ ???ˆì§ˆ ê²€ì¦ê¸° ?¬ìš©)"""
        filtered_pairs = []
        
        for qa in qa_pairs:
            # ?ˆì§ˆ ê²€ì¦ê¸°ë¡?ê²€ì¦?
            validation_result = self.quality_validator.validate_qa_pair(qa)
            
            if validation_result['is_valid']:
                # ê²€ì¦ëœ ?ˆì§ˆ ?ìˆ˜?€ ? ë¢°???…ë°?´íŠ¸
                qa['quality_score'] = validation_result['quality_score']
                qa['confidence'] = validation_result['confidence']
                qa['validation_issues'] = validation_result['issues']
                qa['validation_suggestions'] = validation_result['suggestions']
                
                filtered_pairs.append(qa)
            else:
                self.logger.debug(f"Q&A ?„í„°ë§ë¨: {validation_result['issues']}")
        
        return filtered_pairs
    
    def _calculate_quality_score(self, qa_pair: Dict[str, Any]) -> float:
        """Q&A ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°"""
        score = 0.0
        
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        # ê¸°ë³¸ ?ìˆ˜
        score += 0.2
        
        # ì§ˆë¬¸ ?ˆì§ˆ ?ìˆ˜
        if 15 <= len(question) <= 100:
            score += 0.25
        elif 100 < len(question) <= 150:
            score += 0.15
        
        # ?µë? ?ˆì§ˆ ?ìˆ˜
        if 30 <= len(answer) <= 400:
            score += 0.3
        elif 400 < len(answer) <= 600:
            score += 0.2
        
        # ì§ˆë¬¸ ? í˜• ?ìˆ˜
        question_type = qa_pair.get('type', '')
        if question_type in ['?¤ì œ ?ìš©', '?¤ë¬´ ?ìš©', '?ˆì‹œ']:
            score += 0.15
        elif question_type in ['ê°œë… ?¤ëª…', '?”ê±´/?¨ê³¼']:
            score += 0.1
        
        # ? ë¢°???ìˆ˜
        confidence = qa_pair.get('confidence', 0.5)
        score += confidence * 0.1
        
        return min(score, 1.0)
    
    def _remove_duplicates(self):
        """ì¤‘ë³µ Q&A ?œê±°"""
        self.logger.info("ì¤‘ë³µ Q&A ?œê±° ì¤?..")
        
        # ì¤‘ë³µ ê°ì?
        duplicate_groups = self.quality_validator.detect_duplicates(self.qa_pairs)
        
        removed_count = 0
        for group in duplicate_groups:
            if len(group) > 1:
                # ê·¸ë£¹ ?´ì—??ê°€???’ì? ?ˆì§ˆ ?ìˆ˜ë¥?ê°€ì§?Q&Aë§?? ì?
                best_qa = max(group, key=lambda i: self.qa_pairs[i].get('quality_score', 0))
                
                # ?˜ë¨¸ì§€ ?œê±°
                for i in sorted(group, reverse=True):
                    if i != best_qa:
                        del self.qa_pairs[i]
                        removed_count += 1
        
        self.logger.info(f"ì¤‘ë³µ ?œê±° ?„ë£Œ: {removed_count}ê°??œê±°, ?¨ì? Q&A: {len(self.qa_pairs)}ê°?)
    
    def generate_dataset(
        self, 
        data_dir: str = "data/processed", 
        output_dir: str = "data/qa_dataset/llm_generated",
        data_types: List[str] = None,
        max_items_per_type: int = 50
    ) -> bool:
        """?„ì²´ LLM ê¸°ë°˜ Q&A ?°ì´?°ì…‹ ?ì„±"""
        try:
            data_path = Path(data_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if data_types is None:
                data_types = ['laws', 'precedents', 'constitutional_decisions']
            
            self.logger.info("LLM ê¸°ë°˜ Q&A ?°ì´?°ì…‹ ?ì„± ?œì‘...")
            
            total_generated = 0
            
            for data_type in data_types:
                self.logger.info(f"{data_type} ?°ì´??ì²˜ë¦¬ ì¤?..")
                
                data_files = list(data_path.glob(f"{data_type}/*.json"))
                processed_count = 0
                type_generated = 0
                
                for file_path in data_files:
                    if processed_count >= max_items_per_type:
                        break
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # ?°ì´?°ê? ë°°ì—´??ê²½ìš° ê°???ª©ë³„ë¡œ ì²˜ë¦¬
                        if isinstance(data, list):
                            for item in data:
                                if processed_count >= max_items_per_type:
                                    break
                                
                                if not isinstance(item, dict):
                                    continue
                                
                                # ?°ì´???€?…ì— ?°ë¥¸ Q&A ?ì„±
                                if data_type == 'laws':
                                    qa_pairs = self.generate_law_qa_pairs(item)
                                elif data_type == 'precedents':
                                    qa_pairs = self.generate_precedent_qa_pairs(item)
                                elif data_type == 'constitutional_decisions':
                                    qa_pairs = self.generate_constitutional_qa_pairs(item)
                                else:
                                    continue
                                
                                self.qa_pairs.extend(qa_pairs)
                                type_generated += len(qa_pairs)
                                processed_count += 1
                                
                                # ì§„í–‰ ?í™© ë¡œê¹…
                                if processed_count % 10 == 0:
                                    self.logger.info(f"{data_type}: {processed_count}ê°???ª© ì²˜ë¦¬ ?„ë£Œ, ?„ì¬ Q&A: {len(self.qa_pairs)}ê°?)
                        else:
                            # ?¨ì¼ ê°ì²´??ê²½ìš°
                            if data_type == 'laws':
                                qa_pairs = self.generate_law_qa_pairs(data)
                            elif data_type == 'precedents':
                                qa_pairs = self.generate_precedent_qa_pairs(data)
                            elif data_type == 'constitutional_decisions':
                                qa_pairs = self.generate_constitutional_qa_pairs(data)
                            else:
                                continue
                            
                            self.qa_pairs.extend(qa_pairs)
                            type_generated += len(qa_pairs)
                            processed_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"?Œì¼ ì²˜ë¦¬ ì¤??¤ë¥˜ {file_path}: {e}")
                        continue
                
                self.logger.info(f"{data_type} ì²˜ë¦¬ ?„ë£Œ: {processed_count}ê°???ª©, {type_generated}ê°?Q&A ?ì„±")
                total_generated += type_generated
            
            # ì¤‘ë³µ ?œê±°
            self._remove_duplicates()
            
            # ?ˆì§ˆ ?ìˆ˜ë³??•ë ¬
            self.qa_pairs.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # ?°ì´?°ì…‹ ?€??
            self._save_dataset(output_path)
            
            # ?µê³„ ?ì„±
            self._generate_statistics(output_path)
            
            self.logger.info(f"LLM ê¸°ë°˜ Q&A ?°ì´?°ì…‹ ?ì„± ?„ë£Œ: {len(self.qa_pairs)}ê°???)
            return True
            
        except Exception as e:
            self.logger.error(f"?°ì´?°ì…‹ ?ì„± ì¤??¤ë¥˜: {e}")
            return False
    
    def _save_dataset(self, output_path: Path):
        """?°ì´?°ì…‹ ?€??""
        # ?„ì²´ ?°ì´?°ì…‹ ?€??
        with open(output_path / "llm_qa_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=2)
        
        # ?ˆì§ˆë³?ë¶„í•  ?€??
        high_quality = [qa for qa in self.qa_pairs if qa.get('quality_score', 0) >= 0.8]
        medium_quality = [qa for qa in self.qa_pairs if 0.6 <= qa.get('quality_score', 0) < 0.8]
        low_quality = [qa for qa in self.qa_pairs if qa.get('quality_score', 0) < 0.6]
        
        with open(output_path / "llm_qa_dataset_high_quality.json", 'w', encoding='utf-8') as f:
            json.dump(high_quality, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "llm_qa_dataset_medium_quality.json", 'w', encoding='utf-8') as f:
            json.dump(medium_quality, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "llm_qa_dataset_low_quality.json", 'w', encoding='utf-8') as f:
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
            'question_type_distribution': {},
            'generated_at': datetime.now().isoformat(),
            'model_used': self.ollama_client.model,
            'temperature': self.temperature
        }
        
        # ?ŒìŠ¤ë³?ë¶„í¬
        for qa in self.qa_pairs:
            source = qa.get('source', 'unknown')
            stats['source_distribution'][source] = stats['source_distribution'].get(source, 0) + 1
        
        # ?œì´?„ë³„ ë¶„í¬
        for qa in self.qa_pairs:
            difficulty = qa.get('difficulty', 'unknown')
            stats['difficulty_distribution'][difficulty] = stats['difficulty_distribution'].get(difficulty, 0) + 1
        
        # ì§ˆë¬¸ ? í˜•ë³?ë¶„í¬
        for qa in self.qa_pairs:
            q_type = qa.get('type', 'unknown')
            stats['question_type_distribution'][q_type] = stats['question_type_distribution'].get(q_type, 0) + 1
        
        with open(output_path / "llm_qa_dataset_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    """?ŒìŠ¤???¨ìˆ˜"""
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    try:
        # LLM Q&A ?ì„±ê¸??ì„±
        generator = LLMQAGenerator()
        
        # ?Œê·œëª??ŒìŠ¤???°ì´?°ì…‹ ?ì„±
        success = generator.generate_dataset(
            data_dir="data/processed",
            output_dir="data/qa_dataset/llm_test",
            data_types=['laws'],
            max_items_per_type=5
        )
        
        if success:
            print("??LLM ê¸°ë°˜ Q&A ?°ì´?°ì…‹ ?ì„± ?±ê³µ")
        else:
            print("??LLM ê¸°ë°˜ Q&A ?°ì´?°ì…‹ ?ì„± ?¤íŒ¨")
            
    except Exception as e:
        print(f"???¤ë¥˜ ë°œìƒ: {e}")


if __name__ == "__main__":
    main()
