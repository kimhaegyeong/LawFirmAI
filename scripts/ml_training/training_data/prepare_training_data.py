#!/usr/bin/env python3
"""
?ˆë ¨ ?°ì´??ì¤€ë¹??¤í¬ë¦½íŠ¸
?ë³¸ ë²•ë¥  ?°ì´?°ì? ì²˜ë¦¬???°ì´?°ë? ë§¤ì¹­?˜ì—¬ ?ˆë ¨ ?°ì´???ì„±
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# ë¡œê¹… ?¤ì •
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataPreparer:
    """?ˆë ¨ ?°ì´??ì¤€ë¹??´ë˜??""
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        """
        ì´ˆê¸°??
        
        Args:
            raw_data_dir: ?ë³¸ ?°ì´???”ë ‰? ë¦¬
            processed_data_dir: ì²˜ë¦¬???°ì´???”ë ‰? ë¦¬
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
    def prepare_training_data(self) -> List[Dict[str, Any]]:
        """?ˆë ¨ ?°ì´??ì¤€ë¹?""
        training_samples = []
        
        # ì²˜ë¦¬???°ì´???Œì¼??ì°¾ê¸°
        processed_files = list(self.processed_data_dir.glob("**/*.json"))
        logger.info(f"Found {len(processed_files)} processed files")
        
        processed_count = 0
        success_count = 0
        
        for i, processed_file in enumerate(processed_files):
            processed_count += 1
            
            # ì§„í–‰ ?í™© ?œì‹œ
            if i % 100 == 0:
                print(f"Processing file {i+1}/{len(processed_files)}: {processed_file.name}")
            
            try:
                # ì²˜ë¦¬???°ì´??ë¡œë“œ
                with open(processed_file, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                
                if 'articles' not in processed_data:
                    continue
                
                # ?ë³¸ ?°ì´??ì°¾ê¸°
                raw_content = self._find_raw_content(processed_data.get('law_id', ''))
                if not raw_content:
                    continue
                
                # ê°?ì¡°ë¬¸???€???ˆë ¨ ?˜í”Œ ?ì„±
                for article in processed_data['articles']:
                    sample = self._create_training_sample(article, raw_content)
                    if sample:
                        training_samples.append(sample)
                        success_count += 1
                        
            except Exception as e:
                logger.warning(f"Error processing {processed_file}: {e}")
                continue
        
        print(f"\nProcessed {processed_count} files, generated {success_count} samples")
        logger.info(f"Prepared {len(training_samples)} training samples")
        return training_samples
    
    def _find_raw_content(self, law_id: str) -> Optional[str]:
        """?ë³¸ ë²•ë¥  ?´ìš© ì°¾ê¸°"""
        if not law_id:
            return None
            
        # ?ë³¸ ?°ì´???Œì¼?¤ì—???´ë‹¹ ë²•ë¥  ì°¾ê¸°
        raw_files = list(self.raw_data_dir.glob("**/*.json"))
        
        # ìºì‹œë¥??¬ìš©?˜ì—¬ ?±ëŠ¥ ê°œì„ 
        if not hasattr(self, '_raw_content_cache'):
            self._raw_content_cache = {}
        
        if law_id in self._raw_content_cache:
            return self._raw_content_cache[law_id]
        
        for raw_file in raw_files:
            try:
                with open(raw_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                if isinstance(raw_data, dict) and 'laws' in raw_data:
                    for law in raw_data['laws']:
                        if law.get('law_id') == law_id:
                            content = law.get('law_content', '')
                            self._raw_content_cache[law_id] = content
                            return content
                elif isinstance(raw_data, list):
                    for law in raw_data:
                        if law.get('law_id') == law_id:
                            content = law.get('law_content', '')
                            self._raw_content_cache[law_id] = content
                            return content
                            
            except Exception as e:
                logger.warning(f"Error reading {raw_file}: {e}")
                continue
        
        # ì°¾ì? ëª»í•œ ê²½ìš° ìºì‹œ??None ?€??
        self._raw_content_cache[law_id] = None
        return None
    
    def _create_training_sample(self, article: Dict[str, Any], raw_content: str) -> Optional[Dict[str, Any]]:
        """?ˆë ¨ ?˜í”Œ ?ì„±"""
        article_number = article.get('article_number', '')
        article_title = article.get('article_title', '')
        
        if not article_number:
            return None
        
        # ì¡°ë¬¸ ?„ì¹˜ ì°¾ê¸°
        position = self._find_article_position(raw_content, article_number)
        if position == -1:
            return None
        
        # ?¹ì„± ì¶”ì¶œ
        features = self._extract_features(raw_content, position, article_number)
        
        # ?ˆì´ë¸?ê²°ì •
        label = 'real_article' if article_title else 'reference'
        
        return {
            'features': features,
            'label': label,
            'article_number': article_number,
            'article_title': article_title,
            'position': position,
            'raw_content': raw_content[:1000]  # ?”ë²„ê¹…ìš©?¼ë¡œ ?¼ë?ë§??€??
        }
    
    def _find_article_position(self, content: str, article_number: str) -> int:
        """ì¡°ë¬¸ ?„ì¹˜ ì°¾ê¸°"""
        # ?•í™•??ì¡°ë¬¸ ë²ˆí˜¸ë¡?ê²€??
        pattern = re.escape(article_number)
        match = re.search(pattern, content)
        return match.start() if match else -1
    
    def _extract_features(self, content: str, position: int, article_number: str) -> Dict[str, Any]:
        """?¹ì„± ì¶”ì¶œ"""
        features = {}
        
        # 1. ?„ì¹˜ ê¸°ë°˜ ?¹ì„±
        features['position_ratio'] = position / len(content) if len(content) > 0 else 0
        features['is_at_start'] = 1 if position < 200 else 0
        features['is_at_end'] = 1 if position > len(content) * 0.8 else 0
        
        # 2. ë¬¸ë§¥ ê¸°ë°˜ ?¹ì„±
        context_before = content[max(0, position - 200):position]
        context_after = content[position:min(len(content), position + 200)]
        
        # ë¬¸ì¥ ???¨í„´
        features['has_sentence_end'] = 1 if re.search(r'[.!?]\s*$', context_before) else 0
        
        # ì¡°ë¬¸ ì°¸ì¡° ?¨í„´
        reference_patterns = [
            r'??d+ì¡°ì—\s*?°ë¼',
            r'??d+ì¡°ì œ\d+??,
            r'??d+ì¡°ì˜\d+',
            r'??d+ì¡?*???s*?˜í•˜??,
            r'??d+ì¡?*???s*?°ë¼',
        ]
        
        features['has_reference_pattern'] = 0
        for pattern in reference_patterns:
            if re.search(pattern, context_before):
                features['has_reference_pattern'] = 1
                break
        
        # 3. ì¡°ë¬¸ ë²ˆí˜¸ ?¹ì„±
        article_num = int(re.search(r'\d+', article_number).group()) if re.search(r'\d+', article_number) else 0
        features['article_number'] = article_num
        features['is_supplementary'] = 1 if 'ë¶€ì¹? in article_number else 0
        
        # 4. ?ìŠ¤??ê¸¸ì´ ?¹ì„±
        features['context_before_length'] = len(context_before)
        features['context_after_length'] = len(context_after)
        
        # 5. ì¡°ë¬¸ ?œëª© ? ë¬´
        title_match = re.search(r'??d+ì¡?s*\(([^)]+)\)', context_after)
        features['has_title'] = 1 if title_match else 0
        
        # 6. ?¹ìˆ˜ ë¬¸ì ?¨í„´
        features['has_parentheses'] = 1 if '(' in context_after[:50] else 0
        features['has_quotes'] = 1 if '"' in context_after[:50] or "'" in context_after[:50] else 0
        
        # 7. ë²•ë¥  ?©ì–´ ?¨í„´
        legal_terms = [
            'ë²•ë¥ ', 'ë²•ë ¹', 'ê·œì •', 'ì¡°í•­', '??, '??, 'ëª?,
            '?œí–‰', 'ê³µí¬', 'ê°œì •', '?ì?', '?œì •'
        ]
        
        features['legal_term_count'] = sum(1 for term in legal_terms if term in context_after[:100])
        
        # 8. ?«ì ?¨í„´
        features['number_count'] = len(re.findall(r'\d+', context_after[:100]))
        
        # 9. ì¡°ë¬¸ ?´ìš© ê¸¸ì´ (?¤ìŒ ì¡°ë¬¸ê¹Œì???ê±°ë¦¬)
        next_article_match = re.search(r'??d+ì¡?, content[position + 1:])
        if next_article_match:
            features['article_length'] = next_article_match.start()
        else:
            features['article_length'] = len(content) - position
        
        # 10. ë¬¸ë§¥ ë°€??(ì¡°ë¬¸ ì°¸ì¡° ë¹ˆë„)
        article_refs_in_context = len(re.findall(r'??d+ì¡?, context_before))
        features['reference_density'] = article_refs_in_context / max(len(context_before), 1) * 1000
        
        return features
    
    def save_training_data(self, training_samples: List[Dict[str, Any]], output_file: str):
        """?ˆë ¨ ?°ì´???€??""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_samples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training data saved to {output_file}")


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    # ?°ì´??ì¤€ë¹„ê¸° ?ì„±
    preparer = TrainingDataPreparer(
        raw_data_dir="data/raw/assembly/law/2025101201",
        processed_data_dir="data/processed/assembly/law"
    )
    
    # ?ˆë ¨ ?°ì´??ì¤€ë¹?
    training_samples = preparer.prepare_training_data()
    
    if len(training_samples) == 0:
        logger.error("No training samples generated")
        return
    
    # ?ˆë ¨ ?°ì´???€??
    output_file = "data/training/article_classification_training_data.json"
    preparer.save_training_data(training_samples, output_file)
    
    # ?µê³„ ì¶œë ¥
    real_articles = sum(1 for sample in training_samples if sample['label'] == 'real_article')
    references = sum(1 for sample in training_samples if sample['label'] == 'reference')
    
    print(f"\n=== Training Data Statistics ===")
    print(f"Total samples: {len(training_samples)}")
    print(f"Real articles: {real_articles}")
    print(f"References: {references}")
    print(f"Real article ratio: {real_articles/len(training_samples):.3f}")


if __name__ == "__main__":
    main()
