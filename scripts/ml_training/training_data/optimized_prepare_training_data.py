#!/usr/bin/env python3
"""
ìµœì ?”ëœ ?ˆë ¨ ?°ì´??ì¤€ë¹??¤í¬ë¦½íŠ¸
?ë³¸ ë²•ë¥  ?°ì´?°ì? ì²˜ë¦¬???°ì´?°ë? ë§¤ì¹­?˜ì—¬ ?ˆë ¨ ?°ì´???ì„± (?±ëŠ¥ ìµœì ??ë²„ì „)
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import time

# ë¡œê¹… ?¤ì •
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedTrainingDataPreparer:
    """ìµœì ?”ëœ ?ˆë ¨ ?°ì´??ì¤€ë¹??´ë˜??""
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        """
        ì´ˆê¸°??
        
        Args:
            raw_data_dir: ?ë³¸ ?°ì´???”ë ‰? ë¦¬
            processed_data_dir: ì²˜ë¦¬???°ì´???”ë ‰? ë¦¬
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # ?ë³¸ ?°ì´???¸ë±??ìºì‹œ
        self.raw_content_index = {}
        self._build_raw_content_index()
    
    def _build_raw_content_index(self):
        """?ë³¸ ?°ì´???¸ë±??êµ¬ì¶• (??ë²ˆë§Œ ?¤í–‰)"""
        logger.info("Building raw content index...")
        start_time = time.time()
        
        raw_files = list(self.raw_data_dir.glob("**/*.json"))
        logger.info(f"Found {len(raw_files)} raw files")
        
        for i, raw_file in enumerate(raw_files):
            if i % 100 == 0:
                logger.info(f"Indexing raw file {i+1}/{len(raw_files)}: {raw_file.name}")
            
            try:
                with open(raw_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                if isinstance(raw_data, dict) and 'laws' in raw_data:
                    for law in raw_data['laws']:
                        law_id = law.get('cont_id')  # ?ë³¸ ?°ì´?°ì—?œëŠ” cont_id ?¬ìš©
                        law_name = law.get('law_name', '')  # ë²•ë¥ ëª…ë„ ?€??
                        if law_id and law_name:
                            # cont_id?€ ë²•ë¥ ëª?ëª¨ë‘ë¡??¸ë±??
                            self.raw_content_index[law_id] = law.get('law_content', '')
                            self.raw_content_index[law_name] = law.get('law_content', '')
                elif isinstance(raw_data, list):
                    for law in raw_data:
                        law_id = law.get('cont_id')  # ?ë³¸ ?°ì´?°ì—?œëŠ” cont_id ?¬ìš©
                        law_name = law.get('law_name', '')  # ë²•ë¥ ëª…ë„ ?€??
                        if law_id and law_name:
                            # cont_id?€ ë²•ë¥ ëª?ëª¨ë‘ë¡??¸ë±??
                            self.raw_content_index[law_id] = law.get('law_content', '')
                            self.raw_content_index[law_name] = law.get('law_content', '')
                            
            except Exception as e:
                logger.warning(f"Error reading {raw_file}: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        logger.info(f"Raw content index built in {elapsed_time:.2f} seconds")
        logger.info(f"Indexed {len(self.raw_content_index)} laws")
    
    def prepare_training_data(self) -> List[Dict[str, Any]]:
        """?ˆë ¨ ?°ì´??ì¤€ë¹?(ìµœì ?”ëœ ë²„ì „)"""
        training_samples = []
        
        # ì²˜ë¦¬???°ì´???Œì¼??ì°¾ê¸°
        processed_files = list(self.processed_data_dir.glob("**/*.json"))
        logger.info(f"Found {len(processed_files)} processed files")
        
        processed_count = 0
        success_count = 0
        start_time = time.time()
        
        for i, processed_file in enumerate(processed_files):
            processed_count += 1
            
            # ì§„í–‰ ?í™© ?œì‹œ (???ì£¼)
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(processed_files) - i) / rate if rate > 0 else 0
                print(f"Processing file {i+1}/{len(processed_files)}: {processed_file.name}")
                print(f"  Rate: {rate:.1f} files/sec, ETA: {eta/60:.1f} minutes")
            
            try:
                # ì²˜ë¦¬???°ì´??ë¡œë“œ
                with open(processed_file, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                
                if 'articles' not in processed_data:
                    continue
                
                # ?ë³¸ ?°ì´??ì°¾ê¸° (?¸ë±???¬ìš©)
                law_id = processed_data.get('law_id', '')
                law_name = processed_data.get('law_name', '')
                
                # ë²•ë¥ ëª…ìœ¼ë¡?ë¨¼ì? ì°¾ê¸°
                raw_content = self.raw_content_index.get(law_name)
                if not raw_content:
                    # law_idë¡?ì°¾ê¸° ?œë„
                    raw_content = self.raw_content_index.get(law_id)
                
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
        
        elapsed_time = time.time() - start_time
        print(f"\nProcessed {processed_count} files, generated {success_count} samples")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Rate: {processed_count/elapsed_time:.1f} files/sec")
        logger.info(f"Prepared {len(training_samples)} training samples")
        return training_samples
    
    def _create_training_sample(self, article: Dict[str, Any], raw_content: str) -> Optional[Dict[str, Any]]:
        """?ˆë ¨ ?˜í”Œ ?ì„± (ìµœì ?”ëœ ë²„ì „)"""
        article_number = article.get('article_number', '')
        article_title = article.get('article_title', '')
        
        if not article_number:
            return None
        
        # ì¡°ë¬¸ ?„ì¹˜ ì°¾ê¸° (?•ê·œ??ìµœì ??
        position = self._find_article_position_optimized(raw_content, article_number)
        if position == -1:
            return None
        
        # ?¹ì„± ì¶”ì¶œ (ìµœì ?”ëœ ë²„ì „)
        features = self._extract_features_optimized(raw_content, position, article_number)
        
        # ?ˆì´ë¸?ê²°ì •
        label = 'real_article' if article_title else 'reference'
        
        return {
            'features': features,
            'label': label,
            'article_number': article_number,
            'article_title': article_title,
            'position': position,
            'raw_content': raw_content[:500]  # ?”ë²„ê¹…ìš©?¼ë¡œ ???ê²Œ ?€??
        }
    
    def _find_article_position_optimized(self, content: str, article_number: str) -> int:
        """ì¡°ë¬¸ ?„ì¹˜ ì°¾ê¸° (ìµœì ?”ëœ ë²„ì „)"""
        # ?•í™•??ì¡°ë¬¸ ë²ˆí˜¸ë¡?ê²€??(?•ê·œ??ì»´íŒŒ??
        pattern = re.compile(re.escape(article_number))
        match = pattern.search(content)
        return match.start() if match else -1
    
    def _extract_features_optimized(self, content: str, position: int, article_number: str) -> Dict[str, Any]:
        """?¹ì„± ì¶”ì¶œ (ìµœì ?”ëœ ë²„ì „)"""
        features = {}
        
        # 1. ?„ì¹˜ ê¸°ë°˜ ?¹ì„±
        features['position_ratio'] = position / len(content) if len(content) > 0 else 0
        features['is_at_start'] = 1 if position < 200 else 0
        features['is_at_end'] = 1 if position > len(content) * 0.8 else 0
        
        # 2. ë¬¸ë§¥ ê¸°ë°˜ ?¹ì„± (???‘ì? ?ˆë„???¬ìš©)
        context_before = content[max(0, position - 100):position]
        context_after = content[position:min(len(content), position + 100)]
        
        # ë¬¸ì¥ ???¨í„´ (ì»´íŒŒ?¼ëœ ?•ê·œ???¬ìš©)
        sentence_end_pattern = re.compile(r'[.!?]\s*$')
        features['has_sentence_end'] = 1 if sentence_end_pattern.search(context_before) else 0
        
        # ì¡°ë¬¸ ì°¸ì¡° ?¨í„´ (ì»´íŒŒ?¼ëœ ?•ê·œ???¬ìš©)
        reference_patterns = [
            re.compile(r'??d+ì¡°ì—\s*?°ë¼'),
            re.compile(r'??d+ì¡°ì œ\d+??),
            re.compile(r'??d+ì¡°ì˜\d+'),
            re.compile(r'??d+ì¡?*???s*?˜í•˜??),
            re.compile(r'??d+ì¡?*???s*?°ë¼'),
        ]
        
        features['has_reference_pattern'] = 0
        for pattern in reference_patterns:
            if pattern.search(context_before):
                features['has_reference_pattern'] = 1
                break
        
        # 3. ì¡°ë¬¸ ë²ˆí˜¸ ?¹ì„±
        article_num_match = re.search(r'\d+', article_number)
        article_num = int(article_num_match.group()) if article_num_match else 0
        features['article_number'] = article_num
        features['is_supplementary'] = 1 if 'ë¶€ì¹? in article_number else 0
        
        # 4. ?ìŠ¤??ê¸¸ì´ ?¹ì„±
        features['context_before_length'] = len(context_before)
        features['context_after_length'] = len(context_after)
        
        # 5. ì¡°ë¬¸ ?œëª© ? ë¬´ (ì»´íŒŒ?¼ëœ ?•ê·œ???¬ìš©)
        title_pattern = re.compile(r'??d+ì¡?s*\(([^)]+)\)')
        title_match = title_pattern.search(context_after)
        features['has_title'] = 1 if title_match else 0
        
        # 6. ?¹ìˆ˜ ë¬¸ì ?¨í„´
        features['has_parentheses'] = 1 if '(' in context_after[:30] else 0
        features['has_quotes'] = 1 if '"' in context_after[:30] or "'" in context_after[:30] else 0
        
        # 7. ë²•ë¥  ?©ì–´ ?¨í„´ (???ì? ?©ì–´ë¡?
        legal_terms = ['ë²•ë¥ ', 'ë²•ë ¹', 'ê·œì •', 'ì¡°í•­', '??, '??, 'ëª?]
        features['legal_term_count'] = sum(1 for term in legal_terms if term in context_after[:50])
        
        # 8. ?«ì ?¨í„´ (ì»´íŒŒ?¼ëœ ?•ê·œ???¬ìš©)
        number_pattern = re.compile(r'\d+')
        features['number_count'] = len(number_pattern.findall(context_after[:50]))
        
        # 9. ì¡°ë¬¸ ?´ìš© ê¸¸ì´ (?¤ìŒ ì¡°ë¬¸ê¹Œì???ê±°ë¦¬)
        next_article_pattern = re.compile(r'??d+ì¡?)
        next_article_match = next_article_pattern.search(content[position + 1:])
        if next_article_match:
            features['article_length'] = next_article_match.start()
        else:
            features['article_length'] = len(content) - position
        
        # 10. ë¬¸ë§¥ ë°€??(ì¡°ë¬¸ ì°¸ì¡° ë¹ˆë„)
        article_ref_pattern = re.compile(r'??d+ì¡?)
        article_refs_in_context = len(article_ref_pattern.findall(context_before))
        features['reference_density'] = article_refs_in_context / max(len(context_before), 1) * 1000
        
        return features
    
    def save_training_data(self, training_samples: List[Dict[str, Any]], output_file: str):
        """?ˆë ¨ ?°ì´???€??""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_samples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training data saved to {output_file}")
    
    def save_training_data_batch(self, training_samples: List[Dict[str, Any]], output_file: str, batch_size: int = 1000):
        """ë°°ì¹˜ë¡??ˆë ¨ ?°ì´???€??(ë©”ëª¨ë¦??¨ìœ¨??"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ë°°ì¹˜ë¡??€??
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('[\n')
            
            for i, sample in enumerate(training_samples):
                if i > 0:
                    f.write(',\n')
                
                json.dump(sample, f, ensure_ascii=False, indent=2)
                
                # ë©”ëª¨ë¦??•ë¦¬
                if i % batch_size == 0:
                    f.flush()
            
            f.write('\n]')
        
        logger.info(f"Training data saved to {output_file} in batches")


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    start_time = time.time()
    
    # ?°ì´??ì¤€ë¹„ê¸° ?ì„±
    preparer = OptimizedTrainingDataPreparer(
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
    preparer.save_training_data_batch(training_samples, output_file)
    
    # ?µê³„ ì¶œë ¥
    real_articles = sum(1 for sample in training_samples if sample['label'] == 'real_article')
    references = sum(1 for sample in training_samples if sample['label'] == 'reference')
    
    total_time = time.time() - start_time
    
    print(f"\n=== Training Data Statistics ===")
    print(f"Total samples: {len(training_samples)}")
    print(f"Real articles: {real_articles}")
    print(f"References: {references}")
    print(f"Real article ratio: {real_articles/len(training_samples):.3f}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time/len(training_samples):.4f} seconds")


if __name__ == "__main__":
    main()
