# -*- coding: utf-8 -*-
"""
ë²•ë¥  ?©ì–´ ?¬ì „ ?ˆì§ˆ ê²€ì¦?ë°?ë¶„ì„
"""

import json
import os
from typing import Dict, List, Any
from collections import Counter
import statistics

def analyze_legal_term_dictionary(file_path: str) -> Dict[str, Any]:
    """ë²•ë¥  ?©ì–´ ?¬ì „ ë¶„ì„"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dictionary = data.get('dictionary', {})
    metadata = data.get('metadata', {})
    
    # ê¸°ë³¸ ?µê³„
    total_terms = len(dictionary)
    domains = metadata.get('domains', [])
    
    # ?©ì–´ë³?ë¶„ì„
    term_stats = {
        'total_terms': total_terms,
        'domains': domains,
        'synonyms_count': [],
        'related_terms_count': [],
        'precedent_keywords_count': [],
        'confidence_scores': [],
        'domain_distribution': {},
        'quality_metrics': {}
    }
    
    # ?„ë©”?¸ë³„ ë¶„í¬ ê³„ì‚°
    domain_counts = Counter()
    
    for term, expansion in dictionary.items():
        # ê°?ì¹´í…Œê³ ë¦¬ë³??©ì–´ ??
        synonyms_count = len(expansion.get('synonyms', []))
        related_count = len(expansion.get('related_terms', []))
        keywords_count = len(expansion.get('precedent_keywords', []))
        confidence = expansion.get('confidence', 0.0)
        
        term_stats['synonyms_count'].append(synonyms_count)
        term_stats['related_terms_count'].append(related_count)
        term_stats['precedent_keywords_count'].append(keywords_count)
        term_stats['confidence_scores'].append(confidence)
        
        # ?„ë©”?¸ë³„ ë¶„ë¥˜ (?©ì–´ëª…ìœ¼ë¡?ì¶”ì •)
        domain = classify_term_domain(term)
        domain_counts[domain] += 1
    
    term_stats['domain_distribution'] = dict(domain_counts)
    
    # ?ˆì§ˆ ë©”íŠ¸ë¦?ê³„ì‚°
    quality_metrics = {
        'avg_synonyms_per_term': statistics.mean(term_stats['synonyms_count']),
        'avg_related_terms_per_term': statistics.mean(term_stats['related_terms_count']),
        'avg_keywords_per_term': statistics.mean(term_stats['precedent_keywords_count']),
        'avg_confidence': statistics.mean(term_stats['confidence_scores']),
        'min_confidence': min(term_stats['confidence_scores']),
        'max_confidence': max(term_stats['confidence_scores']),
        'terms_with_high_confidence': len([c for c in term_stats['confidence_scores'] if c >= 0.9]),
        'terms_with_medium_confidence': len([c for c in term_stats['confidence_scores'] if 0.7 <= c < 0.9]),
        'terms_with_low_confidence': len([c for c in term_stats['confidence_scores'] if c < 0.7])
    }
    
    term_stats['quality_metrics'] = quality_metrics
    
    return term_stats

def classify_term_domain(term: str) -> str:
    """?©ì–´ë¥??„ë©”?¸ë³„ë¡?ë¶„ë¥˜"""
    
    # ë¯¼ì‚¬ë²?ê´€???©ì–´
    civil_terms = ['?í•´ë°°ìƒ', 'ê³„ì•½', '?Œìœ ê¶?, '?„ë?ì°?, 'ë¶ˆë²•?‰ìœ„', 'ì±„ê¶Œ', 'ì±„ë¬´', '?´ë³´', 'ë³´ì¦', '?°ë?', 'ë¶ˆê?ë¶?, 'ë¶„í• ', '?ì†', '? ì–¸', '? ì¦', 'ë¶€??, '?¼ì¸', '?´í˜¼', 'ì¹œì']
    
    # ?•ì‚¬ë²?ê´€???©ì–´
    criminal_terms = ['?´ì¸', '?ˆë„', '?¬ê¸°', 'ê°•ë„', 'ê°•ê°„', '??–‰', '?í•´', '?‘ë°•', 'ê°ê¸ˆ', '?½ì·¨', '? ì¸', 'ê°•ì œì¶”í–‰', 'ëª…ì˜ˆ?¼ì†', 'ëª¨ë…', 'ì£¼ê±°ì¹¨ì…', 'ë°©í™”', 'ê³µê°ˆ', '?¡ë ¹', 'ë°°ì„']
    
    # ?ì‚¬ë²?ê´€???©ì–´
    commercial_terms = ['ì£¼ì‹?Œì‚¬', '? í•œ?Œì‚¬', '?í–‰??, '?´ìŒ', '?˜í‘œ', 'ë³´í—˜', '?´ìƒ', '??³µ', '?´ì†¡', '?„ì„', '?„ê¸‰', '?„ì¹˜', 'ì¡°í•©', '?©ì', '?©ëª…', '?í˜¸', '?í‘œ', '?¹í—ˆ', '?€?‘ê¶Œ']
    
    # ?‰ì •ë²?ê´€???©ì–´
    administrative_terms = ['?‰ì •ì²˜ë¶„', '?‰ì •ì§€??, '?ˆê?', '?¸ê?', '?¹ì¸', '? ê³ ', '? ì²­', 'ì²?›', '?´ì˜? ì²­', '?‰ì •?¬íŒ', '?‰ì •?Œì†¡', 'êµ??ë°°ìƒ', '?ì‹¤ë³´ìƒ', '?‰ì •ê·œì¹™', '?‰ì •ê³„íš', '?‰ì •ê³„ì•½', 'ê³µë²•ê´€ê³?, '?¬ë²•ê´€ê³?]
    
    # ?¸ë™ë²?ê´€???©ì–´
    labor_terms = ['ê·¼ë¡œê³„ì•½', '?„ê¸ˆ', 'ê·¼ë¡œ?œê°„', '?´ê³ ', 'ë¶€?¹í•´ê³?, '?´ì§ê¸?, '?¤ì—…ê¸‰ì—¬', '?°ì—…?¬í•´', '?°ì—…?ˆì „', '?¸ë™ì¡°í•©', '?¨ì²´êµì„­', '?¨ì²´?‘ì•½', '?ì˜?‰ìœ„', '?Œì—…', 'ì§ì¥?ì‡„', '?¸ë™?ì˜', 'ê·¼ë¡œê¸°ì?', 'ìµœì??„ê¸ˆ', '?°ì¥ê·¼ë¡œ', '?´ê²Œ?œê°„']
    
    if term in civil_terms:
        return 'ë¯¼ì‚¬ë²?
    elif term in criminal_terms:
        return '?•ì‚¬ë²?
    elif term in commercial_terms:
        return '?ì‚¬ë²?
    elif term in administrative_terms:
        return '?‰ì •ë²?
    elif term in labor_terms:
        return '?¸ë™ë²?
    else:
        return 'ê¸°í?'

def generate_quality_report(stats: Dict[str, Any]) -> str:
    """?ˆì§ˆ ë³´ê³ ???ì„±"""
    
    report = []
    report.append("=" * 60)
    report.append("ë²•ë¥  ?©ì–´ ?¬ì „ ?ˆì§ˆ ë¶„ì„ ë³´ê³ ??)
    report.append("=" * 60)
    
    # ê¸°ë³¸ ?•ë³´
    report.append(f"\n?“Š ê¸°ë³¸ ?µê³„:")
    report.append(f"  ??ì´??©ì–´ ?? {stats['total_terms']}ê°?)
    report.append(f"  ???„ë©”???? {len(stats['domains'])}ê°?)
    report.append(f"  ???„ë©”?? {', '.join(stats['domains'])}")
    
    # ?„ë©”?¸ë³„ ë¶„í¬
    report.append(f"\n?“ˆ ?„ë©”?¸ë³„ ë¶„í¬:")
    for domain, count in stats['domain_distribution'].items():
        percentage = (count / stats['total_terms']) * 100
        report.append(f"  ??{domain}: {count}ê°?({percentage:.1f}%)")
    
    # ?ˆì§ˆ ë©”íŠ¸ë¦?
    metrics = stats['quality_metrics']
    report.append(f"\n?¯ ?ˆì§ˆ ë©”íŠ¸ë¦?")
    report.append(f"  ???‰ê·  ?™ì˜???? {metrics['avg_synonyms_per_term']:.2f}ê°?)
    report.append(f"  ???‰ê·  ê´€???©ì–´ ?? {metrics['avg_related_terms_per_term']:.2f}ê°?)
    report.append(f"  ???‰ê·  ?ë? ?¤ì›Œ???? {metrics['avg_keywords_per_term']:.2f}ê°?)
    report.append(f"  ???‰ê·  ? ë¢°?? {metrics['avg_confidence']:.3f}")
    report.append(f"  ??ìµœì†Œ ? ë¢°?? {metrics['min_confidence']:.3f}")
    report.append(f"  ??ìµœë? ? ë¢°?? {metrics['max_confidence']:.3f}")
    
    # ? ë¢°??ë¶„í¬
    report.append(f"\n?“Š ? ë¢°??ë¶„í¬:")
    report.append(f"  ??ê³ ì‹ ë¢°ë„ (??.9): {metrics['terms_with_high_confidence']}ê°?)
    report.append(f"  ??ì¤‘ì‹ ë¢°ë„ (0.7-0.9): {metrics['terms_with_medium_confidence']}ê°?)
    report.append(f"  ???€? ë¢°??(<0.7): {metrics['terms_with_low_confidence']}ê°?)
    
    # ?ˆì§ˆ ?‰ê?
    report.append(f"\nâ­??ˆì§ˆ ?‰ê?:")
    
    # ?„ì²´?ì¸ ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°
    quality_score = 0
    
    # ? ë¢°???ìˆ˜ (40%)
    avg_confidence = metrics['avg_confidence']
    confidence_score = avg_confidence * 40
    quality_score += confidence_score
    
    # ?©ì–´ ?¤ì–‘???ìˆ˜ (30%)
    avg_total_terms = (metrics['avg_synonyms_per_term'] + 
                      metrics['avg_related_terms_per_term'] + 
                      metrics['avg_keywords_per_term']) / 3
    diversity_score = min(avg_total_terms / 5, 1.0) * 30
    quality_score += diversity_score
    
    # ?„ë©”??ê· í˜• ?ìˆ˜ (20%)
    domain_balance = 1.0 - (max(stats['domain_distribution'].values()) - min(stats['domain_distribution'].values())) / stats['total_terms']
    balance_score = domain_balance * 20
    quality_score += balance_score
    
    # ?„ì„±???ìˆ˜ (10%)
    completion_score = 10  # ëª¨ë“  ?©ì–´ê°€ ì²˜ë¦¬?˜ì—ˆ?¼ë?ë¡?
    quality_score += completion_score
    
    report.append(f"  ???„ì²´ ?ˆì§ˆ ?ìˆ˜: {quality_score:.1f}/100")
    report.append(f"    - ? ë¢°???ìˆ˜: {confidence_score:.1f}/40")
    report.append(f"    - ?¤ì–‘???ìˆ˜: {diversity_score:.1f}/30")
    report.append(f"    - ê· í˜• ?ìˆ˜: {balance_score:.1f}/20")
    report.append(f"    - ?„ì„±???ìˆ˜: {completion_score:.1f}/10")
    
    # ?±ê¸‰ ?‰ê?
    if quality_score >= 90:
        grade = "A+ (?°ìˆ˜)"
    elif quality_score >= 80:
        grade = "A (?‘í˜¸)"
    elif quality_score >= 70:
        grade = "B (ë³´í†µ)"
    elif quality_score >= 60:
        grade = "C (ê°œì„  ?„ìš”)"
    else:
        grade = "D (?¬ì‘???„ìš”)"
    
    report.append(f"  ???±ê¸‰: {grade}")
    
    report.append(f"\n??ê²°ë¡ :")
    report.append(f"  ë²•ë¥  ?©ì–´ ?¬ì „???±ê³µ?ìœ¼ë¡?êµ¬ì¶•?˜ì—ˆ?µë‹ˆ??")
    report.append(f"  ì´?{stats['total_terms']}ê°??©ì–´ê°€ {len(stats['domains'])}ê°??„ë©”?¸ì— ê±¸ì³ ?•ì¥?˜ì—ˆ?¼ë©°,")
    report.append(f"  ?‰ê·  ? ë¢°??{metrics['avg_confidence']:.3f}ë¡??’ì? ?ˆì§ˆ??ë³´ì…?ˆë‹¤.")
    
    return "\n".join(report)

def safe_print(text: str):
    """?ˆì „???œê? ì¶œë ¥ ?¨ìˆ˜"""
    try:
        # ?Œì¼ë¡?ì¶œë ¥?˜ì—¬ ?œê? ë¬¸ì œ ?´ê²°
        with open('quality_analysis_output.txt', 'a', encoding='utf-8') as f:
            f.write(text + '\n')
        
        # ì½˜ì†” ì¶œë ¥?€ ASCIIë¡?ë³€?˜í•˜??ê¹¨ì§ ë°©ì?
        try:
            ascii_text = text.encode('ascii', 'ignore').decode('ascii')
            if ascii_text.strip():
                print(ascii_text)
        except:
            print("[?œê? ì¶œë ¥ - quality_analysis_output.txt ?Œì¼ ì°¸ì¡°]")
    except Exception:
        # ê¸°í? ?¤ë¥˜ ???ë³¸ ì¶œë ¥
        print(text)

def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    
    # ë¶„ì„???Œì¼ ê²½ë¡œ
    file_path = "data/comprehensive_legal_term_dictionary.json"
    
    if not os.path.exists(file_path):
        safe_print(f"?Œì¼??ì°¾ì„ ???†ìŠµ?ˆë‹¤: {file_path}")
        return
    
    safe_print("ë²•ë¥  ?©ì–´ ?¬ì „ ?ˆì§ˆ ë¶„ì„ ?œì‘...")
    
    try:
        # ë¶„ì„ ?¤í–‰
        stats = analyze_legal_term_dictionary(file_path)
        
        # ë³´ê³ ???ì„±
        report = generate_quality_report(stats)
        
        # ë³´ê³ ??ì¶œë ¥
        safe_print(report)
        
        # ë³´ê³ ???Œì¼ë¡??€??
        with open("data/quality_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        safe_print(f"\n?ì„¸ ë³´ê³ ?œê? ?€?¥ë˜?ˆìŠµ?ˆë‹¤: data/quality_analysis_report.txt")
        
    except Exception as e:
        safe_print(f"ë¶„ì„ ì¤??¤ë¥˜ ë°œìƒ: {e}")

if __name__ == "__main__":
    main()
