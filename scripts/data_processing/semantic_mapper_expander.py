#!/usr/bin/env python3
"""
SemanticKeywordMapper ?•ì¥ ?¤í¬ë¦½íŠ¸
ì¶”ì¶œ??ë²•ë¥  ?©ì–´?¤ì„ ê¸°ë°˜?¼ë¡œ ?˜ë???ê´€ê³„ë? ?•ì¥?©ë‹ˆ??
"""

import json
import os
from typing import Dict, List, Set
from collections import defaultdict, Counter

class SemanticKeywordMapperExpander:
    """SemanticKeywordMapper ?•ì¥ê¸?""
    
    def __init__(self):
        self.extracted_terms_file = "data/extracted_terms/extracted_legal_terms.json"
        self.semantic_relations_file = "data/extracted_terms/semantic_relations.json"
        self.output_file = "source/services/langgraph/enhanced_semantic_relations.py"
        
    def load_extracted_data(self) -> tuple:
        """ì¶”ì¶œ???°ì´??ë¡œë“œ"""
        print("ì¶”ì¶œ???©ì–´ ?°ì´??ë¡œë“œ ì¤?..")
        
        with open(self.extracted_terms_file, 'r', encoding='utf-8') as f:
            extracted_terms = json.load(f)
        
        with open(self.semantic_relations_file, 'r', encoding='utf-8') as f:
            semantic_relations = json.load(f)
        
        print(f"ë¡œë“œ???©ì–´ ?? {len(extracted_terms)}")
        print(f"ë¡œë“œ???˜ë???ê´€ê³??? {len(semantic_relations)}")
        
        return extracted_terms, semantic_relations
    
    def expand_semantic_relations(self, extracted_terms: Dict, semantic_relations: Dict) -> Dict:
        """?˜ë???ê´€ê³??•ì¥"""
        print("?˜ë???ê´€ê³??•ì¥ ì¤?..")
        
        # ê¸°ì¡´ ê´€ê³„ë? ê¸°ë°˜?¼ë¡œ ?•ì¥??ê´€ê³??ì„±
        expanded_relations = {}
        
        # ?„ë©”?¸ë³„ ?©ì–´ ê·¸ë£¹??
        domain_groups = defaultdict(list)
        for term, term_data in extracted_terms.items():
            domain = term_data.get('domain', 'ê¸°í?')
            domain_groups[domain].append((term, term_data))
        
        # ê°??„ë©”?¸ë³„ë¡??˜ë???ê´€ê³??ì„±
        for domain, terms in domain_groups.items():
            if len(terms) < 5:  # ìµœì†Œ 5ê°??´ìƒ???©ì–´ê°€ ?ˆëŠ” ?„ë©”?¸ë§Œ ì²˜ë¦¬
                continue
            
            # ë¹ˆë„??ê¸°ì??¼ë¡œ ?•ë ¬
            terms.sort(key=lambda x: x[1].get('frequency', 0), reverse=True)
            
            # ?ìœ„ ?©ì–´?¤ì„ ?€???©ì–´ë¡?? íƒ
            representative_terms = terms[:10]  # ?ìœ„ 10ê°??©ì–´
            
            for i, (main_term, main_data) in enumerate(representative_terms):
                if main_term in expanded_relations:
                    continue
                
                # ?™ì˜??ì°¾ê¸° (ê°™ì? ì¹´í…Œê³ ë¦¬, ? ì‚¬??ë¹ˆë„??
                synonyms = []
                for j, (synonym_term, synonym_data) in enumerate(representative_terms):
                    if i != j and synonym_term != main_term:
                        if (synonym_data.get('category') == main_data.get('category') and
                            abs(synonym_data.get('frequency', 0) - main_data.get('frequency', 0)) <= 5):
                            synonyms.append(synonym_term)
                            if len(synonyms) >= 3:  # ìµœë? 3ê°??™ì˜??
                                break
                
                # ê´€???©ì–´ ì°¾ê¸° (ê°™ì? ?„ë©”?? ?’ì? ë¹ˆë„??
                related_terms = []
                for term, term_data in terms:
                    if term != main_term and term not in synonyms:
                        if term_data.get('frequency', 0) >= 5:  # ë¹ˆë„??5 ?´ìƒ
                            related_terms.append(term)
                            if len(related_terms) >= 8:  # ìµœë? 8ê°?ê´€???©ì–´
                                break
                
                # ì»¨í…?¤íŠ¸ ?©ì–´ (ë²•ë¥ ëª? ê¸°ê?ëª???
                context_terms = []
                for term, term_data in terms:
                    if (term != main_term and 
                        term_data.get('category') in ['ë²•ë¥ ëª?, 'ê¸°ê?'] and
                        term_data.get('frequency', 0) >= 3):
                        context_terms.append(term)
                        if len(context_terms) >= 5:  # ìµœë? 5ê°?ì»¨í…?¤íŠ¸ ?©ì–´
                            break
                
                expanded_relations[main_term] = {
                    "synonyms": synonyms,
                    "related": related_terms,
                    "context": context_terms
                }
        
        print(f"?•ì¥???˜ë???ê´€ê³??? {len(expanded_relations)}")
        return expanded_relations
    
    def generate_enhanced_semantic_mapper(self, expanded_relations: Dict):
        """?¥ìƒ??SemanticKeywordMapper ì½”ë“œ ?ì„±"""
        print("?¥ìƒ??SemanticKeywordMapper ì½”ë“œ ?ì„± ì¤?..")
        
        # Python ì½”ë“œ ?œí”Œë¦?
        code_template = '''#!/usr/bin/env python3
"""
?¥ìƒ??SemanticKeywordMapper
ì¶”ì¶œ??ë²•ë¥  ?©ì–´ë¥?ê¸°ë°˜?¼ë¡œ ?•ì¥???˜ë???ê´€ê³„ë? ?œê³µ?©ë‹ˆ??
"""

from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import json

class EnhancedSemanticKeywordMapper:
    """?¥ìƒ???˜ë????¤ì›Œ??ë§¤í•‘ ?œìŠ¤??""
    
    def __init__(self):
        # ?•ì¥??ë²•ë¥  ?©ì–´ ê°„ì˜ ?˜ë???ê´€ê³??•ì˜
        self.semantic_relations = {semantic_relations_dict}
        
        # ?¤ì›Œ??ê°??˜ë???ê±°ë¦¬ ë§¤íŠ¸ë¦?Š¤
        self.semantic_distance = self._build_semantic_distance_matrix()
        
        # ?„ë©”?¸ë³„ ?©ì–´ ê·¸ë£¹
        self.domain_groups = self._build_domain_groups()
    
    def _build_semantic_distance_matrix(self) -> Dict[str, Dict[str, float]]:
        """?˜ë???ê±°ë¦¬ ë§¤íŠ¸ë¦?Š¤ êµ¬ì¶•"""
        distance_matrix = {{}}
        
        for term, relations in self.semantic_relations.items():
            distance_matrix[term] = {{}}
            
            # ?™ì˜?´ëŠ” ê±°ë¦¬ 0.1
            for synonym in relations["synonyms"]:
                distance_matrix[term][synonym] = 0.1
            
            # ê´€???©ì–´??ê±°ë¦¬ 0.3
            for related in relations["related"]:
                distance_matrix[term][related] = 0.3
            
            # ì»¨í…?¤íŠ¸ ?©ì–´??ê±°ë¦¬ 0.5
            for context in relations["context"]:
                distance_matrix[term][context] = 0.5
            
            # ?ê¸° ?ì‹ ?€ ê±°ë¦¬ 0
            distance_matrix[term][term] = 0.0
        
        return distance_matrix
    
    def _build_domain_groups(self) -> Dict[str, List[str]]:
        """?„ë©”?¸ë³„ ?©ì–´ ê·¸ë£¹ êµ¬ì¶•"""
        domain_groups = defaultdict(list)
        
        for term, relations in self.semantic_relations.items():
            # ì»¨í…?¤íŠ¸?ì„œ ?„ë©”??ì¶”ì¶œ
            for context in relations["context"]:
                if context in ["?•ì‚¬ë²?, "ë¯¼ì‚¬ë²?, "ê°€ì¡±ë²•", "?ì‚¬ë²?, "?¸ë™ë²?, "ë¶€?™ì‚°ë²?, "?¹í—ˆë²?, "?‰ì •ë²?]:
                    domain_groups[context].append(term)
                    break
            else:
                domain_groups["ê¸°í?"].append(term)
        
        return dict(domain_groups)
    
    def calculate_semantic_similarity(self, keyword1: str, keyword2: str) -> float:
        """???¤ì›Œ??ê°„ì˜ ?˜ë???? ì‚¬??ê³„ì‚°"""
        # ì§ì ‘?ì¸ ?˜ë???ê´€ê³??•ì¸
        for term, relations in self.semantic_relations.items():
            if keyword1 == term:
                if keyword2 in relations["synonyms"]:
                    return 0.9
                elif keyword2 in relations["related"]:
                    return 0.7
                elif keyword2 in relations["context"]:
                    return 0.5
            
            if keyword2 == term:
                if keyword1 in relations["synonyms"]:
                    return 0.9
                elif keyword1 in relations["related"]:
                    return 0.7
                elif keyword1 in relations["context"]:
                    return 0.5
        
        # ë¶€ë¶?ë¬¸ì??ë§¤ì¹­
        if keyword1 in keyword2 or keyword2 in keyword1:
            return 0.6
        
        # ê³µí†µ ë¬¸ì ê¸°ë°˜ ? ì‚¬??
        common_chars = set(keyword1) & set(keyword2)
        if common_chars:
            similarity = len(common_chars) / max(len(keyword1), len(keyword2))
            return similarity * 0.4
        
        return 0.0
    
    def find_semantic_related_keywords(self, target_keyword: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """?˜ë??ìœ¼ë¡?ê´€?¨ëœ ?¤ì›Œ??ì°¾ê¸°"""
        related_keywords = []
        
        for keyword, relations in self.semantic_relations.items():
            similarity = self.calculate_semantic_similarity(target_keyword, keyword)
            if similarity >= threshold:
                related_keywords.append((keyword, similarity))
        
        # ? ì‚¬??ê¸°ì??¼ë¡œ ?•ë ¬
        related_keywords.sort(key=lambda x: x[1], reverse=True)
        return related_keywords
    
    def expand_keywords_semantically(self, keywords: List[str], expansion_factor: float = 0.7) -> List[str]:
        """?¤ì›Œ?œë? ?˜ë??ìœ¼ë¡??•ì¥"""
        expanded_keywords = set(keywords)
        
        for keyword in keywords:
            related_keywords = self.find_semantic_related_keywords(keyword, expansion_factor)
            for related_keyword, similarity in related_keywords[:5]:  # ?ìœ„ 5ê°œë§Œ ì¶”ê?
                expanded_keywords.add(related_keyword)
        
        return list(expanded_keywords)
    
    def get_semantic_keyword_clusters(self, keywords: List[str]) -> Dict[str, List[str]]:
        """?¤ì›Œ?œì˜ ?˜ë????´ëŸ¬?¤í„° ?ì„±"""
        clusters = defaultdict(list)
        
        for keyword in keywords:
            # ê°€??? ì‚¬???€???¤ì›Œ??ì°¾ê¸°
            best_match = None
            best_similarity = 0.0
            
            for cluster_center, relations in self.semantic_relations.items():
                similarity = self.calculate_semantic_similarity(keyword, cluster_center)
                if similarity > best_similarity and similarity >= 0.5:
                    best_similarity = similarity
                    best_match = cluster_center
            
            if best_match:
                clusters[best_match].append(keyword)
            else:
                clusters[keyword].append(keyword)  # ?…ë¦½ ?´ëŸ¬?¤í„°
        
        return dict(clusters)
    
    def analyze_keyword_semantic_coverage(self, answer: str, keywords: List[str]) -> Dict[str, any]:
        """?¤ì›Œ?œì˜ ?˜ë???ì»¤ë²„ë¦¬ì? ë¶„ì„"""
        answer_lower = answer.lower()
        
        # ì§ì ‘ ë§¤ì¹­
        direct_matches = [kw for kw in keywords if kw.lower() in answer_lower]
        
        # ?˜ë???ë§¤ì¹­
        semantic_matches = []
        for keyword in keywords:
            related_keywords = self.find_semantic_related_keywords(keyword, 0.6)
            for related_kw, similarity in related_keywords:
                if related_kw.lower() in answer_lower:
                    semantic_matches.append((keyword, related_kw, similarity))
        
        # ?´ëŸ¬?¤í„° ë¶„ì„
        clusters = self.get_semantic_keyword_clusters(keywords)
        cluster_coverage = {}
        for cluster_center, cluster_keywords in clusters.items():
            cluster_matches = [kw for kw in cluster_keywords if kw.lower() in answer_lower]
            cluster_coverage[cluster_center] = {
                "total_keywords": len(cluster_keywords),
                "matched_keywords": len(cluster_matches),
                "coverage_ratio": len(cluster_matches) / len(cluster_keywords) if cluster_keywords else 0
            }
        
        return {
            "direct_matches": direct_matches,
            "semantic_matches": semantic_matches,
            "cluster_coverage": cluster_coverage,
            "overall_coverage": len(direct_matches) / len(keywords) if keywords else 0,
            "semantic_coverage": len(set([match[0] for match in semantic_matches])) / len(keywords) if keywords else 0
        }
    
    def get_semantic_keyword_recommendations(self, question: str, query_type: str, base_keywords: List[str]) -> Dict[str, any]:
        """?˜ë????¤ì›Œ??ì¶”ì²œ"""
        # ì§ˆë¬¸?ì„œ ?„ë©”??ì¶”ì¶œ
        question_domains = []
        for domain, domain_keywords in self.domain_groups.items():
            for keyword in domain_keywords:
                if keyword in question:
                    question_domains.append(domain)
                    break
        
        # ?„ë©”?¸ë³„ ê´€???¤ì›Œ??ì¶”ì²œ
        domain_recommendations = {}
        for domain in question_domains:
            if domain in self.domain_groups:
                domain_keywords = self.domain_groups[domain]
                # ê¸°ì¡´ ?¤ì›Œ?œì? ? ì‚¬???„ë©”???¤ì›Œ??ì¶”ì²œ
                recommended = []
                for base_kw in base_keywords:
                    for domain_kw in domain_keywords:
                        similarity = self.calculate_semantic_similarity(base_kw, domain_kw)
                        if similarity >= 0.6 and domain_kw not in base_keywords:
                            recommended.append((domain_kw, similarity))
                
                recommended.sort(key=lambda x: x[1], reverse=True)
                domain_recommendations[domain] = [kw for kw, sim in recommended[:5]]
        
        # ?˜ë????•ì¥ ?¤ì›Œ??
        expanded_keywords = self.expand_keywords_semantically(base_keywords, 0.7)
        new_keywords = [kw for kw in expanded_keywords if kw not in base_keywords]
        
        return {
            "domain_recommendations": domain_recommendations,
            "expanded_keywords": new_keywords,
            "semantic_clusters": self.get_semantic_keyword_clusters(base_keywords),
            "recommended_keywords": new_keywords[:10]  # ?ìœ„ 10ê°?ì¶”ì²œ
        }
    
    def get_domain_statistics(self) -> Dict[str, any]:
        """?„ë©”?¸ë³„ ?µê³„ ?•ë³´"""
        stats = {}
        for domain, keywords in self.domain_groups.items():
            stats[domain] = {
                "total_keywords": len(keywords),
                "top_keywords": keywords[:5],  # ?ìœ„ 5ê°??¤ì›Œ??
                "semantic_relations_count": len([kw for kw in keywords if kw in self.semantic_relations])
            }
        return stats
    
    def export_semantic_relations(self, output_file: str):
        """?˜ë???ê´€ê³„ë? JSON ?Œì¼ë¡??´ë³´?´ê¸°"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.semantic_relations, f, ensure_ascii=False, indent=2)
    
    def load_semantic_relations(self, input_file: str):
        """JSON ?Œì¼?ì„œ ?˜ë???ê´€ê³?ë¡œë“œ"""
        with open(input_file, 'r', encoding='utf-8') as f:
            self.semantic_relations = json.load(f)
        self.semantic_distance = self._build_semantic_distance_matrix()
        self.domain_groups = self._build_domain_groups()

# ?¬ìš© ?ˆì‹œ
if __name__ == "__main__":
    mapper = EnhancedSemanticKeywordMapper()
    
    # ?ŒìŠ¤??
    test_keywords = ["ê³„ì•½", "?í•´ë°°ìƒ", "?Œì†¡"]
    expanded = mapper.expand_keywords_semantically(test_keywords)
    print(f"?•ì¥???¤ì›Œ?? {{expanded}}")
    
    clusters = mapper.get_semantic_keyword_clusters(test_keywords)
    print(f"?¤ì›Œ???´ëŸ¬?¤í„°: {{clusters}}")
    
    stats = mapper.get_domain_statistics()
    print(f"?„ë©”???µê³„: {{stats}}")
'''
        
        # ?˜ë???ê´€ê³??•ì…”?ˆë¦¬ë¥?ë¬¸ì?´ë¡œ ë³€??
        relations_str = json.dumps(expanded_relations, ensure_ascii=False, indent=8)
        
        # ì½”ë“œ ?ì„±
        generated_code = code_template.replace("{semantic_relations_dict}", relations_str)
        
        # ?Œì¼ ?€??
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(generated_code)
        
        print(f"?¥ìƒ??SemanticKeywordMapper ì½”ë“œ ?ì„± ?„ë£Œ: {self.output_file}")
    
    def generate_integration_script(self, expanded_relations: Dict):
        """ê¸°ì¡´ keyword_mapper.py?€ ?µí•©?˜ëŠ” ?¤í¬ë¦½íŠ¸ ?ì„±"""
        print("?µí•© ?¤í¬ë¦½íŠ¸ ?ì„± ì¤?..")
        
        integration_script = '''#!/usr/bin/env python3
"""
ê¸°ì¡´ keyword_mapper.py??SemanticKeywordMapperë¥??•ì¥??ë²„ì „?¼ë¡œ êµì²´?˜ëŠ” ?¤í¬ë¦½íŠ¸
"""

import os
import shutil
from datetime import datetime

def backup_original_file():
    """?ë³¸ ?Œì¼ ë°±ì—…"""
    original_file = "source/services/langgraph/keyword_mapper.py"
    backup_file = f"source/services/langgraph/keyword_mapper_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    if os.path.exists(original_file):
        shutil.copy2(original_file, backup_file)
        print(f"?ë³¸ ?Œì¼ ë°±ì—… ?„ë£Œ: {backup_file}")
        return True
    return False

def integrate_enhanced_semantic_mapper():
    """?¥ìƒ??SemanticKeywordMapper ?µí•©"""
    try:
        # ë°±ì—… ?ì„±
        if not backup_original_file():
            print("?ë³¸ ?Œì¼??ì°¾ì„ ???†ìŠµ?ˆë‹¤.")
            return False
        
        # ê¸°ì¡´ ?Œì¼ ?½ê¸°
        with open("source/services/langgraph/keyword_mapper.py", 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # ?¥ìƒ??SemanticKeywordMapper ?´ë˜???½ê¸°
        with open("source/services/langgraph/enhanced_semantic_relations.py", 'r', encoding='utf-8') as f:
            enhanced_content = f.read()
        
        # EnhancedSemanticKeywordMapper ?´ë˜??ì¶”ì¶œ
        start_marker = "class EnhancedSemanticKeywordMapper:"
        end_marker = "# ?¬ìš© ?ˆì‹œ"
        
        start_idx = enhanced_content.find(start_marker)
        end_idx = enhanced_content.find(end_marker)
        
        if start_idx == -1 or end_idx == -1:
            print("?¥ìƒ???´ë˜?¤ë? ì°¾ì„ ???†ìŠµ?ˆë‹¤.")
            return False
        
        enhanced_class = enhanced_content[start_idx:end_idx].strip()
        
        # ê¸°ì¡´ SemanticKeywordMapper ?´ë˜??êµì²´
        old_start = original_content.find("class SemanticKeywordMapper:")
        if old_start == -1:
            print("ê¸°ì¡´ SemanticKeywordMapper ?´ë˜?¤ë? ì°¾ì„ ???†ìŠµ?ˆë‹¤.")
            return False
        
        # ê¸°ì¡´ ?´ë˜?¤ì˜ ??ì°¾ê¸°
        old_end = original_content.find("class EnhancedKeywordMapper:", old_start)
        if old_end == -1:
            print("ê¸°ì¡´ ?´ë˜?¤ì˜ ?ì„ ì°¾ì„ ???†ìŠµ?ˆë‹¤.")
            return False
        
        # ?ˆë¡œ???´ìš© ?ì„±
        new_content = (
            original_content[:old_start] +
            enhanced_class + "\n\n" +
            original_content[old_end:]
        )
        
        # ?Œì¼ ?€??
        with open("source/services/langgraph/keyword_mapper.py", 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("?¥ìƒ??SemanticKeywordMapper ?µí•© ?„ë£Œ")
        return True
        
    except Exception as e:
        print(f"?µí•© ì¤??¤ë¥˜ ë°œìƒ: {e}")
        return False

if __name__ == "__main__":
    print("SemanticKeywordMapper ?•ì¥ ?µí•© ?œì‘")
    
    if integrate_enhanced_semantic_mapper():
        print("?µí•© ?„ë£Œ!")
    else:
        print("?µí•© ?¤íŒ¨!")
'''
        
        with open("scripts/data_processing/integrate_enhanced_semantic_mapper.py", 'w', encoding='utf-8') as f:
            f.write(integration_script)
        
        print("?µí•© ?¤í¬ë¦½íŠ¸ ?ì„± ?„ë£Œ: scripts/data_processing/integrate_enhanced_semantic_mapper.py")
    
    def run_expansion(self):
        """?•ì¥ ?¤í–‰"""
        print("SemanticKeywordMapper ?•ì¥ ?œì‘")
        
        try:
            # ?°ì´??ë¡œë“œ
            extracted_terms, semantic_relations = self.load_extracted_data()
            
            # ?˜ë???ê´€ê³??•ì¥
            expanded_relations = self.expand_semantic_relations(extracted_terms, semantic_relations)
            
            # ?¥ìƒ??SemanticKeywordMapper ì½”ë“œ ?ì„±
            self.generate_enhanced_semantic_mapper(expanded_relations)
            
            # ?µí•© ?¤í¬ë¦½íŠ¸ ?ì„±
            self.generate_integration_script(expanded_relations)
            
            print("SemanticKeywordMapper ?•ì¥ ?„ë£Œ!")
            
        except Exception as e:
            print(f"?•ì¥ ì¤??¤ë¥˜ ë°œìƒ: {e}")
            raise

def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    expander = SemanticKeywordMapperExpander()
    expander.run_expansion()

if __name__ == "__main__":
    main()
