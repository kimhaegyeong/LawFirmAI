#!/usr/bin/env python3
"""
SemanticKeywordMapper 확장 스크립트
추출된 법률 용어들을 기반으로 의미적 관계를 확장합니다.
"""

import json
import os
from typing import Dict, List, Set
from collections import defaultdict, Counter

class SemanticKeywordMapperExpander:
    """SemanticKeywordMapper 확장기"""
    
    def __init__(self):
        self.extracted_terms_file = "data/extracted_terms/extracted_legal_terms.json"
        self.semantic_relations_file = "data/extracted_terms/semantic_relations.json"
        self.output_file = "source/services/langgraph/enhanced_semantic_relations.py"
        
    def load_extracted_data(self) -> tuple:
        """추출된 데이터 로드"""
        print("추출된 용어 데이터 로드 중...")
        
        with open(self.extracted_terms_file, 'r', encoding='utf-8') as f:
            extracted_terms = json.load(f)
        
        with open(self.semantic_relations_file, 'r', encoding='utf-8') as f:
            semantic_relations = json.load(f)
        
        print(f"로드된 용어 수: {len(extracted_terms)}")
        print(f"로드된 의미적 관계 수: {len(semantic_relations)}")
        
        return extracted_terms, semantic_relations
    
    def expand_semantic_relations(self, extracted_terms: Dict, semantic_relations: Dict) -> Dict:
        """의미적 관계 확장"""
        print("의미적 관계 확장 중...")
        
        # 기존 관계를 기반으로 확장된 관계 생성
        expanded_relations = {}
        
        # 도메인별 용어 그룹화
        domain_groups = defaultdict(list)
        for term, term_data in extracted_terms.items():
            domain = term_data.get('domain', '기타')
            domain_groups[domain].append((term, term_data))
        
        # 각 도메인별로 의미적 관계 생성
        for domain, terms in domain_groups.items():
            if len(terms) < 5:  # 최소 5개 이상의 용어가 있는 도메인만 처리
                continue
            
            # 빈도수 기준으로 정렬
            terms.sort(key=lambda x: x[1].get('frequency', 0), reverse=True)
            
            # 상위 용어들을 대표 용어로 선택
            representative_terms = terms[:10]  # 상위 10개 용어
            
            for i, (main_term, main_data) in enumerate(representative_terms):
                if main_term in expanded_relations:
                    continue
                
                # 동의어 찾기 (같은 카테고리, 유사한 빈도수)
                synonyms = []
                for j, (synonym_term, synonym_data) in enumerate(representative_terms):
                    if i != j and synonym_term != main_term:
                        if (synonym_data.get('category') == main_data.get('category') and
                            abs(synonym_data.get('frequency', 0) - main_data.get('frequency', 0)) <= 5):
                            synonyms.append(synonym_term)
                            if len(synonyms) >= 3:  # 최대 3개 동의어
                                break
                
                # 관련 용어 찾기 (같은 도메인, 높은 빈도수)
                related_terms = []
                for term, term_data in terms:
                    if term != main_term and term not in synonyms:
                        if term_data.get('frequency', 0) >= 5:  # 빈도수 5 이상
                            related_terms.append(term)
                            if len(related_terms) >= 8:  # 최대 8개 관련 용어
                                break
                
                # 컨텍스트 용어 (법률명, 기관명 등)
                context_terms = []
                for term, term_data in terms:
                    if (term != main_term and 
                        term_data.get('category') in ['법률명', '기관'] and
                        term_data.get('frequency', 0) >= 3):
                        context_terms.append(term)
                        if len(context_terms) >= 5:  # 최대 5개 컨텍스트 용어
                            break
                
                expanded_relations[main_term] = {
                    "synonyms": synonyms,
                    "related": related_terms,
                    "context": context_terms
                }
        
        print(f"확장된 의미적 관계 수: {len(expanded_relations)}")
        return expanded_relations
    
    def generate_enhanced_semantic_mapper(self, expanded_relations: Dict):
        """향상된 SemanticKeywordMapper 코드 생성"""
        print("향상된 SemanticKeywordMapper 코드 생성 중...")
        
        # Python 코드 템플릿
        code_template = '''#!/usr/bin/env python3
"""
향상된 SemanticKeywordMapper
추출된 법률 용어를 기반으로 확장된 의미적 관계를 제공합니다.
"""

from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import json

class EnhancedSemanticKeywordMapper:
    """향상된 의미적 키워드 매핑 시스템"""
    
    def __init__(self):
        # 확장된 법률 용어 간의 의미적 관계 정의
        self.semantic_relations = {semantic_relations_dict}
        
        # 키워드 간 의미적 거리 매트릭스
        self.semantic_distance = self._build_semantic_distance_matrix()
        
        # 도메인별 용어 그룹
        self.domain_groups = self._build_domain_groups()
    
    def _build_semantic_distance_matrix(self) -> Dict[str, Dict[str, float]]:
        """의미적 거리 매트릭스 구축"""
        distance_matrix = {{}}
        
        for term, relations in self.semantic_relations.items():
            distance_matrix[term] = {{}}
            
            # 동의어는 거리 0.1
            for synonym in relations["synonyms"]:
                distance_matrix[term][synonym] = 0.1
            
            # 관련 용어는 거리 0.3
            for related in relations["related"]:
                distance_matrix[term][related] = 0.3
            
            # 컨텍스트 용어는 거리 0.5
            for context in relations["context"]:
                distance_matrix[term][context] = 0.5
            
            # 자기 자신은 거리 0
            distance_matrix[term][term] = 0.0
        
        return distance_matrix
    
    def _build_domain_groups(self) -> Dict[str, List[str]]:
        """도메인별 용어 그룹 구축"""
        domain_groups = defaultdict(list)
        
        for term, relations in self.semantic_relations.items():
            # 컨텍스트에서 도메인 추출
            for context in relations["context"]:
                if context in ["형사법", "민사법", "가족법", "상사법", "노동법", "부동산법", "특허법", "행정법"]:
                    domain_groups[context].append(term)
                    break
            else:
                domain_groups["기타"].append(term)
        
        return dict(domain_groups)
    
    def calculate_semantic_similarity(self, keyword1: str, keyword2: str) -> float:
        """두 키워드 간의 의미적 유사도 계산"""
        # 직접적인 의미적 관계 확인
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
        
        # 부분 문자열 매칭
        if keyword1 in keyword2 or keyword2 in keyword1:
            return 0.6
        
        # 공통 문자 기반 유사도
        common_chars = set(keyword1) & set(keyword2)
        if common_chars:
            similarity = len(common_chars) / max(len(keyword1), len(keyword2))
            return similarity * 0.4
        
        return 0.0
    
    def find_semantic_related_keywords(self, target_keyword: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """의미적으로 관련된 키워드 찾기"""
        related_keywords = []
        
        for keyword, relations in self.semantic_relations.items():
            similarity = self.calculate_semantic_similarity(target_keyword, keyword)
            if similarity >= threshold:
                related_keywords.append((keyword, similarity))
        
        # 유사도 기준으로 정렬
        related_keywords.sort(key=lambda x: x[1], reverse=True)
        return related_keywords
    
    def expand_keywords_semantically(self, keywords: List[str], expansion_factor: float = 0.7) -> List[str]:
        """키워드를 의미적으로 확장"""
        expanded_keywords = set(keywords)
        
        for keyword in keywords:
            related_keywords = self.find_semantic_related_keywords(keyword, expansion_factor)
            for related_keyword, similarity in related_keywords[:5]:  # 상위 5개만 추가
                expanded_keywords.add(related_keyword)
        
        return list(expanded_keywords)
    
    def get_semantic_keyword_clusters(self, keywords: List[str]) -> Dict[str, List[str]]:
        """키워드의 의미적 클러스터 생성"""
        clusters = defaultdict(list)
        
        for keyword in keywords:
            # 가장 유사한 대표 키워드 찾기
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
                clusters[keyword].append(keyword)  # 독립 클러스터
        
        return dict(clusters)
    
    def analyze_keyword_semantic_coverage(self, answer: str, keywords: List[str]) -> Dict[str, any]:
        """키워드의 의미적 커버리지 분석"""
        answer_lower = answer.lower()
        
        # 직접 매칭
        direct_matches = [kw for kw in keywords if kw.lower() in answer_lower]
        
        # 의미적 매칭
        semantic_matches = []
        for keyword in keywords:
            related_keywords = self.find_semantic_related_keywords(keyword, 0.6)
            for related_kw, similarity in related_keywords:
                if related_kw.lower() in answer_lower:
                    semantic_matches.append((keyword, related_kw, similarity))
        
        # 클러스터 분석
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
        """의미적 키워드 추천"""
        # 질문에서 도메인 추출
        question_domains = []
        for domain, domain_keywords in self.domain_groups.items():
            for keyword in domain_keywords:
                if keyword in question:
                    question_domains.append(domain)
                    break
        
        # 도메인별 관련 키워드 추천
        domain_recommendations = {}
        for domain in question_domains:
            if domain in self.domain_groups:
                domain_keywords = self.domain_groups[domain]
                # 기존 키워드와 유사한 도메인 키워드 추천
                recommended = []
                for base_kw in base_keywords:
                    for domain_kw in domain_keywords:
                        similarity = self.calculate_semantic_similarity(base_kw, domain_kw)
                        if similarity >= 0.6 and domain_kw not in base_keywords:
                            recommended.append((domain_kw, similarity))
                
                recommended.sort(key=lambda x: x[1], reverse=True)
                domain_recommendations[domain] = [kw for kw, sim in recommended[:5]]
        
        # 의미적 확장 키워드
        expanded_keywords = self.expand_keywords_semantically(base_keywords, 0.7)
        new_keywords = [kw for kw in expanded_keywords if kw not in base_keywords]
        
        return {
            "domain_recommendations": domain_recommendations,
            "expanded_keywords": new_keywords,
            "semantic_clusters": self.get_semantic_keyword_clusters(base_keywords),
            "recommended_keywords": new_keywords[:10]  # 상위 10개 추천
        }
    
    def get_domain_statistics(self) -> Dict[str, any]:
        """도메인별 통계 정보"""
        stats = {}
        for domain, keywords in self.domain_groups.items():
            stats[domain] = {
                "total_keywords": len(keywords),
                "top_keywords": keywords[:5],  # 상위 5개 키워드
                "semantic_relations_count": len([kw for kw in keywords if kw in self.semantic_relations])
            }
        return stats
    
    def export_semantic_relations(self, output_file: str):
        """의미적 관계를 JSON 파일로 내보내기"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.semantic_relations, f, ensure_ascii=False, indent=2)
    
    def load_semantic_relations(self, input_file: str):
        """JSON 파일에서 의미적 관계 로드"""
        with open(input_file, 'r', encoding='utf-8') as f:
            self.semantic_relations = json.load(f)
        self.semantic_distance = self._build_semantic_distance_matrix()
        self.domain_groups = self._build_domain_groups()

# 사용 예시
if __name__ == "__main__":
    mapper = EnhancedSemanticKeywordMapper()
    
    # 테스트
    test_keywords = ["계약", "손해배상", "소송"]
    expanded = mapper.expand_keywords_semantically(test_keywords)
    print(f"확장된 키워드: {{expanded}}")
    
    clusters = mapper.get_semantic_keyword_clusters(test_keywords)
    print(f"키워드 클러스터: {{clusters}}")
    
    stats = mapper.get_domain_statistics()
    print(f"도메인 통계: {{stats}}")
'''
        
        # 의미적 관계 딕셔너리를 문자열로 변환
        relations_str = json.dumps(expanded_relations, ensure_ascii=False, indent=8)
        
        # 코드 생성
        generated_code = code_template.replace("{semantic_relations_dict}", relations_str)
        
        # 파일 저장
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(generated_code)
        
        print(f"향상된 SemanticKeywordMapper 코드 생성 완료: {self.output_file}")
    
    def generate_integration_script(self, expanded_relations: Dict):
        """기존 keyword_mapper.py와 통합하는 스크립트 생성"""
        print("통합 스크립트 생성 중...")
        
        integration_script = '''#!/usr/bin/env python3
"""
기존 keyword_mapper.py의 SemanticKeywordMapper를 확장된 버전으로 교체하는 스크립트
"""

import os
import shutil
from datetime import datetime

def backup_original_file():
    """원본 파일 백업"""
    original_file = "source/services/langgraph/keyword_mapper.py"
    backup_file = f"source/services/langgraph/keyword_mapper_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    if os.path.exists(original_file):
        shutil.copy2(original_file, backup_file)
        print(f"원본 파일 백업 완료: {backup_file}")
        return True
    return False

def integrate_enhanced_semantic_mapper():
    """향상된 SemanticKeywordMapper 통합"""
    try:
        # 백업 생성
        if not backup_original_file():
            print("원본 파일을 찾을 수 없습니다.")
            return False
        
        # 기존 파일 읽기
        with open("source/services/langgraph/keyword_mapper.py", 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # 향상된 SemanticKeywordMapper 클래스 읽기
        with open("source/services/langgraph/enhanced_semantic_relations.py", 'r', encoding='utf-8') as f:
            enhanced_content = f.read()
        
        # EnhancedSemanticKeywordMapper 클래스 추출
        start_marker = "class EnhancedSemanticKeywordMapper:"
        end_marker = "# 사용 예시"
        
        start_idx = enhanced_content.find(start_marker)
        end_idx = enhanced_content.find(end_marker)
        
        if start_idx == -1 or end_idx == -1:
            print("향상된 클래스를 찾을 수 없습니다.")
            return False
        
        enhanced_class = enhanced_content[start_idx:end_idx].strip()
        
        # 기존 SemanticKeywordMapper 클래스 교체
        old_start = original_content.find("class SemanticKeywordMapper:")
        if old_start == -1:
            print("기존 SemanticKeywordMapper 클래스를 찾을 수 없습니다.")
            return False
        
        # 기존 클래스의 끝 찾기
        old_end = original_content.find("class EnhancedKeywordMapper:", old_start)
        if old_end == -1:
            print("기존 클래스의 끝을 찾을 수 없습니다.")
            return False
        
        # 새로운 내용 생성
        new_content = (
            original_content[:old_start] +
            enhanced_class + "\n\n" +
            original_content[old_end:]
        )
        
        # 파일 저장
        with open("source/services/langgraph/keyword_mapper.py", 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("향상된 SemanticKeywordMapper 통합 완료")
        return True
        
    except Exception as e:
        print(f"통합 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    print("SemanticKeywordMapper 확장 통합 시작")
    
    if integrate_enhanced_semantic_mapper():
        print("통합 완료!")
    else:
        print("통합 실패!")
'''
        
        with open("scripts/data_processing/integrate_enhanced_semantic_mapper.py", 'w', encoding='utf-8') as f:
            f.write(integration_script)
        
        print("통합 스크립트 생성 완료: scripts/data_processing/integrate_enhanced_semantic_mapper.py")
    
    def run_expansion(self):
        """확장 실행"""
        print("SemanticKeywordMapper 확장 시작")
        
        try:
            # 데이터 로드
            extracted_terms, semantic_relations = self.load_extracted_data()
            
            # 의미적 관계 확장
            expanded_relations = self.expand_semantic_relations(extracted_terms, semantic_relations)
            
            # 향상된 SemanticKeywordMapper 코드 생성
            self.generate_enhanced_semantic_mapper(expanded_relations)
            
            # 통합 스크립트 생성
            self.generate_integration_script(expanded_relations)
            
            print("SemanticKeywordMapper 확장 완료!")
            
        except Exception as e:
            print(f"확장 중 오류 발생: {e}")
            raise

def main():
    """메인 실행 함수"""
    expander = SemanticKeywordMapperExpander()
    expander.run_expansion()

if __name__ == "__main__":
    main()
