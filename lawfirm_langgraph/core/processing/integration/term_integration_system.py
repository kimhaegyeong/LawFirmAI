import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class TermIntegrator:
    """용어 통합 및 중복 제거 시스템"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
    
    def calculate_similarity(self, term1: str, term2: str) -> float:
        """용어 간 유사도 계산"""
        # 자카드 유사도
        set1 = set(term1)
        set2 = set(term2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        
        # 길이 차이 페널티
        length_penalty = 1.0 - abs(len(term1) - len(term2)) / max(len(term1), len(term2))
        
        # 최종 유사도
        final_similarity = jaccard_similarity * length_penalty
        
        return final_similarity
    
    def group_similar_terms(self, terms: List[str]) -> List[List[str]]:
        """유사 용어 그룹화"""
        groups = []
        used_terms = set()
        
        for term in terms:
            if term in used_terms:
                continue
            
            # 새 그룹 시작
            current_group = [term]
            used_terms.add(term)
            
            # 유사한 용어 찾기
            for other_term in terms:
                if other_term in used_terms:
                    continue
                
                similarity = self.calculate_similarity(term, other_term)
                if similarity > self.similarity_threshold:
                    current_group.append(other_term)
                    used_terms.add(other_term)
            
            groups.append(current_group)
        
        return groups
    
    def select_representative(self, term_group: List[str]) -> str:
        """대표 용어 선택"""
        if not term_group:
            return ""
        
        if len(term_group) == 1:
            return term_group[0]
        
        # 가장 짧고 일반적인 용어 선택
        # 우선순위: 길이, 빈도, 사전 순
        term_scores = []
        for term in term_group:
            score = (
                -len(term),  # 길이가 짧을수록 좋음
                -term_group.count(term),  # 빈도가 높을수록 좋음
                term  # 사전 순
            )
            term_scores.append((score, term))
        
        term_scores.sort()
        return term_scores[0][1]
    
    def integrate_terms(self, extracted_terms: List[str]) -> List[Dict[str, Any]]:
        """추출된 용어 통합"""
        # 1. 중복 제거
        unique_terms = list(set(extracted_terms))
        self.logger.info(f"중복 제거 후 {len(unique_terms)}개 용어")
        
        # 2. 유사 용어 그룹화
        term_groups = self.group_similar_terms(unique_terms)
        self.logger.info(f"{len(term_groups)}개 용어 그룹 생성")
        
        # 3. 대표 용어 선택 및 통합 정보 생성
        integrated_terms = []
        for group in term_groups:
            representative = self.select_representative(group)
            
            # 그룹 정보
            group_info = {
                "representative_term": representative,
                "all_variants": group,
                "group_size": len(group),
                "similarity_score": self.calculate_group_similarity(group)
            }
            
            integrated_terms.append(group_info)
        
        return integrated_terms
    
    def calculate_group_similarity(self, group: List[str]) -> float:
        """그룹 내 용어들의 평균 유사도 계산"""
        if len(group) <= 1:
            return 1.0
        
        similarities = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                sim = self.calculate_similarity(group[i], group[j])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0

class QualityFilter:
    """품질 기반 용어 필터링"""
    
    def __init__(self, min_quality_score: int = 70, min_confidence: float = 0.7):
        self.min_quality_score = min_quality_score
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)
    
    def filter_terms(self, validated_terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """품질 기준으로 용어 필터링"""
        high_quality_terms = []
        filtered_out = []
        
        for term_data in validated_terms:
            quality_score = term_data.get("quality_score", 0)
            confidence = term_data.get("final_confidence", 0.0)
            is_valid = term_data.get("is_valid", False)
            
            # 필터링 조건
            if (is_valid and 
                quality_score >= self.min_quality_score and 
                confidence >= self.min_confidence):
                high_quality_terms.append(term_data)
            else:
                filtered_out.append({
                    "term": term_data.get("term", ""),
                    "reason": f"품질점수: {quality_score}, 신뢰도: {confidence:.2f}, 유효성: {is_valid}"
                })
        
        self.logger.info(f"고품질 용어: {len(high_quality_terms)}개, 필터링됨: {len(filtered_out)}개")
        
        return high_quality_terms, filtered_out
    
    def calculate_term_priority(self, term_data: Dict[str, Any]) -> float:
        """용어 우선순위 계산"""
        quality_score = term_data.get("quality_score", 0)
        confidence = term_data.get("final_confidence", 0.0)
        weight = term_data.get("weight", 0.5)
        
        # 우선순위 점수 계산
        priority_score = (
            quality_score * 0.4 +  # 품질 점수 40%
            confidence * 100 * 0.3 +  # 신뢰도 30%
            weight * 100 * 0.3  # 가중치 30%
        )
        
        return priority_score

class DatabaseUpdater:
    """데이터베이스 업데이트 시스템"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self.existing_terms = self.load_existing_terms()
    
    def load_existing_terms(self) -> Dict[str, Any]:
        """기존 용어 데이터 로드"""
        if not self.db_path.exists():
            self.logger.info("기존 데이터베이스가 없습니다. 새로 생성합니다.")
            return {}
        
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"기존 데이터베이스 로드 완료: {len(data)}개 도메인")
            return data
        except Exception as e:
            self.logger.error(f"데이터베이스 로드 중 오류: {e}")
            return {}
    
    def is_duplicate(self, term_data: Dict[str, Any]) -> bool:
        """중복 용어 확인"""
        term = term_data.get("term", "")
        domain = term_data.get("domain", "기타/일반")
        
        # 도메인별로 중복 확인
        if domain in self.existing_terms:
            return term in self.existing_terms[domain]
        
        return False
    
    def add_term(self, term_data: Dict[str, Any]) -> None:
        """새 용어 추가"""
        term = term_data.get("term", "")
        domain = term_data.get("domain", "기타/일반")
        
        # 도메인이 없으면 생성
        if domain not in self.existing_terms:
            self.existing_terms[domain] = {}
        
        # 용어 데이터 구성
        term_info = {
            "weight": term_data.get("weight", 0.5),
            "synonyms": term_data.get("synonyms", []),
            "related_terms": term_data.get("related_terms", []),
            "context_keywords": term_data.get("context_keywords", []),
            "source": "nlp_extraction",
            "confidence": term_data.get("final_confidence", 0.0),
            "verified": True,
            "added_date": datetime.now().isoformat(),
            "quality_score": term_data.get("quality_score", 0),
            "definition": term_data.get("definition", "")
        }
        
        self.existing_terms[domain][term] = term_info
        self.logger.info(f"새 용어 추가: {domain} - {term}")
    
    def update_existing_term(self, term_data: Dict[str, Any]) -> None:
        """기존 용어 정보 업데이트"""
        term = term_data.get("term", "")
        domain = term_data.get("domain", "기타/일반")
        
        if domain in self.existing_terms and term in self.existing_terms[domain]:
            # 기존 정보와 새 정보 병합
            existing = self.existing_terms[domain][term]
            
            # 신뢰도가 높은 경우에만 업데이트
            if term_data.get("final_confidence", 0.0) > existing.get("confidence", 0.0):
                existing.update({
                    "weight": term_data.get("weight", existing.get("weight", 0.5)),
                    "synonyms": list(set(existing.get("synonyms", []) + term_data.get("synonyms", []))),
                    "related_terms": list(set(existing.get("related_terms", []) + term_data.get("related_terms", []))),
                    "context_keywords": list(set(existing.get("context_keywords", []) + term_data.get("context_keywords", []))),
                    "confidence": term_data.get("final_confidence", existing.get("confidence", 0.0)),
                    "updated_date": datetime.now().isoformat(),
                    "quality_score": term_data.get("quality_score", existing.get("quality_score", 0)),
                    "definition": term_data.get("definition", existing.get("definition", ""))
                })
                self.logger.info(f"용어 정보 업데이트: {domain} - {term}")
    
    def save_database(self) -> None:
        """데이터베이스 저장"""
        try:
            # 디렉토리 생성
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 데이터 저장
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.existing_terms, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"데이터베이스 저장 완료: {self.db_path}")
        except Exception as e:
            self.logger.error(f"데이터베이스 저장 중 오류: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계"""
        total_terms = sum(len(terms) for terms in self.existing_terms.values())
        domain_stats = {
            domain: len(terms) for domain, terms in self.existing_terms.items()
        }
        
        return {
            "total_terms": total_terms,
            "domain_count": len(self.existing_terms),
            "domain_stats": domain_stats
        }

class TermIntegrationSystem:
    """용어 통합 시스템 메인 클래스"""
    
    def __init__(self, db_path: str = "data/legal_terms_database.json"):
        self.integrator = TermIntegrator()
        self.quality_filter = QualityFilter()
        self.db_updater = DatabaseUpdater(db_path)
        self.logger = logging.getLogger(__name__)
    
    def process_extracted_terms(self, extracted_terms: List[str]) -> List[Dict[str, Any]]:
        """추출된 용어 처리"""
        # 1. 용어 통합
        integrated_terms = self.integrator.integrate_terms(extracted_terms)
        
        # 2. 대표 용어만 추출
        representative_terms = [term["representative_term"] for term in integrated_terms]
        
        return representative_terms
    
    def process_validated_terms(self, validated_terms: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """검증된 용어 처리"""
        # 1. 품질 필터링
        high_quality_terms, filtered_out = self.quality_filter.filter_terms(validated_terms)
        
        # 2. 우선순위 계산
        for term_data in high_quality_terms:
            term_data["priority_score"] = self.quality_filter.calculate_term_priority(term_data)
        
        # 3. 우선순위별 정렬
        high_quality_terms.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return high_quality_terms, filtered_out
    
    def update_database(self, processed_terms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """데이터베이스 업데이트"""
        added_count = 0
        updated_count = 0
        
        for term_data in processed_terms:
            if not self.db_updater.is_duplicate(term_data):
                self.db_updater.add_term(term_data)
                added_count += 1
            else:
                self.db_updater.update_existing_term(term_data)
                updated_count += 1
        
        # 데이터베이스 저장
        self.db_updater.save_database()
        
        # 통계 반환
        stats = self.db_updater.get_database_stats()
        stats.update({
            "added_terms": added_count,
            "updated_terms": updated_count
        })
        
        return stats
    
    def full_pipeline(self, extracted_terms: List[str], validated_terms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        self.logger.info("용어 통합 파이프라인 시작")
        
        # 1. 추출된 용어 처리
        processed_extracted = self.process_extracted_terms(extracted_terms)
        
        # 2. 검증된 용어 처리
        high_quality_terms, filtered_out = self.process_validated_terms(validated_terms)
        
        # 3. 데이터베이스 업데이트
        update_stats = self.update_database(high_quality_terms)
        
        # 4. 결과 반환
        result = {
            "extracted_terms_count": len(extracted_terms),
            "processed_extracted_count": len(processed_extracted),
            "validated_terms_count": len(validated_terms),
            "high_quality_terms_count": len(high_quality_terms),
            "filtered_out_count": len(filtered_out),
            "database_stats": update_stats,
            "high_quality_terms": high_quality_terms,
            "filtered_out": filtered_out
        }
        
        self.logger.info(f"파이프라인 완료: {result}")
        return result
