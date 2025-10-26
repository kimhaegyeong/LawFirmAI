#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Q&A 품질 검증 모듈

LLM 생성 결과의 품질을 검증하고 개선하는 모듈
"""

import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import difflib

logger = logging.getLogger(__name__)


class QAQualityValidator:
    """Q&A 품질 검증기"""
    
    def __init__(self):
        """품질 검증기 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 품질 기준 정의
        self.quality_criteria = {
            # 길이 기준
            'min_question_length': 10,
            'max_question_length': 200,
            'min_answer_length': 20,
            'max_answer_length': 1000,
            
            # 품질 점수 기준
            'min_quality_score': 0.6,
            'high_quality_threshold': 0.8,
            
            # 중복 검사 기준
            'similarity_threshold': 0.8,
            
            # 법률 용어 검증
            'legal_terms_weight': 0.1,
            'citation_weight': 0.15,
            'completeness_weight': 0.2,
            'clarity_weight': 0.15,
            'relevance_weight': 0.2,
            'uniqueness_weight': 0.1,
            'format_weight': 0.1
        }
        
        # 법률 용어 패턴
        self.legal_patterns = [
            r'법\s*[령규]',
            r'조\s*문',
            r'항\s*목',
            r'규\s*정',
            r'의\s*미',
            r'적\s*용',
            r'효\s*력',
            r'요\s*건',
            r'절\s*차',
            r'권\s*리',
            r'의\s*무',
            r'책\s*임',
            r'손\s*해',
            r'배\s*상',
            r'계\s*약',
            r'소\s*송',
            r'판\s*결',
            r'판\s*례',
            r'헌\s*법',
            r'기\s*본\s*권'
        ]
        
        # 부적절한 패턴
        self.inappropriate_patterns = [
            r'모르겠',
            r'잘\s*모르',
            r'확실하지',
            r'정확하지',
            r'아마도',
            r'아마',
            r'추측',
            r'개인적',
            r'의견',
            r'생각',
            r'불확실'
        ]
    
    def validate_qa_pair(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """단일 Q&A 쌍 품질 검증"""
        validation_result = {
            'is_valid': True,
            'quality_score': 0.0,
            'issues': [],
            'suggestions': [],
            'confidence': 0.0
        }
        
        try:
            # 기본 필드 검증
            if not self._validate_basic_fields(qa_pair, validation_result):
                return validation_result
            
            # 길이 검증
            self._validate_lengths(qa_pair, validation_result)
            
            # 내용 품질 검증
            self._validate_content_quality(qa_pair, validation_result)
            
            # 법률 정확성 검증
            self._validate_legal_accuracy(qa_pair, validation_result)
            
            # 형식 검증
            self._validate_format(qa_pair, validation_result)
            
            # 최종 품질 점수 계산
            validation_result['quality_score'] = self._calculate_final_score(qa_pair, validation_result)
            
            # 신뢰도 계산
            validation_result['confidence'] = self._calculate_confidence(qa_pair, validation_result)
            
            # 유효성 최종 판정
            validation_result['is_valid'] = (
                validation_result['quality_score'] >= self.quality_criteria['min_quality_score'] and
                len(validation_result['issues']) == 0
            )
            
        except Exception as e:
            self.logger.error(f"Q&A 검증 중 오류: {e}")
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"검증 오류: {str(e)}")
        
        return validation_result
    
    def _validate_basic_fields(self, qa_pair: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """기본 필드 검증"""
        required_fields = ['question', 'answer']
        
        for field in required_fields:
            if field not in qa_pair or not qa_pair[field]:
                result['issues'].append(f"필수 필드 누락: {field}")
                return False
            
            if not isinstance(qa_pair[field], str):
                result['issues'].append(f"필드 타입 오류: {field}는 문자열이어야 함")
                return False
        
        return True
    
    def _validate_lengths(self, qa_pair: Dict[str, Any], result: Dict[str, Any]):
        """길이 검증"""
        question = qa_pair['question'].strip()
        answer = qa_pair['answer'].strip()
        
        # 질문 길이 검증
        if len(question) < self.quality_criteria['min_question_length']:
            result['issues'].append(f"질문이 너무 짧음: {len(question)}자 (최소 {self.quality_criteria['min_question_length']}자)")
        elif len(question) > self.quality_criteria['max_question_length']:
            result['issues'].append(f"질문이 너무 김: {len(question)}자 (최대 {self.quality_criteria['max_question_length']}자)")
        
        # 답변 길이 검증
        if len(answer) < self.quality_criteria['min_answer_length']:
            result['issues'].append(f"답변이 너무 짧음: {len(answer)}자 (최소 {self.quality_criteria['min_answer_length']}자)")
        elif len(answer) > self.quality_criteria['max_answer_length']:
            result['issues'].append(f"답변이 너무 김: {len(answer)}자 (최대 {self.quality_criteria['max_answer_length']}자)")
    
    def _validate_content_quality(self, qa_pair: Dict[str, Any], result: Dict[str, Any]):
        """내용 품질 검증"""
        question = qa_pair['question'].strip()
        answer = qa_pair['answer'].strip()
        
        # 질문 품질 검증
        if not question.endswith(('?', '요', '까', '인가요', '습니까')):
            result['suggestions'].append("질문이 의문문으로 끝나지 않음")
        
        # 부적절한 표현 검증
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                result['issues'].append(f"부적절한 표현 발견: {pattern}")
        
        # 법률 용어 사용 검증
        legal_term_count = sum(1 for pattern in self.legal_patterns if re.search(pattern, answer, re.IGNORECASE))
        if legal_term_count == 0:
            result['suggestions'].append("법률 용어 사용을 늘려주세요")
        
        # 완전성 검증
        if len(answer.split('.')) < 2:
            result['suggestions'].append("답변이 너무 간단함. 더 자세한 설명이 필요합니다")
    
    def _validate_legal_accuracy(self, qa_pair: Dict[str, Any], result: Dict[str, Any]):
        """법률 정확성 검증"""
        answer = qa_pair['answer'].strip()
        
        # 법령 인용 검증
        law_citations = re.findall(r'[가-힣]+법\s*제?\d+조', answer)
        if law_citations:
            result['confidence'] += 0.1  # 법령 인용 시 신뢰도 증가
        
        # 판례 인용 검증
        precedent_citations = re.findall(r'[가-힣]+법원\s*\d+[가-힣]+\d+', answer)
        if precedent_citations:
            result['confidence'] += 0.1  # 판례 인용 시 신뢰도 증가
        
        # 구체적 조문 언급 검증
        article_mentions = re.findall(r'제?\d+조', answer)
        if article_mentions:
            result['confidence'] += 0.05
    
    def _validate_format(self, qa_pair: Dict[str, Any], result: Dict[str, Any]):
        """형식 검증"""
        question = qa_pair['question'].strip()
        answer = qa_pair['answer'].strip()
        
        # 질문 형식 검증
        if question.startswith(('질문:', 'Q:', 'Question:')):
            result['suggestions'].append("질문에서 접두사 제거 권장")
        
        # 답변 형식 검증
        if answer.startswith(('답변:', 'A:', 'Answer:')):
            result['suggestions'].append("답변에서 접두사 제거 권장")
        
        # 문장 부호 검증
        if not answer.endswith(('.', '다', '요', '니다')):
            result['suggestions'].append("답변이 적절한 문장 부호로 끝나지 않음")
    
    def _calculate_final_score(self, qa_pair: Dict[str, Any], result: Dict[str, Any]) -> float:
        """최종 품질 점수 계산"""
        score = 0.0
        
        # 기본 점수
        score += 0.2
        
        # 길이 점수 (질문)
        question_len = len(qa_pair['question'])
        if 15 <= question_len <= 100:
            score += 0.15
        elif 100 < question_len <= 150:
            score += 0.1
        
        # 길이 점수 (답변)
        answer_len = len(qa_pair['answer'])
        if 30 <= answer_len <= 400:
            score += 0.2
        elif 400 < answer_len <= 600:
            score += 0.15
        
        # 법률 용어 점수
        legal_term_count = sum(1 for pattern in self.legal_patterns 
                              if re.search(pattern, qa_pair['answer'], re.IGNORECASE))
        score += min(legal_term_count * 0.05, 0.15)
        
        # 신뢰도 점수
        confidence = qa_pair.get('confidence', 0.5)
        score += confidence * 0.1
        
        # 이슈 페널티
        issue_penalty = len(result['issues']) * 0.1
        score -= issue_penalty
        
        return max(0.0, min(1.0, score))
    
    def _calculate_confidence(self, qa_pair: Dict[str, Any], result: Dict[str, Any]) -> float:
        """신뢰도 계산"""
        confidence = qa_pair.get('confidence', 0.5)
        
        # 이슈가 있으면 신뢰도 감소
        if result['issues']:
            confidence -= len(result['issues']) * 0.1
        
        # 제안사항이 있으면 신뢰도 약간 감소
        if result['suggestions']:
            confidence -= len(result['suggestions']) * 0.05
        
        return max(0.0, min(1.0, confidence))
    
    def validate_dataset(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """전체 데이터셋 품질 검증"""
        validation_results = {
            'total_pairs': len(qa_pairs),
            'valid_pairs': 0,
            'invalid_pairs': 0,
            'high_quality_pairs': 0,
            'medium_quality_pairs': 0,
            'low_quality_pairs': 0,
            'average_quality_score': 0.0,
            'common_issues': {},
            'quality_distribution': {},
            'recommendations': [],
            'validated_at': datetime.now().isoformat()
        }
        
        valid_pairs = []
        quality_scores = []
        
        for i, qa_pair in enumerate(qa_pairs):
            # 개별 Q&A 검증
            validation = self.validate_qa_pair(qa_pair)
            
            if validation['is_valid']:
                validation_results['valid_pairs'] += 1
                valid_pairs.append(qa_pair)
                
                # 품질 점수별 분류
                quality_score = validation['quality_score']
                quality_scores.append(quality_score)
                
                if quality_score >= self.quality_criteria['high_quality_threshold']:
                    validation_results['high_quality_pairs'] += 1
                elif quality_score >= self.quality_criteria['min_quality_score']:
                    validation_results['medium_quality_pairs'] += 1
                else:
                    validation_results['low_quality_pairs'] += 1
            else:
                validation_results['invalid_pairs'] += 1
                
                # 공통 이슈 수집
                for issue in validation['issues']:
                    validation_results['common_issues'][issue] = validation_results['common_issues'].get(issue, 0) + 1
        
        # 평균 품질 점수 계산
        if quality_scores:
            validation_results['average_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        # 품질 분포 계산
        if quality_scores:
            validation_results['quality_distribution'] = {
                'min': min(quality_scores),
                'max': max(quality_scores),
                'median': sorted(quality_scores)[len(quality_scores) // 2],
                'std_dev': self._calculate_std_dev(quality_scores)
            }
        
        # 권장사항 생성
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        
        return validation_results, valid_pairs
    
    def _calculate_std_dev(self, scores: List[float]) -> float:
        """표준편차 계산"""
        if len(scores) < 2:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance ** 0.5
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # 유효성 비율 기반 권장사항
        validity_rate = results['valid_pairs'] / results['total_pairs'] if results['total_pairs'] > 0 else 0
        if validity_rate < 0.8:
            recommendations.append("유효성 비율이 낮습니다. 데이터 품질을 개선해주세요.")
        
        # 품질 점수 기반 권장사항
        if results['average_quality_score'] < 0.7:
            recommendations.append("평균 품질 점수가 낮습니다. LLM 프롬프트를 개선해주세요.")
        
        # 공통 이슈 기반 권장사항
        if '질문이 너무 짧음' in results['common_issues']:
            recommendations.append("질문 길이를 늘려주세요.")
        
        if '답변이 너무 짧음' in results['common_issues']:
            recommendations.append("답변을 더 자세히 작성해주세요.")
        
        return recommendations
    
    def detect_duplicates(self, qa_pairs: List[Dict[str, Any]]) -> List[List[int]]:
        """중복 Q&A 감지"""
        duplicates = []
        processed = set()
        
        for i, qa1 in enumerate(qa_pairs):
            if i in processed:
                continue
                
            duplicate_group = [i]
            
            for j, qa2 in enumerate(qa_pairs[i+1:], i+1):
                if j in processed:
                    continue
                
                # 질문 유사도 계산
                question_similarity = difflib.SequenceMatcher(
                    None, 
                    qa1['question'].lower(), 
                    qa2['question'].lower()
                ).ratio()
                
                # 답변 유사도 계산
                answer_similarity = difflib.SequenceMatcher(
                    None, 
                    qa1['answer'].lower(), 
                    qa2['answer'].lower()
                ).ratio()
                
                # 전체 유사도 계산
                overall_similarity = (question_similarity + answer_similarity) / 2
                
                if overall_similarity >= self.quality_criteria['similarity_threshold']:
                    duplicate_group.append(j)
                    processed.add(j)
            
            if len(duplicate_group) > 1:
                duplicates.append(duplicate_group)
                processed.update(duplicate_group)
        
        return duplicates


def main():
    """테스트 함수"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트 Q&A 데이터
    test_qa_pairs = [
        {
            "question": "개인정보 보호법 제2조의 정의는 무엇인가요?",
            "answer": "개인정보 보호법 제2조제1항에 따르면, '개인정보'란 살아 있는 개인에 관한 정보로서 성명, 주민등록번호 및 영상 등을 통하여 개인을 알아볼 수 있는 정보를 말합니다.",
            "source": "law_definition_llm",
            "confidence": 0.9,
            "type": "개념 설명"
        },
        {
            "question": "개인정보란?",
            "answer": "개인정보는 개인에 관한 정보입니다.",
            "source": "law_definition_llm",
            "confidence": 0.5,
            "type": "개념 설명"
        }
    ]
    
    # 품질 검증기 생성
    validator = QAQualityValidator()
    
    # 개별 Q&A 검증 테스트
    print("개별 Q&A 검증 테스트:")
    for i, qa in enumerate(test_qa_pairs):
        result = validator.validate_qa_pair(qa)
        print(f"\n{i+1}. 질문: {qa['question']}")
        print(f"   유효성: {result['is_valid']}")
        print(f"   품질 점수: {result['quality_score']:.3f}")
        print(f"   이슈: {result['issues']}")
        print(f"   제안: {result['suggestions']}")
    
    # 데이터셋 검증 테스트
    print("\n데이터셋 검증 테스트:")
    validation_results, valid_pairs = validator.validate_dataset(test_qa_pairs)
    
    print(f"총 Q&A: {validation_results['total_pairs']}개")
    print(f"유효한 Q&A: {validation_results['valid_pairs']}개")
    print(f"무효한 Q&A: {validation_results['invalid_pairs']}개")
    print(f"평균 품질 점수: {validation_results['average_quality_score']:.3f}")
    print(f"공통 이슈: {validation_results['common_issues']}")
    print(f"권장사항: {validation_results['recommendations']}")


if __name__ == "__main__":
    main()
