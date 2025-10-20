#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 기반 Q&A 데이터셋 생성기

Ollama Qwen2.5:7b 모델을 사용하여 다양하고 자연스러운 법률 Q&A를 생성
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

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.utils.ollama_client import OllamaClient
from source.utils.qa_quality_validator import QAQualityValidator

# 로깅 설정
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
    """LLM 기반 Q&A 생성기"""
    
    def __init__(
        self, 
        model: str = "qwen2.5:7b",
        temperature: float = 0.7,
        max_tokens: int = 1500
    ):
        """
        LLM Q&A 생성기 초기화
        
        Args:
            model: 사용할 Ollama 모델명
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
        """
        self.ollama_client = OllamaClient(model=model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.qa_pairs = []
        self.logger = logging.getLogger(__name__)
        
        # 품질 검증기 초기화
        self.quality_validator = QAQualityValidator()
        
        # 질문 유형 정의
        self.question_types = [
            "개념 설명", "실제 적용", "요건/효과", 
            "비교/차이", "절차", "예시", "주의사항",
            "법적 근거", "실무 적용", "예외 사항"
        ]
        
        # 품질 필터링 기준
        self.quality_criteria = {
            "min_question_length": 10,
            "max_question_length": 200,
            "min_answer_length": 20,
            "max_answer_length": 1000,
            "min_quality_score": 0.6
        }
    
    def generate_law_qa_pairs(self, law_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """법령 데이터에서 LLM 기반 Q&A 쌍 생성"""
        qa_pairs = []
        
        try:
            law_name = law_data.get('law_name', '')
            articles = law_data.get('articles', [])
            cleaned_content = law_data.get('cleaned_content', '')
            
            if not law_name:
                return qa_pairs
            
            self.logger.info(f"법령 '{law_name}' 처리 중... (조문 {len(articles)}개)")
            
            # 법령 전체 정의 Q&A 생성
            if cleaned_content:
                definition_qa = self._generate_law_definition_qa(law_name, cleaned_content)
                qa_pairs.extend(definition_qa)
            
            # 각 조문별 Q&A 생성
            for article in articles[:10]:  # 처음 10개 조문만 처리
                article_qa = self._generate_article_qa(law_name, article)
                qa_pairs.extend(article_qa)
            
            # 품질 검증 및 필터링
            filtered_qa = self._filter_qa_pairs(qa_pairs)
            
            self.logger.info(f"법령 '{law_name}'에서 {len(filtered_qa)}개 Q&A 생성")
            return filtered_qa
            
        except Exception as e:
            self.logger.error(f"법령 Q&A 생성 중 오류: {e}")
            return []
    
    def generate_precedent_qa_pairs(self, precedent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """판례 데이터에서 LLM 기반 Q&A 쌍 생성"""
        qa_pairs = []
        
        try:
            case_name = precedent_data.get('case_name', '')
            issue = precedent_data.get('issue', '')
            reasoning = precedent_data.get('reasoning', '')
            conclusion = precedent_data.get('conclusion', '')
            court = precedent_data.get('court', '')
            
            if not case_name:
                return qa_pairs
            
            self.logger.info(f"판례 '{case_name}' 처리 중...")
            
            # 판례 컨텍스트 구성
            context = f"""
            사건명: {case_name}
            법원: {court}
            쟁점: {issue}
            판결 요지: {reasoning[:500] if reasoning else ''}
            결론: {conclusion[:300] if conclusion else ''}
            """
            
            # 다양한 관점에서 Q&A 생성
            precedent_qa = self._generate_precedent_qa(context, case_name)
            qa_pairs.extend(precedent_qa)
            
            # 품질 검증 및 필터링
            filtered_qa = self._filter_qa_pairs(qa_pairs)
            
            self.logger.info(f"판례 '{case_name}'에서 {len(filtered_qa)}개 Q&A 생성")
            return filtered_qa
            
        except Exception as e:
            self.logger.error(f"판례 Q&A 생성 중 오류: {e}")
            return []
    
    def generate_constitutional_qa_pairs(self, decision_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """헌재결정례 데이터에서 LLM 기반 Q&A 쌍 생성"""
        qa_pairs = []
        
        try:
            case_name = decision_data.get('case_name', '')
            issue = decision_data.get('issue', '')
            reasoning = decision_data.get('reasoning', '')
            decision_type = decision_data.get('decision_type', '')
            
            if not case_name:
                return qa_pairs
            
            self.logger.info(f"헌재결정례 '{case_name}' 처리 중...")
            
            # 헌재결정례 컨텍스트 구성
            context = f"""
            사건명: {case_name}
            결정 유형: {decision_type}
            헌법적 쟁점: {issue}
            헌법재판소 판단: {reasoning[:500] if reasoning else ''}
            """
            
            # 헌법적 관점에서 Q&A 생성
            constitutional_qa = self._generate_constitutional_qa(context, case_name)
            qa_pairs.extend(constitutional_qa)
            
            # 품질 검증 및 필터링
            filtered_qa = self._filter_qa_pairs(qa_pairs)
            
            self.logger.info(f"헌재결정례 '{case_name}'에서 {len(filtered_qa)}개 Q&A 생성")
            return filtered_qa
            
        except Exception as e:
            self.logger.error(f"헌재결정례 Q&A 생성 중 오류: {e}")
            return []
    
    def _generate_law_definition_qa(self, law_name: str, content: str) -> List[Dict[str, Any]]:
        """법령 정의 기반 Q&A 생성"""
        context = f"""
        법령명: {law_name}
        법령 내용: {content[:800]}
        """
        
        qa_pairs = self.ollama_client.generate_qa_pairs(
            context=context,
            qa_count=3,
            question_types=["개념 설명", "목적", "적용 범위"],
            temperature=self.temperature
        )
        
        # 메타데이터 추가
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
        """조문 기반 Q&A 생성"""
        article_number = article.get('article_number', '')
        content = article.get('content', '')
        title = article.get('title', '')
        
        if not content and not title:
            return []
        
        context = f"""
        법령명: {law_name}
        조문 번호: 제{article_number}조
        조문 제목: {title}
        조문 내용: {content[:600]}
        """
        
        qa_pairs = self.ollama_client.generate_qa_pairs(
            context=context,
            qa_count=2,
            question_types=["실제 적용", "요건/효과", "절차", "주의사항"],
            temperature=self.temperature
        )
        
        # 메타데이터 추가
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
        """판례 기반 Q&A 생성"""
        qa_pairs = self.ollama_client.generate_qa_pairs(
            context=context,
            qa_count=3,
            question_types=["실무 적용", "시사점", "예방 조치", "유사 사례"],
            temperature=self.temperature
        )
        
        # 메타데이터 추가
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
        """헌재결정례 기반 Q&A 생성"""
        qa_pairs = self.ollama_client.generate_qa_pairs(
            context=context,
            qa_count=2,
            question_types=["헌법적 의미", "기본권", "법적 효과"],
            temperature=self.temperature
        )
        
        # 메타데이터 추가
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
        """Q&A 쌍 품질 필터링 (개선된 품질 검증기 사용)"""
        filtered_pairs = []
        
        for qa in qa_pairs:
            # 품질 검증기로 검증
            validation_result = self.quality_validator.validate_qa_pair(qa)
            
            if validation_result['is_valid']:
                # 검증된 품질 점수와 신뢰도 업데이트
                qa['quality_score'] = validation_result['quality_score']
                qa['confidence'] = validation_result['confidence']
                qa['validation_issues'] = validation_result['issues']
                qa['validation_suggestions'] = validation_result['suggestions']
                
                filtered_pairs.append(qa)
            else:
                self.logger.debug(f"Q&A 필터링됨: {validation_result['issues']}")
        
        return filtered_pairs
    
    def _calculate_quality_score(self, qa_pair: Dict[str, Any]) -> float:
        """Q&A 품질 점수 계산"""
        score = 0.0
        
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        # 기본 점수
        score += 0.2
        
        # 질문 품질 점수
        if 15 <= len(question) <= 100:
            score += 0.25
        elif 100 < len(question) <= 150:
            score += 0.15
        
        # 답변 품질 점수
        if 30 <= len(answer) <= 400:
            score += 0.3
        elif 400 < len(answer) <= 600:
            score += 0.2
        
        # 질문 유형 점수
        question_type = qa_pair.get('type', '')
        if question_type in ['실제 적용', '실무 적용', '예시']:
            score += 0.15
        elif question_type in ['개념 설명', '요건/효과']:
            score += 0.1
        
        # 신뢰도 점수
        confidence = qa_pair.get('confidence', 0.5)
        score += confidence * 0.1
        
        return min(score, 1.0)
    
    def _remove_duplicates(self):
        """중복 Q&A 제거"""
        self.logger.info("중복 Q&A 제거 중...")
        
        # 중복 감지
        duplicate_groups = self.quality_validator.detect_duplicates(self.qa_pairs)
        
        removed_count = 0
        for group in duplicate_groups:
            if len(group) > 1:
                # 그룹 내에서 가장 높은 품질 점수를 가진 Q&A만 유지
                best_qa = max(group, key=lambda i: self.qa_pairs[i].get('quality_score', 0))
                
                # 나머지 제거
                for i in sorted(group, reverse=True):
                    if i != best_qa:
                        del self.qa_pairs[i]
                        removed_count += 1
        
        self.logger.info(f"중복 제거 완료: {removed_count}개 제거, 남은 Q&A: {len(self.qa_pairs)}개")
    
    def generate_dataset(
        self, 
        data_dir: str = "data/processed", 
        output_dir: str = "data/qa_dataset/llm_generated",
        data_types: List[str] = None,
        max_items_per_type: int = 50
    ) -> bool:
        """전체 LLM 기반 Q&A 데이터셋 생성"""
        try:
            data_path = Path(data_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if data_types is None:
                data_types = ['laws', 'precedents', 'constitutional_decisions']
            
            self.logger.info("LLM 기반 Q&A 데이터셋 생성 시작...")
            
            total_generated = 0
            
            for data_type in data_types:
                self.logger.info(f"{data_type} 데이터 처리 중...")
                
                data_files = list(data_path.glob(f"{data_type}/*.json"))
                processed_count = 0
                type_generated = 0
                
                for file_path in data_files:
                    if processed_count >= max_items_per_type:
                        break
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 데이터가 배열인 경우 각 항목별로 처리
                        if isinstance(data, list):
                            for item in data:
                                if processed_count >= max_items_per_type:
                                    break
                                
                                if not isinstance(item, dict):
                                    continue
                                
                                # 데이터 타입에 따른 Q&A 생성
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
                                
                                # 진행 상황 로깅
                                if processed_count % 10 == 0:
                                    self.logger.info(f"{data_type}: {processed_count}개 항목 처리 완료, 현재 Q&A: {len(self.qa_pairs)}개")
                        else:
                            # 단일 객체인 경우
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
                        self.logger.error(f"파일 처리 중 오류 {file_path}: {e}")
                        continue
                
                self.logger.info(f"{data_type} 처리 완료: {processed_count}개 항목, {type_generated}개 Q&A 생성")
                total_generated += type_generated
            
            # 중복 제거
            self._remove_duplicates()
            
            # 품질 점수별 정렬
            self.qa_pairs.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # 데이터셋 저장
            self._save_dataset(output_path)
            
            # 통계 생성
            self._generate_statistics(output_path)
            
            self.logger.info(f"LLM 기반 Q&A 데이터셋 생성 완료: {len(self.qa_pairs)}개 쌍")
            return True
            
        except Exception as e:
            self.logger.error(f"데이터셋 생성 중 오류: {e}")
            return False
    
    def _save_dataset(self, output_path: Path):
        """데이터셋 저장"""
        # 전체 데이터셋 저장
        with open(output_path / "llm_qa_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=2)
        
        # 품질별 분할 저장
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
        """통계 정보 생성"""
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
        
        # 소스별 분포
        for qa in self.qa_pairs:
            source = qa.get('source', 'unknown')
            stats['source_distribution'][source] = stats['source_distribution'].get(source, 0) + 1
        
        # 난이도별 분포
        for qa in self.qa_pairs:
            difficulty = qa.get('difficulty', 'unknown')
            stats['difficulty_distribution'][difficulty] = stats['difficulty_distribution'].get(difficulty, 0) + 1
        
        # 질문 유형별 분포
        for qa in self.qa_pairs:
            q_type = qa.get('type', 'unknown')
            stats['question_type_distribution'][q_type] = stats['question_type_distribution'].get(q_type, 0) + 1
        
        with open(output_path / "llm_qa_dataset_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    """테스트 함수"""
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    try:
        # LLM Q&A 생성기 생성
        generator = LLMQAGenerator()
        
        # 소규모 테스트 데이터셋 생성
        success = generator.generate_dataset(
            data_dir="data/processed",
            output_dir="data/qa_dataset/llm_test",
            data_types=['laws'],
            max_items_per_type=5
        )
        
        if success:
            print("✅ LLM 기반 Q&A 데이터셋 생성 성공")
        else:
            print("❌ LLM 기반 Q&A 데이터셋 생성 실패")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
