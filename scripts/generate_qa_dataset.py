#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q&A 데이터셋 생성 스크립트

수집된 법률 데이터를 기반으로 Q&A 데이터셋을 생성합니다.
- 자동 Q&A 생성 파이프라인
- 법률 전문가 검토용 데이터 준비
- 품질 점수 매기기
- 데이터셋 최종 검증
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

from source.data.data_processor import LegalDataProcessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/generate_qa_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Q&A 생성 템플릿
QA_TEMPLATES = {
    'law_definition': [
        "{law_name}이란 무엇인가요?",
        "{law_name}의 정의를 설명해주세요.",
        "{law_name}의 목적은 무엇인가요?",
        "{law_name}의 적용 범위는 어떻게 되나요?"
    ],
    'law_article': [
        "{law_name} 제{article}조의 내용을 설명해주세요.",
        "{law_name} 제{article}조에서 규정하는 내용은 무엇인가요?",
        "{law_name} 제{article}조의 요건은 무엇인가요?",
        "{law_name} 제{article}조의 효과는 무엇인가요?"
    ],
    'precedent_issue': [
        "{case_name} 사건의 쟁점은 무엇인가요?",
        "{case_name} 사건에서 다룬 문제는 무엇인가요?",
        "{case_name} 사건의 핵심 쟁점을 설명해주세요.",
        "{case_name} 사건의 법적 쟁점은 무엇인가요?"
    ],
    'precedent_decision': [
        "{case_name} 사건의 판결 내용은 무엇인가요?",
        "{case_name} 사건에서 법원이 내린 결론은 무엇인가요?",
        "{case_name} 사건의 판결 요지는 무엇인가요?",
        "{case_name} 사건의 법원 판단을 설명해주세요."
    ],
    'constitutional_issue': [
        "{case_name} 사건의 헌법적 쟁점은 무엇인가요?",
        "{case_name} 사건에서 다룬 기본권 문제는 무엇인가요?",
        "{case_name} 사건의 헌법재판소 판단 대상은 무엇인가요?",
        "{case_name} 사건의 헌법적 의미는 무엇인가요?"
    ],
    'constitutional_decision': [
        "{case_name} 사건의 헌법재판소 결정은 무엇인가요?",
        "{case_name} 사건에서 헌법재판소가 내린 결론은 무엇인가요?",
        "{case_name} 사건의 헌법재판소 판단을 설명해주세요.",
        "{case_name} 사건의 헌법적 판단은 무엇인가요?"
    ],
    'interpretation_question': [
        "{topic}에 대한 법령해석은 어떻게 되나요?",
        "{topic}의 법적 해석 기준은 무엇인가요?",
        "{topic}에 대한 중앙부처의 해석은 무엇인가요?",
        "{topic}의 법령 적용 기준은 무엇인가요?"
    ],
    'general_legal': [
        "{keyword}에 대한 법적 근거는 무엇인가요?",
        "{keyword}의 법적 요건은 무엇인가요?",
        "{keyword}의 법적 효과는 무엇인가요?",
        "{keyword}에 대한 법적 해석은 어떻게 되나요?"
    ]
}

# 답변 생성 템플릿
ANSWER_TEMPLATES = {
    'law_definition': [
        "{law_name}은 {definition}을 목적으로 하는 법률입니다.",
        "{law_name}는 {definition}에 관한 사항을 규정한 법률입니다.",
        "{law_name}의 목적은 {definition}입니다.",
        "{law_name}은 {definition}을 규정하는 법률입니다."
    ],
    'law_article': [
        "{law_name} 제{article}조에 따르면, {content}입니다.",
        "제{article}조에서는 {content}라고 규정하고 있습니다.",
        "{law_name} 제{article}조의 내용은 {content}입니다.",
        "제{article}조에 규정된 내용은 {content}입니다."
    ],
    'precedent_issue': [
        "{case_name} 사건의 쟁점은 {issue}입니다.",
        "이 사건에서 다룬 문제는 {issue}입니다.",
        "법원이 판단한 쟁점은 {issue}입니다.",
        "사건의 핵심 쟁점은 {issue}입니다."
    ],
    'precedent_decision': [
        "{case_name} 사건에서 법원은 {decision}라고 판단했습니다.",
        "법원의 판결 내용은 {decision}입니다.",
        "이 사건의 판결 요지는 {decision}입니다.",
        "법원이 내린 결론은 {decision}입니다."
    ],
    'constitutional_issue': [
        "{case_name} 사건의 헌법적 쟁점은 {issue}입니다.",
        "이 사건에서 다룬 기본권 문제는 {issue}입니다.",
        "헌법재판소가 판단한 대상은 {issue}입니다.",
        "사건의 헌법적 의미는 {issue}입니다."
    ],
    'constitutional_decision': [
        "{case_name} 사건에서 헌법재판소는 {decision}라고 결정했습니다.",
        "헌법재판소의 결정 내용은 {decision}입니다.",
        "이 사건의 헌법재판소 판단은 {decision}입니다.",
        "헌법재판소가 내린 결론은 {decision}입니다."
    ],
    'interpretation_question': [
        "{topic}에 대한 법령해석은 {interpretation}입니다.",
        "{topic}의 법적 해석 기준은 {interpretation}입니다.",
        "중앙부처의 해석에 따르면 {interpretation}입니다.",
        "{topic}의 법령 적용 기준은 {interpretation}입니다."
    ],
    'general_legal': [
        "{keyword}에 대한 법적 근거는 {basis}입니다.",
        "{keyword}의 법적 요건은 {requirement}입니다.",
        "{keyword}의 법적 효과는 {effect}입니다.",
        "{keyword}에 대한 법적 해석은 {interpretation}입니다."
    ]
}


class QADatasetGenerator:
    """Q&A 데이터셋 생성 클래스"""
    
    def __init__(self):
        self.processor = LegalDataProcessor()
        self.qa_pairs = []
        self.logger = logging.getLogger(__name__)
        
    def generate_law_qa_pairs(self, law_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """법령 데이터에서 Q&A 쌍 생성"""
        qa_pairs = []
        
        try:
            law_name = law_data.get('law_name', '')
            articles = law_data.get('articles', [])
            cleaned_content = law_data.get('cleaned_content', '')
            
            # 1. 법령 정의 관련 Q&A
            if law_name and cleaned_content:
                # 법령 정의 추출 (첫 번째 문장 또는 첫 번째 조문)
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
            
            # 2. 조문별 Q&A
            for article in articles:
                article_number = article.get('article_number', '')
                content = article.get('content', '')
                title = article.get('title', '')
                
                if article_number and content:
                    # 조문 내용 Q&A
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
                    
                    # 조문 제목 Q&A
                    if title:
                        question = f"{law_name} 제{article_number}조의 제목은 무엇인가요?"
                        answer = f"{law_name} 제{article_number}조의 제목은 '{title}'입니다."
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'source': 'law_article_title',
                            'law_name': law_name,
                            'article_number': article_number,
                            'confidence': 0.95,
                            'difficulty': 'easy'
                        })
            
            # 3. 키워드 기반 Q&A
            entities = law_data.get('entities', {})
            keywords = entities.get('keywords', [])
            for keyword in keywords[:5]:  # 상위 5개 키워드만 사용
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
        """판례 데이터에서 Q&A 쌍 생성"""
        qa_pairs = []
        
        try:
            case_name = precedent_data.get('case_name', '')
            issue = precedent_data.get('issue', '')
            reasoning = precedent_data.get('reasoning', '')
            conclusion = precedent_data.get('conclusion', '')
            court = precedent_data.get('court', '')
            
            # 1. 쟁점 관련 Q&A
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
            
            # 2. 판결 내용 Q&A
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
            
            # 3. 결론 Q&A
            if case_name and conclusion:
                question = f"{case_name} 사건의 결론은 무엇인가요?"
                answer = f"{case_name} 사건에서 {conclusion}라고 판단했습니다."
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
        """헌재결정례 데이터에서 Q&A 쌍 생성"""
        qa_pairs = []
        
        try:
            case_name = decision_data.get('case_name', '')
            issue = decision_data.get('issue', '')
            reasoning = decision_data.get('reasoning', '')
            conclusion = decision_data.get('conclusion', '')
            decision_type = decision_data.get('decision_type', '')
            
            # 1. 헌법적 쟁점 Q&A
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
            
            # 2. 헌법재판소 결정 Q&A
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
        """법령해석례 데이터에서 Q&A 쌍 생성"""
        qa_pairs = []
        
        try:
            case_name = interpretation_data.get('case_name', '')
            issue = interpretation_data.get('issue', '')
            reasoning = interpretation_data.get('reasoning', '')
            topic = interpretation_data.get('topic', '')
            ministry = interpretation_data.get('ministry', '')
            
            # 1. 해석 주제 Q&A
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
            
            # 2. 구체적 해석 Q&A
            if case_name and reasoning:
                question = f"{topic}에 대한 {ministry}의 해석은 무엇인가요?"
                answer = f"{ministry}의 해석에 따르면 {reasoning[:200]}...입니다."
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
        """법령 정의 추출"""
        # 첫 번째 문장에서 정의 추출
        sentences = content.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 20:  # 충분한 길이의 문장만
                return first_sentence
        return ""
    
    def _generate_keyword_answer(self, keyword: str, law_name: str, content: str) -> str:
        """키워드 기반 답변 생성"""
        # 키워드가 포함된 문장 찾기
        sentences = content.split('.')
        for sentence in sentences:
            if keyword in sentence and len(sentence) > 20:
                return f"{law_name}에 따르면 {sentence.strip()}입니다."
        return ""
    
    def calculate_quality_score(self, qa_pair: Dict[str, Any]) -> float:
        """Q&A 쌍의 품질 점수 계산"""
        score = 0.0
        
        # 기본 점수
        score += 0.3
        
        # 질문 길이 점수
        question_length = len(qa_pair.get('question', ''))
        if 10 <= question_length <= 100:
            score += 0.2
        elif 100 < question_length <= 200:
            score += 0.1
        
        # 답변 길이 점수
        answer_length = len(qa_pair.get('answer', ''))
        if 20 <= answer_length <= 500:
            score += 0.3
        elif 500 < answer_length <= 1000:
            score += 0.2
        
        # 신뢰도 점수
        confidence = qa_pair.get('confidence', 0.5)
        score += confidence * 0.2
        
        return min(score, 1.0)
    
    def generate_dataset(self, data_dir: str = "data/processed", output_dir: str = "data/qa_dataset") -> bool:
        """전체 Q&A 데이터셋 생성"""
        try:
            data_path = Path(data_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Q&A 데이터셋 생성 시작...")
            
            # 각 데이터 타입별 처리
            data_types = ['laws', 'precedents', 'constitutional_decisions', 'legal_interpretations']
            
            for data_type in data_types:
                self.logger.info(f"{data_type} 데이터 처리 중...")
                
                data_files = list(data_path.glob(f"{data_type}/*.json"))
                for file_path in data_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 데이터 타입에 따른 Q&A 생성
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
                        
                        # 품질 점수 계산
                        for qa_pair in qa_pairs:
                            qa_pair['quality_score'] = self.calculate_quality_score(qa_pair)
                            qa_pair['generated_at'] = datetime.now().isoformat()
                        
                        self.qa_pairs.extend(qa_pairs)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
                        continue
            
            # 품질 점수별 정렬
            self.qa_pairs.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # 데이터셋 저장
            self._save_dataset(output_path)
            
            # 통계 생성
            self._generate_statistics(output_path)
            
            self.logger.info(f"Q&A 데이터셋 생성 완료: {len(self.qa_pairs)}개 쌍")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating dataset: {e}")
            return False
    
    def _save_dataset(self, output_path: Path):
        """데이터셋 저장"""
        # 전체 데이터셋 저장
        with open(output_path / "qa_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=2)
        
        # 품질별 분할 저장
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
        """통계 정보 생성"""
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
        
        # 소스별 분포
        for qa in self.qa_pairs:
            source = qa.get('source', 'unknown')
            stats['source_distribution'][source] = stats['source_distribution'].get(source, 0) + 1
        
        # 난이도별 분포
        for qa in self.qa_pairs:
            difficulty = qa.get('difficulty', 'unknown')
            stats['difficulty_distribution'][difficulty] = stats['difficulty_distribution'].get(difficulty, 0) + 1
        
        with open(output_path / "qa_dataset_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    """메인 함수"""
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Q&A 데이터셋 생성
    generator = QADatasetGenerator()
    success = generator.generate_dataset()
    
    if success:
        logger.info("Q&A 데이터셋 생성이 완료되었습니다.")
    else:
        logger.error("Q&A 데이터셋 생성에 실패했습니다.")


if __name__ == "__main__":
    main()
