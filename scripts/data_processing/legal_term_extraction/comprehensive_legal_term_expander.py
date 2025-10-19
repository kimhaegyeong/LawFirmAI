# -*- coding: utf-8 -*-
"""
전체 법률 용어 확장 및 저장 시스템
배치 처리, 품질 관리, 데이터베이스 저장 기능 포함
"""

import os
import sys
import json
import logging
import sqlite3
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 한글 출력을 위한 환경 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['GRPC_PYTHON_LOG_VERBOSITY'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Google API 관련 경고 비활성화
import warnings
warnings.filterwarnings('ignore')

# 환경변수 로드
env_path = r"D:\project\LawFirmAI\LawFirmAI\.env"
load_dotenv(env_path)

logger = logging.getLogger(__name__)

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class LegalTermExpansion(BaseModel):
    """법률 용어 확장 결과 모델"""
    synonyms: List[str] = Field(description="동의어 목록")
    related_terms: List[str] = Field(description="관련 용어 목록")
    precedent_keywords: List[str] = Field(description="판례 키워드 목록")
    confidence: float = Field(description="신뢰도 점수 (0.0-1.0)")


class LegalTermBatchExpander:
    """법률 용어 배치 확장기"""
    
    def __init__(self, 
                 model_name: str = "gemini-2.0-flash-exp",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 batch_size: int = 10,
                 delay_between_batches: float = 2.0):
        """배치 확장기 초기화"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.CRITICAL)
        
        # 환경변수에서 API 키 확인
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key or api_key == "your_google_api_key_here":
            raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일에 유효한 Google API 키를 설정하세요.")
        
        if not api_key.startswith("AIza"):
            raise ValueError("유효하지 않은 Google API 키 형식입니다. Google API 키는 'AIza'로 시작해야 합니다.")
        
        # Gemini 모델 초기화
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        # 출력 파서 초기화
        self.output_parser = JsonOutputParser(pydantic_object=LegalTermExpansion)
        
        # 프롬프트 템플릿 설정
        self.prompt_template = self._create_prompt_template()
        
        # 법률 도메인별 프롬프트 로드
        self.domain_prompts = self._load_domain_prompts()
        
        # 배치 처리 설정
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        
        self.logger.info(f"LegalTermBatchExpander 초기화 완료: {model_name}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """프롬프트 템플릿 생성"""
        template = """
당신은 한국 법률 전문가입니다. 주어진 법률 용어에 대해 정확하고 전문적인 용어 확장을 수행해주세요.

법률 도메인: {domain}
기본 용어: {base_term}

{domain_context}

다음 형식으로 JSON 응답을 생성해주세요:
{format_instructions}

요구사항:
1. 동의어는 의미가 동일하거나 매우 유사한 용어만 포함
2. 관련 용어는 법률적으로 연관된 개념들
3. 판례 키워드는 실제 판례에서 사용되는 키워드
4. 신뢰도는 생성된 용어들의 정확성을 평가 (0.0-1.0)
5. 모든 용어는 한국어로 작성
6. 각 카테고리당 최대 5개 용어 생성
"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _load_domain_prompts(self) -> Dict[str, str]:
        """법률 도메인별 프롬프트 로드"""
        domain_prompts = {
            "민사법": """
민사법 도메인 특화 지침:
- 계약, 불법행위, 소유권, 상속, 가족관계 등 민사 분쟁 관련 용어
- 손해배상, 계약해지, 소유권이전, 상속분할 등 구체적 법률 행위
- 민법 조문과 연관된 전문 용어 사용
""",
            "형사법": """
형사법 도메인 특화 지침:
- 범죄 유형, 처벌, 소송절차, 관련인물 등 형사 사건 관련 용어
- 살인, 절도, 사기, 강도, 강간 등 구체적 범죄 유형
- 형법 조문과 연관된 전문 용어 사용
""",
            "가족법": """
가족법 도메인 특화 지침:
- 혼인관계, 친자관계, 재산관계 등 가족 관련 법률 용어
- 이혼, 양육권, 친권, 상속 등 구체적 가족법 사안
- 가족법 조문과 연관된 전문 용어 사용
""",
            "상사법": """
상사법 도메인 특화 지침:
- 회사법, 상행위, 어음수표 등 상업 관련 법률 용어
- 주식회사, 유한회사, 상행위, 어음, 수표 등 구체적 상사법 개념
- 상법 조문과 연관된 전문 용어 사용
""",
            "행정법": """
행정법 도메인 특화 지침:
- 행정행위, 행정절차, 행정소송 등 행정 관련 법률 용어
- 행정처분, 행정지도, 행정심판 등 구체적 행정법 개념
- 행정법 조문과 연관된 전문 용어 사용
""",
            "노동법": """
노동법 도메인 특화 지침:
- 근로계약, 임금, 근로시간, 휴가 등 근로 관련 법률 용어
- 해고, 부당해고, 실업급여 등 구체적 노동법 개념
- 근로기준법, 노동조합법 등 관련 법령 용어 사용
""",
            "기타": """
일반 법률 도메인 지침:
- 다양한 법률 분야에 공통적으로 사용되는 용어
- 법률 일반론, 법원, 검사, 변호사 등 법률 제도 관련 용어
- 헌법, 국제법 등 기타 법률 분야 용어
"""
        }
        
        return domain_prompts
    
    def expand_term(self, base_term: str, domain: str = "민사법") -> Dict[str, Any]:
        """단일 용어 확장"""
        try:
            # 도메인 컨텍스트 가져오기
            domain_context = self.domain_prompts.get(domain, self.domain_prompts["기타"])
            
            # 프롬프트 생성
            prompt = self.prompt_template.format_messages(
                domain=domain,
                base_term=base_term,
                domain_context=domain_context,
                format_instructions=self.output_parser.get_format_instructions()
            )
            
            # LLM 호출
            response = self.model.invoke(prompt)
            
            # 응답 파싱
            parsed_response = self.output_parser.parse(response.content)
            
            # 결과 검증 및 정제
            validated_result = self._validate_and_refine_result(parsed_response, base_term)
            
            return validated_result
            
        except Exception as e:
            print(f"용어 확장 중 오류 발생: {e}")
            return {
                "synonyms": [],
                "related_terms": [],
                "precedent_keywords": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _validate_and_refine_result(self, result: Dict[str, Any], base_term: str) -> Dict[str, Any]:
        """결과 검증 및 정제"""
        try:
            validated_result = {
                "synonyms": [],
                "related_terms": [],
                "precedent_keywords": [],
                "confidence": 0.0,
                "expanded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 동의어 검증
            if "synonyms" in result and isinstance(result["synonyms"], list):
                validated_synonyms = []
                for synonym in result["synonyms"]:
                    if self._is_valid_legal_term(synonym) and synonym != base_term:
                        validated_synonyms.append(synonym.strip())
                validated_result["synonyms"] = validated_synonyms[:5]  # 최대 5개
            
            # 관련 용어 검증
            if "related_terms" in result and isinstance(result["related_terms"], list):
                validated_related = []
                for term in result["related_terms"]:
                    if self._is_valid_legal_term(term) and term != base_term:
                        validated_related.append(term.strip())
                validated_result["related_terms"] = validated_related[:5]  # 최대 5개
            
            # 판례 키워드 검증
            if "precedent_keywords" in result and isinstance(result["precedent_keywords"], list):
                validated_keywords = []
                for keyword in result["precedent_keywords"]:
                    if self._is_valid_legal_term(keyword) and keyword != base_term:
                        validated_keywords.append(keyword.strip())
                validated_result["precedent_keywords"] = validated_keywords[:5]  # 최대 5개
            
            # 신뢰도 계산
            confidence = result.get("confidence", 0.0)
            if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
                validated_result["confidence"] = float(confidence)
            else:
                # 자동 신뢰도 계산
                validated_result["confidence"] = self._calculate_confidence(validated_result, base_term)
            
            return validated_result
            
        except Exception as e:
            self.logger.error(f"결과 검증 중 오류 발생: {e}")
            return {
                "synonyms": [],
                "related_terms": [],
                "precedent_keywords": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _is_valid_legal_term(self, term: str) -> bool:
        """법률 용어 유효성 검증"""
        if not term or not isinstance(term, str):
            return False
        
        term = term.strip()
        
        # 기본 길이 검증
        if len(term) < 2 or len(term) > 20:
            return False
        
        # 한글 포함 검증
        if not re.search(r'[가-힣]', term):
            return False
        
        # 특수문자 제한
        if re.search(r'[^\w가-힣\s]', term):
            return False
        
        # 법률 도메인 키워드 검증
        legal_keywords = [
            '법', '권', '책임', '손해', '계약', '소송', '처벌', '제재',
            '배상', '보상', '청구', '제기', '해지', '위반', '침해',
            '절차', '신청', '심리', '판결', '항소', '상고',
            '법원', '검사', '변호사', '피고인', '원고', '피고'
        ]
        
        # 법률 키워드가 포함되어 있거나 법률 관련 용어인지 확인
        has_legal_keyword = any(keyword in term for keyword in legal_keywords)
        is_legal_concept = re.search(r'[가-힣]{2,6}(?:법|권|책임|손해|계약|소송)', term)
        
        return has_legal_keyword or is_legal_concept or len(term) >= 3
    
    def _calculate_confidence(self, result: Dict[str, Any], base_term: str) -> float:
        """신뢰도 자동 계산"""
        try:
            total_terms = len(result.get("synonyms", [])) + len(result.get("related_terms", [])) + len(result.get("precedent_keywords", []))
            
            if total_terms == 0:
                return 0.0
            
            # 기본 점수
            base_score = 0.5
            
            # 용어 수에 따른 점수
            term_count_score = min(total_terms / 15, 0.3)  # 최대 0.3점
            
            # 용어 품질 점수
            quality_score = 0.0
            for category in ["synonyms", "related_terms", "precedent_keywords"]:
                for term in result.get(category, []):
                    if self._is_valid_legal_term(term):
                        quality_score += 0.1
            
            quality_score = min(quality_score, 0.2)  # 최대 0.2점
            
            final_score = base_score + term_count_score + quality_score
            return min(final_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"신뢰도 계산 중 오류 발생: {e}")
            return 0.5
    
    def expand_all_terms(self, terms: List[str], domain: str = "민사법") -> Dict[str, Any]:
        """전체 용어를 배치로 확장"""
        try:
            self.logger.info(f"전체 용어 확장 시작: {len(terms)}개 용어 ({domain})")
            
            results = {}
            successful_expansions = 0
            failed_expansions = 0
            
            # 배치 단위로 처리
            for i in range(0, len(terms), self.batch_size):
                batch_terms = terms[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (len(terms) + self.batch_size - 1) // self.batch_size
                
                self.logger.info(f"배치 {batch_num}/{total_batches} 처리 중: {len(batch_terms)}개 용어")
                
                # 배치 내에서 순차 처리
                for j, term in enumerate(batch_terms):
                    try:
                        self.logger.info(f"용어 확장: {term} ({i+j+1}/{len(terms)})")
                        
                        expansion_result = self.expand_term(term, domain)
                        results[term] = expansion_result
                        
                        if "error" not in expansion_result:
                            successful_expansions += 1
                        else:
                            failed_expansions += 1
                        
                        # 진행률 로깅
                        progress = (i + j + 1) / len(terms) * 100
                        self.logger.info(f"진행률: {progress:.1f}% ({i+j+1}/{len(terms)})")
                        
                    except Exception as e:
                        self.logger.error(f"용어 '{term}' 확장 중 오류: {e}")
                        results[term] = {
                            "synonyms": [],
                            "related_terms": [],
                            "precedent_keywords": [],
                            "confidence": 0.0,
                            "error": str(e)
                        }
                        failed_expansions += 1
                
                # 배치 간 지연
                if i + self.batch_size < len(terms):
                    self.logger.info(f"배치 간 지연: {self.delay_between_batches}초")
                    time.sleep(self.delay_between_batches)
                
                # API 할당량 관리를 위한 추가 지연
                if i + self.batch_size < len(terms):
                    self.logger.info("API 할당량 관리를 위한 추가 2초 지연...")
                    time.sleep(2)
            
            # 전체 결과 요약
            batch_summary = {
                "total_terms": len(terms),
                "successful_expansions": successful_expansions,
                "failed_expansions": failed_expansions,
                "success_rate": successful_expansions / len(terms) if terms else 0,
                "domain": domain,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": results
            }
            
            self.logger.info(f"전체 확장 완료: {successful_expansions}/{len(terms)} 성공 ({batch_summary['success_rate']:.1%})")
            
            return batch_summary
            
        except Exception as e:
            self.logger.error(f"전체 확장 중 오류 발생: {e}")
            return {
                "total_terms": len(terms),
                "successful_expansions": 0,
                "failed_expansions": len(terms),
                "success_rate": 0.0,
                "error": str(e),
                "results": {}
            }
    
    def save_progress(self, results: Dict[str, Any], checkpoint_file: str):
        """진행 상황 저장"""
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"진행 상황 저장 완료: {checkpoint_file}")
        except Exception as e:
            self.logger.error(f"진행 상황 저장 중 오류: {e}")
    
    def resume_from_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """중단된 작업 재개"""
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                self.logger.info(f"체크포인트에서 복구: {checkpoint_file}")
                return results
            else:
                self.logger.info(f"체크포인트 파일이 존재하지 않음: {checkpoint_file}")
                return {}
        except Exception as e:
            self.logger.error(f"체크포인트 복구 중 오류: {e}")
            return {}


class QualityValidator:
    """용어 확장 품질 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.CRITICAL)
    
    def validate_expansion_quality(self, original_term: str, expansion_result: Dict) -> float:
        """확장 결과 품질 검증"""
        try:
            quality_score = 0.0
            
            # 기본 점수
            base_score = 0.3
            
            # 용어 수 점수
            total_terms = len(expansion_result.get("synonyms", [])) + \
                         len(expansion_result.get("related_terms", [])) + \
                         len(expansion_result.get("precedent_keywords", []))
            
            term_count_score = min(total_terms / 15, 0.3)  # 최대 0.3점
            
            # 용어 품질 점수
            quality_score = 0.0
            for category in ["synonyms", "related_terms", "precedent_keywords"]:
                for term in expansion_result.get(category, []):
                    if self._is_high_quality_term(term):
                        quality_score += 0.1
            
            quality_score = min(quality_score, 0.4)  # 최대 0.4점
            
            final_score = base_score + term_count_score + quality_score
            return min(final_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"품질 검증 중 오류: {e}")
            return 0.5
    
    def _is_high_quality_term(self, term: str) -> bool:
        """고품질 용어 판단"""
        if not term or len(term) < 2:
            return False
        
        # 법률 전문 용어 키워드
        legal_keywords = [
            '법', '권', '책임', '손해', '계약', '소송', '처벌', '제재',
            '배상', '보상', '청구', '제기', '해지', '위반', '침해',
            '절차', '신청', '심리', '판결', '항소', '상고'
        ]
        
        return any(keyword in term for keyword in legal_keywords)
    
    def filter_low_quality_terms(self, results: Dict[str, Any], threshold: float = 0.7) -> Dict[str, Any]:
        """저품질 용어 필터링"""
        try:
            filtered_results = {}
            
            for term, expansion_result in results.items():
                quality_score = self.validate_expansion_quality(term, expansion_result)
                
                if quality_score >= threshold:
                    expansion_result["quality_score"] = quality_score
                    filtered_results[term] = expansion_result
                else:
                    self.logger.info(f"저품질 용어 제외: {term} (품질점수: {quality_score:.2f})")
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"품질 필터링 중 오류: {e}")
            return results
    
    def generate_quality_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """품질 보고서 생성"""
        try:
            total_terms = len(results)
            quality_scores = []
            
            for term, expansion_result in results.items():
                quality_score = self.validate_expansion_quality(term, expansion_result)
                quality_scores.append(quality_score)
            
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                min_quality = min(quality_scores)
                max_quality = max(quality_scores)
            else:
                avg_quality = min_quality = max_quality = 0.0
            
            report = {
                "total_terms": total_terms,
                "average_quality": avg_quality,
                "min_quality": min_quality,
                "max_quality": max_quality,
                "high_quality_terms": len([s for s in quality_scores if s >= 0.8]),
                "medium_quality_terms": len([s for s in quality_scores if 0.6 <= s < 0.8]),
                "low_quality_terms": len([s for s in quality_scores if s < 0.6]),
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"품질 보고서 생성 중 오류: {e}")
            return {}


class LegalTermDatabase:
    """법률 용어 데이터베이스 관리자"""
    
    def __init__(self, db_path: str = "data/legal_terms.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.CRITICAL)
        self._create_tables()
    
    def _create_tables(self):
        """데이터베이스 테이블 생성"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 용어 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS legal_terms (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        term TEXT UNIQUE NOT NULL,
                        domain TEXT NOT NULL,
                        synonyms TEXT,
                        related_terms TEXT,
                        precedent_keywords TEXT,
                        confidence REAL,
                        quality_score REAL,
                        expanded_at TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 확장 이력 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS expansion_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        term TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        expansion_result TEXT,
                        success BOOLEAN,
                        error_message TEXT,
                        processed_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info("데이터베이스 테이블 생성 완료")
                
        except Exception as e:
            self.logger.error(f"데이터베이스 테이블 생성 중 오류: {e}")
    
    def save_expanded_terms(self, terms: Dict[str, Any], domain: str = "민사법"):
        """확장된 용어 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for term, expansion_result in terms.items():
                    # 기존 용어 확인
                    cursor.execute("SELECT id FROM legal_terms WHERE term = ?", (term,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # 기존 용어 업데이트
                        cursor.execute('''
                            UPDATE legal_terms SET
                                synonyms = ?, related_terms = ?, precedent_keywords = ?,
                                confidence = ?, quality_score = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE term = ?
                        ''', (
                            json.dumps(expansion_result.get("synonyms", []), ensure_ascii=False),
                            json.dumps(expansion_result.get("related_terms", []), ensure_ascii=False),
                            json.dumps(expansion_result.get("precedent_keywords", []), ensure_ascii=False),
                            expansion_result.get("confidence", 0.0),
                            expansion_result.get("quality_score", 0.0),
                            term
                        ))
                    else:
                        # 새 용어 삽입
                        cursor.execute('''
                            INSERT INTO legal_terms 
                            (term, domain, synonyms, related_terms, precedent_keywords, 
                             confidence, quality_score, expanded_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            term,
                            domain,
                            json.dumps(expansion_result.get("synonyms", []), ensure_ascii=False),
                            json.dumps(expansion_result.get("related_terms", []), ensure_ascii=False),
                            json.dumps(expansion_result.get("precedent_keywords", []), ensure_ascii=False),
                            expansion_result.get("confidence", 0.0),
                            expansion_result.get("quality_score", 0.0),
                            expansion_result.get("expanded_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        ))
                
                conn.commit()
                self.logger.info(f"용어 저장 완료: {len(terms)}개")
                
        except Exception as e:
            self.logger.error(f"용어 저장 중 오류: {e}")
    
    def get_terms_by_domain(self, domain: str) -> List[Dict]:
        """도메인별 용어 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM legal_terms WHERE domain = ?", (domain,))
                rows = cursor.fetchall()
                
                terms = []
                for row in rows:
                    term_data = {
                        "term": row["term"],
                        "domain": row["domain"],
                        "synonyms": json.loads(row["synonyms"]) if row["synonyms"] else [],
                        "related_terms": json.loads(row["related_terms"]) if row["related_terms"] else [],
                        "precedent_keywords": json.loads(row["precedent_keywords"]) if row["precedent_keywords"] else [],
                        "confidence": row["confidence"],
                        "quality_score": row["quality_score"],
                        "expanded_at": row["expanded_at"]
                    }
                    terms.append(term_data)
                
                return terms
                
        except Exception as e:
            self.logger.error(f"용어 조회 중 오류: {e}")
            return []
    
    def get_all_terms(self) -> List[Dict]:
        """모든 용어 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM legal_terms ORDER BY domain, term")
                rows = cursor.fetchall()
                
                terms = []
                for row in rows:
                    term_data = {
                        "term": row["term"],
                        "domain": row["domain"],
                        "synonyms": json.loads(row["synonyms"]) if row["synonyms"] else [],
                        "related_terms": json.loads(row["related_terms"]) if row["related_terms"] else [],
                        "precedent_keywords": json.loads(row["precedent_keywords"]) if row["precedent_keywords"] else [],
                        "confidence": row["confidence"],
                        "quality_score": row["quality_score"],
                        "expanded_at": row["expanded_at"]
                    }
                    terms.append(term_data)
                
                return terms
                
        except Exception as e:
            self.logger.error(f"전체 용어 조회 중 오류: {e}")
            return []
    
    def update_term_quality(self, term: str, quality_score: float):
        """용어 품질 점수 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE legal_terms SET quality_score = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE term = ?
                ''', (quality_score, term))
                
                conn.commit()
                self.logger.info(f"용어 품질 점수 업데이트: {term} -> {quality_score}")
                
        except Exception as e:
            self.logger.error(f"품질 점수 업데이트 중 오류: {e}")
    
    def export_to_json(self, output_file: str):
        """JSON 파일로 내보내기"""
        try:
            terms = self.get_all_terms()
            
            export_data = {
                "metadata": {
                    "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_terms": len(terms),
                    "domains": list(set(term["domain"] for term in terms))
                },
                "dictionary": {}
            }
            
            for term_data in terms:
                export_data["dictionary"][term_data["term"]] = {
                    "synonyms": term_data["synonyms"],
                    "related_terms": term_data["related_terms"],
                    "precedent_keywords": term_data["precedent_keywords"],
                    "confidence": term_data["confidence"],
                    "quality_score": term_data["quality_score"],
                    "domain": term_data["domain"],
                    "expanded_at": term_data["expanded_at"]
                }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"JSON 내보내기 완료: {output_file}")
            
        except Exception as e:
            self.logger.error(f"JSON 내보내기 중 오류: {e}")


class ProgressMonitor:
    """진행 상황 모니터링"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.CRITICAL)
        self.total_terms = 0
        self.processed_terms = 0
        self.successful_expansions = 0
        self.failed_expansions = 0
        self.start_time = None
    
    def start_monitoring(self, total_terms: int):
        """모니터링 시작"""
        self.total_terms = total_terms
        self.processed_terms = 0
        self.successful_expansions = 0
        self.failed_expansions = 0
        self.start_time = datetime.now()
        self.logger.info(f"모니터링 시작: {total_terms}개 용어")
    
    def update_progress(self, batch_results: Dict[str, Any]):
        """진행 상황 업데이트"""
        try:
            if "results" in batch_results:
                batch_size = len(batch_results["results"])
                self.processed_terms += batch_size
                self.successful_expansions += batch_results.get("successful_expansions", 0)
                self.failed_expansions += batch_results.get("failed_expansions", 0)
                
                progress = self.processed_terms / self.total_terms * 100
                self.logger.info(f"진행률: {progress:.1f}% ({self.processed_terms}/{self.total_terms})")
                
        except Exception as e:
            self.logger.error(f"진행 상황 업데이트 중 오류: {e}")
    
    def generate_progress_report(self) -> Dict[str, Any]:
        """진행 상황 보고서 생성"""
        try:
            if self.start_time:
                elapsed_time = datetime.now() - self.start_time
                elapsed_seconds = elapsed_time.total_seconds()
                
                if self.processed_terms > 0:
                    avg_time_per_term = elapsed_seconds / self.processed_terms
                    remaining_terms = self.total_terms - self.processed_terms
                    estimated_remaining_time = remaining_terms * avg_time_per_term
                else:
                    avg_time_per_term = 0
                    estimated_remaining_time = 0
            else:
                elapsed_seconds = 0
                avg_time_per_term = 0
                estimated_remaining_time = 0
            
            report = {
                "total_terms": self.total_terms,
                "processed_terms": self.processed_terms,
                "successful_expansions": self.successful_expansions,
                "failed_expansions": self.failed_expansions,
                "success_rate": self.successful_expansions / self.processed_terms if self.processed_terms > 0 else 0,
                "progress_percentage": self.processed_terms / self.total_terms * 100 if self.total_terms > 0 else 0,
                "elapsed_time_seconds": elapsed_seconds,
                "average_time_per_term": avg_time_per_term,
                "estimated_remaining_time": estimated_remaining_time,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"진행 상황 보고서 생성 중 오류: {e}")
            return {}


def safe_print(text: str):
    """안전한 한글 출력 함수"""
    try:
        # 파일로 출력하여 한글 문제 해결
        with open('legal_term_expansion_output.txt', 'a', encoding='utf-8') as f:
            f.write(text + '\n')
        
        # 콘솔 출력은 ASCII로 변환하여 깨짐 방지
        try:
            ascii_text = text.encode('ascii', 'ignore').decode('ascii')
            if ascii_text.strip():
                print(ascii_text)
        except:
            print("[한글 출력 - legal_term_expansion_output.txt 파일 참조]")
    except Exception:
        # 기타 오류 시 원본 출력
        print(text)


def main():
    """메인 실행 함수"""
    # 로깅 비활성화
    logging.getLogger().setLevel(logging.CRITICAL)
    
    safe_print("전체 법률 용어 확장 및 저장 시스템")
    safe_print("=" * 50)
    
    try:
        # 시스템 초기화
        expander = LegalTermBatchExpander()
        validator = QualityValidator()
        database = LegalTermDatabase()
        monitor = ProgressMonitor()
        
        # 테스트용 법률 용어 목록 (실제로는 더 많은 용어 사용)
        test_terms = [
            # 민사법 도메인
            "손해배상", "계약", "소유권", "임대차", "불법행위", "상속", "이혼", "교통사고", "근로",
            "물권", "채권", "가족관계", "친족", "양육권", "친권", "위자료", "재산분할",
            
            # 형사법 도메인
            "살인", "절도", "사기", "강도", "강간", "폭행", "상해", "명예훼손", "모독",
            "도주", "증거인멸", "위증", "무고", "공갈", "횡령", "배임", "뇌물",
            
            # 상사법 도메인
            "주식회사", "유한회사", "합명회사", "합자회사", "상행위", "어음", "수표", "해상",
            "보험", "운송", "위임", "대리", "중개", "도급", "임치", "사용대차",
            
            # 행정법 도메인
            "행정처분", "행정지도", "행정심판", "행정소송", "허가", "인가", "승인", "면허",
            "등록", "신고", "신청", "고지", "공고", "공시", "조사", "검사",
            
            # 노동법 도메인
            "근로계약", "임금", "근로시간", "휴가", "해고", "부당해고", "실업급여", "산업재해",
            "노동조합", "단체교섭", "단체협약", "파업", "로크아웃", "분쟁조정", "중재", "조정"
        ]
        
        safe_print(f"확장할 용어 수: {len(test_terms)}개")
        safe_print(f"도메인별 분류:")
        safe_print(f"  - 민사법: 18개")
        safe_print(f"  - 형사법: 18개")
        safe_print(f"  - 상사법: 16개")
        safe_print(f"  - 행정법: 16개")
        safe_print(f"  - 노동법: 16개")
        safe_print("-" * 30)
        
        # 모니터링 시작
        monitor.start_monitoring(len(test_terms))
        
        # 도메인별 확장
        domains = {
            "민사법": test_terms[:18],
            "형사법": test_terms[18:36],
            "상사법": test_terms[36:52],
            "행정법": test_terms[52:68],
            "노동법": test_terms[68:84]
        }
        
        all_results = {}
        
        for domain, terms in domains.items():
            safe_print(f"\n{domain} 도메인 확장 시작: {len(terms)}개 용어")
            safe_print("-" * 30)
            
            # 용어 확장
            domain_results = expander.expand_all_terms(terms, domain)
            
            # 품질 검증
            validated_results = validator.filter_low_quality_terms(domain_results["results"], threshold=0.6)
            
            # 데이터베이스 저장
            database.save_expanded_terms(validated_results, domain)
            
            # 결과 통합
            all_results.update(validated_results)
            
            # 진행 상황 업데이트
            monitor.update_progress(domain_results)
            
            safe_print(f"{domain} 도메인 확장 완료:")
            safe_print(f"  성공: {domain_results['successful_expansions']}개")
            safe_print(f"  실패: {domain_results['failed_expansions']}개")
            safe_print(f"  성공률: {domain_results['success_rate']:.1%}")
        
        # 전체 품질 보고서 생성
        quality_report = validator.generate_quality_report(all_results)
        
        # 진행 상황 보고서 생성
        progress_report = monitor.generate_progress_report()
        
        # JSON 파일로 내보내기
        database.export_to_json("data/comprehensive_legal_term_dictionary.json")
        
        # 최종 결과 출력
        safe_print(f"\n전체 법률 용어 확장 완료!")
        safe_print("=" * 50)
        safe_print(f"총 처리 용어: {progress_report['total_terms']}개")
        safe_print(f"성공: {progress_report['successful_expansions']}개")
        safe_print(f"실패: {progress_report['failed_expansions']}개")
        safe_print(f"성공률: {progress_report['success_rate']:.1%}")
        safe_print(f"평균 품질: {quality_report.get('average_quality', 0):.2f}")
        safe_print(f"처리 시간: {progress_report['elapsed_time_seconds']:.1f}초")
        safe_print(f"저장 위치: data/comprehensive_legal_term_dictionary.json")
        
    except ValueError as e:
        safe_print(f"초기화 오류: {e}")
        safe_print("해결 방법: .env 파일에 유효한 GOOGLE_API_KEY를 설정하세요.")
    except Exception as e:
        safe_print(f"실행 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
