# -*- coding: utf-8 -*-
"""
법령 및 판례 기반 법률 용어 추출 시스템
Gemini 2.5 Flash Lite를 활용한 용어 검증
"""

import os
import json
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import requests
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re
from datetime import datetime

from .gemini_client import GeminiClient, GeminiResponse

logger = get_logger(__name__)


@dataclass
class LegalTerm:
    """법률 용어 정보"""
    term: str
    domain: str
    weight: float
    synonyms: List[str]
    related_terms: List[str]
    context_keywords: List[str]
    source: str  # "legal_act", "precedent", "dictionary"
    confidence: float
    verified: bool = False


class LegalActTermExtractor:
    """법령 기반 용어 추출기"""
    
    def __init__(self):
        """법령 기반 용어 추출기 초기화"""
        self.logger = get_logger(__name__)
        
        # 주요 법령별 핵심 용어 정의
        self.legal_acts_terms = {
            "민법": {
                "domain": "민사법",
                "terms": {
                    "친권": {
                        "weight": 0.9,
                        "synonyms": ["친권자", "친권행사", "친권상실", "친권제한"],
                        "related_terms": ["양육권", "면접교섭권", "친생자", "양자"],
                        "context_keywords": ["부모", "자녀", "권리", "의무", "행사", "상실"]
                    },
                    "양육권": {
                        "weight": 0.9,
                        "synonyms": ["양육", "양육권자", "양육권자격"],
                        "related_terms": ["친권", "면접교섭권", "양육비", "이혼"],
                        "context_keywords": ["자녀", "부모", "이혼", "양육비", "면접교섭", "결정"]
                    },
                    "면접교섭권": {
                        "weight": 0.8,
                        "synonyms": ["면접교섭", "면접권", "교섭권"],
                        "related_terms": ["양육권", "친권", "이혼", "자녀"],
                        "context_keywords": ["자녀", "면접", "교섭", "이혼", "부모", "권리"]
                    },
                    "친자확인": {
                        "weight": 0.8,
                        "synonyms": ["친자관계", "친생자", "친자확인소송"],
                        "related_terms": ["친자부인", "인지", "친권", "양육권"],
                        "context_keywords": ["혈연", "부모", "자녀", "확인", "소송", "관계"]
                    },
                    "상속": {
                        "weight": 0.9,
                        "synonyms": ["상속분", "상속인", "상속재산", "상속포기"],
                        "related_terms": ["유언", "유증", "상속회복청구권", "상속분할"],
                        "context_keywords": ["사망", "유족", "재산", "분할", "포기", "승인"]
                    }
                }
            },
            "형사소송법": {
                "domain": "형사소송법",
                "terms": {
                    "변호인": {
                        "weight": 0.9,
                        "synonyms": ["변호사", "선임", "변호인선임", "변호인선임권"],
                        "related_terms": ["국선변호인", "변호인접견", "변호인조력"],
                        "context_keywords": ["방어", "선임", "변호", "재판", "수사", "피의자", "피고인"]
                    },
                    "증거능력": {
                        "weight": 0.8,
                        "synonyms": ["증거", "능력", "증거력", "증거조사"],
                        "related_terms": ["증거제출", "자백배제법칙", "증거금지"],
                        "context_keywords": ["증거", "재판", "수사", "조사", "제출", "능력"]
                    },
                    "항소": {
                        "weight": 0.8,
                        "synonyms": ["항소심", "항소제기", "항소권", "항소이유"],
                        "related_terms": ["상고", "재심", "불복신청", "항소기각"],
                        "context_keywords": ["판결", "불복", "재판", "심급", "제기", "기각"]
                    },
                    "상고": {
                        "weight": 0.8,
                        "synonyms": ["상고심", "상고제기", "상고권", "상고이유"],
                        "related_terms": ["항소", "재심", "불복신청", "상고기각"],
                        "context_keywords": ["판결", "불복", "재판", "심급", "대법원", "제기"]
                    },
                    "공소제기": {
                        "weight": 0.9,
                        "synonyms": ["기소", "기소권", "기소절차", "공소"],
                        "related_terms": ["검사", "기소유예", "공소취소", "공소보류"],
                        "context_keywords": ["범죄", "수사", "재판", "처벌", "기소", "공소"]
                    }
                }
            },
            "특허법": {
                "domain": "지적재산권법",
                "terms": {
                    "디자인권": {
                        "weight": 0.8,
                        "synonyms": ["디자인", "디자인보호", "디자인출원", "디자인등록"],
                        "related_terms": ["특허", "상표", "저작권", "실용신안"],
                        "context_keywords": ["디자인", "보호", "출원", "등록", "침해", "권리"]
                    },
                    "실용신안": {
                        "weight": 0.7,
                        "synonyms": ["실용신안권", "실용신안출원", "실용신안등록"],
                        "related_terms": ["특허", "디자인", "발명", "기술"],
                        "context_keywords": ["실용", "신안", "기술", "출원", "등록", "보호"]
                    },
                    "지적재산권": {
                        "weight": 0.9,
                        "synonyms": ["지재권", "IP", "지적재산", "지적재산권보호"],
                        "related_terms": ["특허", "상표", "저작권", "디자인", "실용신안"],
                        "context_keywords": ["지적재산", "보호", "권리", "침해", "구제", "권리"]
                    }
                }
            },
            "민사소송법": {
                "domain": "민사소송법",
                "terms": {
                    "소장": {
                        "weight": 0.8,
                        "synonyms": ["소송장", "소제기", "소장작성"],
                        "related_terms": ["답변서", "준비서면", "소송절차"],
                        "context_keywords": ["소송", "제기", "작성", "법원", "절차"]
                    },
                    "증거제출": {
                        "weight": 0.7,
                        "synonyms": ["증거", "제출", "증거조사"],
                        "related_terms": ["증거능력", "증거력", "증거조사"],
                        "context_keywords": ["증거", "제출", "재판", "조사", "법원"]
                    },
                    "준비서면": {
                        "weight": 0.7,
                        "synonyms": ["서면", "준비", "서면준비"],
                        "related_terms": ["소장", "답변서", "소송절차"],
                        "context_keywords": ["소송", "준비", "서면", "법원", "절차"]
                    }
                }
            },
            "세법": {
                "domain": "세법",
                "terms": {
                    "세무조사": {
                        "weight": 0.8,
                        "synonyms": ["조사", "세무", "세무조사권"],
                        "related_terms": ["세무서", "조세불복", "세무조정"],
                        "context_keywords": ["세무", "조사", "세금", "신고", "납부"]
                    },
                    "조세불복": {
                        "weight": 0.7,
                        "synonyms": ["불복", "세무불복", "조세불복청구"],
                        "related_terms": ["세무조사", "세무조정", "행정심판"],
                        "context_keywords": ["세금", "불복", "청구", "조정", "심판"]
                    }
                }
            }
        }
    
    def extract_terms_from_legal_acts(self) -> List[LegalTerm]:
        """법령에서 용어 추출"""
        extracted_terms = []
        
        for act_name, act_data in self.legal_acts_terms.items():
            domain = act_data["domain"]
            
            for term_name, term_data in act_data["terms"].items():
                legal_term = LegalTerm(
                    term=term_name,
                    domain=domain,
                    weight=term_data["weight"],
                    synonyms=term_data["synonyms"],
                    related_terms=term_data["related_terms"],
                    context_keywords=term_data["context_keywords"],
                    source="legal_act",
                    confidence=0.9  # 법령 기반이므로 높은 신뢰도
                )
                extracted_terms.append(legal_term)
        
        self.logger.info(f"법령에서 {len(extracted_terms)}개 용어 추출 완료")
        return extracted_terms


class PrecedentTermExtractor:
    """판례 기반 용어 추출기"""
    
    def __init__(self):
        """판례 기반 용어 추출기 초기화"""
        self.logger = get_logger(__name__)
        
        # 판례에서 자주 등장하는 용어들
        self.precedent_terms = {
            "가족법": {
                "양육권자": {
                    "weight": 0.8,
                    "synonyms": ["양육권자격", "양육권자"],
                    "related_terms": ["양육권", "친권", "면접교섭권"],
                    "context_keywords": ["자녀", "양육", "권리", "이혼", "부모"]
                },
                "양육비": {
                    "weight": 0.8,
                    "synonyms": ["양육비용", "양육지원비"],
                    "related_terms": ["양육권", "면접교섭권", "이혼"],
                    "context_keywords": ["자녀", "양육", "비용", "지원", "이혼"]
                },
                "면접교섭": {
                    "weight": 0.7,
                    "synonyms": ["면접", "교섭", "면접권"],
                    "related_terms": ["양육권", "친권", "이혼"],
                    "context_keywords": ["자녀", "면접", "교섭", "이혼", "부모"]
                },
                "가족관계등록부": {
                    "weight": 0.8,
                    "synonyms": ["호적", "등록부", "가족관계등록"],
                    "related_terms": ["호적정정", "가족관계등록부정정"],
                    "context_keywords": ["신고", "등록", "정정", "가족", "관계"]
                }
            },
            "형사소송법": {
                "변호인선임권": {
                    "weight": 0.8,
                    "synonyms": ["변호인선임", "선임권"],
                    "related_terms": ["변호인", "국선변호인", "변호인접견"],
                    "context_keywords": ["변호", "선임", "권리", "피의자", "피고인"]
                },
                "자백배제법칙": {
                    "weight": 0.7,
                    "synonyms": ["자백배제", "배제법칙"],
                    "related_terms": ["증거능력", "증거금지", "자백"],
                    "context_keywords": ["자백", "배제", "증거", "능력", "금지"]
                },
                "공소보류": {
                    "weight": 0.7,
                    "synonyms": ["보류", "공소보류제도"],
                    "related_terms": ["공소제기", "기소유예", "공소취소"],
                    "context_keywords": ["공소", "보류", "기소", "제기", "취소"]
                }
            },
            "지적재산권법": {
                "디자인침해": {
                    "weight": 0.8,
                    "synonyms": ["디자인권침해", "디자인침해행위"],
                    "related_terms": ["디자인권", "침해", "손해배상"],
                    "context_keywords": ["디자인", "침해", "권리", "손해", "배상"]
                },
                "저작권침해": {
                    "weight": 0.8,
                    "synonyms": ["저작권침해행위", "저작권침해"],
                    "related_terms": ["저작권", "침해", "손해배상"],
                    "context_keywords": ["저작권", "침해", "권리", "손해", "배상"]
                },
                "상표침해": {
                    "weight": 0.8,
                    "synonyms": ["상표권침해", "상표침해행위"],
                    "related_terms": ["상표권", "침해", "손해배상"],
                    "context_keywords": ["상표", "침해", "권리", "손해", "배상"]
                },
                "특허침해": {
                    "weight": 0.8,
                    "synonyms": ["특허권침해", "특허침해행위"],
                    "related_terms": ["특허권", "침해", "손해배상"],
                    "context_keywords": ["특허", "침해", "권리", "손해", "배상"]
                }
            }
        }
    
    def extract_terms_from_precedents(self) -> List[LegalTerm]:
        """판례에서 용어 추출"""
        extracted_terms = []
        
        for domain, terms in self.precedent_terms.items():
            for term_name, term_data in terms.items():
                legal_term = LegalTerm(
                    term=term_name,
                    domain=domain,
                    weight=term_data["weight"],
                    synonyms=term_data["synonyms"],
                    related_terms=term_data["related_terms"],
                    context_keywords=term_data["context_keywords"],
                    source="precedent",
                    confidence=0.8  # 판례 기반이므로 중간 신뢰도
                )
                extracted_terms.append(legal_term)
        
        self.logger.info(f"판례에서 {len(extracted_terms)}개 용어 추출 완료")
        return extracted_terms


class GeminiTermValidator:
    """Gemini 2.5 Flash Lite를 활용한 용어 검증기"""
    
    def __init__(self):
        """Gemini 용어 검증기 초기화"""
        self.gemini_client = GeminiClient()
        self.logger = get_logger(__name__)
    
    def validate_term(self, term: LegalTerm) -> Tuple[bool, float, str]:
        """개별 용어 검증"""
        try:
            validation_prompt = f"""
당신은 대한민국 법률 전문가입니다. 다음 법률 용어의 정확성을 검증해주세요.

용어: {term.term}
도메인: {term.domain}
동의어: {', '.join(term.synonyms)}
관련 용어: {', '.join(term.related_terms)}
문맥 키워드: {', '.join(term.context_keywords)}

검증 기준:
1. 용어가 해당 법률 도메인에 적합한가?
2. 동의어가 정확한가?
3. 관련 용어가 적절한가?
4. 문맥 키워드가 적절한가?
5. 가중치({term.weight})가 적절한가?

다음 형식으로 답변해주세요:
정확성: [정확함/부정확함/부분적정확함]
신뢰도: [0.0-1.0]
개선사항: [구체적인 개선사항이나 문제점]
"""
            
            response = self.gemini_client.generate(validation_prompt)
            
            # 응답 파싱
            accuracy, confidence, improvement = self._parse_validation_response(response)
            
            return accuracy, confidence, improvement
            
        except Exception as e:
            self.logger.error(f"용어 검증 중 오류 발생: {e}")
            return False, 0.0, f"검증 오류: {str(e)}"
    
    def _parse_validation_response(self, response: GeminiResponse) -> Tuple[bool, float, str]:
        """검증 응답 파싱"""
        try:
            lines = response.response.strip().split('\n')
            accuracy = False
            confidence = 0.0
            improvement = ""
            
            for line in lines:
                if line.startswith("정확성:"):
                    accuracy_text = line.split(":")[1].strip()
                    accuracy = "정확함" in accuracy_text
                elif line.startswith("신뢰도:"):
                    confidence_text = line.split(":")[1].strip()
                    confidence = float(confidence_text)
                elif line.startswith("개선사항:"):
                    improvement = line.split(":", 1)[1].strip()
            
            return accuracy, confidence, improvement
            
        except Exception as e:
            self.logger.error(f"응답 파싱 중 오류: {e}")
            return False, 0.0, f"파싱 오류: {str(e)}"
    
    def batch_validate_terms(self, terms: List[LegalTerm]) -> List[LegalTerm]:
        """용어 일괄 검증"""
        validated_terms = []
        
        for i, term in enumerate(terms):
            self.logger.info(f"용어 검증 중 ({i+1}/{len(terms)}): {term.term}")
            
            accuracy, confidence, improvement = self.validate_term(term)
            
            # 검증 결과 반영
            term.verified = accuracy
            term.confidence = confidence
            
            if improvement:
                self.logger.info(f"개선사항: {improvement}")
            
            validated_terms.append(term)
            
            # API 호출 제한을 위한 대기
            time.sleep(1)
        
        return validated_terms


class LegalTermDatabaseUpdater:
    """법률 용어 데이터베이스 업데이터"""
    
    def __init__(self, database_path: str = "data/legal_terms_database.json"):
        """데이터베이스 업데이터 초기화"""
        self.database_path = Path(database_path)
        self.logger = get_logger(__name__)
    
    def update_database(self, new_terms: List[LegalTerm]) -> bool:
        """데이터베이스 업데이트"""
        try:
            # 기존 데이터베이스 로드
            existing_data = self._load_existing_database()
            
            # 새 용어 추가
            for term in new_terms:
                if term.verified:  # 검증된 용어만 추가
                    self._add_term_to_database(existing_data, term)
            
            # 데이터베이스 저장
            self._save_database(existing_data)
            
            self.logger.info(f"데이터베이스 업데이트 완료: {len(new_terms)}개 용어 추가")
            return True
            
        except Exception as e:
            self.logger.error(f"데이터베이스 업데이트 실패: {e}")
            return False
    
    def _load_existing_database(self) -> Dict[str, Any]:
        """기존 데이터베이스 로드"""
        if self.database_path.exists():
            with open(self.database_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    
    def _add_term_to_database(self, database: Dict[str, Any], term: LegalTerm):
        """데이터베이스에 용어 추가"""
        domain_name = term.domain
        
        if domain_name not in database:
            database[domain_name] = {}
        
        database[domain_name][term.term] = {
            "weight": term.weight,
            "synonyms": term.synonyms,
            "related_terms": term.related_terms,
            "context_keywords": term.context_keywords,
            "source": term.source,
            "confidence": term.confidence,
            "verified": term.verified,
            "added_date": datetime.now().isoformat()
        }
    
    def _save_database(self, database: Dict[str, Any]):
        """데이터베이스 저장"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.database_path, 'w', encoding='utf-8') as f:
            json.dump(database, f, ensure_ascii=False, indent=2)


class LegalTermExtractionPipeline:
    """법률 용어 추출 파이프라인"""
    
    def __init__(self):
        """파이프라인 초기화"""
        self.logger = get_logger(__name__)
        
        # 컴포넌트 초기화
        self.legal_act_extractor = LegalActTermExtractor()
        self.precedent_extractor = PrecedentTermExtractor()
        self.validator = GeminiTermValidator()
        self.database_updater = LegalTermDatabaseUpdater()
    
    def run_extraction_pipeline(self) -> bool:
        """용어 추출 파이프라인 실행"""
        try:
            self.logger.info("법률 용어 추출 파이프라인 시작")
            
            # 1. 법령에서 용어 추출
            self.logger.info("1단계: 법령에서 용어 추출")
            legal_act_terms = self.legal_act_extractor.extract_terms_from_legal_acts()
            
            # 2. 판례에서 용어 추출
            self.logger.info("2단계: 판례에서 용어 추출")
            precedent_terms = self.precedent_extractor.extract_terms_from_precedents()
            
            # 3. 용어 통합
            all_terms = legal_act_terms + precedent_terms
            self.logger.info(f"총 {len(all_terms)}개 용어 추출 완료")
            
            # 4. Gemini로 용어 검증
            self.logger.info("3단계: Gemini로 용어 검증")
            validated_terms = self.validator.batch_validate_terms(all_terms)
            
            # 5. 검증 결과 통계
            verified_count = sum(1 for term in validated_terms if term.verified)
            self.logger.info(f"검증 완료: {verified_count}/{len(validated_terms)}개 용어 검증됨")
            
            # 6. 데이터베이스 업데이트
            self.logger.info("4단계: 데이터베이스 업데이트")
            success = self.database_updater.update_database(validated_terms)
            
            if success:
                self.logger.info("용어 추출 파이프라인 완료")
                return True
            else:
                self.logger.error("데이터베이스 업데이트 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"용어 추출 파이프라인 실패: {e}")
            return False


def run_legal_term_extraction():
    """법률 용어 추출 실행"""
    pipeline = LegalTermExtractionPipeline()
    return pipeline.run_extraction_pipeline()


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 용어 추출 실행
    success = run_legal_term_extraction()
    
    if success:
        print("법률 용어 추출이 성공적으로 완료되었습니다.")
    else:
        print("법률 용어 추출 중 오류가 발생했습니다.")
