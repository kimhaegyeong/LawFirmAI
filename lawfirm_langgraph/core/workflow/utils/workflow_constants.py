# -*- coding: utf-8 -*-
"""
워크플로우 상수 정의
리팩토링: legal_workflow_enhanced.py에서 상수 분리
"""

import os


class WorkflowConstants:
    """워크플로우 상수 정의"""

    # LLM 설정
    # Gemini 2.5 Flash Lite 최대 출력 토큰: 65,536
    # 환경 변수로 설정 가능, 기본값: 65,536 (Gemini 2.5 Flash Lite 최대값)
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "65536"))  # 최대값 (약 260,000자, 한국어 기준 1토큰≈4자)
    TEMPERATURE = 0.3
    
    # Timeout 설정 (Google Gemini 가이드라인 기준)
    TIMEOUT = 30  # RAG QA용 기본 timeout (20~30초 권장)
    TIMEOUT_RAG_QA = 30  # RAG QA: 20~30초
    TIMEOUT_LONG_TEXT = 60  # 긴 글/코드 생성: 30~60초

    # 검색 설정 (성능 최적화: 결과 수 제한)
    SEMANTIC_SEARCH_K = 12  # 15 -> 12 (성능 최적화)
    KEYWORD_SEARCH_K = 8  # 10 -> 8 (성능 최적화)
    MAX_DOCUMENTS = 10  # 최종 문서 수
    CATEGORY_SEARCH_LIMIT = 4  # 5 -> 4 (성능 최적화)

    # 재시도 설정
    MAX_RETRIES = 3
    RETRY_DELAY = 1

    # 신뢰도 설정
    LLM_CLASSIFICATION_CONFIDENCE = 0.85
    FALLBACK_CONFIDENCE = 0.7
    DEFAULT_CONFIDENCE = 0.6

    # 답변 길이 임계값
    MIN_ANSWER_LENGTH_GENERATION = 100  # 생성 단계 최소 길이 (50 -> 100)
    MIN_ANSWER_LENGTH_VALIDATION = 100  # 검증 단계 최소 길이 (50 -> 100)


class RetryConfig:
    """재시도 설정 상수"""
    MAX_GENERATION_RETRIES = 4
    MAX_VALIDATION_RETRIES = 1
    MAX_TOTAL_RETRIES = 6


class QualityThresholds:
    """품질 임계값 상수"""
    QUALITY_PASS_THRESHOLD = 0.75
    HIGH_QUALITY_THRESHOLD = 0.80
    MEDIUM_QUALITY_THRESHOLD = 0.60
    HIGH_QUALITY_MIN_LENGTH = 50  # 고품질 답변 최소 길이
    MEDIUM_QUALITY_MIN_LENGTH = 80  # 중품질 답변 최소 길이
    LOW_QUALITY_MIN_LENGTH = 100  # 낮은 품질 답변 최소 길이


class AnswerExtractionPatterns:
    """답변 추출을 위한 정규식 패턴 상수"""

    # 추론 과정 섹션 패턴 (개선: 더 많은 패턴 추가)
    REASONING_SECTION_PATTERNS = [
        r'##\s*🧠\s*추론\s*과정\s*작성',
        r'##\s*🧠\s*추론\s*과정',
        r'##\s*추론\s*과정\s*작성',
        r'##\s*추론\s*과정',
        r'##\s*Chain-of-Thought',
        r'##\s*CoT',
        r'###\s*🧠\s*추론',
        r'##\s*추론',  # 추가: 간단한 추론 패턴
        r'###\s*추론',  # 추가: 3단계 추론 패턴
        r'##\s*사고\s*과정',  # 추가: 사고 과정 패턴
        r'##\s*분석\s*과정',  # 추가: 분석 과정 패턴
    ]

    # 출력 섹션 패턴 (실제 답변)
    OUTPUT_SECTION_PATTERNS = [
        r'##\s*📤\s*출력',
        r'##\s*📤\s*출력\s*형식',
        r'##\s*출력',
        r'##\s*최종\s*답변',
        r'##\s*답변\s*내용',
        r'##\s*답변\s*결과',
    ]

    # Step 패턴 (추론 과정 내부)
    STEP_PATTERNS = [
        r'###\s*Step\s*1[:：]',
        r'###\s*Step\s*2[:：]',
        r'###\s*Step\s*3[:：]',
        r'###\s*단계\s*1[:：]',
        r'###\s*단계\s*2[:：]',
        r'###\s*단계\s*3[:：]',
        r'###\s*Step\s*1\s*[:：]',
        r'###\s*Step\s*2\s*[:：]',
        r'###\s*Step\s*3\s*[:：]',
    ]

    # 답변 섹션 패턴 (일반적인 답변 헤더)
    ANSWER_SECTION_PATTERNS = [
        r'##\s*답변',
        r'##\s*💬\s*답변',
        r'###\s*답변',
    ]
