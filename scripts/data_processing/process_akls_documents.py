#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AKLS 문서 처리 스크립트
법률전문대학원협의회 PDF 문서들을 처리하여 벡터 임베딩 및 검색 인덱스 생성
"""

import os
import sys
import logging
import json
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.akls_processor import AKLSProcessor
from source.services.akls_search_engine import AKLSSearchEngine

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/akls_processing.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "data/processed/akls",
        "data/embeddings/akls_precedents",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"디렉토리 생성: {directory}")


def process_akls_documents():
    """AKLS 문서 처리 메인 함수"""
    try:
        logger.info("=== AKLS 문서 처리 시작 ===")
        
        # 디렉토리 생성
        create_directories()
        
        # AKLS 프로세서 초기화
        processor = AKLSProcessor()
        
        # AKLS 디렉토리 경로
        akls_dir = "data/raw/akls"
        output_dir = "data/processed/akls"
        
        if not os.path.exists(akls_dir):
            logger.error(f"AKLS 디렉토리를 찾을 수 없습니다: {akls_dir}")
            return False
        
        # PDF 파일 확인
        pdf_files = [f for f in os.listdir(akls_dir) if f.lower().endswith('.pdf')]
        logger.info(f"처리할 PDF 파일 수: {len(pdf_files)}")
        
        if not pdf_files:
            logger.warning("처리할 PDF 파일이 없습니다.")
            return False
        
        # 문서 처리
        logger.info("AKLS 문서 처리 시작...")
        processed_docs = processor.process_akls_directory(akls_dir)
        
        if not processed_docs:
            logger.error("처리된 문서가 없습니다.")
            return False
        
        # 처리 결과 저장
        logger.info("처리된 문서 저장 중...")
        processor.save_processed_documents(processed_docs, output_dir)
        
        # 처리 결과 요약
        logger.info(f"=== AKLS 문서 처리 완료 ===")
        logger.info(f"처리된 문서 수: {len(processed_docs)}")
        
        # 법률 영역별 통계
        area_stats = {}
        for doc in processed_docs:
            area = doc.law_area
            area_stats[area] = area_stats.get(area, 0) + 1
        
        logger.info("법률 영역별 문서 수:")
        for area, count in area_stats.items():
            logger.info(f"  {area}: {count}개")
        
        return True
        
    except Exception as e:
        logger.error(f"AKLS 문서 처리 중 오류 발생: {e}")
        return False


def create_search_index():
    """AKLS 검색 인덱스 생성"""
    try:
        logger.info("=== AKLS 검색 인덱스 생성 시작 ===")
        
        # 처리된 문서 로드
        processed_dir = "data/processed/akls"
        if not os.path.exists(processed_dir):
            logger.error(f"처리된 문서 디렉토리를 찾을 수 없습니다: {processed_dir}")
            return False
        
        # JSON 파일들 로드
        documents = []
        json_files = [f for f in os.listdir(processed_dir) if f.endswith('.json')]
        
        logger.info(f"로드할 JSON 파일 수: {len(json_files)}")
        
        for file in json_files:
            file_path = os.path.join(processed_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    documents.append(doc)
            except Exception as e:
                logger.error(f"문서 로드 실패 {file}: {e}")
        
        if not documents:
            logger.error("로드된 AKLS 문서가 없습니다.")
            return False
        
        # 검색 엔진 초기화 및 인덱스 생성
        logger.info("AKLS 검색 엔진 초기화...")
        search_engine = AKLSSearchEngine()
        
        logger.info("FAISS 인덱스 생성 중...")
        search_engine.create_index_from_documents(documents)
        
        logger.info("=== AKLS 검색 인덱스 생성 완료 ===")
        logger.info(f"인덱스된 문서 수: {len(documents)}")
        
        # 통계 정보 출력
        stats = search_engine.get_law_area_statistics()
        logger.info("법률 영역별 문서 수:")
        for area, count in stats.items():
            korean_name = search_engine.law_area_mapping.get(area, area)
            logger.info(f"  {korean_name}: {count}개")
        
        return True
        
    except Exception as e:
        logger.error(f"AKLS 검색 인덱스 생성 중 오류 발생: {e}")
        return False


def test_search_functionality():
    """검색 기능 테스트"""
    try:
        logger.info("=== AKLS 검색 기능 테스트 시작 ===")
        
        # 검색 엔진 초기화
        search_engine = AKLSSearchEngine()
        
        # 테스트 쿼리
        test_queries = [
            "계약 해지",
            "손해배상",
            "형법",
            "대법원",
            "민사소송"
        ]
        
        logger.info("테스트 쿼리 실행 중...")
        
        for query in test_queries:
            logger.info(f"\n--- 검색 쿼리: '{query}' ---")
            
            try:
                results = search_engine.search(query, top_k=3)
                
                logger.info(f"검색 결과 수: {len(results)}")
                
                for i, result in enumerate(results, 1):
                    logger.info(f"{i}. 점수: {result.score:.3f}")
                    logger.info(f"   법률영역: {search_engine.law_area_mapping.get(result.law_area, result.law_area)}")
                    if result.case_number:
                        logger.info(f"   사건번호: {result.case_number}")
                    if result.court:
                        logger.info(f"   법원: {result.court}")
                    logger.info(f"   파일명: {result.metadata.get('filename', 'N/A')}")
                
            except Exception as e:
                logger.error(f"쿼리 '{query}' 실행 실패: {e}")
        
        logger.info("=== AKLS 검색 기능 테스트 완료 ===")
        return True
        
    except Exception as e:
        logger.error(f"AKLS 검색 기능 테스트 중 오류 발생: {e}")
        return False


def main():
    """메인 실행 함수"""
    print("=== AKLS 문서 처리 및 검색 인덱스 생성 ===")
    
    # 1단계: AKLS 문서 처리
    print("\n1단계: AKLS PDF 문서 처리")
    if not process_akls_documents():
        print("AKLS 문서 처리 실패")
        return False
    
    # 2단계: 검색 인덱스 생성
    print("\n2단계: AKLS 검색 인덱스 생성")
    if not create_search_index():
        print("AKLS 검색 인덱스 생성 실패")
        return False
    
    # 3단계: 검색 기능 테스트
    print("\n3단계: AKLS 검색 기능 테스트")
    if not test_search_functionality():
        print("AKLS 검색 기능 테스트 실패")
        return False
    
    print("\n=== AKLS 통합 완료 ===")
    print("AKLS 표준판례 자료가 성공적으로 LawFirmAI 시스템에 통합되었습니다.")
    print("이제 Enhanced RAG Service를 통해 AKLS 자료를 검색할 수 있습니다.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
