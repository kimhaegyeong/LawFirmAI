#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 기반 Q&A 데이터셋 생성 실행 스크립트

Ollama Qwen2.5:7b 모델을 사용하여 법률 Q&A 데이터셋을 생성합니다.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.llm_qa_generator import LLMQAGenerator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/generate_qa_with_llm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="LLM 기반 법률 Q&A 데이터셋 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 실행
  python scripts/generate_qa_with_llm.py

  # 특정 모델과 데이터 타입 지정
  python scripts/generate_qa_with_llm.py --model qwen2.5:7b --data-type laws precedents

  # 출력 디렉토리와 목표 개수 지정
  python scripts/generate_qa_with_llm.py --output data/qa_dataset/llm_generated --target 3000

  # 테스트 모드 (소규모 데이터)
  python scripts/generate_qa_with_llm.py --dry-run --max-items 10
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='qwen2.5:7b',
        help='사용할 Ollama 모델명 (기본값: qwen2.5:7b)'
    )
    
    parser.add_argument(
        '--data-type',
        nargs='+',
        choices=['laws', 'precedents', 'constitutional_decisions', 'legal_interpretations'],
        default=['laws', 'precedents'],
        help='처리할 데이터 유형 (기본값: laws precedents)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='입력 데이터 디렉토리 (기본값: data/processed)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/qa_dataset/llm_generated',
        help='출력 디렉토리 (기본값: data/qa_dataset/llm_generated)'
    )
    
    parser.add_argument(
        '--target',
        type=int,
        default=3000,
        help='목표 Q&A 개수 (기본값: 3000)'
    )
    
    parser.add_argument(
        '--max-items',
        type=int,
        default=100,
        help='데이터 타입별 최대 처리 항목 수 (기본값: 100)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM 생성 온도 (0.0-1.0, 기본값: 0.7)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1500,
        help='최대 토큰 수 (기본값: 1500)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='배치 처리 크기 (기본값: 10)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='테스트 모드 (실제 생성하지 않고 설정만 확인)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='상세 로그 출력'
    )
    
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=0.6,
        help='품질 점수 임계값 (기본값: 0.6)'
    )
    
    return parser.parse_args()


def validate_environment():
    """환경 검증"""
    logger.info("환경 검증 중...")
    
    # 필요한 디렉토리 확인
    data_dir = Path("data/processed")
    if not data_dir.exists():
        logger.error(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        return False
    
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.info("환경 검증 완료")
    return True


def test_ollama_connection(model: str) -> bool:
    """Ollama 연결 테스트"""
    logger.info(f"Ollama 모델 '{model}' 연결 테스트 중...")
    
    try:
        from source.utils.ollama_client import OllamaClient
        
        client = OllamaClient(model=model)
        success = client.test_connection()
        
        if success:
            logger.info("✅ Ollama 연결 성공")
            return True
        else:
            logger.error("❌ Ollama 연결 실패")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ollama 연결 테스트 중 오류: {e}")
        return False


def generate_qa_dataset(args):
    """Q&A 데이터셋 생성"""
    logger.info("LLM 기반 Q&A 데이터셋 생성 시작...")
    
    try:
        # LLM Q&A 생성기 초기화
        generator = LLMQAGenerator(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # 품질 기준 업데이트
        generator.quality_criteria['min_quality_score'] = args.quality_threshold
        
        # 데이터셋 생성
        success = generator.generate_dataset(
            data_dir=args.data_dir,
            output_dir=args.output,
            data_types=args.data_type,
            max_items_per_type=args.max_items
        )
        
        if success:
            logger.info("✅ Q&A 데이터셋 생성 완료")
            
            # 결과 요약 출력
            print_summary(args.output)
            return True
        else:
            logger.error("❌ Q&A 데이터셋 생성 실패")
            return False
            
    except Exception as e:
        logger.error(f"Q&A 데이터셋 생성 중 오류: {e}")
        return False


def print_summary(output_dir: str):
    """결과 요약 출력"""
    try:
        import json
        
        stats_file = Path(output_dir) / "llm_qa_dataset_statistics.json"
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            print("\n" + "="*60)
            print("📊 LLM 기반 Q&A 데이터셋 생성 결과")
            print("="*60)
            print(f"총 Q&A 쌍 수: {stats['total_pairs']:,}개")
            print(f"고품질 Q&A: {stats['high_quality_pairs']:,}개 (품질점수 ≥0.8)")
            print(f"중품질 Q&A: {stats['medium_quality_pairs']:,}개 (품질점수 0.6-0.8)")
            print(f"저품질 Q&A: {stats['low_quality_pairs']:,}개 (품질점수 <0.6)")
            print(f"평균 품질 점수: {stats['average_quality_score']:.3f}")
            print(f"사용 모델: {stats['model_used']}")
            print(f"생성 온도: {stats['temperature']}")
            
            print("\n📈 소스별 분포:")
            for source, count in stats['source_distribution'].items():
                print(f"  - {source}: {count:,}개")
            
            print("\n📈 난이도별 분포:")
            for difficulty, count in stats['difficulty_distribution'].items():
                print(f"  - {difficulty}: {count:,}개")
            
            print("\n📈 질문 유형별 분포:")
            for q_type, count in stats['question_type_distribution'].items():
                print(f"  - {q_type}: {count:,}개")
            
            print("\n📁 생성된 파일:")
            output_path = Path(output_dir)
            for file_path in output_path.glob("*.json"):
                print(f"  - {file_path.name}")
            
            print("="*60)
            
    except Exception as e:
        logger.error(f"결과 요약 출력 중 오류: {e}")


def main():
    """메인 함수"""
    # 인수 파싱
    args = parse_arguments()
    
    # 로그 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("LLM 기반 Q&A 데이터셋 생성 스크립트 시작")
    logger.info(f"설정: {vars(args)}")
    
    # 환경 검증
    if not validate_environment():
        logger.error("환경 검증 실패")
        return 1
    
    # Ollama 연결 테스트
    if not test_ollama_connection(args.model):
        logger.error("Ollama 연결 실패")
        return 1
    
    # 테스트 모드 확인
    if args.dry_run:
        logger.info("🔍 테스트 모드: 실제 생성하지 않음")
        logger.info("설정 확인 완료. --dry-run 옵션을 제거하고 실행하세요.")
        return 0
    
    # Q&A 데이터셋 생성
    success = generate_qa_dataset(args)
    
    if success:
        logger.info("🎉 LLM 기반 Q&A 데이터셋 생성 완료!")
        return 0
    else:
        logger.error("❌ LLM 기반 Q&A 데이터셋 생성 실패")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
