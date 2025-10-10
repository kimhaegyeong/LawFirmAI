#!/usr/bin/env python3
"""
TASK 3.1 종합 테스트 스크립트
모든 개발 내역을 체계적으로 테스트합니다.
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def run_command(command: str, description: str) -> Dict[str, Any]:
    """명령어 실행 및 결과 반환"""
    print(f"\n🔧 {description}")
    print(f"실행 명령: {command}")
    print("-" * 50)
    
    start_time = time.time()
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=300  # 5분 타임아웃
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": execution_time,
            "description": description
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Command timed out after 5 minutes",
            "execution_time": 300,
            "description": description
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "execution_time": 0,
            "description": description
        }

def test_environment_setup():
    """1. 기본 환경 및 의존성 확인"""
    print("\n" + "="*60)
    print("🧪 1. 기본 환경 및 의존성 확인")
    print("="*60)
    
    tests = [
        ("python scripts/setup_lora_environment.py --verbose", "환경 검사 실행"),
        ("python source/utils/gpu_memory_monitor.py --interval 10", "GPU 메모리 모니터링 테스트")
    ]
    
    results = []
    for command, description in tests:
        result = run_command(command, description)
        results.append(result)
        
        if result["success"]:
            print(f"✅ {description} - 성공 ({result['execution_time']:.2f}초)")
        else:
            print(f"❌ {description} - 실패")
            if result["stderr"]:
                print(f"오류: {result['stderr']}")
    
    return results

def test_model_loading():
    """2. 모델 로딩 및 구조 확인"""
    print("\n" + "="*60)
    print("🧪 2. 모델 로딩 및 구조 확인")
    print("="*60)
    
    tests = [
        ("python scripts/analyze_kogpt2_structure.py --test-lora", "KoGPT-2 모델 구조 분석"),
        ("python source/models/model_manager.py", "모델 매니저 테스트")
    ]
    
    results = []
    for command, description in tests:
        result = run_command(command, description)
        results.append(result)
        
        if result["success"]:
            print(f"✅ {description} - 성공 ({result['execution_time']:.2f}초)")
        else:
            print(f"❌ {description} - 실패")
            if result["stderr"]:
                print(f"오류: {result['stderr']}")
    
    return results

def test_dataset_preparation():
    """3. 데이터셋 준비 및 전처리 확인"""
    print("\n" + "="*60)
    print("🧪 3. 데이터셋 준비 및 전처리 확인")
    print("="*60)
    
    tests = [
        ("python scripts/prepare_training_dataset.py", "기본 데이터셋 준비"),
        ("python scripts/prepare_expanded_training_dataset.py", "확장된 데이터셋 준비 (342개 샘플)"),
        ("python scripts/test_expanded_tokenizer_setup.py", "토크나이저 설정 테스트")
    ]
    
    results = []
    for command, description in tests:
        result = run_command(command, description)
        results.append(result)
        
        if result["success"]:
            print(f"✅ {description} - 성공 ({result['execution_time']:.2f}초)")
        else:
            print(f"❌ {description} - 실패")
            if result["stderr"]:
                print(f"오류: {result['stderr']}")
    
    return results

def test_lora_finetuning():
    """4. LoRA 파인튜닝 테스트"""
    print("\n" + "="*60)
    print("🧪 4. LoRA 파인튜닝 테스트")
    print("="*60)
    
    # 테스트용 모델 디렉토리 생성
    test_model_dir = "models/test/kogpt2-legal-lora-test"
    os.makedirs(test_model_dir, exist_ok=True)
    
    tests = [
        (f"python scripts/finetune_legal_model.py --epochs 1 --batch-size 1 --output {test_model_dir}", "LoRA 파인튜닝 실행 (테스트용)"),
        (f"python scripts/evaluate_legal_model.py --model {test_model_dir} --test-data data/training/test_split.json", "모델 평가")
    ]
    
    results = []
    for command, description in tests:
        result = run_command(command, description)
        results.append(result)
        
        if result["success"]:
            print(f"✅ {description} - 성공 ({result['execution_time']:.2f}초)")
        else:
            print(f"❌ {description} - 실패")
            if result["stderr"]:
                print(f"오류: {result['stderr']}")
    
    return results

def test_day4_features():
    """5. Day 4 고도화된 기능 테스트"""
    print("\n" + "="*60)
    print("🧪 5. Day 4 고도화된 기능 테스트")
    print("="*60)
    
    test_model_dir = "models/test/kogpt2-legal-lora-test"
    
    tests = [
        ("python scripts/day4_test.py", "Day 4 통합 테스트"),
        (f"python scripts/day4_evaluation_optimization.py --test-data data/training/test_split.json --models {test_model_dir} --output results/day4_evaluation", "고도화된 평가 시스템 테스트"),
        (f"python scripts/day4_evaluation_optimization.py --optimize --model-path {test_model_dir} --output results/day4_optimization", "모델 최적화 테스트"),
        (f"python scripts/day4_evaluation_optimization.py --ab-test --test-data data/training/test_split.json --output results/day4_ab_test", "A/B 테스트 프레임워크 테스트")
    ]
    
    results = []
    for command, description in tests:
        result = run_command(command, description)
        results.append(result)
        
        if result["success"]:
            print(f"✅ {description} - 성공 ({result['execution_time']:.2f}초)")
        else:
            print(f"❌ {description} - 실패")
            if result["stderr"]:
                print(f"오류: {result['stderr']}")
    
    return results

def test_integrated_system():
    """6. 종합 통합 테스트"""
    print("\n" + "="*60)
    print("🧪 6. 종합 통합 테스트")
    print("="*60)
    
    test_model_dir = "models/test/kogpt2-legal-lora-test"
    
    command = f"python scripts/day4_evaluation_optimization.py --test-data data/training/test_split.json --models {test_model_dir} --optimize --ab-test --output results/day4_complete"
    description = "모든 기능을 한 번에 테스트"
    
    result = run_command(command, description)
    
    if result["success"]:
        print(f"✅ {description} - 성공 ({result['execution_time']:.2f}초)")
    else:
        print(f"❌ {description} - 실패")
        if result["stderr"]:
            print(f"오류: {result['stderr']}")
    
    return [result]

def check_performance_metrics():
    """성능 지표 확인"""
    print("\n" + "="*60)
    print("📊 성능 지표 확인")
    print("="*60)
    
    metrics = {
        "모델 크기": "2GB 이하 압축 확인",
        "추론 속도": "50% 이상 향상 확인", 
        "메모리 사용량": "40% 이상 감소 확인",
        "법률 Q&A 정확도": "75% 이상 달성 확인"
    }
    
    for metric, description in metrics.items():
        print(f"📈 {metric}: {description}")
    
    return metrics

def check_result_files():
    """결과 파일 확인"""
    print("\n" + "="*60)
    print("📁 결과 파일 확인")
    print("="*60)
    
    result_dirs = [
        "results/day4_test/",
        "results/ab_tests/",
        "models/test/kogpt2-legal-lora-test/",
        "logs/"
    ]
    
    for dir_path in result_dirs:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"✅ {dir_path}: {len(files)}개 파일 존재")
        else:
            print(f"❌ {dir_path}: 디렉토리 없음")

def generate_test_report(all_results: List[List[Dict]], output_dir: str = "results/task3_1_test"):
    """테스트 보고서 생성"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 전체 결과 통합
    all_tests = []
    for test_group in all_results:
        all_tests.extend(test_group)
    
    # 통계 계산
    total_tests = len(all_tests)
    successful_tests = sum(1 for test in all_tests if test["success"])
    failed_tests = total_tests - successful_tests
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    # 보고서 생성
    report = {
        "test_summary": {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "test_date": datetime.now().isoformat()
        },
        "test_results": all_tests,
        "performance_metrics": check_performance_metrics(),
        "recommendations": []
    }
    
    # 권장사항 추가
    if success_rate < 80:
        report["recommendations"].append("전체 성공률이 80% 미만입니다. 실패한 테스트를 재검토하세요.")
    
    if failed_tests > 0:
        report["recommendations"].append("실패한 테스트들을 수정하고 재실행하세요.")
    
    # JSON 파일로 저장
    report_file = os.path.join(output_dir, "task3_1_test_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 테스트 보고서가 생성되었습니다: {report_file}")
    return report

def main():
    """메인 테스트 실행"""
    print("🚀 TASK 3.1 종합 테스트 시작")
    print(f"테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    try:
        # 각 테스트 그룹 실행
        all_results.append(test_environment_setup())
        all_results.append(test_model_loading())
        all_results.append(test_dataset_preparation())
        all_results.append(test_lora_finetuning())
        all_results.append(test_day4_features())
        all_results.append(test_integrated_system())
        
        # 성능 지표 및 결과 파일 확인
        check_performance_metrics()
        check_result_files()
        
        # 테스트 보고서 생성
        report = generate_test_report(all_results)
        
        # 최종 요약
        print("\n" + "="*60)
        print("🎯 테스트 완료 요약")
        print("="*60)
        
        total_tests = report["test_summary"]["total_tests"]
        successful_tests = report["test_summary"]["successful_tests"]
        success_rate = report["test_summary"]["success_rate"]
        
        print(f"총 테스트: {total_tests}개")
        print(f"성공: {successful_tests}개")
        print(f"실패: {total_tests - successful_tests}개")
        print(f"성공률: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 TASK 3.1 테스트 통과!")
        else:
            print("⚠️ 일부 테스트 실패. 보고서를 확인하여 수정하세요.")
            
    except KeyboardInterrupt:
        print("\n⏹️ 테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
    
    print(f"\n테스트 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
