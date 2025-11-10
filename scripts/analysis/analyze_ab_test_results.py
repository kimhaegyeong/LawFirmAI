#!/usr/bin/env python3
"""
A/B 테스트 결과 분석 스크립트
워크플로우 최적화 효과를 검증하기 위한 A/B 테스트 결과 분석
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    env_file = project_root / "api" / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
except:
    pass

from lawfirm_langgraph.core.services.ab_test_manager import ABTestManager
from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig


def analyze_ab_test_results():
    """A/B 테스트 결과 분석"""
    print("=== A/B 테스트 결과 분석 ===\n")
    
    # A/B 테스트 관리자 초기화
    ab_manager = ABTestManager()
    
    # 워크플로우 서비스에서 A/B 테스트 관리자 가져오기
    try:
        config = LangGraphConfig.from_env()
        service = LangGraphWorkflowService(config)
        if service.ab_test_manager:
            ab_manager = service.ab_test_manager
            print("✅ 워크플로우 서비스에서 A/B 테스트 관리자 로드 완료\n")
        else:
            print("⚠️  워크플로우 서비스에 A/B 테스트 관리자가 없습니다.")
            print("   ENABLE_AB_TESTING=true로 설정하세요.\n")
    except Exception as e:
        print(f"⚠️  워크플로우 서비스에서 A/B 테스트 관리자를 가져올 수 없습니다: {e}")
        print("   로컬 A/B 테스트 관리자를 사용합니다.\n")
    
    # 전체 실험 요약
    summary = ab_manager.get_summary()
    if not summary:
        print("❌ 분석할 실험 결과가 없습니다.")
        print("   A/B 테스트를 활성화하고 워크플로우를 실행하세요.")
        return
    
    print("📊 전체 실험 요약:")
    for experiment, info in summary.items():
        print(f"\n  실험: {experiment}")
        print(f"    변형: {', '.join(info['variants'])}")
        print(f"    총 결과 수: {info['total_results']}")
        print(f"    메트릭: {', '.join(info['metrics'])}")
    
    # 각 실험별 상세 분석
    print("\n\n=== 실험별 상세 분석 ===\n")
    
    for experiment in summary.keys():
        print(f"📌 실험: {experiment}")
        print("=" * 60)
        
        results = ab_manager.get_results(experiment)
        if not results:
            print("  ❌ 결과가 없습니다.\n")
            continue
        
        # 변형별 통계 출력
        for variant, metrics in results.items():
            print(f"\n  변형: {variant}")
            print("  " + "-" * 58)
            
            for metric, stats in metrics.items():
                print(f"    메트릭: {metric}")
                print(f"      평균: {stats['mean']:.4f}")
                print(f"      중앙값: {stats['median']:.4f}")
                print(f"      최소값: {stats['min']:.4f}")
                print(f"      최대값: {stats['max']:.4f}")
                print(f"      표준편차: {stats['std']:.4f}")
                print(f"      샘플 수: {stats['count']}")
        
        # 변형 비교
        print("\n  📊 변형 비교:")
        print("  " + "-" * 58)
        
        variants = list(results.keys())
        if len(variants) >= 2:
            # control vs variant_a 비교
            if "control" in variants and "variant_a" in variants:
                comparison = ab_manager.compare_variants(
                    experiment, "execution_time", "control", "variant_a"
                )
                if comparison:
                    print(f"    Control vs Variant A:")
                    print(f"      Control 평균: {comparison['variant1']['mean']:.4f}s")
                    print(f"      Variant A 평균: {comparison['variant2']['mean']:.4f}s")
                    print(f"      개선율: {comparison['improvement']:.2f}%")
                    print(f"      절대 개선: {comparison['improvement_abs']:.4f}s")
        
        print()
    
    # 통계적 유의성 검정 (t-test)
    print("\n\n=== 통계적 유의성 검정 ===\n")
    
    try:
        from scipy import stats
        
        for experiment in summary.keys():
            results = ab_manager.get_results(experiment)
            if not results or "execution_time" not in results.get("control", {}):
                continue
            
            control_times = []
            variant_a_times = []
            
            # 실험 결과에서 execution_time 값 추출
            for result in ab_manager.results:
                if result.experiment == experiment and result.metric == "execution_time":
                    if result.variant == "control":
                        control_times.append(result.value)
                    elif result.variant == "variant_a":
                        variant_a_times.append(result.value)
            
            if len(control_times) > 0 and len(variant_a_times) > 0:
                t_stat, p_value = stats.ttest_ind(control_times, variant_a_times)
                
                print(f"실험: {experiment}")
                print(f"  t-statistic: {t_stat:.4f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  유의수준 0.05 기준: {'유의함' if p_value < 0.05 else '유의하지 않음'}")
                print()
    except ImportError:
        print("⚠️  scipy가 설치되지 않았습니다. 통계적 유의성 검정을 건너뜁니다.")
        print("   설치: pip install scipy\n")
    except Exception as e:
        print(f"⚠️  통계적 유의성 검정 중 오류: {e}\n")
    
    # 권장 사항
    print("\n=== 권장 사항 ===\n")
    
    for experiment in summary.keys():
        results = ab_manager.get_results(experiment)
        if not results:
            continue
        
        if "execution_time" in results.get("control", {}):
            control_mean = results["control"]["execution_time"]["mean"]
            variant_a_mean = results.get("variant_a", {}).get("execution_time", {}).get("mean", 0)
            
            if variant_a_mean > 0 and variant_a_mean < control_mean:
                improvement = ((control_mean - variant_a_mean) / control_mean) * 100
                print(f"✅ {experiment}: Variant A가 Control보다 {improvement:.2f}% 빠릅니다.")
                print(f"   권장: Variant A 채택")
            elif variant_a_mean > control_mean:
                degradation = ((variant_a_mean - control_mean) / control_mean) * 100
                print(f"⚠️  {experiment}: Variant A가 Control보다 {degradation:.2f}% 느립니다.")
                print(f"   권장: Control 유지")
            else:
                print(f"ℹ️  {experiment}: 유의미한 차이가 없습니다.")
                print(f"   권장: 추가 테스트 필요")
            print()


if __name__ == "__main__":
    analyze_ab_test_results()

