# -*- coding: utf-8 -*-
"""
비교 리포트 생성기
벤치마크 및 비교 결과 리포트 생성
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)

logger = get_logger(__name__)


class ReportGenerator:
    """비교 리포트 생성기"""
    
    def __init__(self, output_dir: Path):
        """
        리포트 생성기 초기화
        
        Args:
            output_dir: 출력 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from lawfirm_langgraph.core.utils.logger import get_logger
            self.logger = get_logger(__name__)
        except ImportError:
            self.logger = logging.getLogger(__name__)
    
    def generate_performance_report(
        self,
        benchmark_results: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> bool:
        """
        성능 비교 리포트 생성
        
        Args:
            benchmark_results: 벤치마크 결과
            output_path: 출력 파일 경로 (None이면 자동 생성)
        
        Returns:
            성공 여부
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"performance_report_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"성능 리포트 생성: {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"성능 리포트 생성 실패: {e}")
            return False
    
    def generate_comparison_report(
        self,
        comparison_results: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> bool:
        """
        검색 결과 비교 리포트 생성
        
        Args:
            comparison_results: 비교 결과
            output_path: 출력 파일 경로 (None이면 자동 생성)
        
        Returns:
            성공 여부
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"comparison_report_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"비교 리포트 생성: {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"비교 리포트 생성 실패: {e}")
            return False
    
    def generate_summary_report(
        self,
        all_results: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> bool:
        """
        종합 리포트 생성
        
        Args:
            all_results: 모든 결과 (벤치마크 + 비교)
            output_path: 출력 파일 경로 (None이면 자동 생성)
        
        Returns:
            성공 여부
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"summary_report_{timestamp}.json"
        
        try:
            # 요약 정보 추가
            summary = {
                "timestamp": datetime.now().isoformat(),
                "benchmark": all_results.get("benchmark", {}),
                "comparison": all_results.get("comparison", {}),
                "recommendations": self._generate_recommendations(all_results)
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"종합 리포트 생성: {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"종합 리포트 생성 실패: {e}")
            return False
    
    def _generate_recommendations(
        self,
        all_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        권장사항 생성
        
        Args:
            all_results: 모든 결과
        
        Returns:
            권장사항
        """
        recommendations = {
            "pgvector": [],
            "faiss": [],
            "hybrid": []
        }
        
        benchmark = all_results.get("benchmark", {})
        comparison = all_results.get("comparison", {})
        
        # 성능 기반 권장사항
        if benchmark:
            pgvector_perf = benchmark.get("pgvector", {})
            faiss_perf = benchmark.get("faiss", {})
            
            if pgvector_perf and faiss_perf:
                pgvector_avg = pgvector_perf.get("avg_time", float('inf'))
                faiss_avg = faiss_perf.get("avg_time", float('inf'))
                
                if pgvector_avg < faiss_avg:
                    recommendations["pgvector"].append(
                        "pgvector가 검색 속도가 더 빠릅니다."
                    )
                else:
                    recommendations["faiss"].append(
                        "FAISS가 검색 속도가 더 빠릅니다."
                    )
        
        # 정확도 기반 권장사항
        if comparison:
            avg_overlap_ratio = comparison.get("avg_overlap_ratio", 0)
            
            if avg_overlap_ratio > 0.8:
                recommendations["hybrid"].append(
                    "두 시스템의 검색 결과가 매우 유사합니다. "
                    "운영 편의성을 고려하여 선택하세요."
                )
            elif avg_overlap_ratio < 0.5:
                recommendations["hybrid"].append(
                    "두 시스템의 검색 결과가 다릅니다. "
                    "하이브리드 접근 방법을 고려하세요."
                )
        
        return recommendations

