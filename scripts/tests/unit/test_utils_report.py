#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_utils 모듈 단위 테스트
"""

import tempfile
from pathlib import Path

from scripts.utils.report_utils import (
    print_section_header,
    print_metrics,
    print_table,
    print_improvements,
    save_text_report,
    generate_markdown_report
)


class TestReportUtils:
    """report_utils 모듈 테스트"""
    
    def test_print_section_header(self, capsys):
        """섹션 헤더 출력 테스트"""
        print_section_header("테스트 섹션")
        captured = capsys.readouterr()
        
        assert "테스트 섹션" in captured.out
        assert "=" in captured.out
    
    def test_print_metrics(self, capsys):
        """메트릭 출력 테스트"""
        metrics = {
            "평균 점수": 0.85,
            "총 개수": 100
        }
        print_metrics(metrics)
        captured = capsys.readouterr()
        
        assert "평균 점수" in captured.out
        assert "0.85" in captured.out
    
    def test_print_table(self, capsys):
        """테이블 출력 테스트"""
        data = [
            {"이름": "테스트1", "값": 10},
            {"이름": "테스트2", "값": 20}
        ]
        print_table(data)
        captured = capsys.readouterr()
        
        assert "테스트1" in captured.out
        assert "테스트2" in captured.out
    
    def test_print_improvements_empty(self, capsys):
        """빈 개선 사항 출력 테스트"""
        print_improvements([])
        captured = capsys.readouterr()
        
        assert "추가 개선 사항이 없습니다" in captured.out
    
    def test_print_improvements(self, capsys):
        """개선 사항 출력 테스트"""
        improvements = [
            {
                "category": "테스트 카테고리",
                "priority": "HIGH",
                "current": "80%",
                "target": "90%",
                "description": "테스트 설명",
                "recommendation": "테스트 권장사항"
            }
        ]
        print_improvements(improvements)
        captured = capsys.readouterr()
        
        assert "테스트 카테고리" in captured.out
        assert "HIGH" in captured.out
    
    def test_save_text_report(self):
        """텍스트 리포트 저장 테스트"""
        content = "테스트 리포트 내용"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "report.txt"
            save_text_report(content, file_path)
            
            assert file_path.exists()
            assert file_path.read_text(encoding='utf-8') == content
    
    def test_generate_markdown_report(self):
        """마크다운 리포트 생성 테스트"""
        sections = [
            {
                "title": "테스트 섹션",
                "type": "metrics",
                "data": {
                    "평균": 0.85,
                    "총계": 100
                }
            }
        ]
        
        markdown = generate_markdown_report("테스트 리포트", sections)
        
        assert "# 테스트 리포트" in markdown
        assert "## 테스트 섹션" in markdown
        assert "평균" in markdown
    
    def test_generate_markdown_report_with_output(self):
        """마크다운 리포트 생성 및 저장 테스트"""
        sections = [
            {
                "title": "테스트",
                "type": "text",
                "data": {"content": "테스트 내용"}
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            markdown = generate_markdown_report("테스트", sections, output_path)
            
            assert output_path.exists()
            assert markdown is not None

