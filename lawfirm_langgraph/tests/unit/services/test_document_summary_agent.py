# -*- coding: utf-8 -*-
"""
DocumentSummaryAgent 단위 테스트
"""

import unittest
from typing import Dict, Any

try:
    from lawfirm_langgraph.core.agents.handlers.document_summary_agent import DocumentSummaryAgent
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.agents.handlers.document_summary_agent import DocumentSummaryAgent


class TestDocumentSummaryAgent(unittest.TestCase):
    """DocumentSummaryAgent 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.agent = DocumentSummaryAgent()
        self.query = "계약서 작성 시 주의할 사항은 무엇인가요?"
    
    def test_summarize_law_document(self):
        """법령 문서 요약 테스트"""
        doc = {
            "law_name": "민법",
            "article_no": "543",
            "content": "계약 해지에 관한 긴 조문 내용입니다. " * 100  # 2000자 이상
        }
        
        summary = self.agent.summarize_document(doc, self.query, use_llm=False)
        
        self.assertIn('summary', summary)
        self.assertIn('key_points', summary)
        self.assertIn('document_type', summary)
        self.assertEqual(summary['document_type'], 'law')
        self.assertLessEqual(len(summary['summary']), 200)
        self.assertGreater(summary['original_length'], summary['summary_length'])
    
    def test_summarize_case_document(self):
        """판례 문서 요약 테스트"""
        doc = {
            "court": "대법원",
            "case_name": "손해배상",
            "case_number": "2018다12345",
            "content": "판례 내용입니다. 계약 해지 시 손해배상 범위에 대한 판결입니다. " * 50  # 1500자 이상
        }
        
        summary = self.agent.summarize_document(doc, self.query, use_llm=False)
        
        self.assertIn('summary', summary)
        self.assertIn('key_points', summary)
        self.assertEqual(summary['document_type'], 'case')
        self.assertLessEqual(len(summary['summary']), 200)
    
    def test_summarize_commentary_document(self):
        """해설 문서 요약 테스트"""
        doc = {
            "title": "계약법 해설",
            "content": "계약서 작성에 대한 해설입니다. " * 100  # 1000자 이상
        }
        
        summary = self.agent.summarize_document(doc, self.query, use_llm=False)
        
        self.assertIn('summary', summary)
        self.assertEqual(summary['document_type'], 'commentary')
        self.assertLessEqual(len(summary['summary']), 200)
    
    def test_summarize_batch(self):
        """배치 요약 테스트"""
        docs = [
            {
                "law_name": "민법",
                "article_no": "543",
                "content": "긴 조문 내용 " * 100
            },
            {
                "court": "대법원",
                "case_name": "판례",
                "content": "판례 내용 " * 80
            }
        ]
        
        summaries = self.agent.summarize_batch(docs, self.query, use_llm=False)
        
        self.assertEqual(len(summaries), 2)
        self.assertIn('summary', summaries[0])
        self.assertIn('summary', summaries[1])
    
    def test_get_document_type(self):
        """문서 유형 판단 테스트"""
        law_doc = {"law_name": "민법", "article_no": "543"}
        case_doc = {"court": "대법원", "case_name": "판례"}
        commentary_doc = {"title": "해설", "type": "commentary"}
        
        self.assertEqual(self.agent._get_document_type(law_doc), 'law')
        self.assertEqual(self.agent._get_document_type(case_doc), 'case')
        self.assertEqual(self.agent._get_document_type(commentary_doc), 'commentary')


if __name__ == '__main__':
    unittest.main()

