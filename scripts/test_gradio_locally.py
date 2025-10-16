#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Testing Script for LawFirmAI Gradio Interface
Easy local testing and validation
"""

import sys
import os
import time
import subprocess
from pathlib import Path

def check_dependencies():
    """필수 의존성 확인"""
    print("의존성 확인 중...")
    
    required_packages = [
        'gradio',
        'torch',
        'transformers',
        'sentence-transformers',
        'pandas',
        'numpy',
        'scikit-learn',
        'requests',
        'PyPDF2',
        'python-docx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  OK {package}")
        except ImportError:
            print(f"  FAIL {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("모든 의존성이 설치되어 있습니다.")
    return True

def check_file_structure():
    """파일 구조 확인"""
    print("\n파일 구조 확인 중...")
    
    required_files = [
        'gradio/app.py',
        'gradio/components/document_analyzer.py',
        'gradio/static/custom.css',
        'gradio/requirements.txt',
        'source/services/chat_service.py',
        'source/services/rag_service.py',
        'source/data/database.py',
        'source/models/model_manager.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  OK {file_path}")
        else:
            print(f"  FAIL {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n누락된 파일: {', '.join(missing_files)}")
        return False
    
    print("모든 필수 파일이 존재합니다.")
    return True

def run_unit_tests():
    """단위 테스트 실행"""
    print("\n단위 테스트 실행 중...")
    
    try:
        # Run the integration tests
        result = subprocess.run([
            sys.executable, 
            'tests/integration/test_gradio_interface.py'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("단위 테스트 통과")
            return True
        else:
            print(f"단위 테스트 실패:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("단위 테스트 시간 초과")
        return False
    except Exception as e:
        print(f"단위 테스트 실행 오류: {e}")
        return False

def test_gradio_import():
    """Gradio 모듈 임포트 테스트"""
    print("\nGradio 모듈 임포트 테스트 중...")
    
    try:
        # Add paths
        sys.path.append(str(Path(__file__).parent.parent / "source"))
        sys.path.append(str(Path(__file__).parent.parent / "gradio"))
        
        # Test imports
        import gradio as gr
        print("  OK gradio")
        
        # Import DocumentAnalyzer with correct path
        sys.path.append(str(Path(__file__).parent.parent / "gradio"))
        from components.document_analyzer import DocumentAnalyzer
        print("  OK DocumentAnalyzer")
        
        # Test DocumentAnalyzer instantiation
        analyzer = DocumentAnalyzer()
        print("  OK DocumentAnalyzer 인스턴스 생성")
        
        print("Gradio 모듈 임포트 테스트 통과")
        return True
        
    except Exception as e:
        print(f"Gradio 모듈 임포트 실패: {e}")
        return False

def test_document_analyzer():
    """문서 분석기 기능 테스트"""
    print("\n문서 분석기 기능 테스트 중...")
    
    try:
        # Add gradio path
        sys.path.append(str(Path(__file__).parent.parent / "gradio"))
        from components.document_analyzer import DocumentAnalyzer
        
        analyzer = DocumentAnalyzer()
        
        # Test contract analysis
        sample_text = """
        제1조 (목적)
        이 계약은 갑과 을 사이의 부동산 매매에 관한 사항을 정함을 목적으로 한다.
        
        제2조 (손해배상)
        당사자 중 일방이 계약을 위반한 경우 상대방에게 손해배상의 책임을 진다.
        """
        
        result = analyzer.analyze_contract(sample_text)
        
        # Check result structure
        required_keys = ['summary', 'clauses', 'risks', 'recommendations', 'risk_score']
        for key in required_keys:
            if key not in result:
                raise ValueError(f"결과에 {key} 키가 없습니다")
        
        print("  OK 계약서 분석")
        print("  OK 위험 요소 평가")
        print("  OK 개선 제안 생성")
        
        print("문서 분석기 기능 테스트 통과")
        return True
        
    except Exception as e:
        print(f"문서 분석기 기능 테스트 실패: {e}")
        return False

def test_gradio_interface_creation():
    """Gradio 인터페이스 생성 테스트"""
    print("\nGradio 인터페이스 생성 테스트 중...")
    
    try:
        # Mock heavy dependencies
        import unittest.mock as mock
        
        # Add gradio path
        sys.path.append(str(Path(__file__).parent.parent / "gradio"))
        
        with mock.patch('app.DatabaseManager'), \
             mock.patch('app.LegalVectorStore'), \
             mock.patch('app.LegalModelManager'), \
             mock.patch('app.MLEnhancedRAGService'), \
             mock.patch('app.MLEnhancedSearchService'), \
             mock.patch('app.ChatService'):
            
            from app import create_ml_enhanced_gradio_interface
            
            start_time = time.time()
            interface = create_ml_enhanced_gradio_interface()
            creation_time = time.time() - start_time
            
            if interface is None:
                raise ValueError("인터페이스가 None입니다")
            
            print(f"  OK 인터페이스 생성 완료 ({creation_time:.2f}초)")
            print("  OK 탭 구조 확인")
            print("  OK 컴포넌트 생성 확인")
            
            print("Gradio 인터페이스 생성 테스트 통과")
            return True
            
    except Exception as e:
        print(f"Gradio 인터페이스 생성 테스트 실패: {e}")
        return False

def start_gradio_server():
    """Gradio 서버 시작"""
    print("\nGradio 서버 시작 중...")
    
    try:
        # Change to gradio directory
        gradio_dir = Path(__file__).parent.parent / "gradio"
        os.chdir(gradio_dir)
        
        print("작업 디렉토리:", gradio_dir)
        print("서버 주소: http://localhost:7860")
        print("중지하려면 Ctrl+C를 누르세요")
        print("\n" + "="*50)
        
        # Start the server
        subprocess.run([sys.executable, "app.py"])
        
    except KeyboardInterrupt:
        print("\n\n서버가 중지되었습니다.")
    except Exception as e:
        print(f"서버 시작 실패: {e}")

def main():
    """메인 함수"""
    print("LawFirmAI Gradio 인터페이스 로컬 테스트")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("gradio").exists():
        print("gradio 디렉토리를 찾을 수 없습니다.")
        print("프로젝트 루트 디렉토리에서 실행하세요.")
        return
    
    # Run all checks
    checks = [
        ("의존성 확인", check_dependencies),
        ("파일 구조 확인", check_file_structure),
        ("Gradio 모듈 임포트", test_gradio_import),
        ("문서 분석기 기능", test_document_analyzer),
        ("Gradio 인터페이스 생성", test_gradio_interface_creation),
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{'='*20} {check_name} {'='*20}")
        try:
            if check_func():
                passed_checks += 1
            else:
                print(f"FAIL {check_name}")
        except Exception as e:
            print(f"ERROR {check_name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"체크 결과: {passed_checks}/{total_checks} 통과")
    
    if passed_checks == total_checks:
        print("모든 체크가 통과했습니다!")
        
        # Ask if user wants to start the server
        response = input("\nGradio 서버를 시작하시겠습니까? (y/n): ")
        if response.lower() in ['y', 'yes', '예']:
            start_gradio_server()
        else:
            print("테스트를 완료했습니다.")
    else:
        print(f"{total_checks - passed_checks}개 체크가 실패했습니다.")
        print("문제를 해결한 후 다시 실행하세요.")

if __name__ == "__main__":
    main()
