#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
개별 테스트 파일 직접 실행
"""
import sys
import os

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
parent_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, parent_root)

os.chdir(project_root)

# 테스트할 파일명
test_file = sys.argv[1] if len(sys.argv) > 1 else "test_processing_reasoning_extractor.py"

print(f"Running test file: {test_file}")
print("=" * 80)

# 테스트 파일 경로
test_path = os.path.join(project_root, "tests", "langgraph_core", test_file)

if not os.path.exists(test_path):
    print(f"Error: Test file not found: {test_path}")
    sys.exit(1)

# 모듈 경로로 변환
module_path = f"tests.langgraph_core.{test_file.replace('.py', '')}"

try:
    # 모듈 import
    module = __import__(module_path, fromlist=[''])
    
    # 테스트 클래스 찾기
    test_classes = [attr for attr in dir(module) if attr.startswith('Test')]
    
    print(f"Found test classes: {test_classes}")
    
    # 각 테스트 클래스의 테스트 메서드 실행
    total_tests = 0
    passed = 0
    failed = 0
    
    for class_name in test_classes:
        test_class = getattr(module, class_name)
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        print(f"\n[{class_name}]")
        print("-" * 80)
        
        for method_name in test_methods:
            total_tests += 1
            test_method = getattr(test_class, method_name)
            instance = test_class()
            
            try:
                test_method(instance)
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                failed += 1
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {total_tests} tests")
    print("=" * 80)
    
    sys.exit(0 if failed == 0 else 1)
    
except Exception as e:
    print(f"Error loading test module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

