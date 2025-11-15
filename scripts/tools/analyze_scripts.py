"""
Scripts 폴더의 파일들을 분석하고 분류하는 도구
"""
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def analyze_scripts_directory(root_dir: str = None) -> Dict[str, List[str]]:
    """스크립트 디렉토리 분석"""
    if root_dir is None:
        root_dir = Path(__file__).parent.parent.parent
    
    root_path = Path(root_dir)
    scripts_path = root_path / "scripts"
    
    if not scripts_path.exists():
        print(f"Error: {scripts_path} does not exist")
        return {}
    
    categories = defaultdict(list)
    root_files = []
    
    for file_path in scripts_path.iterdir():
        if file_path.is_file() and file_path.suffix in [".py", ".ps1", ".sh"]:
            root_files.append(file_path.name)
    
    for file_path in scripts_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in [".py", ".ps1", ".sh"]:
            relative_path = file_path.relative_to(scripts_path)
            
            if len(relative_path.parts) > 1:
                category = relative_path.parts[0]
                if category not in [".venv", "__pycache__"]:
                    categories[category].append(str(relative_path))
    
    return {
        "root_files": sorted(root_files),
        "categorized": {k: sorted(v) for k, v in sorted(categories.items())}
    }

def classify_root_files(files: List[str]) -> Dict[str, List[str]]:
    """루트 레벨 파일들을 카테고리별로 분류"""
    classification = {
        "test": [],
        "verify": [],
        "check": [],
        "monitor": [],
        "analyze": [],
        "create": [],
        "assign": [],
        "wait": [],
        "migrate": [],
        "setup": [],
        "wrapper": [],
        "other": []
    }
    
    for file in files:
        name_lower = file.lower()
        
        if name_lower.startswith("test_"):
            classification["test"].append(file)
        elif name_lower.startswith("verify_"):
            classification["verify"].append(file)
        elif name_lower.startswith("check_"):
            classification["check"].append(file)
        elif name_lower.startswith("monitor_"):
            classification["monitor"].append(file)
        elif name_lower.startswith("analyze_"):
            classification["analyze"].append(file)
        elif name_lower.startswith("create_"):
            classification["create"].append(file)
        elif name_lower.startswith("assign_"):
            classification["assign"].append(file)
        elif name_lower.startswith("wait_"):
            classification["wait"].append(file)
        elif name_lower.startswith("migrate_"):
            classification["migrate"].append(file)
        elif name_lower.startswith("init_"):
            classification["migrate"].append(file)
        elif name_lower.startswith("setup_"):
            classification["setup"].append(file)
        elif file.endswith((".ps1", ".sh")):
            classification["wrapper"].append(file)
        else:
            classification["other"].append(file)
    
    return {k: v for k, v in classification.items() if v}

def generate_migration_plan(classification: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
    """파일 이동 계획 생성"""
    plan = {
        "testing/": [],
        "verification/": [],
        "checks/": [],
        "monitoring/": [],
        "analysis/": [],
        "tools/": [],
        "migrations/": [],
        "setup/": [],
        "scripts/": []
    }
    
    for file in classification.get("test", []):
        plan["testing/"].append((file, f"testing/{file}"))
    
    for file in classification.get("verify", []):
        plan["verification/"].append((file, f"verification/{file}"))
    
    for file in classification.get("check", []):
        plan["checks/"].append((file, f"checks/{file}"))
    
    for file in classification.get("monitor", []):
        plan["monitoring/"].append((file, f"monitoring/{file}"))
    
    for file in classification.get("analyze", []):
        plan["analysis/"].append((file, f"analysis/{file}"))
    
    for file in classification.get("create", []) + classification.get("assign", []) + classification.get("wait", []):
        plan["tools/"].append((file, f"tools/{file}"))
    
    for file in classification.get("migrate", []):
        plan["migrations/"].append((file, f"migrations/{file}"))
    
    for file in classification.get("setup", []):
        plan["setup/"].append((file, f"setup/{file}"))
    
    for file in classification.get("wrapper", []):
        plan["scripts/"].append((file, f"scripts/{file}"))
    
    for file in classification.get("other", []):
        plan["tools/"].append((file, f"tools/{file}"))
    
    return {k: v for k, v in plan.items() if v}

def print_analysis_report(analysis: Dict, classification: Dict, plan: Dict):
    """분석 결과 리포트 출력"""
    print("=" * 80)
    print("Scripts 폴더 분석 결과")
    print("=" * 80)
    print()
    
    print(f"루트 레벨 파일 수: {len(analysis['root_files'])}")
    print(f"카테고리별 폴더 수: {len(analysis['categorized'])}")
    print()
    
    print("루트 레벨 파일 분류:")
    print("-" * 80)
    for category, files in sorted(classification.items()):
        if files:
            print(f"\n{category.upper()} ({len(files)}개):")
            for file in files:
                print(f"  - {file}")
    
    print("\n" + "=" * 80)
    print("파일 이동 계획")
    print("=" * 80)
    for target_dir, moves in sorted(plan.items()):
        if moves:
            print(f"\n{target_dir} ({len(moves)}개):")
            for source, target in moves:
                print(f"  {source} -> {target}")

def main():
    """메인 함수"""
    analysis = analyze_scripts_directory()
    
    if not analysis:
        return
    
    classification = classify_root_files(analysis["root_files"])
    plan = generate_migration_plan(classification)
    
    print_analysis_report(analysis, classification, plan)
    
    print("\n" + "=" * 80)
    print("카테고리별 파일 통계")
    print("=" * 80)
    for category, files in sorted(analysis["categorized"].items()):
        print(f"{category}: {len(files)}개")

if __name__ == "__main__":
    main()

