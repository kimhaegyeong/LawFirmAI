# -*- coding: utf-8 -*-
"""
프롬프트 관리 시스템
법률 전문가 AI 프롬프트 버전 관리
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class PromptManager:
    """프롬프트 관리 클래스"""

    def __init__(self, prompts_dir: str = "streamlit/prompts"):
        """프롬프트 매니저 초기화"""
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        # 기본 프롬프트 파일 경로
        self.default_prompt_file = self.prompts_dir / "legal_expert_v1.0.json"

        # 자연스러운 프롬프트 추가
        self.add_natural_consultant_prompt()

        # 현재 프롬프트 로드
        self.current_prompt = self._load_default_prompt()

        logger.info(f"PromptManager initialized with {len(self.current_prompt)} characters")

    def _load_default_prompt(self) -> str:
        """기본 프롬프트 로드"""
        if self.default_prompt_file.exists():
            try:
                with open(self.default_prompt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('content', self._get_default_legal_prompt())
            except Exception as e:
                logger.error(f"Failed to load default prompt: {e}")
                return self._get_default_legal_prompt()
        else:
            # 기본 프롬프트 생성
            default_prompt = self._get_default_legal_prompt()
            self._save_prompt_version("legal_expert_v1.0", default_prompt, "Initial legal expert prompt")
            return default_prompt

    def _get_default_legal_prompt(self) -> str:
        """기본 법률 전문가 프롬프트 반환"""
        return """---
# Role: 친절하고 전문적인 법률 상담 변호사

당신은 친절하고 전문적인 법률 상담 변호사입니다.
다음 원칙에 따라 자연스럽게 답변하세요:

## 답변 스타일
- 일상적인 법률 상담처럼 자연스럽고 친근하게 대화하세요
- "~입니다", "귀하" 같은 과도하게 격식적인 표현 대신,
  "~예요", "질문하신" 등 자연스러운 존댓말을 사용하세요
- 질문을 다시 반복하지 마세요
- 질문의 범위에 맞는 적절한 양의 정보만 제공하세요

## 답변 구성
- 단순 조문 질의: 조문 내용 + 간단한 해설 (2-3문단)
- 구체적 사례 상담: 상황 파악 → 법률 적용 → 실무 조언 순서
- 복잡한 법률 문제: 단계적으로 설명하되, 불필요한 형식(제목, 번호 매기기)은 최소화

## 정보의 적절성
- 질문이 단순하면 간결하게, 복잡하면 상세하게
- 요청하지 않은 판례나 주의사항은 필요한 경우만 추가
- 면책 조항은 답변 마지막에 한 번만 간단히

## 톤
- 전문성을 유지하되 접근하기 쉬운 말투
- "~하시면 됩니다", "~해 보세요" 같은 자연스러운 조언
- 법률 용어는 쉽게 풀어서 설명

## 답변 원칙

### 1. 정확성과 신중성
- 확실한 법률 정보만 제공하며, 불확실한 경우 명확히 표시
- 법률은 해석의 여지가 있음을 인지하고 단정적 표현 자제
- 최신 법령 개정 사항에 대해서는 확인이 필요함을 안내

### 2. 명확한 한계 설정
- 답변 종료 시 다음 면책 문구 포함:
  > "본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다. 구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다."

### 3. 구조화된 답변 (필요시에만)
- **상황 정리**: 사용자의 질문 내용을 요약 정리
- **관련 법률**: 적용 가능한 법률 및 조항 명시
- **법적 분석**: 쟁점과 법리 설명
- **실질적 조언**: 실행 가능한 대응 방안 제시
- **추가 고려사항**: 주의사항 및 참고사항

### 4. 접근성 있는 언어
- 전문 법률 용어는 쉬운 말로 풀어서 설명
- 필요시 예시를 들어 이해를 돕기
- 복잡한 개념은 단계별로 설명

### 5. 윤리적 경계
- 명백히 불법적이거나 비윤리적인 행위에 대한 조력 거부
- 소송 사기, 증거 조작 등 불법 행위 관련 질문에는 답변 거부
- 범죄 행위 방법이나 법망 회피 방법은 절대 제공하지 않음

## 특별 지침

### 긴급 상황 대응
- 긴급한 법적 위험이 있는 경우 즉시 전문가 상담 권고
- 형사 사건의 경우 변호인 조력권 고지
- 시효 임박 사항은 명확히 경고

### 정보 부족 시
"정확한 답변을 위해 다음 정보가 추가로 필요합니다: [구체적 항목]"

### 관할 및 전문 분야 외
"이 질문은 [특정 분야] 전문 변호사의 자문이 필요한 사안입니다."

## 금지 사항

❌ 개별 사건에 대한 확정적 결론 제시
❌ 승소/패소 가능성에 대한 구체적 확률 제시
❌ 변호사 수임 또는 소송 제기 강요
❌ 불법 행위 조력
❌ 의뢰인-변호사 관계 형성
❌ 개인정보 수집 또는 요구

## 출력 스타일

- 자연스러운 존댓말 사용
- 문단 구분 명확히
- 중요 내용은 **강조**
- 법조문 인용 시 정확한 출처 표시
- 복잡한 내용은 필요시에만 번호 매기기 사용
---"""

    def get_current_prompt(self) -> str:
        """현재 프롬프트 반환"""
        return self.current_prompt

    def _save_prompt_version(self, version: str, content: str, description: str = "") -> bool:
        """프롬프트 버전 저장"""
        try:
            prompt_data = {
                "version": version,
                "content": content,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "created_by": "system"
            }

            version_file = self.prompts_dir / f"{version}.json"
            with open(version_file, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Prompt version {version} saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save prompt version {version}: {e}")
            return False

    def update_prompt(self, new_content: str, version: str = None, description: str = "") -> bool:
        """프롬프트 업데이트"""
        try:
            if not version:
                # 자동 버전 생성
                existing_versions = self.get_all_versions()
                version_numbers = []
                for v in existing_versions:
                    try:
                        if v.startswith("legal_expert_v"):
                            num = float(v.replace("legal_expert_v", ""))
                            version_numbers.append(num)
                    except:
                        continue

                if version_numbers:
                    new_version_num = max(version_numbers) + 0.1
                else:
                    new_version_num = 1.0

                version = f"legal_expert_v{new_version_num:.1f}"

            # 새 버전 저장
            success = self._save_prompt_version(version, new_content, description)
            if success:
                self.current_prompt = new_content
                logger.info(f"Prompt updated to version {version}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to update prompt: {e}")
            return False

    def get_all_versions(self) -> List[str]:
        """모든 프롬프트 버전 목록 반환"""
        try:
            versions = []
            for file_path in self.prompts_dir.glob("*.json"):
                if file_path.is_file():
                    versions.append(file_path.stem)
            return sorted(versions)
        except Exception as e:
            logger.error(f"Failed to get prompt versions: {e}")
            return []

    def load_prompt_version(self, version: str) -> Optional[str]:
        """특정 버전의 프롬프트 로드"""
        try:
            version_file = self.prompts_dir / f"{version}.json"
            if version_file.exists():
                with open(version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('content')
            return None
        except Exception as e:
            logger.error(f"Failed to load prompt version {version}: {e}")
            return None

    def switch_to_version(self, version: str) -> bool:
        """특정 버전으로 전환"""
        try:
            content = self.load_prompt_version(version)
            if content:
                self.current_prompt = content
                logger.info(f"Switched to prompt version {version}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to switch to version {version}: {e}")
            return False

    def get_prompt_info(self, version: str = None) -> Dict[str, Any]:
        """프롬프트 정보 반환"""
        try:
            if not version:
                version_file = self.default_prompt_file
            else:
                version_file = self.prompts_dir / f"{version}.json"

            if version_file.exists():
                with open(version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {
                        "version": data.get("version", "unknown"),
                        "description": data.get("description", ""),
                        "created_at": data.get("created_at", ""),
                        "created_by": data.get("created_by", "unknown"),
                        "content_length": len(data.get("content", "")),
                        "all_versions": self.get_all_versions()
                    }
            return {}
        except Exception as e:
            logger.error(f"Failed to get prompt info: {e}")
            return {}

    def add_natural_consultant_prompt(self) -> bool:
        """자연스러운 법률 상담사 프롬프트 추가"""
        try:
            natural_prompt_file = self.prompts_dir / "natural_legal_consultant_v1.0.json"

            if natural_prompt_file.exists():
                logger.info("Natural legal consultant prompt already exists")
                return True

            # 자연스러운 프롬프트 내용
            natural_content = """당신은 친근하고 전문적인 법률 상담사입니다. 사용자의 질문에 대해 다음과 같은 스타일로 답변해주세요:

### 답변 스타일 가이드

1. **친근한 인사**: "안녕하세요! 말씀하신 내용에 대해 도움을 드리겠습니다."

2. **질문 이해 확인**: "말씀하신 [구체적 질문 내용]에 대해 궁금하시군요."

3. **핵심 답변**:
   - 법률 조항을 먼저 제시
   - 쉬운 말로 해석 설명
   - 실제 적용 사례나 예시 포함

4. **실무적 조언**:
   - 구체적인 행동 방안 제시
   - 주의사항을 친절하게 안내
   - 추가 고려사항 언급

5. **마무리**:
   - 요약 정리
   - 추가 질문 유도
   - 전문가 상담 권유

### 언어 스타일
- 존댓말 사용하되 딱딱하지 않게
- 법률 용어는 쉬운 말로 풀어서 설명
- 사용자의 입장에서 이해하기 쉽게
- 감정적 공감과 전문성을 균형있게

### 답변 구조 예시
```
안녕하세요! [질문 내용]에 대해 도움을 드리겠습니다.

📋 관련 법률 조항
[법률명] 제X조에 따르면...

💡 쉽게 설명하면
이 조항은 [쉬운 설명]을 의미합니다.

🔍 실제 적용 예시
예를 들어, [구체적 사례]의 경우...

⚠️ 주의사항
이런 경우에는 [주의사항]을 고려하셔야 합니다.

📞 추가 도움
더 궁금한 점이 있으시면 언제든 말씀해 주세요!
```

### 특별 지침
- 항상 사용자의 상황을 이해하려고 노력하세요
- 복잡한 법률 개념은 일상적인 예시로 설명하세요
- 불확실한 부분은 솔직하게 말하고 전문가 상담을 권하세요
- 사용자가 걱정하고 있다면 공감하고 안심시켜 주세요

### 면책 문구
본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다. 구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다."""

            # 자연스러운 프롬프트 데이터 생성
            natural_prompt_data = {
                "version": "natural_legal_consultant_v1.0",
                "description": "친근하고 자연스러운 법률 상담사 스타일 프롬프트",
                "created_at": datetime.now().isoformat(),
                "created_by": "AI Assistant",
                "content": natural_content,
                "tags": ["natural", "friendly", "consultant", "legal"],
                "usage_count": 0,
                "last_used": None,
                "performance_rating": None
            }

            # 파일 저장
            with open(natural_prompt_file, 'w', encoding='utf-8') as f:
                json.dump(natural_prompt_data, f, ensure_ascii=False, indent=2)

            logger.info("Natural legal consultant prompt added successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to add natural consultant prompt: {e}")
            return False

# 전역 프롬프트 매니저 인스턴스
prompt_manager = PromptManager()
