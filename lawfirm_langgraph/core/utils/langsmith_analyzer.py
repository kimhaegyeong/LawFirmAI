import os
import uuid
import sys
import time
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from collections import defaultdict, OrderedDict

try:
    from langsmith import Client
    from langsmith.schemas import Run
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Run = None
    Client = None

logger = get_logger(__name__)

STATE_GROUP_NAMES = [
    "input", "classification", "search", "analysis", "answer",
    "document", "multi_turn", "validation", "control", "common"
]

MODEL_PRICING = {
    "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
    "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
    "gpt-3.5-turbo": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
    "gemini-pro": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
    "gemini-1.5-pro": {"input": 0.00125 / 1000, "output": 0.005 / 1000},
    "claude-3-opus": {"input": 0.015 / 1000, "output": 0.075 / 1000},
    "claude-3-sonnet": {"input": 0.003 / 1000, "output": 0.015 / 1000},
    "claude-3-haiku": {"input": 0.00025 / 1000, "output": 0.00125 / 1000},
}


class LangGraphQueryAnalyzer:
    """LangGraph 질의 분석 및 개선 제안을 위한 LangSmith 분석기"""
    
    def __init__(self, api_key: Optional[str] = None, project_name: Optional[str] = None):
        """
        초기화
        
        Args:
            api_key: LangSmith API 키 (없으면 환경 변수에서 가져옴)
            project_name: 프로젝트 이름 (없으면 환경 변수에서 가져옴)
        """
        if not LANGSMITH_AVAILABLE:
            raise ImportError("langsmith package is not installed. Install with: pip install langsmith")
        
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        self.project_name = project_name or os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "LawFirmAI")
        
        if not self.api_key:
            raise ValueError("LangSmith API key is required. Set LANGSMITH_API_KEY or LANGCHAIN_API_KEY environment variable.")
        
        self.client = Client(api_key=self.api_key)
        self._max_cache_size = int(os.getenv("LANGSMITH_CACHE_SIZE", "1000"))
        self._run_cache: OrderedDict[str, Run] = OrderedDict()
        self._tree_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cache_stats = {"hits": 0, "misses": 0}
        self._analysis_id = str(uuid.uuid4())[:8]
        self._max_retries = int(os.getenv("LANGSMITH_MAX_RETRIES", "3"))
        self._retry_delay = float(os.getenv("LANGSMITH_RETRY_DELAY", "2.0"))
        self._rate_limit_delay = float(os.getenv("LANGSMITH_RATE_LIMIT_DELAY", "5.0"))
        logger.info(f"LangGraphQueryAnalyzer initialized for project: {self.project_name} (analysis_id: {self._analysis_id})")
    
    def _read_run_with_retry(self, run_id: str) -> Optional[Run]:
        """Rate limit을 고려한 재시도 로직이 포함된 run 읽기"""
        for attempt in range(self._max_retries):
            try:
                return self.client.read_run(run_id)
            except Exception as e:
                error_str = str(e).lower()
                
                if "rate limit" in error_str or "429" in error_str:
                    delay = self._rate_limit_delay * (attempt + 1)
                    logger.warning(f"[{self._analysis_id}] Rate limit hit for run {run_id}, waiting {delay:.1f}s (attempt {attempt + 1}/{self._max_retries})")
                    time.sleep(delay)
                elif attempt < self._max_retries - 1:
                    delay = self._retry_delay * (attempt + 1)
                    logger.warning(f"[{self._analysis_id}] Error reading run {run_id}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self._max_retries}): {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"[{self._analysis_id}] Failed to read run {run_id} after {self._max_retries} attempts: {e}")
        
        return None
    
    def clear_cache(self):
        """캐시 클리어"""
        self._run_cache.clear()
        self._tree_cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
        logger.info(f"[{self._analysis_id}] Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (self._cache_stats["hits"] / total * 100) if total > 0 else 0
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_rate": hit_rate,
            "run_cache_size": len(self._run_cache),
            "tree_cache_size": len(self._tree_cache)
        }
    
    def _manage_cache_size(self, cache: OrderedDict, max_size: int):
        """캐시 크기 관리 (LRU 방식)"""
        while len(cache) >= max_size:
            cache.popitem(last=False)
    
    def _calculate_cost(self, tokens: int, model_name: Optional[str] = None, token_usage: Optional[Dict[str, int]] = None) -> float:
        """토큰 사용량을 기반으로 비용 계산"""
        if not tokens or tokens <= 0:
            return 0.0
        
        if not model_name:
            model_name = "gpt-3.5-turbo"
        
        model_key = model_name.lower()
        pricing = None
        
        for key, price in MODEL_PRICING.items():
            if key in model_key:
                pricing = price
                break
        
        if not pricing:
            pricing = MODEL_PRICING.get("gpt-3.5-turbo", {"input": 0.0015 / 1000, "output": 0.002 / 1000})
        
        if token_usage and isinstance(token_usage, dict):
            input_tokens = token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
            output_tokens = token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))
            if input_tokens == 0 and output_tokens == 0:
                input_tokens = int(tokens * 0.7)
                output_tokens = tokens - input_tokens
        else:
            input_tokens = int(tokens * 0.7)
            output_tokens = tokens - input_tokens
        
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
        return cost
    
    def _extract_token_usage(self, run: Run) -> Optional[Dict[str, int]]:
        """다양한 소스에서 토큰 사용량 추출 (개선된 버전)"""
        token_usage = {}
        
        try:
            # 1. usage_metadata 직접 확인
            if hasattr(run, 'usage_metadata') and run.usage_metadata:
                if hasattr(run.usage_metadata, 'input_tokens'):
                    token_usage['input_tokens'] = run.usage_metadata.input_tokens
                if hasattr(run.usage_metadata, 'output_tokens'):
                    token_usage['output_tokens'] = run.usage_metadata.output_tokens
                if hasattr(run.usage_metadata, 'total_tokens'):
                    token_usage['total_tokens'] = run.usage_metadata.total_tokens
            
            # 2. extra.usage_metadata 확인
            if hasattr(run, 'extra') and run.extra:
                if isinstance(run.extra, dict):
                    usage = run.extra.get('usage_metadata', {})
                    if isinstance(usage, dict):
                        if 'input_tokens' in usage:
                            token_usage['input_tokens'] = usage['input_tokens']
                        if 'output_tokens' in usage:
                            token_usage['output_tokens'] = usage['output_tokens']
                        if 'total_tokens' in usage:
                            token_usage['total_tokens'] = usage['total_tokens']
                        if 'prompt_tokens' in usage:
                            token_usage['prompt_tokens'] = usage['prompt_tokens']
                        if 'completion_tokens' in usage:
                            token_usage['completion_tokens'] = usage['completion_tokens']
            
            # 3. extra.token_usage 확인 (다양한 형식)
            if hasattr(run, 'extra') and run.extra:
                if isinstance(run.extra, dict):
                    tokens = run.extra.get('token_usage', {})
                    if isinstance(tokens, dict):
                        if 'input_tokens' in tokens:
                            token_usage['input_tokens'] = tokens['input_tokens']
                        if 'output_tokens' in tokens:
                            token_usage['output_tokens'] = tokens['output_tokens']
                        if 'total_tokens' in tokens:
                            token_usage['total_tokens'] = tokens['total_tokens']
                        if 'prompt_tokens' in tokens:
                            token_usage['prompt_tokens'] = tokens['prompt_tokens']
                        if 'completion_tokens' in tokens:
                            token_usage['completion_tokens'] = tokens['completion_tokens']
            
            # 4. total_tokens 직접 확인
            if hasattr(run, 'total_tokens') and run.total_tokens:
                token_usage['total_tokens'] = run.total_tokens
            
            # 5. total_tokens가 없으면 input/output 합산
            if 'total_tokens' not in token_usage or token_usage.get('total_tokens', 0) == 0:
                input_tokens = token_usage.get('input_tokens', token_usage.get('prompt_tokens', 0))
                output_tokens = token_usage.get('output_tokens', token_usage.get('completion_tokens', 0))
                if input_tokens > 0 or output_tokens > 0:
                    token_usage['total_tokens'] = input_tokens + output_tokens
            
            return token_usage if token_usage else None
            
        except Exception as e:
            logger.debug(f"[{self._analysis_id}] Error extracting token usage for run {run.id if hasattr(run, 'id') else 'unknown'}: {e}")
            return None
    
    def _extract_model_name(self, run: Run) -> Optional[str]:
        """Run에서 모델 이름 추출"""
        if hasattr(run, "extra") and run.extra:
            if isinstance(run.extra, dict):
                invocation_params = run.extra.get("invocation_params", {})
                if isinstance(invocation_params, dict):
                    model = invocation_params.get("model") or invocation_params.get("model_name")
                    if model:
                        return str(model)
        
        if hasattr(run, "run_type") and run.run_type:
            if "gpt" in run.run_type.lower():
                return "gpt-3.5-turbo"
            elif "gemini" in run.run_type.lower():
                return "gemini-pro"
            elif "claude" in run.run_type.lower():
                return "claude-3-sonnet"
        
        return None
    
    def _validate_run_id(self, run_id: str) -> bool:
        """Run ID 형식 검증"""
        if not run_id:
            return False
        try:
            uuid.UUID(run_id)
            return True
        except (ValueError, AttributeError):
            return False
    
    def get_recent_runs(
        self,
        hours: int = 24,
        limit: int = 100,
        filter_query: Optional[str] = None
    ) -> List[Run]:
        """
        최근 실행된 LangGraph runs 조회
        
        Args:
            hours: 조회할 시간 범위 (시간 단위)
            limit: 최대 조회 개수
            filter_query: 필터 쿼리 (예: "tags:production")
        
        Returns:
            List[Run]: 실행된 runs 리스트
        """
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time,
                limit=limit,
                filter=filter_query
            ))
            
            logger.info(f"[{self._analysis_id}] Found {len(runs)} runs in the last {hours} hours")
            return runs
        except Exception as e:
            logger.error(f"[{self._analysis_id}] Error fetching runs: {e}", exc_info=True)
            return []
    
    def analyze_run_performance(self, run: Run, tree: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        단일 run의 성능 분석
        
        Args:
            run: 분석할 Run 객체
            tree: 이미 조회한 RunTree (선택적, 중복 조회 방지)
        
        Returns:
            Dict[str, Any]: 성능 분석 결과
        """
        analysis = {
            "run_id": str(run.id),
            "query": self._extract_query(run),
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "duration": None,
            "status": run.status,
            "error": str(run.error) if run.error else None,
            "nodes": [],
            "total_tokens": 0,
            "total_cost": 0.0,
            "bottlenecks": [],
            "recommendations": [],
            "state_info": self._extract_state_info(run)
        }
        
        if run.start_time and run.end_time:
            analysis["duration"] = (run.end_time - run.start_time).total_seconds()
        
        child_runs = self._get_child_runs(str(run.id), recursive=True)
        node_analysis = self._analyze_nodes(child_runs)
        analysis["nodes"] = node_analysis["nodes"]
        analysis["total_tokens"] = node_analysis["total_tokens"]
        analysis["total_cost"] = node_analysis["total_cost"]
        analysis["bottlenecks"] = self._identify_bottlenecks(node_analysis)
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _extract_state_info(self, run: Run) -> Dict[str, Any]:
        """Run에서 state 정보 추출 (개선된 버전)"""
        state_info = {
            "has_inputs": bool(run.inputs),
            "has_outputs": bool(run.outputs),
            "input_keys": [],
            "output_keys": [],
            "state_snapshot": None,
            "input_state": None,
            "output_state": None,
            "state_structure": {},
            "state_changes": {}
        }
        
        run_id_str = str(run.id) if hasattr(run, 'id') else None
        
        if run.inputs and isinstance(run.inputs, dict):
            state_info["input_keys"] = list(run.inputs.keys())
            
            # State 찾기: "state" 키 또는 state가 포함된 키, 또는 dict 타입 값
            input_state = None
            for key in run.inputs.keys():
                if key == "state" or "state" in key.lower():
                    input_state = run.inputs.get(key)
                    state_info["state_snapshot"] = "present"
                    break
                elif isinstance(run.inputs.get(key), dict) and len(run.inputs.get(key)) > 0:
                    # 큰 dict는 state일 가능성이 높음
                    potential_state = run.inputs.get(key)
                    if isinstance(potential_state, dict) and len(potential_state) > 3:
                        input_state = potential_state
                        state_info["state_snapshot"] = "present"
                        break
            
            if input_state:
                state_info["input_state"] = self._analyze_state_structure(input_state, "input", run_id=run_id_str)
                state_info["state_structure"]["input"] = state_info["input_state"]
        
        if run.outputs and isinstance(run.outputs, dict):
            state_info["output_keys"] = list(run.outputs.keys())
            
            # State 찾기: "state" 키 또는 state가 포함된 키, 또는 dict 타입 값
            output_state = None
            for key in run.outputs.keys():
                if key == "state" or "state" in key.lower():
                    output_state = run.outputs.get(key)
                    state_info["state_snapshot"] = "updated"
                    break
                elif isinstance(run.outputs.get(key), dict) and len(run.outputs.get(key)) > 0:
                    # 큰 dict는 state일 가능성이 높음
                    potential_state = run.outputs.get(key)
                    if isinstance(potential_state, dict) and len(potential_state) > 3:
                        output_state = potential_state
                        state_info["state_snapshot"] = "updated"
                        break
            
            if output_state:
                state_info["output_state"] = self._analyze_state_structure(output_state, "output", run_id=run_id_str)
                state_info["state_structure"]["output"] = state_info["output_state"]
                
                if state_info["input_state"] and state_info["output_state"]:
                    node_name = run.name or "unknown"
                    state_info["state_changes"] = self._detect_state_changes(
                        state_info["input_state"],
                        state_info["output_state"],
                        node_name=node_name
                    )
        
        return state_info
    
    def _analyze_state_structure(self, state: Any, context: str = "", run_id: Optional[str] = None) -> Dict[str, Any]:
        """State 구조 분석"""
        if state is None:
            return {"exists": False}
        
        try:
            analysis = {
                "exists": True,
                "type": type(state).__name__,
                "is_dict": isinstance(state, dict),
                "keys": [],
                "key_count": 0,
                "state_groups": {},
                "size_estimate": 0
            }
            
            if isinstance(state, dict):
                analysis["keys"] = list(state.keys())
                analysis["key_count"] = len(state.keys())
                
                for key, value in state.items():
                    try:
                        value_type = type(value).__name__
                        value_size = self._estimate_size(value)
                        analysis["size_estimate"] += value_size
                        
                        if key in STATE_GROUP_NAMES:
                            analysis["state_groups"][key] = {
                                "exists": True,
                                "type": value_type,
                                "size": value_size,
                                "keys": list(value.keys()) if isinstance(value, dict) else None
                            }
                    except Exception as e:
                        logger.debug(f"[{self._analysis_id}] Error analyzing state key {key}: {e}")
                        continue
                
                logger.debug(f"[{self._analysis_id}] State structure ({context}): {analysis['key_count']} keys, "
                           f"groups: {list(analysis['state_groups'].keys())}, "
                           f"size: ~{analysis['size_estimate']} bytes")
                
                if analysis['state_groups']:
                    logger.debug(f"[{self._analysis_id}] State groups detail ({context}):")
                    for group_name, group_info in analysis['state_groups'].items():
                        keys_info = f", keys: {group_info.get('keys', [])[:5]}" if group_info.get('keys') else ""
                        logger.debug(f"[{self._analysis_id}]   - {group_name}: type={group_info.get('type')}, "
                                   f"size={group_info.get('size')} bytes{keys_info}")
            
            return analysis
        except Exception as e:
            logger.warning(f"[{self._analysis_id}] Error analyzing state structure (context: {context}, run_id: {run_id}): {e}", exc_info=True)
            return {"exists": False, "error": str(e)}
    
    def _estimate_size(self, obj: Any, max_depth: int = 3, current_depth: int = 0) -> int:
        """객체 크기 추정 (바이트, 재귀적)"""
        if current_depth >= max_depth:
            return 0
        
        try:
            size = sys.getsizeof(obj)
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    size += self._estimate_size(key, max_depth, current_depth + 1)
                    size += self._estimate_size(value, max_depth, current_depth + 1)
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    size += self._estimate_size(item, max_depth, current_depth + 1)
            
            return size
        except Exception:
            try:
                return len(str(obj))
            except Exception:
                return 0
    
    def _detect_state_changes(self, input_state: Dict[str, Any], output_state: Dict[str, Any], node_name: str = "unknown") -> Dict[str, Any]:
        """State 변경 사항 감지"""
        changes = {
            "keys_added": [],
            "keys_removed": [],
            "keys_modified": [],
            "groups_modified": [],
            "total_changes": 0
        }
        
        if not input_state.get("exists") or not output_state.get("exists"):
            return changes
        
        input_keys = set(input_state.get("keys", []))
        output_keys = set(output_state.get("keys", []))
        
        changes["keys_added"] = list(output_keys - input_keys)
        changes["keys_removed"] = list(input_keys - output_keys)
        changes["keys_modified"] = list(input_keys & output_keys)
        
        input_groups = input_state.get("state_groups", {})
        output_groups = output_state.get("state_groups", {})
        
        all_group_names = input_groups.keys() | output_groups.keys()
        
        for group_name in all_group_names:
            input_group = input_groups.get(group_name, {})
            output_group = output_groups.get(group_name, {})
            
            if input_group.get("exists") != output_group.get("exists"):
                changes["groups_modified"].append({
                    "group": group_name,
                    "change": "created" if output_group.get("exists") else "removed"
                })
            elif input_group.get("exists") and output_group.get("exists"):
                input_group_keys = set(input_group.get("keys", []) or [])
                output_group_keys = set(output_group.get("keys", []) or [])
                if input_group_keys != output_group_keys:
                    changes["groups_modified"].append({
                        "group": group_name,
                        "change": "modified",
                        "keys_added": list(output_group_keys - input_group_keys),
                        "keys_removed": list(input_group_keys - output_group_keys)
                    })
        
        changes["total_changes"] = (
            len(changes["keys_added"]) +
            len(changes["keys_removed"]) +
            len(changes["keys_modified"]) +
            len(changes["groups_modified"])
        )
        
        if changes["total_changes"] > 0:
            logger.debug(f"[{self._analysis_id}] State changes at {node_name}: {changes['total_changes']} total changes")
            if changes.get('keys_added'):
                logger.debug(f"[{self._analysis_id}]   - Keys added ({len(changes['keys_added'])}): {changes['keys_added'][:5]}")
            if changes.get('keys_removed'):
                logger.debug(f"[{self._analysis_id}]   - Keys removed ({len(changes['keys_removed'])}): {changes['keys_removed'][:5]}")
            if changes.get('groups_modified'):
                logger.debug(f"[{self._analysis_id}]   - Groups modified: {[g['group'] for g in changes['groups_modified']]}")
        
        return changes
    
    def analyze_state_flow(self, run_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """RunTree 전체에서 State 흐름 분석"""
        tree = self.get_run_tree(run_id, max_depth, show_progress=True)
        if not tree or tree.get("error"):
            logger.error(f"[{self._analysis_id}] Cannot analyze state flow: {tree.get('error', 'Unknown error')}", exc_info=False)
            return {}
        
        state_flow = {
            "nodes_with_state": [],
            "state_transitions": [],
            "state_groups_usage": defaultdict(int),
            "state_size_evolution": [],
            "state_changes_summary": {
                "total_nodes": 0,
                "nodes_with_state": 0,
                "nodes_with_changes": 0,
                "total_changes": 0
            }
        }
        
        def traverse_tree(node: Dict[str, Any], depth: int = 0):
            state_flow["state_changes_summary"]["total_nodes"] += 1
            
            state_info = node.get("state_info", {})
            if not isinstance(state_info, dict):
                state_info = {}
            
            if state_info.get("state_snapshot"):
                state_flow["state_changes_summary"]["nodes_with_state"] += 1
                
                # 안전하게 state 정보 추출
                input_state = state_info.get("input_state")
                if input_state is None:
                    input_state = {}
                elif not isinstance(input_state, dict):
                    input_state = {}
                
                output_state = state_info.get("output_state")
                if output_state is None:
                    output_state = {}
                elif not isinstance(output_state, dict):
                    output_state = {}
                
                state_structure = state_info.get("state_structure", {})
                if not isinstance(state_structure, dict):
                    state_structure = {}
                
                output_structure = state_structure.get("output", {})
                if not isinstance(output_structure, dict):
                    output_structure = {}
                
                state_groups = output_structure.get("state_groups", {})
                if not isinstance(state_groups, dict):
                    state_groups = {}
                
                # State 전달 정보 확인
                state_inherited = state_info.get("state_inherited", False)
                inherited_keys = state_info.get("inherited_keys", [])
                
                node_info = {
                    "node_name": node.get("name", "unknown"),
                    "run_id": node.get("run_id"),
                    "depth": depth,
                    "has_input_state": input_state.get("exists", False),
                    "has_output_state": output_state.get("exists", False),
                    "state_groups": list(state_groups.keys()),
                    "state_size": output_state.get("size_estimate", 0),
                    "state_inherited": state_inherited,
                    "inherited_keys": inherited_keys if state_inherited else []
                }
                
                state_flow["nodes_with_state"].append(node_info)
                
                for group in node_info["state_groups"]:
                    state_flow["state_groups_usage"][group] += 1
                
                if node_info["state_size"] > 0:
                    state_flow["state_size_evolution"].append({
                        "node": node_info["node_name"],
                        "depth": depth,
                        "size": node_info["state_size"]
                    })
                
                changes = state_info.get("state_changes", {})
                if changes.get("total_changes", 0) > 0:
                    state_flow["state_changes_summary"]["nodes_with_changes"] += 1
                    state_flow["state_changes_summary"]["total_changes"] += changes["total_changes"]
                    
                    state_flow["state_transitions"].append({
                        "node": node_info["node_name"],
                        "run_id": node_info["run_id"],
                        "changes": changes
                    })
                    
                    logger.debug(f"[{self._analysis_id}] State transition at {node_info['node_name']} (depth {depth}): "
                              f"{changes['total_changes']} changes "
                              f"(keys: +{len(changes['keys_added'])}, "
                              f"-{len(changes['keys_removed'])}, "
                              f"~{len(changes['keys_modified'])})")
            
            for child in node.get("children", []):
                traverse_tree(child, depth + 1)
        
        traverse_tree(tree)
        
        state_flow["state_groups_usage"] = dict(state_flow["state_groups_usage"])
        
        cache_stats = self.get_cache_stats()
        logger.info(f"[{self._analysis_id}] State flow analysis complete:")
        logger.info(f"[{self._analysis_id}]   - Total nodes: {state_flow['state_changes_summary']['total_nodes']}")
        logger.info(f"[{self._analysis_id}]   - Nodes with state: {state_flow['state_changes_summary']['nodes_with_state']}")
        logger.info(f"[{self._analysis_id}]   - Nodes with changes: {state_flow['state_changes_summary']['nodes_with_changes']}")
        logger.info(f"[{self._analysis_id}]   - Total state changes: {state_flow['state_changes_summary']['total_changes']}")
        logger.info(f"[{self._analysis_id}]   - State groups used: {list(state_flow['state_groups_usage'].keys())}")
        logger.info(f"[{self._analysis_id}]   - Cache stats: hit_rate={cache_stats['hit_rate']:.1f}%, "
                   f"hits={cache_stats['hits']}, misses={cache_stats['misses']}")
        
        return state_flow
    
    def analyze_query_patterns(
        self,
        runs: List[Run],
        min_occurrences: int = 2
    ) -> Dict[str, Any]:
        """
        질의 패턴 분석
        
        Args:
            runs: 분석할 runs 리스트
            min_occurrences: 최소 발생 횟수
        
        Returns:
            Dict[str, Any]: 패턴 분석 결과
        """
        patterns = {
            "slow_queries": [],
            "error_queries": [],
            "common_nodes": {},
            "average_durations": {},
            "token_usage": {
                "total": 0,
                "average_per_run": 0,
                "max": 0
            }
        }
        
        node_durations = {}
        node_counts = {}
        total_tokens = 0
        max_tokens = 0
        
        for run in runs:
            query = self._extract_query(run)
            duration = None
            if run.start_time and run.end_time:
                duration = (run.end_time - run.start_time).total_seconds()
            
            if run.status == "error":
                patterns["error_queries"].append({
                    "query": query,
                    "error": str(run.error) if run.error else "Unknown error",
                    "run_id": str(run.id)
                })
            
            if duration and duration > 30:
                patterns["slow_queries"].append({
                    "query": query,
                    "duration": duration,
                    "run_id": str(run.id)
                })
            
            child_runs = self._get_child_runs(str(run.id), recursive=True)
            run_tokens = 0
            
            for child_run in child_runs:
                node_name = child_run.name or "unknown"
                node_duration = None
                
                if child_run.start_time and child_run.end_time:
                    node_duration = (child_run.end_time - child_run.start_time).total_seconds()
                
                if node_name not in node_durations:
                    node_durations[node_name] = []
                    node_counts[node_name] = 0
                
                if node_duration:
                    node_durations[node_name].append(node_duration)
                node_counts[node_name] += 1
                
                if hasattr(child_run, "extra") and child_run.extra:
                    tokens = child_run.extra.get("token_usage", {})
                    if isinstance(tokens, dict):
                        total = tokens.get("total_tokens", 0)
                        run_tokens += total
            
            total_tokens += run_tokens
            max_tokens = max(max_tokens, run_tokens)
        
        patterns["token_usage"]["total"] = total_tokens
        patterns["token_usage"]["average_per_run"] = total_tokens / len(runs) if runs else 0
        patterns["token_usage"]["max"] = max_tokens
        
        for node_name, durations in node_durations.items():
            if node_counts[node_name] >= min_occurrences:
                patterns["average_durations"][node_name] = sum(durations) / len(durations)
                patterns["common_nodes"][node_name] = node_counts[node_name]
        
        return patterns
    
    def get_improvement_suggestions(
        self,
        analysis: Dict[str, Any],
        patterns: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        개선 제안 생성
        
        Args:
            analysis: 단일 run 분석 결과
            patterns: 패턴 분석 결과 (선택적)
        
        Returns:
            List[str]: 개선 제안 리스트
        """
        suggestions = []
        
        if analysis.get("error"):
            suggestions.append(f"❌ 오류 발생: {analysis['error']} - 오류 처리 로직 개선 필요")
        
        if analysis.get("duration") and analysis["duration"] > 30:
            suggestions.append(f"⏱️ 실행 시간이 {analysis['duration']:.2f}초로 느립니다 - 병목 지점 최적화 필요")
        
        bottlenecks = analysis.get("bottlenecks", [])
        for bottleneck in bottlenecks:
            node_name = bottleneck.get("node")
            duration = bottleneck.get("duration", 0)
            suggestions.append(f"🐌 병목 지점: {node_name} 노드가 {duration:.2f}초 소요 - 캐싱 또는 병렬 처리 고려")
        
        if analysis.get("total_tokens", 0) > 10000:
            suggestions.append(f"💰 토큰 사용량이 {analysis['total_tokens']}로 높습니다 - 프롬프트 최적화 또는 모델 변경 고려")
        
        if patterns:
            slow_nodes = sorted(
                patterns.get("average_durations", {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for node_name, avg_duration in slow_nodes:
                if avg_duration > 5:
                    suggestions.append(f"📊 {node_name} 노드의 평균 실행 시간이 {avg_duration:.2f}초입니다 - 최적화 우선순위 높음")
        
        if not suggestions:
            suggestions.append("✅ 현재 성능이 양호합니다. 추가 모니터링을 계속하세요.")
        
        return suggestions
    
    def _extract_query(self, run: Run) -> str:
        """Run에서 질의 추출"""
        if run.inputs:
            if isinstance(run.inputs, dict):
                return run.inputs.get("query") or run.inputs.get("input", {}).get("query", "N/A")
            return str(run.inputs)
        return "N/A"
    
    def _get_child_runs(self, parent_run_id: str, recursive: bool = False, visited_ids: Optional[Set[str]] = None) -> List[Run]:
        """
        자식 runs 조회
        
        Args:
            parent_run_id: 부모 run ID
            recursive: 재귀적으로 모든 하위 runs를 가져올지 여부
            visited_ids: 방문한 run ID 세트 (순환 참조 방지)
        
        Returns:
            List[Run]: 자식 runs 리스트
        """
        if visited_ids is None:
            visited_ids = set()
        
        if parent_run_id in visited_ids:
            logger.debug(f"[{self._analysis_id}] Circular reference detected for run {parent_run_id}")
            return []
        
        child_runs = []
        
        try:
            if parent_run_id in self._run_cache:
                run = self._run_cache[parent_run_id]
                self._run_cache.move_to_end(parent_run_id)
                self._cache_stats["hits"] += 1
            else:
                run = self._read_run_with_retry(parent_run_id)
                if run:
                    self._run_cache[parent_run_id] = run
                    self._run_cache.move_to_end(parent_run_id)
                    self._manage_cache_size(self._run_cache, self._max_cache_size)
                    self._cache_stats["misses"] += 1
                else:
                    return []
            
            if hasattr(run, 'child_runs') and run.child_runs:
                child_runs = list(run.child_runs)
            elif hasattr(run, 'child_run_ids') and run.child_run_ids:
                for child_id in run.child_run_ids:
                    if child_id in visited_ids:
                        continue
                    try:
                        if child_id in self._run_cache:
                            child_run = self._run_cache[child_id]
                            self._run_cache.move_to_end(child_id)
                            self._cache_stats["hits"] += 1
                        else:
                            child_run = self._read_run_with_retry(child_id)
                            if child_run:
                                self._run_cache[child_id] = child_run
                                self._run_cache.move_to_end(child_id)
                                self._manage_cache_size(self._run_cache, self._max_cache_size)
                                self._cache_stats["misses"] += 1
                                child_runs.append(child_run)
                                visited_ids.add(child_id)
                            else:
                                continue
                    except Exception as e:
                        logger.warning(f"[{self._analysis_id}] Error reading child run {child_id} (parent: {parent_run_id}): {e}", exc_info=False)
        except Exception as e:
            logger.warning(f"[{self._analysis_id}] Error reading run {parent_run_id} for child runs: {e}", exc_info=True)
        
        if not child_runs:
            try:
                runs = list(self.client.list_runs(
                    project_name=self.project_name,
                    parent_run_id=parent_run_id,
                    limit=100
                ))
                if runs:
                    child_runs = runs
                    for r in runs:
                        run_id_str = str(r.id)
                        self._run_cache[run_id_str] = r
                        self._run_cache.move_to_end(run_id_str)
                        self._manage_cache_size(self._run_cache, self._max_cache_size)
                        self._cache_stats["misses"] += 1
            except Exception as e:
                logger.warning(f"[{self._analysis_id}] Error fetching child runs with parent_run_id {parent_run_id}: {e}", exc_info=False)
        
        if recursive and child_runs:
            visited_ids.add(parent_run_id)
            all_descendants = list(child_runs)
            for child_run in child_runs:
                child_id = str(child_run.id)
                if child_id not in visited_ids:
                    descendants = self._get_child_runs(child_id, recursive=True, visited_ids=visited_ids)
                    for desc in descendants:
                        desc_id = str(desc.id)
                        if desc_id not in visited_ids:
                            all_descendants.append(desc)
                            visited_ids.add(desc_id)
            return all_descendants
        
        return child_runs
    
    def get_run_tree(self, run_id: str, max_depth: int = 10, show_progress: bool = False) -> Dict[str, Any]:
        """
        RunTree 구조를 재귀적으로 가져오기
        
        Args:
            run_id: 루트 run ID
            max_depth: 최대 깊이
            show_progress: 진행 상황 표시 여부
        
        Returns:
            Dict[str, Any]: RunTree 구조 (에러 발생 시 error 필드 포함)
        """
        if run_id in self._tree_cache:
            self._tree_cache.move_to_end(run_id)
            self._cache_stats["hits"] += 1
            return self._tree_cache[run_id]
        
        self._cache_stats["misses"] += 1
        
        if not self._validate_run_id(run_id):
            return {"error": f"Invalid run ID format: {run_id}"}
        
        depth_reached = False
        
        def build_tree(run_id: str, depth: int = 0, visited: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
            nonlocal depth_reached
            if visited is None:
                visited = set()
            
            if run_id in visited:
                logger.warning(f"[{self._analysis_id}] Circular reference detected for run {run_id} at depth {depth}")
                return None
            
            if depth > max_depth:
                depth_reached = True
                logger.warning(f"[{self._analysis_id}] Maximum depth {max_depth} reached for run {run_id} at depth {depth}")
                return None
            
            try:
                if show_progress and depth == 0:
                    logger.info(f"[{self._analysis_id}] Building run tree for {run_id}...")
                elif show_progress and depth % 5 == 0:
                    logger.debug(f"[{self._analysis_id}] Processing depth {depth}...")
                
                if run_id in self._run_cache:
                    run = self._run_cache[run_id]
                    self._run_cache.move_to_end(run_id)
                    self._cache_stats["hits"] += 1
                else:
                    run = self._read_run_with_retry(run_id)
                    if not run:
                        return {"error": f"Failed to read run {run_id} after retries", "run_id": run_id, "depth": depth}
                    self._run_cache[run_id] = run
                    self._run_cache.move_to_end(run_id)
                    self._manage_cache_size(self._run_cache, self._max_cache_size)
                    self._cache_stats["misses"] += 1
                
                # State 정보 추출 (안전하게 처리)
                state_info = self._extract_state_info(run)
                if not isinstance(state_info, dict):
                    state_info = {}
                
                tree = {
                    "run_id": str(run.id),
                    "name": run.name or "unknown",
                    "run_type": run.run_type or "unknown",
                    "status": run.status,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "end_time": run.end_time.isoformat() if run.end_time else None,
                    "duration": None,
                    "children": [],
                    "state_info": state_info
                }
                
                if run.start_time and run.end_time:
                    tree["duration"] = (run.end_time - run.start_time).total_seconds()
                
                visited.add(run_id)
                child_runs = self._get_child_runs(str(run.id), recursive=False)
                
                if show_progress and child_runs and depth % 3 == 0:
                    logger.debug(f"[{self._analysis_id}] Found {len(child_runs)} child runs at depth {depth}")
                
                for child_run in child_runs:
                    child_id = str(child_run.id)
                    if child_id not in visited:
                        child_tree = build_tree(child_id, depth + 1, visited)
                        if child_tree:
                            # 부모-자식 간 state 전달 확인
                            child_state_info = child_tree.get("state_info", {})
                            if isinstance(child_state_info, dict):
                                # 부모의 output_state가 자식의 input_state와 일치하는지 확인
                                parent_output_state = state_info.get("output_state")
                                child_input_state = child_state_info.get("input_state")
                                
                                if parent_output_state and child_input_state:
                                    if isinstance(parent_output_state, dict) and isinstance(child_input_state, dict):
                                        parent_keys = set(parent_output_state.get("keys", []))
                                        child_keys = set(child_input_state.get("keys", []))
                                        
                                        # 공통 키가 있으면 state 전달로 간주
                                        common_keys = parent_keys & child_keys
                                        if common_keys:
                                            child_state_info["state_inherited"] = True
                                            child_state_info["inherited_keys"] = list(common_keys)
                                            child_tree["state_info"] = child_state_info
                            
                            tree["children"].append(child_tree)
                        visited.discard(child_id)
                
                return tree
            except Exception as e:
                logger.error(f"[{self._analysis_id}] Error building tree for run {run_id} at depth {depth}: {e}", exc_info=True)
                return {"error": str(e), "run_id": run_id, "depth": depth}
        
        result = build_tree(run_id) or {}
        if depth_reached:
            result["max_depth_reached"] = True
        
        self._tree_cache[run_id] = result
        self._tree_cache.move_to_end(run_id)
        self._manage_cache_size(self._tree_cache, self._max_cache_size)
        return result
    
    def _analyze_nodes(self, child_runs: List[Run]) -> Dict[str, Any]:
        """노드별 분석"""
        nodes = []
        total_tokens = 0
        total_cost = 0.0
        
        for child_run in child_runs:
            node_info = {
                "name": child_run.name or "unknown",
                "run_type": child_run.run_type or "unknown",
                "duration": None,
                "tokens": 0,
                "cost": 0.0,
                "status": child_run.status,
                "inputs": str(child_run.inputs)[:100] if child_run.inputs else None,
                "outputs": str(child_run.outputs)[:100] if child_run.outputs else None,
                "state_info": self._extract_state_info(child_run)
            }
            
            if child_run.start_time and child_run.end_time:
                node_info["duration"] = (child_run.end_time - child_run.start_time).total_seconds()
            
            tokens = 0
            model_name = self._extract_model_name(child_run)
            
            token_usage_dict = self._extract_token_usage(child_run)
            if token_usage_dict:
                tokens = token_usage_dict.get("total_tokens", 0)
                if tokens == 0:
                    # input_tokens와 output_tokens 합산
                    input_tokens = token_usage_dict.get("input_tokens", token_usage_dict.get("prompt_tokens", 0))
                    output_tokens = token_usage_dict.get("output_tokens", token_usage_dict.get("completion_tokens", 0))
                    tokens = input_tokens + output_tokens
            
            node_info["tokens"] = tokens
            total_tokens += tokens
            
            cost = self._calculate_cost(tokens, model_name, token_usage=token_usage_dict)
            node_info["cost"] = cost
            total_cost += cost
            
            nodes.append(node_info)
        
        return {
            "nodes": nodes,
            "total_tokens": total_tokens,
            "total_cost": total_cost
        }
    
    def _identify_bottlenecks(self, node_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """병목 지점 식별"""
        bottlenecks = []
        nodes = node_analysis.get("nodes", [])
        
        for node in nodes:
            duration = node.get("duration", 0)
            if duration and duration > 5:
                bottlenecks.append({
                    "node": node.get("name"),
                    "duration": duration,
                    "tokens": node.get("tokens", 0)
                })
        
        return sorted(bottlenecks, key=lambda x: x["duration"], reverse=True)
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """기본 개선 제안 생성"""
        recommendations = []
        
        if analysis.get("error"):
            recommendations.append("오류 처리 및 재시도 로직 강화")
        
        if analysis.get("duration", 0) > 30:
            recommendations.append("비동기 처리 또는 병렬 실행 고려")
        
        bottlenecks = analysis.get("bottlenecks", [])
        if bottlenecks:
            recommendations.append("느린 노드에 대한 캐싱 전략 적용")
        
        return recommendations
    
    def visualize_run_tree(self, run_id: str, max_depth: int = 10) -> str:
        """
        RunTree를 시각화하여 문자열로 반환
        
        Args:
            run_id: 루트 run ID
            max_depth: 최대 깊이
        
        Returns:
            str: 시각화된 RunTree 문자열
        """
        tree = self.get_run_tree(run_id, max_depth)
        if not tree or tree.get("error"):
            error_msg = tree.get("error", "Unknown error") if tree else "Run not found"
            return f"Run {run_id}을(를) 찾을 수 없거나 오류가 발생했습니다: {error_msg}"
        
        lines = []
        
        tree_name = tree.get('name', 'unknown')
        tree_run_id = tree.get('run_id', run_id)
        lines.append(f"RunTree: {tree_name} ({tree_run_id})")
        if tree.get('duration'):
            lines.append(f"Total Duration: {tree['duration']:.2f}s")
        if tree.get('max_depth_reached'):
            lines.append(f"⚠️ Warning: Maximum depth {max_depth} reached. Tree may be truncated.")
        lines.append("")
        
        def format_node(node: Dict[str, Any], prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            duration_str = f" ({node.get('duration', 0):.2f}s)" if node.get('duration') else ""
            status_str = f" [{node.get('status', 'unknown')}]" if node.get('status') else ""
            node_name = node.get('name', 'unknown')
            node_type = node.get('run_type', 'unknown')
            lines.append(f"{prefix}{connector}{node_name} ({node_type}){duration_str}{status_str}")
            
            children = node.get('children', [])
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                extension = "    " if is_last else "│   "
                new_prefix = prefix + extension
                format_node(child, new_prefix, is_last_child)
        
        for i, child in enumerate(tree.get('children', [])):
            is_last = i == len(tree.get('children', [])) - 1
            format_node(child, "", is_last)
        
        return "\n".join(lines)
    
    def get_run_statistics(self, run_id: str, tree: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run의 통계 정보 수집
        
        Args:
            run_id: Run ID
            tree: 이미 조회한 RunTree (선택적, 중복 조회 방지)
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        if tree is None:
            tree = self.get_run_tree(run_id)
        
        if not tree or tree.get("error"):
            return {}
        
        stats = {
            "total_runs": 0,
            "by_type": defaultdict(int),
            "by_status": defaultdict(int),
            "total_duration": 0.0,
            "max_depth": 0,
            "node_durations": {},
            "token_usage": {
                "total": 0,
                "by_node": defaultdict(int)
            },
            "state_updates": {
                "nodes_with_state": 0,
                "state_transitions": 0
            }
        }
        
        def collect_stats(node: Dict[str, Any], depth: int = 0):
            stats["total_runs"] += 1
            stats["by_type"][node.get('run_type', 'unknown')] += 1
            stats["by_status"][node.get('status', 'unknown')] += 1
            stats["max_depth"] = max(stats["max_depth"], depth)
            
            if node.get('duration'):
                stats["total_duration"] += node['duration']
                node_name = node.get('name', 'unknown')
                if node_name not in stats["node_durations"]:
                    stats["node_durations"][node_name] = []
                stats["node_durations"][node_name].append(node['duration'])
            
            state_info = node.get('state_info', {})
            if state_info.get('state_snapshot'):
                stats["state_updates"]["nodes_with_state"] += 1
                if state_info.get('state_snapshot') == "updated":
                    stats["state_updates"]["state_transitions"] += 1
            
            for child in node.get('children', []):
                collect_stats(child, depth + 1)
        
        collect_stats(tree)
        
        stats["by_type"] = dict(stats["by_type"])
        stats["by_status"] = dict(stats["by_status"])
        stats["average_duration"] = stats["total_duration"] / stats["total_runs"] if stats["total_runs"] > 0 else 0
        stats["token_usage"]["by_node"] = dict(stats["token_usage"]["by_node"])
        stats["state_updates"] = dict(stats["state_updates"])
        
        for node_name in stats["node_durations"]:
            durations = stats["node_durations"][node_name]
            stats["node_durations"][node_name] = {
                "count": len(durations),
                "total": sum(durations),
                "average": sum(durations) / len(durations) if durations else 0,
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0
            }
        
        return stats

