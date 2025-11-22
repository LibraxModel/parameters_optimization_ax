"""
LLINBO Agent: Large Language Model for Bayesian Optimization Agent
基于大模型的贝叶斯优化智能体

功能：
1. 接受可配置的优化问题背景和行业描述
2. 支持可配置的参数空间范围
3. 利用历史先验实验数据进行优化建议
4. 使用大模型模拟贝叶斯优化过程
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class ProblemContext:
    """优化问题背景配置"""
    problem_description: str  # 问题描述
    industry: str  # 行业领域
    domain_knowledge: Optional[str] = None  # 领域知识
    constraints: Optional[List[str]] = None  # 约束条件
    optimization_goals: Optional[List[str]] = None  # 优化目标说明


@dataclass
class Parameter:
    """参数空间定义"""
    name: str  # 参数名称
    type: str  # 参数类型: "range" 或 "choice"
    bounds: Optional[List[float]] = None  # 范围参数: [min, max]
    values: Optional[List[Any]] = None  # 选择参数: [value1, value2, ...]
    value_type: str = "float"  # 值类型: "int", "float", "str"
    description: Optional[str] = None  # 参数描述
    unit: Optional[str] = None  # 单位   


@dataclass
class PriorExperiment:
    """先验实验数据"""
    parameters: Dict[str, Any]  # 参数配置
    metrics: Dict[str, float]  # 实验结果指标
    metadata: Optional[Dict[str, Any]] = None  # 额外元数据


@dataclass
class LLMConfig:
    """大模型配置"""
    model_name: str = "gpt-4"  # 模型名称
    api_key: Optional[str] = None  # API密钥
    base_url: Optional[str] = None  # API基础URL
    extra_body: Dict[str, Any] = field(default_factory=lambda: {
        "thinking": {
            "type": "enabled"  # 或 "disabled" / "auto"
        }
    })
    
    


class LLMProvider(ABC):
    """大模型提供者抽象基类"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API 提供者"""
    
    def __init__(self, config: LLMConfig, prompt_config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.prompt_config = prompt_config
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=config.api_key or None,
                base_url=config.base_url
            )
        except ImportError:
            raise ImportError("请安装 openai 库: pip install openai")
        except Exception as e:
            raise RuntimeError(f"初始化 OpenAI 客户端失败: {e}")
    
    def _get_config_value(self, config: Dict[str, Any], key: str, default: str = "") -> str:
        """从配置中获取值，支持数组和字符串格式"""
        value = config.get(key, default) if config else default
        if isinstance(value, list):
            return "\n".join(value)
        return str(value) if value else default
    
    def generate(self, prompt: str, **kwargs) -> str:
        """使用 OpenAI API 生成文本"""
        try:
            # 获取系统消息
            system_message = self._get_config_value(self.prompt_config, "system_message", "") if self.prompt_config else ""
            
            # 构建 API 调用参数
            api_params = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            }
            
            # 如果配置了 extra_body，则添加到 API 参数中
            if self.config.extra_body:
                api_params["extra_body"] = self.config.extra_body
            
            # 合并 kwargs 中的额外参数
            api_params.update(kwargs)
            
            response = self.client.chat.completions.create(**api_params)
        except Exception as e:
            raise RuntimeError(f"调用 OpenAI API 失败: {e}")
        if response.choices[0].message.reasoning_content:
            return response.choices[0].message.reasoning_content + "\n" + response.choices[0].message.content
        else:
            return response.choices[0].message.content


class LLINBOAgent:
    """
    基于大模型的贝叶斯优化智能体
    
    使用大模型来模拟贝叶斯优化过程，结合问题背景、参数空间和先验数据
    生成优化建议。
    """
    
    # 类级别的提示词配置
    _prompt_config = None
    
    @classmethod
    def _load_prompt_config(cls, config_path: str = "promt_config.json"):
        """加载提示词配置文件"""
        if cls._prompt_config is None:
            import os
            # 尝试从当前目录或代码所在目录加载
            possible_paths = [
                config_path,
                os.path.join(os.path.dirname(__file__), config_path),
                os.path.join(os.getcwd(), config_path)
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        cls._prompt_config = json.load(f)
                    break
            
            if cls._prompt_config is None:
                raise FileNotFoundError(f"找不到提示词配置文件: {config_path}")
        
        return cls._prompt_config
    
    def __init__(
        self,
        problem_context: ProblemContext,
        parameters: List[Parameter],
        objectives: Dict[str, Dict[str, bool]],  # {"metric_name": {"minimize": bool}}
        llm_config: Optional[LLMConfig] = None,
        prior_experiments: Optional[List[PriorExperiment]] = None,
        random_seed: Optional[int] = None,
        prompt_config_path: Optional[str] = None
    ):
        """
        初始化 LLINBO Agent
        
        Args:
            problem_context: 优化问题背景
            parameters: 参数定义列表
            objectives: 优化目标配置
            llm_config: 大模型配置（可选）
            prior_experiments: 先验实验数据（可选）
            random_seed: 随机种子（可选）
            prompt_config_path: 提示词配置文件路径（可选，默认使用 promt_config.json）
        """
        self.problem_context = problem_context
        self.parameters = parameters
        self.objectives = objectives
        self.prior_experiments = prior_experiments or []
        self.random_seed = random_seed
        
        # 加载提示词配置
        if prompt_config_path:
            self._load_prompt_config(prompt_config_path)
        else:
            self._load_prompt_config()
        self.prompt_config = self._prompt_config
        
        # 初始化大模型提供者
        if llm_config is None:
            llm_config = LLMConfig()
        self.llm_config = llm_config
        
        # 根据配置选择提供者
        self.llm_provider = OpenAIProvider(llm_config, self.prompt_config)

        # 优化历史记录

        self.optimization_history: List[Dict[str, Any]] = []
    
    def _build_context_prompt(self) -> str:
        """构建包含问题背景的提示词"""
        prompt_parts = [
            "# Optimization Problem Context",
            f"**Problem Description**: {self.problem_context.problem_description}",
            f"**Industry Domain**: {self.problem_context.industry}",
        ]
        
        if self.problem_context.domain_knowledge:
            prompt_parts.append(f"**Domain Knowledge**: {self.problem_context.domain_knowledge}")
        
        if self.problem_context.constraints:
            prompt_parts.append(f"**Constraints**: {', '.join(self.problem_context.constraints)}")
        
        if self.problem_context.optimization_goals:
            prompt_parts.append(f"**Optimization Goals**: {', '.join(self.problem_context.optimization_goals)}")
        
        return "\n".join(prompt_parts)
    
    def _build_parameter_space_prompt(self) -> str:
        """构建参数空间描述提示词"""
        prompt_parts = [
            "# Parameter Space Definition",
            "The following are the parameters to be optimized and their ranges:",
            "",
            "**Important Note**: If a parameter is discrete, you must select from the listed optional values and cannot choose other values; if a parameter is continuous, the value must be within the [minimum, maximum] range."
        ]
        
        for i, param in enumerate(self.parameters, 1):
            param_desc = [f"{i}. **{param.name}**"]
            
            if param.description:
                param_desc.append(f"   Description: {param.description}")
            
            if param.type == "range" and param.bounds:
                param_desc.append(f"   Type: Continuous parameter")
                param_desc.append(f"   Range: [{param.bounds[0]}, {param.bounds[1]}]")
                if param.unit:
                    param_desc.append(f"   Unit: {param.unit}")
            elif param.type == "choice" and param.values:
                param_desc.append(f"   Type: Discrete parameter (must select one from the following values)")
                # 明确列出所有可选值
                values_str = ", ".join([str(v) for v in param.values])
                param_desc.append(f"   Optional values: [{values_str}]")
                param_desc.append(f"   Number of optional values: {len(param.values)}")
                param_desc.append(f"   ⚠️ Important: Only values from the list can be selected, no other values or intermediate values")
            
            if param.value_type:
                param_desc.append(f"   Value type: {param.value_type}")
            
            prompt_parts.append("\n".join(param_desc))
        
        return "\n".join(prompt_parts)
    
    def _build_objectives_prompt(self) -> str:
        """构建优化目标描述提示词"""
        prompt_parts = [
            "# Optimization Objectives",
            "Metrics to be optimized and their directions:"
        ]
        
        for metric_name, config in self.objectives.items():
            minimize = config.get("minimize", True)
            direction = "minimize" if minimize else "maximize"
            prompt_parts.append(f"- **{metric_name}**: {direction}")
        
        # 添加优化方向说明
        if len(self.objectives) > 1:
            prompt_parts.append("\n**Note**: This is a multi-objective optimization problem that requires balancing multiple objectives.")
        
        return "\n".join(prompt_parts)
    
    def _build_optimization_direction_instruction(self) -> str:
        """构建优化方向说明"""
        instructions = []
        
        for metric_name, config in self.objectives.items():
            minimize = config.get("minimize", True)
            if minimize:
                instructions.append(
                    f"- **{metric_name}**: Needs to be **minimized**, prioritize parameter combinations that can reduce this metric value"
                )
            else:
                instructions.append(
                    f"- **{metric_name}**: Needs to be **maximized**, prioritize parameter combinations that can increase this metric value"
                )
        
        return "\n".join(instructions)
    
    def _build_prior_data_prompt(self) -> str:
        """构建先验实验数据提示词"""
        if not self.prior_experiments:
            return "# Prior Experimental Data\nNo prior experimental data available."
        
        prompt_parts = [
            "# Prior Experimental Data",
            f"The following are {len(self.prior_experiments)} historical experimental results:",
            ""
        ]
        
        # 转换为表格格式
        data_rows = []
        for i, exp in enumerate(self.prior_experiments, 1):
            row = {
                "Experiment_ID": i,
                **exp.parameters,
                **exp.metrics
            }
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        prompt_parts.append("```")
        prompt_parts.append(df.to_string(index=False))
        prompt_parts.append("```")
        
        
        return "\n".join(prompt_parts)
    
    def _build_initial_sampling_prompt(self, num_suggestions: int = 1) -> str:
        """构建初始采样提示词（无先验数据时使用）"""
        # 从配置文件读取初始采样提示词
        init_config = self.prompt_config.get("initial_sampling", "")
        
        # 如果是数组，用换行符连接
        if isinstance(init_config, list):
            init_template = "\n".join(init_config)
        else:
            init_template = init_config
        
        # 替换占位符
        init_content = init_template.format(num_suggestions=num_suggestions)
        
        prompt_parts = [
            self._build_context_prompt(),
            "",
            self._build_parameter_space_prompt(),
            "",
            self._build_objectives_prompt(),
            "",
            init_content
        ]
        
        return "\n".join(prompt_parts)
    
    def _build_optimization_prompt(self, num_suggestions: int = 1) -> str:
        """构建完整的优化提示词"""
        # 从配置文件读取优化提示词
        opt_config = self.prompt_config.get("optimization", "")
        
        # 如果是数组，用换行符连接
        if isinstance(opt_config, list):
            opt_template = "\n".join(opt_config)
        else:
            opt_template = opt_config
        
        # 替换占位符
        optimization_direction = self._build_optimization_direction_instruction()
        opt_content = opt_template.format(
            num_suggestions=num_suggestions,
            optimization_direction=optimization_direction
        )
        
        prompt_parts = [
            self._build_context_prompt(),
            "",
            self._build_parameter_space_prompt(),
            "",
            self._build_objectives_prompt(),
            "",
            self._build_prior_data_prompt(),
            "",
            opt_content
        ]
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """解析大模型返回的JSON格式响应"""
        try:
            # 尝试提取JSON部分
            json_str = None
            
            # 方法1: 查找 ```json ... ```
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                if json_end > json_start:
                    json_str = response[json_start:json_end].strip()
            
            # 方法2: 查找 ``` ... ```
            if json_str is None and "```" in response:
                parts = response.split("```")
                for i in range(1, len(parts), 2):
                    candidate = parts[i].strip()
                    if candidate.startswith("json"):
                        candidate = candidate[4:].strip()
                    if candidate.startswith("{") or candidate.startswith("["):
                        json_str = candidate
                        break
            
            # 方法3: 查找第一个 { 或 [
            if json_str is None:
                for start_char in ["{", "["]:
                    start_idx = response.find(start_char)
                    if start_idx >= 0:
                        # 找到匹配的结束字符
                        end_char = "}" if start_char == "{" else "]"
                        depth = 0
                        for i in range(start_idx, len(response)):
                            if response[i] == start_char:
                                depth += 1
                            elif response[i] == end_char:
                                depth -= 1
                                if depth == 0:
                                    json_str = response[start_idx:i+1]
                                    break
                        if json_str:
                            break
            
            # 方法4: 直接解析整个响应
            if json_str is None:
                json_str = response.strip()
            
            result = json.loads(json_str)
            
            if "suggestions" in result:
                return result["suggestions"]
            elif isinstance(result, list):
                return result
            elif isinstance(result, dict):
                # 如果结果是单个字典，检查是否包含参数
                if any(key in result for key in [p.name for p in self.parameters]):
                    return [result]
                else:
                    return []
            else:
                return []
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败: {e}")
            print(f"响应内容（前500字符）: {response[:500]}")
            # 如果解析失败，返回空列表
            return []
        except Exception as e:
            print(f"⚠️ 解析响应时发生错误: {e}")
            return []
    
    def _validate_parameters(self, params: Dict[str, Any]) -> bool:
        """验证参数是否在定义的空间内"""
        for param_def in self.parameters:
            param_name = param_def.name
            if param_name not in params:
                continue  # 允许缺少某些参数
            
            value = params[param_name]
            
            if param_def.type == "range":
                if param_def.bounds is None:
                    continue
                min_val, max_val = param_def.bounds
                if not (min_val <= value <= max_val):
                    return False
                
                # 类型转换检查
                if param_def.value_type == "int":
                    if not isinstance(value, (int, float)) or int(value) != value:
                        return False
            elif param_def.type == "choice":
                if param_def.values is None:
                    continue
                if value not in param_def.values:
                    return False
        
        return True
    
    def _normalize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """规范化参数值（类型转换等）"""
        normalized = {}
        
        for param_def in self.parameters:
            param_name = param_def.name
            if param_name not in params:
                continue
            
            value = params[param_name]
            
            # 类型转换
            if param_def.value_type == "int":
                value = int(float(value))
            elif param_def.value_type == "float":
                value = float(value)
            elif param_def.value_type == "str":
                value = str(value)
            
            normalized[param_name] = value
        
        return normalized
    
    def suggest_initial_parameters(
        self, 
        num_suggestions: int = 1, 
        print_prompt: bool = False, 
        print_response: bool = False
    ) -> List[Dict[str, Any]]:
        """
        生成初始采样参数建议（无先验数据时使用）
        
        该方法专门用于没有先验实验数据的情况，基于领域知识推荐最有希望达到最优解附近的参数组合。
        不需要均匀覆盖参数空间，只需要推荐足够好的参数组合即可。
        
        Args:
            num_suggestions: 需要生成的建议数量
            print_prompt: 是否打印输入大模型的完整提示词
            print_response: 是否打印大模型的原始回答
            
        Returns:
            推荐的参数配置列表
        """
        # 构建初始采样提示词
        prompt = self._build_initial_sampling_prompt(num_suggestions=num_suggestions)
        
        # 根据 print_prompt 参数决定是否打印完整提示词
        if print_prompt:
            print("\n" + "=" * 80)
            print("📝 输入大模型的完整提示词（初始采样模式）:")
            print("=" * 80)
            print(prompt)
            print("=" * 80 + "\n")
        
        # 调用大模型生成建议
        print("🤖 正在使用大模型生成初始采样建议...")
        response = self.llm_provider.generate(prompt)
        
        # 根据 print_response 参数决定是否打印大模型的原始回答
        if print_response:
            print("\n" + "=" * 80)
            print("📤 大模型的原始回答:")
            print("=" * 80)
            print(response)
            print("=" * 80 + "\n")
        
        # 解析响应
        suggestions = self._parse_llm_response(response)
        
        # 验证和规范化参数
        valid_suggestions = []
        for suggestion in suggestions:
            params = {k: v for k, v in suggestion.items() if k != "reason"}
            
            if self._validate_parameters(params):
                normalized = self._normalize_parameters(params)
                valid_suggestions.append(normalized)
            else:
                print(f"⚠️ 参数验证失败，跳过: {params}")
        
        # 如果验证后的建议数量不足，尝试生成更多
        if len(valid_suggestions) < num_suggestions:
            print(f"⚠️ 只生成了 {len(valid_suggestions)} 个有效建议，期望 {num_suggestions} 个")
        
        # 记录到历史
        for suggestion in valid_suggestions:
            self.optimization_history.append({
                "suggestion": suggestion,
                "timestamp": pd.Timestamp.now().isoformat(),
                "type": "initial_sampling"
            })
        
        return valid_suggestions[:num_suggestions]
    
    def suggest_parameters(
        self, 
        num_suggestions: int = 1, 
        print_prompt: bool = False, 
        print_response: bool = False,
        auto_initial_sampling: bool = True
    ) -> List[Dict[str, Any]]:
        """
        生成参数优化建议
        
        Args:
            num_suggestions: 需要生成的建议数量
            print_prompt: 是否打印输入大模型的完整提示词
            print_response: 是否打印大模型的原始回答
            auto_initial_sampling: 如果没有先验数据，是否自动切换到初始采样模式（默认True）
            
        Returns:
            推荐的参数配置列表
        """
        # 如果没有先验数据且启用了自动初始采样，则使用初始采样模式
        if not self.prior_experiments and auto_initial_sampling:
            print("📊 检测到没有先验实验数据，自动切换到初始采样模式...")
            return self.suggest_initial_parameters(
                num_suggestions=num_suggestions,
                print_prompt=print_prompt,
                print_response=print_response
            )
        
        # 构建提示词
        prompt = self._build_optimization_prompt(num_suggestions=num_suggestions)
        
        # 根据 print_prompt 参数决定是否打印完整提示词
        if print_prompt:
            print("\n" + "=" * 80)
            print("📝 输入大模型的完整提示词:")
            print("=" * 80)
            print(prompt)
            print("=" * 80 + "\n")
        
        # 调用大模型生成建议
        print("🤖 正在使用大模型生成优化建议...")
        response = self.llm_provider.generate(prompt)
        
        # 根据 print_response 参数决定是否打印大模型的原始回答
        if print_response:
            print("\n" + "=" * 80)
            print("📤 大模型的原始回答:")
            print("=" * 80)
            print(response)
            print("=" * 80 + "\n")
        
        # 解析响应
        suggestions = self._parse_llm_response(response)
        
        # 验证和规范化参数
        valid_suggestions = []
        for suggestion in suggestions:
            params = {k: v for k, v in suggestion.items() if k != "reason"}
            
            if self._validate_parameters(params):
                normalized = self._normalize_parameters(params)
                valid_suggestions.append(normalized)
            else:
                print(f"⚠️ 参数验证失败，跳过: {params}")
        
        # 如果验证后的建议数量不足，尝试生成更多
        if len(valid_suggestions) < num_suggestions:
            print(f"⚠️ 只生成了 {len(valid_suggestions)} 个有效建议，期望 {num_suggestions} 个")
        
        # 记录到历史
        for suggestion in valid_suggestions:
            self.optimization_history.append({
                "suggestion": suggestion,
                "timestamp": pd.Timestamp.now().isoformat()
            })
        
        return valid_suggestions[:num_suggestions]
    
    def add_experiment_result(
        self,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """添加实验结果到先验数据"""
        experiment = PriorExperiment(
            parameters=parameters,
            metrics=metrics,
            metadata=metadata
        )
        self.prior_experiments.append(experiment)

