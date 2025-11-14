# 参数优化 API - 基于 Ax 框架的贝叶斯优化与大模型调参

一个强大的参数优化框架，支持传统贝叶斯优化和基于大模型的智能调参方法。

## 🚀 主要特性

### 传统贝叶斯优化（基于 Ax 框架）
- **多种初始化采样方式**: Sobol、LHS、Uniform 采样
- **贝叶斯优化**: 基于历史数据的智能参数推荐
- **自定义代理模型**: 支持 SingleTaskGP、MultiTaskGP 等多种高斯过程模型
- **自定义核函数**: 支持 MaternKernel、RBFKernel 等多种核函数
- **自定义采集函数**: 支持单目标和多目标优化的各种采集函数
- **先验数据支持**: 可以集成历史实验数据
- **多目标优化**: 支持帕累托优化和权重优化
- **实验数据分析**: 生成多种可视化图表（并行坐标图、特征重要性图、交叉验证图、切片图、等高线图）
- **RESTful API**: 提供简洁的 HTTP 接口

### 🆕 大模型调参方法（LLINBO）

**LLINBO (Large Language Model for Bayesian Optimization)** 是一种创新的参数优化方法，利用大语言模型的推理能力进行参数推荐。

#### 核心优势

1. **领域知识融合**: 结合问题背景、行业知识和领域专业知识进行推理
2. **先验数据理解**: 深度分析历史实验数据，识别模式和趋势
3. **智能探索策略**: 在探索（exploration）和利用（exploitation）之间智能平衡
4. **多目标优化**: 支持多目标优化，考虑帕累托最优解
5. **可解释性**: 每个推荐都附带详细的推荐理由

#### 工作原理

1. **问题理解**: 接收优化问题描述、行业领域、领域知识等背景信息
2. **参数空间定义**: 支持连续参数（range）和离散参数（choice）
3. **先验数据分析**: 分析历史实验数据，提取有效信息
4. **大模型推理**: 使用大模型基于背景知识和先验数据进行参数推荐
5. **参数验证**: 自动验证推荐参数是否符合参数空间定义

#### 使用示例

```python
from LLINBO_agent import LLINBOAgent, ProblemContext, Parameter, PriorExperiment, LLMConfig

# 1. 定义问题背景
problem_context = ProblemContext(
    problem_description="优化化学反应条件以提高产率和纯度",
    industry="化学合成",
    domain_knowledge="温度对反应速率有显著影响，催化剂选择影响选择性",
    optimization_goals=["最大化产率", "最大化纯度", "最小化副产物"]
)

# 2. 定义参数空间
parameters = [
    Parameter(
        name="temperature",
        type="range",
        bounds=[25, 100],
        value_type="float",
        description="反应温度",
        unit="°C"
    ),
    Parameter(
        name="catalyst",
        type="choice",
        values=["A", "B", "C", "D"],
        description="催化剂类型"
    ),
    Parameter(
        name="reaction_time",
        type="range",
        bounds=[30, 180],
        value_type="int",
        description="反应时间",
        unit="分钟"
    )
]

# 3. 定义优化目标
objectives = {
    "yield": {"minimize": False},
    "purity": {"minimize": False},
    "side_product": {"minimize": True}
}

# 4. 准备先验实验数据（可选）
prior_experiments = [
    PriorExperiment(
        parameters={"temperature": 50, "catalyst": "A", "reaction_time": 60},
        metrics={"yield": 75, "purity": 85, "side_product": 5}
    ),
    PriorExperiment(
        parameters={"temperature": 80, "catalyst": "B", "reaction_time": 120},
        metrics={"yield": 82, "purity": 88, "side_product": 3}
    )
]

# 5. 配置大模型（可选，默认使用 GPT-4o）
llm_config = LLMConfig(
    model_name="gpt-4",
    api_key="your-api-key",  # 可选
    base_url=None  # 可选，使用自定义 API 端点
)

# 6. 初始化 LLINBO Agent
agent = LLINBOAgent(
    problem_context=problem_context,
    parameters=parameters,
    objectives=objectives,
    llm_config=llm_config,
    prior_experiments=prior_experiments
)

# 7. 获取参数推荐
suggestions = agent.suggest_parameters(num_suggestions=3)
for suggestion in suggestions:
    print(f"推荐参数: {suggestion}")
```

### 🆕 融合优化方法（Hybrid Optimizer）

**Hybrid Optimizer** 结合了传统贝叶斯优化（GP）和大模型（LLM）的优势，通过 GP 的采集函数评估 LLM 的推荐，实现更智能的参数优化。

#### 核心策略

1. **LLM 生成推荐**: 使用大模型基于领域知识和先验数据生成候选参数
2. **GP 评估筛选**: 使用高斯过程的采集函数评估 LLM 推荐的点
3. **动态阈值控制**: 基于预测标准差动态调整接受阈值
4. **智能补充**: 如果 LLM 推荐不足，使用 GP 推荐补充

#### 工作原理

```
1. 获取 GP 推荐的最佳点及其采集函数值
2. 使用 LLM 生成多个候选推荐点
3. 使用 GP 采集函数评估每个 LLM 推荐点
4. 计算 LLM 点与 GP 最佳点的采集函数差值
5. 使用动态阈值筛选"不过分差"的 LLM 推荐点
6. 如果还需要更多推荐，使用 GP 推荐补充
```

#### 使用示例

```python
from hybrid_optimizer import HybridOptimizer
from LLINBO_agent import ProblemContext, Parameter, LLMConfig
from ax_optimizer import BayesianOptimizer

# 1. 定义 LLM 参数空间
llm_parameters = [
    Parameter(name="temperature", type="range", bounds=[25, 100]),
    Parameter(name="catalyst", type="choice", values=["A", "B", "C", "D"]),
    Parameter(name="reaction_time", type="range", bounds=[30, 180], value_type="int")
]

# 2. 定义 GP 参数空间（Ax 格式）
gp_search_space = [
    {"name": "temperature", "type": "range", "bounds": [25.0, 100.0]},
    {"name": "catalyst", "type": "choice", "values": ["A", "B", "C", "D"]},
    {"name": "reaction_time", "type": "range", "bounds": [30.0, 180.0]}
]

# 3. 定义优化配置
optimization_config = {
    "objectives": {
        "yield": {"minimize": False},
        "purity": {"minimize": False}
    }
}

# 4. 定义问题背景
problem_context = ProblemContext(
    problem_description="优化化学反应条件",
    industry="化学合成"
)

# 5. 初始化融合优化器
hybrid_optimizer = HybridOptimizer(
    llm_parameters=llm_parameters,
    gp_search_space=gp_search_space,
    optimization_config=optimization_config,
    problem_context=problem_context,
    llm_config=LLMConfig(model_name="gpt-4"),
    # GP 配置
    gp_surrogate_model_class="SingleTaskGP",
    gp_kernel_class="MaternKernel",
    gp_kernel_options={"nu": 2.5},
    gp_acquisition_function_class="qExpectedHypervolumeImprovement",
    # 融合策略参数
    acquisition_threshold=0.1,  # 固定阈值（当 use_dynamic_threshold=False 时使用）
    use_dynamic_threshold=True,  # 使用动态阈值
    threshold_multiplier=1.0  # 动态阈值倍数
)

# 6. 添加历史实验数据
hybrid_optimizer.update_experiment(
    parameters={"temperature": 50, "catalyst": "A", "reaction_time": 60},
    metrics={"yield": 75, "purity": 85}
)

# 7. 获取融合推荐
suggestions = hybrid_optimizer.suggest_parameters(
    num_suggestions=3,
    use_llm=True,
    use_gp=True,
    print_details=True
)

# 8. 更新实验结果
for suggestion in suggestions:
    # 执行实验...
    metrics = run_experiment(suggestion)
    hybrid_optimizer.update_experiment(suggestion, metrics)
```

#### 融合策略参数说明

- **`acquisition_threshold`**: 固定阈值，当 `use_dynamic_threshold=False` 时使用
- **`use_dynamic_threshold`**: 是否使用基于预测标准差的动态阈值（推荐）
- **`threshold_multiplier`**: 动态阈值倍数，阈值 = `threshold_multiplier * 预测标准差`

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/LibraxModel/parameters_optimization_ax.git
cd parameters_optimization_ax

# 创建 conda 环境
conda create -n ax_env python=3.12
conda activate ax_env

# 安装依赖
pip install -r requirements.txt
```

## 🏃‍♂️ 快速开始

### 启动 API 服务器

```bash
python api_parameter_optimizer_v3.py
```

服务器将在 `http://localhost:3320` 启动。

### 传统贝叶斯优化使用

详见原有文档，支持 `/init` 和 `/update` 接口进行传统贝叶斯优化。

## 📚 API 接口文档

### 大模型调参相关接口

#### LLINBO Agent 使用

LLINBO Agent 通过 Python 代码直接使用, HTTP 接口暂未开发。使用方式见上方示例。

#### 融合优化器使用

融合优化器同样通过 Python 代码直接使用，使用方式见上方示例。

### 传统优化接口（简要）

#### POST `/init`
初始化优化，使用传统采样方法（Sobol、LHS、Uniform）

#### POST `/update`
贝叶斯优化接口，基于历史数据推荐参数

#### POST `/analysis`
实验数据分析接口，生成可视化图表

详细接口文档请参考代码注释。

## 🎯 方法选择建议

### 何时使用 LLINBO（大模型调参）

- ✅ 有丰富的领域知识和背景信息
- ✅ 参数空间较小，需要快速获得高质量推荐
- ✅ 需要可解释的推荐理由
- ✅ 先验数据较少，需要利用领域知识

### 何时使用 Hybrid Optimizer（融合优化）

- ✅ 需要结合领域知识和数据驱动方法
- ✅ 希望 LLM 推荐经过 GP 验证
- ✅ 需要平衡探索和利用
- ✅ 参数空间较大，需要更稳健的推荐

### 何时使用传统贝叶斯优化

- ✅ 有大量历史实验数据
- ✅ 参数空间较大，需要系统化探索
- ✅ 不需要领域知识，纯数据驱动
- ✅ 需要精确的数学优化方法

## 📊 性能对比

| 方法 | 优势 | 适用场景 |
|------|------|----------|
| **LLINBO** | 利用领域知识、可解释性强、快速推荐 | 小参数空间、有领域知识、先验数据少 |
| **Hybrid** | 结合知识和数据、稳健可靠 | 中等参数空间、需要平衡探索和利用 |
| **传统 GP** | 数学严谨、系统化探索、适合大数据 | 大参数空间、有丰富历史数据 |

## 🔍 技术细节

### LLINBO Agent 技术特点

- **提示词工程**: 精心设计的提示词，引导大模型进行参数优化推理
- **参数验证**: 自动验证推荐参数是否符合参数空间定义
- **类型转换**: 自动处理参数类型转换（int/float/str）
- **JSON 解析**: 智能解析大模型返回的 JSON 格式响应

### Hybrid Optimizer 技术特点

- **动态阈值**: 基于预测标准差动态调整接受阈值，适应不同不确定性
- **采集函数评估**: 使用 GP 的采集函数评估 LLM 推荐的质量
- **去重机制**: 自动检测和去除重复的参数推荐
- **特殊策略**: 当 GP 信心不足时，更信任 LLM 推荐




