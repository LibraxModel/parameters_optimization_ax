"""
使用 LLINBO Agent 优化化学实验数据集 - Notebook 格式
优化目标：最大化产率（yield），最小化成本（cost）
每个单元格用 # %% 分隔，方便在 notebook 中逐个运行
"""

# %% [markdown]
# # 导入库和设置环境变量

# %%
import pandas as pd
import numpy as np
import os
import sys
import json
import requests
import time
import copy
sys.path.insert(0, '/root/sxw/edit')

# 重新导入模块，确保使用最新代码（如果在 notebook 中运行）
try:
    import importlib
    if 'LLINBO_agent' in sys.modules:
        importlib.reload(sys.modules['LLINBO_agent'])
        print("🔄 重新加载 LLINBO_agent 模块...")
except Exception as e:
    print(f"⚠️ 重新加载模块失败: {e}")

from LLINBO_agent import (
    ProblemContext, Parameter, PriorExperiment, 
    LLMConfig, LLINBOAgent
)

# 检查环境变量
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️ 警告: 未设置 OPENAI_API_KEY 环境变量")
    print("请设置环境变量: export OPENAI_API_KEY='your-api-key'")
    print("或在代码中设置: os.environ['OPENAI_API_KEY'] = 'your-api-key'")

# 可选：设置代理（如果需要）
# os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
# os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")

print("✅ 库导入完成")

# %% [markdown]
# # 定义辅助函数

# %%
def load_chemical_data(csv_path: str):
    """加载化学实验数据"""
    df = pd.read_csv(csv_path)
    return df


def build_parameter_space(df: pd.DataFrame):
    """构建参数空间定义"""
    parameters = [
        Parameter(
            name="base",
            type="choice",
            values=sorted(df["base"].unique().tolist()),
            value_type="str",
            description="碱基类型",
        ),
        Parameter(
            name="ligand",
            type="choice",
            values=sorted(df["ligand"].unique().tolist()),
            value_type="str",
            description="配体类型",
        ),
        Parameter(
            name="solvent",
            type="choice",
            values=sorted(df["solvent"].unique().tolist()),
            value_type="str",
            description="溶剂类型",
        ),
        Parameter(
            name="concentration",
            type="choice",
            values=sorted(df["concentration"].unique().tolist()),
            value_type="float",
            description="反应浓度 (M)",
        ),
        Parameter(
            name="temperature",
            type="choice",
            values=sorted(df["temperature"].unique().tolist()),
            value_type="int",
            description="反应温度 (°C)",
        ),
    ]
    return parameters


def convert_parameters_to_api_format(parameters):
    """将 LLINBO Parameter 转换为 API ParameterSpace 格式"""
    api_params = []
    for param in parameters:
        api_param = {
            "name": param.name,
            "type": param.type,
            "values": param.values
        }
        api_params.append(api_param)
    return api_params


def call_init_api(parameter_space, objectives, batch_size=20, seed=None, api_url="http://localhost:3320"):
    """调用 init 接口进行采样"""
    init_endpoint = f"{api_url}/init"
    
    # 转换参数空间格式
    api_parameter_space = convert_parameters_to_api_format(parameter_space)
    
    # 构建请求数据
    init_request = {
        "parameter_space": api_parameter_space,
        "objectives": list(objectives.keys()),
        "batch": batch_size,
        "seed": seed,
        "sampling_method": "lhs"  # 使用 lhs 采样
    }
    
    print(f"🚀 调用 init 接口，批次大小: {batch_size}")
    
    try:
        response = requests.post(init_endpoint, json=init_request, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Init 接口调用成功")
            print(f"📈 生成参数组合数: {len(result['results'])}")
            print(f"💬 消息: {result['message']}")
            return result
        else:
            print(f"❌ Init 接口调用失败: {response.status_code}")
            print(f"📄 错误信息: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Init 接口调用异常: {str(e)}")
        return None


def simulate_experiment_results(params_list, df, random_seed=None):
    """从真实数据中查找完全匹配的实验结果"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    results = []
    
    for params in params_list:
        # 构建精确匹配的查询条件
        query_conditions = []
        for key, value in params.items():
            if isinstance(value, str):
                query_conditions.append(f"{key} == '{value}'")
            else:
                query_conditions.append(f"{key} == {value}")
        
        # 构建查询字符串
        query_str = " and ".join(query_conditions)
        
        try:
            # 查找完全匹配的数据
            matched_data = df.query(query_str)
            
            if len(matched_data) > 0:
                # 取第一个匹配的结果
                row = matched_data.iloc[0]
                experiment_result = {
                    "parameters": params,
                    "metrics": {
                        "yield": float(row['yield']),
                        "cost": float(row['cost'])
                    }
                }
                results.append(experiment_result)
                print(f"✅ 找到完全匹配: {params} -> yield={row['yield']:.2f}, cost={row['cost']:.3f}")
            else:
                print(f"⚠️ 未找到完全匹配: {params}")
                # 如果找不到，使用随机结果
                random_row = df.sample(1, random_state=random_seed).iloc[0]
                experiment_result = {
                    "parameters": params,
                    "metrics": {
                        "yield": float(random_row['yield']),
                        "cost": float(random_row['cost'])
                    }
                }
                results.append(experiment_result)
                print(f"⚠️ 使用随机结果: {params} -> yield={random_row['yield']:.2f}, cost={random_row['cost']:.3f}")
                
        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")
            # 使用随机结果
            random_row = df.sample(1, random_state=random_seed).iloc[0]
            experiment_result = {
                "parameters": params,
                "metrics": {
                    "yield": float(random_row['yield']),
                    "cost": float(random_row['cost'])
                }
            }
            results.append(experiment_result)
    
    return results


def convert_experiment_results_to_prior_experiments(experiment_results):
    """将实验结果转换为 PriorExperiment 格式"""
    prior_experiments = []
    for result in experiment_results:
        exp = PriorExperiment(
            parameters=result["parameters"],
            metrics=result["metrics"]
        )
        prior_experiments.append(exp)
    return prior_experiments


def create_prior_experiments(df: pd.DataFrame, parameters, objectives, n_samples: int = 50, 
                             api_url: str = "http://localhost:3320", seed: int = None):
    """使用 init 接口进行采样，创建先验实验数据"""
    print(f"\n📚 使用 init 接口进行采样，生成 {n_samples} 个先验实验数据...")
    
    # 调用 init 接口进行采样
    init_result = call_init_api(parameters, objectives, batch_size=n_samples, seed=seed, api_url=api_url)
    
    if not init_result:
        print("⚠️ Init 接口调用失败，使用随机采样作为备选方案")
        # 备选方案：随机采样
        sample_df = df.sample(n=min(n_samples, len(df)), random_state=seed)
        prior_experiments = []
        for _, row in sample_df.iterrows():
            exp = PriorExperiment(
                parameters={
                    "base": str(row["base"]),
                    "ligand": str(row["ligand"]),
                    "solvent": str(row["solvent"]),
                    "concentration": float(row["concentration"]),
                    "temperature": int(row["temperature"]),
                },
                metrics={
                    "yield": float(row["yield"]),
                    "cost": float(row["cost"]),
                }
            )
            prior_experiments.append(exp)
        return prior_experiments
    
    # 从数据集中查找匹配的实验结果
    print(f"\n🔍 从数据集中查找匹配的实验结果...")
    experiment_results = simulate_experiment_results(init_result['results'], df, random_seed=seed)
    
    # 转换为 PriorExperiment 格式
    prior_experiments = convert_experiment_results_to_prior_experiments(experiment_results)
    
    print(f"✅ 成功创建 {len(prior_experiments)} 个先验实验数据")
    
    return prior_experiments

print("✅ 辅助函数定义完成")

# %% [markdown]
# # 单次 LLINBO Agent 测试

# %% [markdown]
# ## 1. 加载数据

# %%
csv_path = "test/1728_BMS_experiments_yield_and_cost.csv"
print(f"📊 加载数据集: {csv_path}")
df = load_chemical_data(csv_path)
print(f"✅ 数据集加载成功: {len(df)} 个实验条件")

# %% [markdown]
# ## 2. 定义问题背景

# %%
problem_context = ProblemContext(
    problem_description="优化 Pd 催化的 C-H 芳基化反应条件，反应物为 N1-甲基-1H-咪唑-4-甲腈和 1-溴-2-氟苯",
    industry="化学合成 - 有机反应优化",
    domain_knowledge="""
    这是一个 Pd 催化的 C-H 芳基化反应优化问题。
    - 碱基（base）影响反应活性和选择性
    - 配体（ligand）对催化剂活性和选择性至关重要
    - 溶剂（solvent）影响反应速率和产物选择性
    - 浓度（concentration）影响反应速率和副反应
    - 温度（temperature）影响反应速率和选择性
    优化目标是同时最大化产率（yield）和最小化成本（cost）。
    """,
    constraints=[
        "所有参数必须从实验验证过的值中选择"
    ],
    optimization_goals=[
        "最大化产率（yield）",
        "最小化成本（cost）"
    ]
)
print("✅ 问题背景定义完成")

# %% [markdown]
# ## 3. 构建参数空间

# %%
print("\n🔧 构建参数空间...")
parameters = build_parameter_space(df)
print(f"✅ 参数空间定义完成: {len(parameters)} 个参数")
for param in parameters:
    print(f"   - {param.name}: {len(param.values)} 个可选值")

# %% [markdown]
# ## 4. 定义优化目标

# %%
objectives = {
    "yield": {"minimize": False},  # 最大化产率
    "cost": {"minimize": True}      # 最小化成本
}
print("✅ 优化目标定义完成")
print(f"   - yield: 最大化")
print(f"   - cost: 最小化")

# %% [markdown]
# ## 5. 创建先验实验数据（使用 init 接口进行采样）

# %%
print("\n📚 创建先验实验数据...")
prior_experiments = create_prior_experiments(
    df, parameters, objectives, 
    n_samples=20,  # 初始批次大小
    api_url="http://localhost:3320",
    seed=42
)
print(f"✅ 先验实验数据: {len(prior_experiments)} 个实验")

# 显示先验数据统计
yield_values = [exp.metrics["yield"] for exp in prior_experiments]
cost_values = [exp.metrics["cost"] for exp in prior_experiments]
print(f"   产率范围: [{min(yield_values):.2f}, {max(yield_values):.2f}], 平均值: {sum(yield_values)/len(yield_values):.2f}")
print(f"   成本范围: [{min(cost_values):.4f}, {max(cost_values):.4f}], 平均值: {sum(cost_values)/len(cost_values):.4f}")

# %% [markdown]
# ## 6. 创建 LLINBO Agent

# %%
print("\n🤖 初始化 LLINBO Agent...")
llm_config = LLMConfig(
    model_name="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"
)

agent = LLINBOAgent(
    problem_context=problem_context,
    parameters=parameters,
    objectives=objectives,
    llm_config=llm_config,
    prior_experiments=prior_experiments
)
print("✅ LLINBO Agent 初始化完成")

# %% [markdown]
# ## 7. 生成优化建议

# %% [markdown]
# **注意**：如果修改了 `LLINBO_agent.py`，请先运行下面的单元格重新导入模块

# %%
# 重新导入模块，确保使用最新代码
try:
    import importlib
    if 'LLINBO_agent' in sys.modules:
        importlib.reload(sys.modules['LLINBO_agent'])
        print("🔄 重新加载 LLINBO_agent 模块...")
        # 重新导入类
        from LLINBO_agent import LLINBOAgent
        # 重新创建 agent（如果需要）
        # agent = LLINBOAgent(...)
except Exception as e:
    print(f"⚠️ 重新加载模块失败: {e}")

# %%
print("\n🎯 生成优化建议...")
num_suggestions = 5
suggestions = agent.suggest_parameters(num_suggestions=num_suggestions)
print(f"✅ 成功生成 {len(suggestions)} 个建议")

# %% [markdown]
# ## 8. 显示结果

# %%
print(f"\n📊 生成的优化建议 ({len(suggestions)} 个):")
print("=" * 80)
for i, suggestion in enumerate(suggestions, 1):
    print(f"\n建议 {i}:")
    for param_name, param_value in suggestion.items():
        print(f"  {param_name}: {param_value}")

# %% [markdown]
# ## 9. 检查建议是否在数据集中

# %%
print("\n🔍 检查建议是否在原始数据集中:")
for i, suggestion in enumerate(suggestions, 1):
    # 构建查询条件
    mask = (
        (df["base"] == suggestion["base"]) &
        (df["ligand"] == suggestion["ligand"]) &
        (df["solvent"] == suggestion["solvent"]) &
        (df["concentration"] == suggestion["concentration"]) &
        (df["temperature"] == suggestion["temperature"])
    )
    matching_rows = df[mask]
    
    if len(matching_rows) > 0:
        row = matching_rows.iloc[0]
        print(f"\n建议 {i} - 在数据集中找到:")
        print(f"  产率: {row['yield']:.2f}%")
        print(f"  成本: {row['cost']:.4f}")
    else:
        print(f"\n建议 {i} - 不在数据集中（新组合）")

# %% [markdown]
# ## 10. 显示优化摘要



# %% [markdown]
# # 多轮优化测试（使用 LLINBO Agent）

# %% [markdown]
# ## 初始化多轮测试参数

# %%
# 重新加载数据（如果需要）
csv_path = "test/1728_BMS_experiments_yield_and_cost.csv"
df_multi = load_chemical_data(csv_path)
parameters_multi = build_parameter_space(df_multi)
objectives_multi = {
    "yield": {"minimize": False},  # 最大化产率
    "cost": {"minimize": True}      # 最小化成本
}

problem_context_multi = ProblemContext(
    problem_description="优化 Pd 催化的 C-H 芳基化反应条件，反应物为 N1-甲基-1H-咪唑-4-甲腈和 1-溴-2-氟苯",
    industry="化学合成 - 有机反应优化",
    domain_knowledge="""
    这是一个 Pd 催化的 C-H 芳基化反应优化问题。
    - 碱基（base）影响反应活性和选择性
    - 配体（ligand）对催化剂活性和选择性至关重要
    - 溶剂（solvent）影响反应速率和产物选择性
    - 浓度（concentration）影响反应速率和副反应
    - 温度（temperature）影响反应速率和选择性
    优化目标是同时最大化产率（yield）和最小化成本（cost）。
    """,
    constraints=[
        "所有参数必须从实验验证过的值中选择"
    ],
    optimization_goals=[
        "最大化产率（yield）",
        "最小化成本（cost）"
    ]
)

# 多轮测试配置
n_rounds = 10
init_batch = 20
update_batch = 3
api_url = "http://localhost:3320"
seed = 42

print("✅ 多轮测试参数初始化完成")
print(f"   总轮次: {n_rounds}")
print(f"   初始批次: {init_batch}")
print(f"   更新批次: {update_batch}")

# %% [markdown]
# ## 定义多轮优化函数

# %%
def run_llinbo_optimization(df, parameters, objectives, problem_context, 
                           n_rounds=10, init_batch=20, update_batch=3, 
                           api_url="http://localhost:3320", seed=42):
    """使用 LLINBO Agent 进行多轮优化，仿照 notebook 的方式"""
    
    print(f"\n🔄 开始 {n_rounds} 轮优化循环（使用 LLINBO Agent）")
    print(f"📊 初始批次: {init_batch}, 更新批次: {update_batch}")
    
    all_experiments = []
    optimization_history = []
    
    # 第一轮：初始化（使用 init 接口）
    print(f"\n=== 第 1 轮：初始化 ===")
    init_result = call_init_api(parameters, objectives, batch_size=init_batch, seed=seed, api_url=api_url)
    
    if not init_result:
        print("❌ 初始化失败，退出优化循环")
        return None, None
    
    # 模拟第一轮实验结果
    round_results = simulate_experiment_results(init_result['results'], df, random_seed=seed)
    round_prior_experiments = convert_experiment_results_to_prior_experiments(round_results)
    all_experiments.extend(round_prior_experiments)
    
    # 记录历史
    optimization_history.append({
        'round': 1,
        'type': 'init',
        'parameters': init_result['results'],
        'results': round_results,
        'best_yield': max([r['metrics']['yield'] for r in round_results]),
        'best_cost': min([r['metrics']['cost'] for r in round_results])
    })
    
    print(f"📈 第 1 轮最佳结果: yield={optimization_history[-1]['best_yield']:.2f}, cost={optimization_history[-1]['best_cost']:.3f}")
    
    # 后续轮次：使用 LLINBO Agent（代替 update 接口）
    for round_num in range(2, n_rounds + 1):
        print(f"\n=== 第 {round_num} 轮：LLINBO 优化 ===")
        
        # 重要：在创建 agent 之前，确认上一轮的结果已经加入
        # 显示当前先验数据数量（确认数据已更新）
        print(f"📊 当前先验实验数据数量: {len(all_experiments)} 个")
        if len(all_experiments) > 0:
            # 显示最近几轮的数据，确认包含上一轮的结果
            recent_count = min(5, len(all_experiments))
            recent_yields = [exp.metrics["yield"] for exp in all_experiments[-recent_count:]]
            recent_costs = [exp.metrics["cost"] for exp in all_experiments[-recent_count:]]
            print(f"   最近 {recent_count} 个实验的产率: {[f'{y:.2f}' for y in recent_yields]}")
            print(f"   最近 {recent_count} 个实验的成本: {[f'{c:.4f}' for c in recent_costs]}")
            
            # 显示上一轮（第 round_num-1 轮）的结果是否在当前先验数据中
            if round_num > 2:
                prev_round_history = [h for h in optimization_history if h['round'] == round_num - 1]
                if prev_round_history:
                    prev_round_params = prev_round_history[0]['parameters']
                    print(f"   🔍 检查上一轮（第 {round_num-1} 轮）的结果是否在先验数据中:")
                    for i, prev_param in enumerate(prev_round_params, 1):
                        # 检查这个参数组合是否在 all_experiments 中
                        found = False
                        for exp in all_experiments:
                            if all(exp.parameters.get(k) == prev_param.get(k) for k in prev_param.keys()):
                                found = True
                                print(f"      上一轮建议 {i}: ✅ 已在先验数据中 (yield={exp.metrics['yield']:.2f}, cost={exp.metrics['cost']:.4f})")
                                break
                        if not found:
                            print(f"      上一轮建议 {i}: ❌ 未在先验数据中！")
        
        # 创建 LLINBO Agent（使用当前所有实验数据作为先验）
        # 重要：确保使用更新后的 all_experiments（包含之前所有轮次的结果）
        # 使用深拷贝确保数据独立
        prior_experiments_for_agent = copy.deepcopy(all_experiments)
        
        print(f"🔧 创建 LLINBO Agent，使用 {len(prior_experiments_for_agent)} 个先验实验数据")
        
        llm_config = LLMConfig(
            model_name="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1"
        )
        
        # 创建新的 agent，传入更新后的先验数据
        agent = LLINBOAgent(
            problem_context=problem_context,
            parameters=parameters,
            objectives=objectives,
            llm_config=llm_config,
            prior_experiments=prior_experiments_for_agent  # 使用深拷贝，确保数据正确传递
        )
        
        # 验证 agent 中的先验数据数量
        print(f"✅ Agent 创建完成，Agent 中的先验数据数量: {len(agent.prior_experiments)} 个")
        
        # 生成优化建议（LLINBO Agent 代替 update 接口）
        print(f"🤖 使用 LLINBO Agent 生成 {update_batch} 个优化建议...")
        
        # 添加延迟，确保不会因为请求过快导致问题
        time.sleep(0.5)  # 在请求前等待，避免请求过快
        
        suggestions = agent.suggest_parameters(num_suggestions=update_batch)
        
        if not suggestions:
            print(f"⚠️ 第 {round_num} 轮未能生成有效建议，跳过")
            continue
        
        print(f"✅ 成功生成 {len(suggestions)} 个建议")
        
        # 模拟实验结果
        round_results = simulate_experiment_results(suggestions, df, random_seed=seed)
        round_prior_experiments = convert_experiment_results_to_prior_experiments(round_results)
        
        # 重要：将本轮结果加入到先验数据中，供下一轮使用
        # 在加入之前，先检查是否有重复
        print(f"📝 准备将第 {round_num} 轮的 {len(round_prior_experiments)} 个实验结果加入到先验数据中")
        for i, new_exp in enumerate(round_prior_experiments, 1):
            # 检查是否已经存在相同的参数组合
            is_duplicate = False
            for existing_exp in all_experiments:
                if all(existing_exp.parameters.get(k) == new_exp.parameters.get(k) 
                       for k in new_exp.parameters.keys()):
                    is_duplicate = True
                    print(f"   ⚠️ 第 {round_num} 轮建议 {i} 的参数组合已存在于先验数据中，跳过重复添加")
                    break
            if not is_duplicate:
                all_experiments.append(new_exp)
                print(f"   ✅ 第 {round_num} 轮建议 {i} 已加入先验数据")
        
        print(f"📊 更新后的先验数据总数: {len(all_experiments)} 个")
        
        # 记录历史
        optimization_history.append({
            'round': round_num,
            'type': 'llinbo',
            'parameters': suggestions,
            'results': round_results,
            'best_yield': max([r['metrics']['yield'] for r in round_results]),
            'best_cost': min([r['metrics']['cost'] for r in round_results])
        })
        
        print(f"📈 第 {round_num} 轮最佳结果: yield={optimization_history[-1]['best_yield']:.2f}, cost={optimization_history[-1]['best_cost']:.3f}")
        
        # 显示本轮生成的建议详情（类似 main 函数中的显示）
        print(f"\n📋 第 {round_num} 轮生成的建议详情:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  建议 {i}: {suggestion}")
        
        # 检查建议是否在数据集中（类似 main 函数中的检查）
        print(f"\n🔍 检查第 {round_num} 轮建议是否在原始数据集中:")
        for i, suggestion in enumerate(suggestions, 1):
            mask = (
                (df["base"] == suggestion["base"]) &
                (df["ligand"] == suggestion["ligand"]) &
                (df["solvent"] == suggestion["solvent"]) &
                (df["concentration"] == suggestion["concentration"]) &
                (df["temperature"] == suggestion["temperature"])
            )
            matching_rows = df[mask]
            
            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                print(f"  建议 {i} - 在数据集中找到: yield={row['yield']:.2f}%, cost={row['cost']:.4f}")
            else:
                print(f"  建议 {i} - 不在数据集中（新组合）")
        
        # 添加延迟，避免请求过快
        time.sleep(1)
    
    return all_experiments, optimization_history

print("✅ 多轮优化函数定义完成")

# %% [markdown]
# ## 运行多轮优化

# %%
llinbo_experiments, llinbo_history = run_llinbo_optimization(
    df_multi, parameters_multi, objectives_multi, problem_context_multi,
    n_rounds=n_rounds,
    init_batch=init_batch,
    update_batch=update_batch,
    api_url=api_url,
    seed=seed
)

# %% [markdown]
# ## 显示优化结果汇总

# %%
if llinbo_history:
    print("\n📊 LLINBO 优化历史汇总:")
    print(f"   总轮次: {len(llinbo_history)}")
    print(f"   总实验数: {len(llinbo_experiments)}")
    
    best_yields = [h['best_yield'] for h in llinbo_history]
    best_costs = [h['best_cost'] for h in llinbo_history]
    
    print(f"   最佳产率: {max(best_yields):.2f} (第 {best_yields.index(max(best_yields)) + 1} 轮)")
    print(f"   最佳成本: {min(best_costs):.4f} (第 {best_costs.index(min(best_costs)) + 1} 轮)")
    print(f"   最终产率: {best_yields[-1]:.2f}")
    print(f"   最终成本: {best_costs[-1]:.4f}")
    
    # 显示每轮结果
    print("\n📈 每轮优化结果:")
    for h in llinbo_history:
        print(f"   第 {h['round']} 轮 ({h['type']}): yield={h['best_yield']:.2f}, cost={h['best_cost']:.3f}")
    
    # 显示所有实验的统计信息（类似 main 函数中的统计）
    print("\n📊 所有实验统计信息:")
    all_yield_values = [exp.metrics["yield"] for exp in llinbo_experiments]
    all_cost_values = [exp.metrics["cost"] for exp in llinbo_experiments]
    print(f"   产率范围: [{min(all_yield_values):.2f}, {max(all_yield_values):.2f}], 平均值: {sum(all_yield_values)/len(all_yield_values):.2f}")
    print(f"   成本范围: [{min(all_cost_values):.4f}, {max(all_cost_values):.4f}], 平均值: {sum(all_cost_values)/len(all_cost_values):.4f}")
    
    # 显示最佳实验组合
    best_yield_idx = all_yield_values.index(max(all_yield_values))
    best_cost_idx = all_cost_values.index(min(all_cost_values))
    print(f"\n🏆 最佳产率实验 (第 {best_yield_idx + 1} 个):")
    print(f"   参数: {llinbo_experiments[best_yield_idx].parameters}")
    print(f"   产率: {llinbo_experiments[best_yield_idx].metrics['yield']:.2f}%")
    print(f"   成本: {llinbo_experiments[best_yield_idx].metrics['cost']:.4f}")
    print(f"\n🏆 最佳成本实验 (第 {best_cost_idx + 1} 个):")
    print(f"   参数: {llinbo_experiments[best_cost_idx].parameters}")
    print(f"   产率: {llinbo_experiments[best_cost_idx].metrics['yield']:.2f}%")
    print(f"   成本: {llinbo_experiments[best_cost_idx].metrics['cost']:.4f}")

