"""
使用 LLINBO Agent 优化化学实验数据集
优化目标：最大化产率（yield），最小化成本（cost）
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import requests
import time
sys.path.insert(0, '/root/sxw/edit')

from LLINBO_agent import (
    ProblemContext, Parameter, PriorExperiment, 
    LLMConfig, LLINBOAgent
)


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


def main():
    """主函数"""
    # 1. 加载数据
    csv_path = "test/1728_BMS_experiments_yield_and_cost.csv"
    print(f"📊 加载数据集: {csv_path}")
    df = load_chemical_data(csv_path)
    print(f"✅ 数据集加载成功: {len(df)} 个实验条件")
    
    # 2. 定义问题背景
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
    
    # 3. 构建参数空间
    print("\n🔧 构建参数空间...")
    parameters = build_parameter_space(df)
    print(f"✅ 参数空间定义完成: {len(parameters)} 个参数")
    for param in parameters:
        print(f"   - {param.name}: {len(param.values)} 个可选值")
    
    # 4. 定义优化目标
    objectives = {
        "yield": {"minimize": False},  # 最大化产率
        "cost": {"minimize": True}      # 最小化成本
    }
    
    # 5. 创建先验实验数据（使用 init 接口进行采样）
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
    
    # 6. 创建 LLINBO Agent
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
    
    # 7. 生成优化建议
    print("\n🎯 生成优化建议...")
    num_suggestions = 5
    suggestions = agent.suggest_parameters(num_suggestions=num_suggestions)
    
    # 8. 显示结果
    print(f"\n📊 生成的优化建议 ({len(suggestions)} 个):")
    print("=" * 80)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n建议 {i}:")
        for param_name, param_value in suggestion.items():
            print(f"  {param_name}: {param_value}")
    
    # 9. 检查建议是否在数据集中
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
    
    # 10. 显示优化摘要
    summary = agent.get_optimization_summary()
    print("\n📈 优化摘要:")
    print(f"   总建议数: {summary['total_suggestions']}")
    print(f"   先验实验数: {summary['total_experiments']}")
    print(f"   参数空间大小: {summary['parameter_space_size']}")
    print(f"   优化目标: {', '.join(summary['objectives'])}")
    
    return suggestions, agent


if __name__ == "__main__":
    # 设置环境变量
    os.environ.setdefault("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
    
    suggestions, agent = main()

