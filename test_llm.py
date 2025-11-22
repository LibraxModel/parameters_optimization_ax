import pandas as pd
import numpy as np
import os
import sys
import json
import requests
import time
import copy
sys.path.insert(0, '/root/sxw/edit')

from LLINBO_agent import (
    ProblemContext, Parameter, PriorExperiment, 
    LLMConfig, LLINBOAgent
)

# 设置环境变量
os.environ.setdefault("OPENAI_API_KEY", "key")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
def load_chemical_data(csv_path: str):
    """加载化学实验数据"""
    df = pd.read_csv(csv_path)
    return df


def build_parameter_space(df: pd.DataFrame):
    """Construct the parameter space definition for the chemical experiment."""

    parameters = [
        Parameter(
            name="base",
            type="choice",
            values=sorted(df["base"].unique().tolist()),
            value_type="str",
            description="Type of base used in the reaction (e.g., CsOAc, KOAc, etc.), which strongly affects the yield and selectivity.",
        ),
        Parameter(
            name="ligand",
            type="choice",
            values=sorted(df["ligand"].unique().tolist()),
            value_type="str",
            description="Type of ligand utilized to stabilize the catalyst; ligands modulate reactivity and product distribution.",
        ),
        Parameter(
            name="solvent",
            type="choice",
            values=sorted(df["solvent"].unique().tolist()),
            value_type="str",
            description="Type of solvent employed during the reaction; chosen for solubility and reaction rate optimization.",
        ),
        Parameter(
            name="concentration",
            type="choice",
            values=sorted(df["concentration"].unique().tolist()),
            value_type="float",
            description="Concentration of reactants in molarity (M); controls the collision rate of molecules and thus the kinetics.",
        ),
        Parameter(
            name="temperature",
            type="choice",
            values=sorted(df["temperature"].unique().tolist()),
            value_type="int",
            description="Reaction temperature in degrees Celsius (°C); higher or lower temperatures can affect reaction speed and outcomes.",
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

def create_prior_experiments(df: pd.DataFrame, parameters, objectives, problem_context,
                             n_samples: int = 50, 
                             llm_config: LLMConfig = None,
                             seed: int = None):
    """使用 LLINBO Agent 的初始采样方法创建先验实验数据"""
    print(f"\n📚 使用 LLINBO Agent 进行初始采样，生成 {n_samples} 个先验实验数据...")
    
    # 如果没有提供 llm_config，创建一个默认的
    if llm_config is None:
        llm_config = LLMConfig(
            model_name="gpt-5-nano",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1"
        )
    
    # 创建 LLINBO Agent（没有先验数据，用于初始采样）
    agent = LLINBOAgent(
        problem_context=problem_context,
        parameters=parameters,
        objectives=objectives,
        llm_config=llm_config,
        prior_experiments=None,  # 没有先验数据
        random_seed=seed
    )
    
    # 使用初始采样方法生成参数建议
    print(f"🤖 使用 LLINBO Agent 生成 {n_samples} 个初始采样建议...")
    suggestions = agent.suggest_initial_parameters(
        num_suggestions=n_samples,
        print_prompt=True,
        print_response=True
    )
    
    if not suggestions:
        print("⚠️ LLINBO Agent 未能生成有效建议，使用随机采样作为备选方案")
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
    experiment_results = simulate_experiment_results(suggestions, df, random_seed=seed)
    
    # 转换为 PriorExperiment 格式
    prior_experiments = convert_experiment_results_to_prior_experiments(experiment_results)
    
    print(f"✅ 成功创建 {len(prior_experiments)} 个先验实验数据")
    
    return prior_experiments

def create_llm_init_sampling(df: pd.DataFrame, parameters, objectives, problem_context,
                             n_samples: int = 50, 
                             llm_config: LLMConfig = None,
                             seed: int = None):
    """使用 LLINBO Agent 的初始采样方法创建先验实验数据"""
    print(f"\n📚 使用 LLINBO Agent 进行初始采样，生成 {n_samples} 个先验实验数据...")
    prior_experiments = create_prior_experiments(df, parameters, objectives, problem_context, n_samples, llm_config, seed)
    return prior_experiments

if __name__ == "__main__":

    df = load_chemical_data("test/1728_BMS_experiments_yield_and_cost.csv")
    parameters = build_parameter_space(df)
    objectives = {
        "yield": {"minimize": False},
        "cost": {"minimize": True}
    }
    problem_context = ProblemContext(
    problem_description="Optimization of the Pd-catalyzed C-H arylation reaction conditions. The reactants are N1-methyl-1H-imidazole-4-carbonitrile and 1-bromo-2-fluorobenzene.",
    industry="Chemical synthesis - organic reaction optimization",
    domain_knowledge="""
    This is an optimization problem for a Pd-catalyzed C-H arylation reaction.
    - The base affects the reactivity and selectivity of the reaction
    - The ligand is crucial to catalytic activity and selectivity
    - The solvent influences the reaction rate and product selectivity
    - Concentration impacts the reaction rate and side reactions
    - Temperature affects both the reaction rate and selectivity
    The optimization objective is to maximize yield and minimize cost simultaneously.
   
    """,
    constraints=[
        "All recommended parameter combinations must conform to basic chemical principles."
    ],
    optimization_goals=[
        "Maximize yield",
        "Minimize cost"
    ])
    import csv

    # 需要做十次实验，每次结果都保存，并标记实验编号
    all_experiments = []
    for trial_id in range(1, 11):
        prior_experiments = create_llm_init_sampling(
            df, parameters, objectives, problem_context,
            n_samples=10,
            llm_config=LLMConfig(
                model_name="gpt-5-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1"
            ),
            seed=42 + trial_id  # 可保证每次实验seed不同
        )
        print(f"✅ 第{trial_id}次实验: {len(prior_experiments)} 个实验")
        # 将实验结果加入到汇总列表，每行加上trial_id
        for exp in prior_experiments:
            row = {
                "trial_id": trial_id,
                **exp.parameters,
                **exp.metrics
            }
            all_experiments.append(row)

    # 保存所有实验数据到csv，并包含trial_id
    output_csv = "all_llm_prior_experiments.csv"
    if all_experiments:
        fieldnames = ["trial_id"] + list(prior_experiments[0].parameters.keys()) + list(prior_experiments[0].metrics.keys())
        with open(output_csv, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_experiments)
        print(f"\n✅ 已将所有{len(all_experiments)}个实验（共10次，每次10个）保存到 {output_csv}")