from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.generation_strategy.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Generators  
import pandas as pd
import numpy as np
from scipy.stats import qmc



# ========================
# 1. Sobol参数生成器
# ========================
def generate_sobol_parameters(search_space, num_points=10, seed=None, prior_experiments=None):
    """
    生成Sobol采样参数组合，支持先验实验数据
    
    参数:
        search_space: 参数搜索空间
        num_points: 需要生成的参数组合数量
        seed: 随机种子
        prior_experiments: 先验实验数据列表，每个元素为字典包含参数和指标
        
    返回:
        list: 包含num_points个参数字典的列表
    """
    ax_client = AxClient(random_seed=seed)
    ax_client.create_experiment(
        name="experiment_design",
        parameters=search_space,
        objectives={"dummy": ObjectiveProperties(minimize=True)}
    )
    
    # 如果有先验实验数据，先添加到实验中
    if prior_experiments:
        for exp in prior_experiments:
            # 分离参数和指标
            params = {}
            metrics = {}
            for key, value in exp.items():
                # 检查是否是参数（在search_space中定义的）
                is_param = any(param["name"] == key for param in search_space)
                if is_param:
                    params[key] = value
                else:
                    metrics[key] = value
            
            # 添加实验数据
            if params and metrics:
                ax_client.attach_trial(parameters=params)
                trial_index = len(ax_client.experiment.trials) - 1
                # 使用第一个指标作为目标值
                first_metric = list(metrics.values())[0]
                ax_client.complete_trial(trial_index, raw_data=first_metric)
    
    # 生成新的参数组合
    trials_dict, _ = ax_client.get_next_trials(max_trials=num_points)
    
    # 收集生成的参数组合
    parameter_sets = list(trials_dict.values())
    
    return parameter_sets

def generate_lhs_parameters(search_space, num_points=10, seed=None):
    """
    生成超立方采样参数组合，支持连续和离散参数
    
    参数:
        search_space: 参数搜索空间，每个参数需要包含：
            - name: 参数名
            - type: 'choice'表示离散值，'range'表示连续值
            - values: 对于'choice'类型是可选值列表，对于'range'类型是[最小值, 最大值]
        num_points: 需要生成的参数组合数量
        seed: 随机种子
        
    返回:
        list: 包含num_points个参数字典的列表
    """
    if seed is not None:
        sampler = qmc.LatinHypercube(d=len(search_space), seed=seed)
    else:
        sampler = qmc.LatinHypercube(d=len(search_space))
        
    sample = sampler.random(n=num_points)
    
    # 将[0,1)映射到实际参数值
    parameter_sets = []
    for row in sample:
        params = {}
        for i, val in enumerate(row):
            param = search_space[i]
            param_name = param["name"]
            param_type = param["type"]
            
            if param_type == "choice":
                # 离散值采样
                values = param["values"]
                idx = int(val * len(values))
                if idx == len(values):  # 防止取到边界
                    idx -= 1
                params[param_name] = values[idx]
            elif param_type == "range":
                # 连续值采样
                min_val, max_val = param["bounds"]
                value = min_val + val * (max_val - min_val)
                # 根据value_type转换类型
                if param.get("value_type") == "int":
                    value = int(round(value))
                params[param_name] = value
            
        parameter_sets.append(params)
    return parameter_sets

# ========================
# 3. Uniform 随机参数生成器
# ========================
def generate_uniform_parameters(search_space, num_points=10, seed=None):
    """
    使用 Ax 的 Generators.UNIFORM 生成纯随机参数组合
    """
    gs = GenerationStrategy(
        name="UNIFORM_ONLY",
        steps=[GenerationStep(model=Generators.UNIFORM, num_trials=-1)]
    )

    ax_client = AxClient(random_seed=seed, generation_strategy=gs)
    ax_client.create_experiment(
        name="experiment_design_uniform",
        parameters=search_space,
        objectives={"dummy": ObjectiveProperties(minimize=True)},
    )

    # 生成新的参数组合
    trials_dict, _ = ax_client.get_next_trials(max_trials=num_points)
    
    # 收集生成的参数组合
    params_list = list(trials_dict.values())
    return params_list


    

def test():
    """
    测试DOE参数生成函数
    """
    # 定义示例搜索空间（混合连续和离散参数）
    search_space = [
        {
            "name": "power",
            "type": "range",  # 连续参数
            "bounds": [1000, 3000],  # 最小值和最大值
            "value_type": "int"  # 指定参数类型
        },
        {
            "name": "speed",
            "type": "range",  # 连续参数
            "bounds": [10, 50],  # 最小值和最大值
            "value_type": "int"  # 指定参数类型
        },
        {
            "name": "frequency",
            "type": "choice",  # 离散参数
            "values": [500, 1000, 1500, 2000],
            "value_type": "int"  # 指定参数类型
        }
    ]
    
    # 定义先验实验数据
    prior_experiments = [
        {"power": 1500, "speed": 25, "frequency": 1000, "accuracy": 0.85},
        {"power": 2500, "speed": 35, "frequency": 1500, "accuracy": 0.92},
        {"power": 2000, "speed": 15, "frequency": 500, "accuracy": 0.78}
    ]
    
    # 测试Sobol采样（无先验数据）
    print("=== Sobol采样测试（无先验数据）===")
    sobol_params = generate_sobol_parameters(search_space, num_points=3, seed=42)
    print("Sobol采样结果（3个样本）:")
    for i, params in enumerate(sobol_params, 1):
        print(f"样本 {i}:", params)
    
    # 测试Sobol采样（有先验数据）
    print("\n=== Sobol采样测试（有先验数据）===")
    sobol_params_with_prior = generate_sobol_parameters(search_space, num_points=3, seed=42, prior_experiments=prior_experiments)
    print("Sobol采样结果（3个样本，基于先验数据）:")
    for i, params in enumerate(sobol_params_with_prior, 1):
        print(f"样本 {i}:", params)
    
    print("\n=== 拉丁超立方采样测试 ===")
    lhs_params = generate_lhs_parameters(search_space, num_points=3, seed=42)
    print("LHS采样结果（3个样本）:")
    for i, params in enumerate(lhs_params, 1):
        print(f"样本 {i}:", params)

    print("\n=== Uniform 结果===")
    for i, p in enumerate(generate_uniform_parameters(search_space, 3, seed=42), 1):
        print(f" 样本 {i}: {p}")

if __name__ == "__main__":
    test()
