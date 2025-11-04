import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any
import json
import numpy as np
import requests
API_BASE_URL = "http://localhost:3320"
INIT_ENDPOINT = f"{API_BASE_URL}/init"
UPDATE_ENDPOINT = f"{API_BASE_URL}/update"


def build_parameter_space(df, parameter_columns):
    """根据数据构建参数空间 - 所有参数都是choice类型"""
    parameter_space = []
    
    for col in parameter_columns:
        unique_values = df[col].unique()
        
        # 转换成原生 Python 类型，避免 np.int64 / np.float64 报错
        converted_values = [v.item() if isinstance(v, (np.generic,)) else v for v in unique_values]
        
        parameter_space.append({
            "name": col,
            "type": "choice",
            "values": converted_values
        })
    
    return parameter_space
def simulate_experiment_results(params_list, df, random_seed=42):
    """从真实数据中查找完全匹配的实验结果"""
    # 设置随机种子确保结果一致性
    np.random.seed(random_seed)
    
    results = []
    
    for params in params_list:
        # 构建精确匹配的查询条件
        query_conditions = []
        for key, value in params.items():
            # 所有参数都进行精确匹配
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
                        "point_hypervolume": float(row['point_hypervolume']),  # 用于优化目标
                        "yield": float(row['yield']),  # 用于跟踪目标
                        "cost": float(row['cost'])
                    }
                }
                results.append(experiment_result)
                print(f"✅ 找到完全匹配: {params} -> yield={row['yield']:.2f}, cost={row['cost']:.3f}, point_hypervolume={row['point_hypervolume']:.6f}")
            else:
                # 没有找到完全匹配，尝试查找最接近的参数组合
                print(f"⚠️ 未找到完全匹配: {params}")
                print(f"   尝试查找最接近的参数组合...")
                
                # 计算每个参数的距离
                best_match = None
                min_distance = float('inf')
                
                for _, row in df.iterrows():
                    distance = 0
                    match = True
                    
                    for key, target_value in params.items():
                        actual_value = row[key]
                        
                        if key in ['base', 'ligand', 'solvent']:
                            # 类别参数必须完全匹配
                            if actual_value != target_value:
                                match = False
                                break
                        else:
                            # 数值参数计算距离
                            distance += abs(actual_value - target_value)
                    
                    if match and distance < min_distance:
                        min_distance = distance
                        best_match = row
                
                if best_match is not None:
                    experiment_result = {
                        "parameters": params,
                        "metrics": {
                            "point_hypervolume": float(best_match['point_hypervolume']),  # 用于优化目标
                            "yield": float(best_match['yield']),  # 用于跟踪目标
                            "cost": float(best_match['cost'])
                        }
                    }
                    results.append(experiment_result)
                    print(f"✅ 找到最接近匹配: {params} -> yield={best_match['yield']:.2f}, cost={best_match['cost']:.3f}, point_hypervolume={best_match['point_hypervolume']:.6f}")
                else:
                    print(f"❌ 未找到任何匹配: {params}")
                    # 如果实在找不到，使用随机结果（这种情况应该很少）
                    random_row = df.sample(1, random_state=random_seed).iloc[0]
                    experiment_result = {
                        "parameters": params,
                        "metrics": {
                            "point_hypervolume": float(random_row['point_hypervolume']),  # 用于优化目标
                            "yield": float(random_row['yield']),  # 用于跟踪目标
                            "cost": float(random_row['cost'])
                        }
                    }
                    results.append(experiment_result)
                    print(f"⚠️ 使用随机结果: {params} -> yield={random_row['yield']:.2f}, cost={random_row['cost']:.3f}, point_hypervolume={random_row['point_hypervolume']:.6f}")
                
        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")
            # 使用随机结果
            random_row = df.sample(1, random_state=random_seed).iloc[0]
            experiment_result = {
                "parameters": params,
                "metrics": {
                    "point_hypervolume": float(random_row['point_hypervolume']),  # 用于优化目标
                    "yield": float(random_row['yield']),  # 用于跟踪目标
                    "cost": float(random_row['cost'])
                }
            }
            results.append(experiment_result)
    
    return results

def call_init_api(parameter_space, objectives, batch_size=5, seed=42):
    """调用init接口初始化优化器"""
    
    # 构建请求数据
    init_request = {
        "parameter_space": parameter_space,
        "objectives": list(objectives.keys()),
        "batch": batch_size,
        "seed": seed,
        "sampling_method": "lhs"  # 使用sobol采样
    }
    
    print(f"🚀 调用init接口，批次大小: {batch_size}")
    print(f"📋 请求数据: {json.dumps(init_request, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(INIT_ENDPOINT, json=init_request)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Init接口调用成功")
            print(f"📊 采样方法: {result['sampling_method']}")
            print(f"📈 生成参数组合数: {len(result['results'])}")
            print(f"💬 消息: {result['message']}")
            return result
        else:
            print(f"❌ Init接口调用失败: {response.status_code}")
            print(f"📄 错误信息: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Init接口调用异常: {str(e)}")
        return None
    

def call_update_api(parameter_space, objectives, completed_experiments, batch_size=3, use_weights=False, 
                    seed=42, objective_weights=None):
    """调用update接口进行贝叶斯优化"""
    
    # 构建请求数据
    update_request = {
        "parameter_space": parameter_space,
        "objectives": objectives,
        "completed_experiments": completed_experiments,
        "batch": batch_size,
        "use_weights": use_weights,
        "objective_weights": objective_weights if use_weights else None,
        "additional_metrics": [],  # yield和cost都是优化目标，不需要额外指标
        "seed": seed
    }
    
    print(f"🔄 调用update接口，批次大小: {batch_size}")
    print(f"📊 已完成实验数: {len(completed_experiments)}")
    print(f"⚖️ 使用权重: {use_weights}")
    
    try:
        response = requests.post(UPDATE_ENDPOINT, json=update_request)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Update接口调用成功")
            print(f"📈 推荐参数组合数: {len(result['results'])}")
            print(f"💬 消息: {result['message']}")
            return result
        else:
            print(f"❌ Update接口调用失败: {response.status_code}")
            print(f"📄 错误信息: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Update接口调用异常: {str(e)}")
        return None


def run_optimization_loop(parameter_space, objectives, df, n_rounds=100, init_batch=15, update_batch=3, seed=42,
                           target_yield=99.81):
    """运行多轮优化循环，记录找到yield=99.81时的实验次数"""
    
    all_experiments = []
    optimization_history = []
    target_yield_record = None  # 记录找到yield=99.81时的实验次数
    total_experiments = 0
    best_hypervolume = -float('inf')
    
    print(f"🔄 开始{n_rounds}轮优化循环")
    print(f"📊 初始批次: {init_batch}, 更新批次: {update_batch}")
    print(f"🎯 跟踪目标: yield={target_yield}")
    
    # 第一轮：初始化
    print("\n=== 第1轮：初始化 ===")
    init_result = call_init_api(parameter_space, objectives, batch_size=init_batch, seed=seed)
    
    if not init_result:
        print("❌ 初始化失败，退出优化循环")
        return None, None, target_yield_record
    
    # 模拟第一轮实验结果
    round_results = simulate_experiment_results(init_result['results'], df, random_seed=seed)
    
    # 逐个处理结果
    for result in round_results:
        all_experiments.append(result)
        total_experiments += 1
        
        hypervolume_val = result['metrics']['point_hypervolume']
        best_hypervolume = max(best_hypervolume, hypervolume_val)
        
        # 检查是否找到目标yield值
        yield_val = result['metrics'].get('yield')
        if yield_val is not None and abs(yield_val - target_yield) < 0.01 and target_yield_record is None:
            target_yield_record = total_experiments
            print(f"🎯 找到目标yield={target_yield:.2f}，当前实验次数: {total_experiments}")
            print(f"✅ 已找到目标yield={target_yield:.2f}，终止优化")
            # 记录历史后立即返回
            optimization_history.append({
                'round': 1,
                'type': 'init',
                'parameters': init_result['results'],
                'results': round_results,
                'best_hypervolume': best_hypervolume,
                'total_experiments': total_experiments
            })
            return all_experiments, optimization_history, target_yield_record
    
    # 记录历史
    optimization_history.append({
        'round': 1,
        'type': 'init',
        'parameters': init_result['results'],
        'results': round_results,
        'best_hypervolume': best_hypervolume,
        'total_experiments': total_experiments
    })
    
    print(f"📈 第1轮最佳结果: point_hypervolume={best_hypervolume:.6f}")
    print(f"📊 累计实验次数: {total_experiments}")
    
    # 后续轮次：贝叶斯优化
    for round_num in range(2, n_rounds + 1):
        # 检查是否已经找到目标，如果找到则终止
        if target_yield_record is not None:
            print(f"\n✅ 已找到目标yield={target_yield:.2f}，终止优化")
            break
            
        print(f"\n=== 第{round_num}轮：贝叶斯优化 ===")
        
        # 调用update接口
        update_result = call_update_api(parameter_space, objectives, all_experiments, batch_size=update_batch, seed=seed)
        
        if not update_result:
            print(f"❌ 第{round_num}轮更新失败，退出优化循环")
            break
        
        # 模拟实验结果
        round_results = simulate_experiment_results(update_result['results'], df, random_seed=seed)
        
        # 逐个处理结果
        found_target = False
        for result in round_results:
            all_experiments.append(result)
            total_experiments += 1
            
            hypervolume_val = result['metrics']['point_hypervolume']
            best_hypervolume = max(best_hypervolume, hypervolume_val)
            
            # 检查是否找到目标yield值
            yield_val = result['metrics'].get('yield')
            if yield_val is not None and abs(yield_val - target_yield) < 0.01 and target_yield_record is None:
                target_yield_record = total_experiments
                print(f"🎯 找到目标yield={target_yield:.2f}，当前实验次数: {total_experiments}")
                found_target = True
                break
        
        # 记录历史
        optimization_history.append({
            'round': round_num,
            'type': 'update',
            'parameters': update_result['results'],
            'results': round_results,
            'best_hypervolume': best_hypervolume,
            'total_experiments': total_experiments
        })
        
        print(f"📈 第{round_num}轮最佳结果: point_hypervolume={best_hypervolume:.6f}")
        print(f"📊 累计实验次数: {total_experiments}")
        
        # 如果找到目标，立即终止
        if found_target:
            print(f"\n✅ 已找到目标yield={target_yield:.2f}，终止优化")
            break
    
    return all_experiments, optimization_history, target_yield_record


def run_batch_test(parameter_space, objectives, df, init_batch=10, seeds=[42, 123, 456, 789, 999], 
                    batch_sizes=[1, 3, 5, 8, 10], target_yield=99.81):
    """
    批量测试函数
    使用不同的随机种子和batch_size测试，统计找到yield=99.81所需的实验次数
    
    参数:
    - init_batch: 初始化采样数量（固定为10）
    - seeds: 随机种子列表
    - batch_sizes: 每轮贝叶斯推荐的参数数量（1, 3, 5, 8, 10）
    - target_yield: 目标yield值（默认99.81）
    
    返回:
    - results_df: 结果DataFrame表格
    """
    
    results = []
    total_tests = len(seeds) * len(batch_sizes)
    current_test = 0
    
    print(f"🚀 开始批量测试")
    print(f"📊 测试配置:")
    print(f"   - 初始化采样: {init_batch}")
    print(f"   - 随机种子数: {len(seeds)}")
    print(f"   - Batch大小: {batch_sizes}")
    print(f"   - 目标yield值: {target_yield}")
    print(f"   - 总测试数: {total_tests}\n")
    
    for seed in seeds:
        for batch_size in batch_sizes:
            current_test += 1
            print(f"\n{'='*60}")
            print(f"测试 {current_test}/{total_tests}: Seed={seed}, Batch Size={batch_size}")
            print(f"{'='*60}")
            
            try:
                # 运行优化循环（打印所有详细信息）
                _, _, target_yield_record = run_optimization_loop(
                    parameter_space, objectives, df, 
                    n_rounds=400,  # 设置足够大的轮次
                    init_batch=init_batch,
                    update_batch=batch_size,
                    seed=seed,
                    target_yield=target_yield
                )
                
                # 记录结果
                result_row = {
                    'seed': seed,
                    'batch_size': batch_size,
                    f'yield_{target_yield}': target_yield_record
                }
                results.append(result_row)
                
                print(f"✅ 测试完成:")
                print(f"   - yield={target_yield}: {target_yield_record}")
                
            except Exception as e:
                print(f"❌ 测试失败: {str(e)}")
                result_row = {
                    'seed': seed,
                    'batch_size': batch_size,
                    f'yield_{target_yield}': None
                }
                results.append(result_row)
    
    # 生成结果表格
    results_df = pd.DataFrame(results)
    
    # 重新排列列的顺序，使其更易读
    col_order = ['seed', 'batch_size', f'yield_{target_yield}']
    results_df = results_df[col_order]
    
    # 按seed和batch_size排序
    results_df = results_df.sort_values(['seed', 'batch_size'])
    
    return results_df


if __name__ == "__main__":
    # 读取实验数据
    data_file = "point_hypervolume.csv"
    df = pd.read_csv(data_file)

    # 分析参数列和目标列
    parameter_columns = ['base', 'ligand', 'solvent', 'concentration', 'temperature']
    objective_columns = ['point_hypervolume']

    # 定义优化目标（单目标优化：最大化point_hypervolume）
    objectives = {
        "point_hypervolume": {"minimize": False}  # 最大化point_hypervolume
    }

    print("🎯 优化目标:")
    for obj, config in objectives.items():
        direction = "最小化" if config["minimize"] else "最大化"
        print(f"  {obj}: {direction}")

    parameter_space = build_parameter_space(df, parameter_columns)
    
    # 运行完整批量测试
    print("\n" + "="*60)
    print("开始完整批量测试")
    print("="*60)
    
    # 完整测试：10种随机种子 × 5种batch_size = 50次测试
    target_yield = 99.81
    results_df = run_batch_test(
        parameter_space=parameter_space,
        objectives=objectives,
        df=df,
        init_batch=10,  # 固定初始化采样为10个
        seeds=[42, 123, 456, 789, 999, 1111, 2222, 3333, 4444, 5555],  # 10种不同随机种子
        batch_sizes=[1, 3, 5, 8, 10],  # 每轮贝叶斯推荐1, 3, 5, 8, 10个点
        target_yield=target_yield  # 目标yield值
    )
    
    # 显示结果表格
    print("\n" + "="*60)
    print("测试结果表格")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # 保存结果到CSV文件
    output_file = "test_results.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存到: {output_file}")
    
    # 计算统计信息
    print("\n" + "="*60)
    print("统计信息")
    print("="*60)
    col_name = f'yield_{target_yield}'
    valid_values = results_df[col_name].dropna()
    if len(valid_values) > 0:
        print(f"\n目标 yield={target_yield}:")
        print(f"  平均实验次数: {valid_values.mean():.2f}")
        print(f"  最小实验次数: {valid_values.min():.0f}")
        print(f"  最大实验次数: {valid_values.max():.0f}")
        print(f"  标准差: {valid_values.std():.2f}")
        print(f"  成功找到次数: {len(valid_values)}/{len(results_df)}")
    else:
        print(f"\n目标 yield={target_yield}: 未找到")
