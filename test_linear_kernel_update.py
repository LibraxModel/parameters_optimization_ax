#!/usr/bin/env python3
"""
测试使用LinearKernel的update接口
"""

import requests
import json
import pandas as pd
import time

def test_linear_kernel_update():
    """测试使用LinearKernel的update接口"""
    print("=== 测试LinearKernel + Update接口 ===")
    
    # 1. 准备参数空间
    parameter_space = [
        {
            "name": "temperature",
            "type": "range",
            "values": [80.0, 120.0]
        },
        {
            "name": "pressure",
            "type": "range", 
            "values": [2.0, 4.0]
        },
        {
            "name": "concentration",
            "type": "range",
            "values": [0.2, 0.8]
        }
    ]
    
    # 2. 准备先验实验数据
    prior_experiments = [
        {
            "parameters": {"temperature": 85.0, "pressure": 2.5, "concentration": 0.3},
            "metrics": {"yield": 72.5, "purity": 90.2}
        },
        {
            "parameters": {"temperature": 95.0, "pressure": 3.0, "concentration": 0.4},
            "metrics": {"yield": 78.3, "purity": 92.1}
        },
        {
            "parameters": {"temperature": 105.0, "pressure": 3.5, "concentration": 0.5},
            "metrics": {"yield": 82.1, "purity": 94.5}
        },
        {
            "parameters": {"temperature": 110.0, "pressure": 2.8, "concentration": 0.6},
            "metrics": {"yield": 79.8, "purity": 91.7}
        },
        {
            "parameters": {"temperature": 90.0, "pressure": 3.2, "concentration": 0.35},
            "metrics": {"yield": 75.6, "purity": 89.8}
        }
    ]
    
    # 3. 准备目标配置 - 多目标优化
    objectives = {
        "yield": {"minimize": False},     # 最大化产率
        "purity": {"minimize": False}     # 最大化纯度
    }
    
    # 4. 测试不同的variance参数值
    variance_values = [0.5, 1.0, 2.0, 5.0]
    
    for variance in variance_values:
        print(f"\n--- 测试 variance={variance} ---")
        
        # 准备update请求
        update_request = {
            "parameter_space": parameter_space,
            "objectives": objectives,
            "completed_experiments": prior_experiments,
            "batch": 3,  # 生成3个新的参数组合
            "use_weights": True,
            "objective_weights": {"yield": 0.6, "purity": 0.4},  # yield权重更高
            "seed": 42,
            
            # 自定义模型配置
            "surrogate_model_class": "SingleTaskGP",
            "kernel_class": "LinearKernel",
            "kernel_options": {"variance": variance}
        }
        
        try:
            # 发送update请求
            print(f"📤 发送update请求 (LinearKernel, variance={variance})...")
            response = requests.post(
                'http://localhost:3320/update',
                json=update_request,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ 请求成功!")
                
                # 解析结果
                success = result.get('success', False)
                message = result.get('message', 'N/A')
                next_parameters = result.get('next_parameters', [])
                
                print(f"成功: {success}")
                print(f"消息: {message}")
                print(f"生成的参数组合数量: {len(next_parameters)}")
                
                # 显示生成的参数
                if next_parameters:
                    print(f"生成的参数组合:")
                    for i, params in enumerate(next_parameters, 1):
                        print(f"  {i}. {params}")
                        
                        # 验证参数是否在合理范围内
                        temp = params.get('temperature', 0)
                        pressure = params.get('pressure', 0)
                        conc = params.get('concentration', 0)
                        
                        temp_valid = 80.0 <= temp <= 120.0
                        pressure_valid = 2.0 <= pressure <= 4.0
                        conc_valid = 0.2 <= conc <= 0.8
                        
                        if temp_valid and pressure_valid and conc_valid:
                            print(f"     ✓ 参数在有效范围内")
                        else:
                            print(f"     ✗ 参数超出范围!")
                            print(f"       Temperature: {temp} ({'✓' if temp_valid else '✗'})")
                            print(f"       Pressure: {pressure} ({'✓' if pressure_valid else '✗'})")
                            print(f"       Concentration: {conc} ({'✓' if conc_valid else '✗'})")
                
                # 检查是否使用了LinearKernel
                if "LinearKernel" in message:
                    print(f"     ✓ 确认使用了LinearKernel")
                else:
                    print(f"     ⚠️ 消息中未提及LinearKernel")
                    
            else:
                print(f"✗ 请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("⚠️  无法连接到API服务器")
            return False
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            return False
    
    return True

def test_compare_kernels():
    """比较不同核函数的效果"""
    print(f"\n=== 比较LinearKernel vs RBFKernel ===")
    
    # 基础配置
    parameter_space = [
        {
            "name": "x1",
            "type": "range",
            "values": [0.0, 10.0]
        },
        {
            "name": "x2",
            "type": "range",
            "values": [0.0, 10.0]
        }
    ]
    
    prior_experiments = [
        {"parameters": {"x1": 1.0, "x2": 1.0}, "metrics": {"y": 2.0}},  # y ≈ x1 + x2
        {"parameters": {"x1": 2.0, "x2": 3.0}, "metrics": {"y": 5.1}},
        {"parameters": {"x1": 4.0, "x2": 2.0}, "metrics": {"y": 5.9}},
        {"parameters": {"x1": 3.0, "x2": 4.0}, "metrics": {"y": 7.2}},
    ]
    
    objectives = {"y": {"minimize": False}}
    
    kernels_to_test = [
        {"name": "LinearKernel", "options": {"variance": 1.0}},
        {"name": "RBFKernel", "options": {"lengthscale": 1.0}},
    ]
    
    results = {}
    
    for kernel_config in kernels_to_test:
        kernel_name = kernel_config["name"]
        kernel_options = kernel_config["options"]
        
        print(f"\n--- 测试 {kernel_name} ---")
        
        update_request = {
            "parameter_space": parameter_space,
            "objectives": objectives,
            "completed_experiments": prior_experiments,
            "batch": 2,
            "seed": 42,
            "surrogate_model_class": "SingleTaskGP",
            "kernel_class": kernel_name,
            "kernel_options": kernel_options
        }
        
        try:
            response = requests.post(
                'http://localhost:3320/update',
                json=update_request,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                next_parameters = result.get('next_parameters', [])
                results[kernel_name] = next_parameters
                
                print(f"✓ {kernel_name} 生成参数:")
                for i, params in enumerate(next_parameters, 1):
                    print(f"  {i}. x1={params['x1']:.3f}, x2={params['x2']:.3f}")
            else:
                print(f"✗ {kernel_name} 失败: {response.status_code}")
                
        except Exception as e:
            print(f"✗ {kernel_name} 异常: {e}")
    
    return results

def main():
    print("开始LinearKernel测试...")
    
    # 检查API服务器
    try:
        response = requests.get('http://localhost:3320/', timeout=5)
        print("✓ API服务器正在运行")
    except:
        print("⚠️  API服务器未运行，请先启动:")
        print("conda activate ax_env && python api_parameter_optimizer_v3.py")
        return
    
    # 测试LinearKernel的不同variance值
    success1 = test_linear_kernel_update()
    
    # 比较不同核函数
    if success1:
        comparison_results = test_compare_kernels()
    
    print(f"\n=== 测试总结 ===")
    print(f"LinearKernel variance参数测试: {'✓ 成功' if success1 else '✗ 失败'}")
    
    print(f"\n💡 LinearKernel使用说明:")
    print(f"  - variance参数控制线性核的方差")
    print(f"  - 较小的variance (0.5-1.0): 更保守的探索")
    print(f"  - 较大的variance (2.0-5.0): 更激进的探索")
    print(f"  - 适用于线性或近似线性的优化问题")
    print(f"  - 示例配置: kernel_options={{'variance': 1.0}}")

if __name__ == "__main__":
    main()
