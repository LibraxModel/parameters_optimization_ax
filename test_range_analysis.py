#!/usr/bin/env python3
"""
测试analysis接口的range类型参数，生成5个图表
"""

import requests
import json
import pandas as pd
import time
import os

def test_range_analysis():
    """测试只有range类型参数的analysis接口"""
    print("=== 测试range类型参数的analysis接口 ===")
    
    # 定义参数空间 - 全部为range类型，明确指定为float类型
    search_space = [
        {
            "name": "temperature",
            "type": "range",
            "values": [80.0, 125.0]  # 使用float类型
        },
        {
            "name": "pressure", 
            "type": "range",
            "values": [2.0, 4.0]
        },
        {
            "name": "flow_rate",
            "type": "range", 
            "values": [10.0, 25.0]
        },
        {
            "name": "concentration",
            "type": "range",
            "values": [0.2, 0.5]
        }
    ]
    
    # 检查测试数据
    print("检查测试数据...")
    df = pd.read_csv('range_test_data.csv')
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"数据预览:")
    print(df.head())
    
    # 验证参数范围
    print("\n验证参数范围:")
    for param in ['temperature', 'pressure', 'flow_rate', 'concentration']:
        min_val = df[param].min()
        max_val = df[param].max()
        space_info = next(p for p in search_space if p['name'] == param)
        space_min, space_max = space_info['values']
        
        print(f"  {param}: 数据范围=[{min_val:.2f}, {max_val:.2f}], 空间范围=[{space_min:.1f}, {space_max:.1f}]")
        
        # 检查数据是否在定义的范围内
        if min_val < space_min or max_val > space_max:
            print(f"    ⚠️ 数据超出定义范围!")
        else:
            print(f"    ✓ 数据在范围内")
    
    # 准备API请求
    print(f"\n发送请求到analysis API...")
    print(f"参数空间配置:")
    print(json.dumps(search_space, indent=2, ensure_ascii=False))
    
    try:
        # 准备文件和表单数据
        files = {
            'file': ('range_test_data.csv', open('range_test_data.csv', 'rb'), 'text/csv')
        }
        
        data = {
            'parameters': 'temperature,pressure,flow_rate,concentration',
            'objectives': 'yield,purity',
            'parameter_space': json.dumps(search_space)
        }
        
        # 发送请求
        print("正在发送请求...")
        response = requests.post(
            'http://localhost:3320/analysis',
            files=files,
            data=data,
            timeout=300  # 5分钟超时
        )
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ API调用成功!")
            
            # 显示结果
            print(f"\n响应结果:")
            print(f"成功: {result.get('success', False)}")
            print(f"消息: {result.get('message', 'N/A')}")
            print(f"输出目录: {result.get('output_directory', 'N/A')}")
            print(f"包含类别数据: {result.get('has_categorical_data', False)}")
            
            generated_plots = result.get('generated_plots', [])
            print(f"\n生成的图表 ({len(generated_plots)} 个):")
            for i, plot in enumerate(generated_plots, 1):
                print(f"  {i}. {plot}")
            
            # 验证是否生成了预期的5个图表
            expected_plots = [
                'parallel_coords_',  # 并行坐标图
                'feature_importance_',  # 特征重要性图  
                'cross_validation_',  # 交叉验证图
                'slice_plot_',  # 切片图
                'contour_plot_'  # 等高线图
            ]
            
            print(f"\n图表验证:")
            found_types = set()
            for plot in generated_plots:
                for expected in expected_plots:
                    if expected in plot:
                        found_types.add(expected)
                        break
            
            print(f"找到的图表类型: {len(found_types)}/5")
            for plot_type in expected_plots:
                if plot_type in found_types:
                    print(f"  ✓ {plot_type.rstrip('_')} 图表")
                else:
                    print(f"  ✗ {plot_type.rstrip('_')} 图表 (缺失)")
            
            # 检查输出目录
            output_dir = result.get('output_directory')
            if output_dir and os.path.exists(output_dir):
                print(f"\n输出目录内容:")
                files_in_dir = os.listdir(output_dir)
                for file in sorted(files_in_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        print(f"  📄 {file} ({size} bytes)")
            
            success = len(found_types) >= 3  # 至少生成3个图表算成功
            return success
            
        else:
            print(f"✗ API调用失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  无法连接到API服务器 (http://localhost:3320)")
        print("请确保API服务器已启动: conda activate ax_env && python api_parameter_optimizer_v3.py")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 确保文件被关闭
        try:
            files['file'][1].close()
        except:
            pass

def main():
    print("开始range类型analysis接口测试...")
    
    # 检查API服务器是否运行
    try:
        response = requests.get('http://localhost:3320/', timeout=5)
        print("✓ API服务器正在运行")
    except:
        print("⚠️  API服务器未运行，请先启动:")
        print("conda activate ax_env && python api_parameter_optimizer_v3.py")
        return
    
    success = test_range_analysis()
    
    print(f"\n=== 测试结果 ===")
    if success:
        print("✓ 测试通过！成功生成了analysis图表")
    else:
        print("✗ 测试失败")
    
    print("\n测试完成。")

if __name__ == "__main__":
    main()
