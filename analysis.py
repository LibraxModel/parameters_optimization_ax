

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Plotly相关导入
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# SHAP相关导入
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Ax相关导入（用于sliceplot）
from ax_optimizer import BayesianOptimizer, ExperimentResult

# 新增：支持自定义代理模型配置的导入
from typing import Type
from botorch.models import SingleTaskGP, MultiTaskGP
from gpytorch.kernels import MaternKernel, RBFKernel

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ParameterOptimizationAnalysis:
    """
    参数优化分析器
    
    提供基于Ax框架的完整分析功能，包括数据可视化、敏感性分析、优化进度跟踪等
    """
    
    def __init__(
        self,
        experiment_data: Optional[pd.DataFrame] = None,
        experiment_file: Optional[str] = None,
        output_dir: str = "analysis_output"
    ):
        """
        初始化分析器
        
        Args:
            experiment_data: 实验数据DataFrame
            experiment_file: 实验数据文件路径
            output_dir: 输出目录
        """
        self.experiment_data = experiment_data
        self.experiment_file = experiment_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 如果提供了文件路径，加载数据
        if experiment_file and experiment_data is None:
            self.load_experiment_data(experiment_file)
        
        # 初始化分析结果存储
        self.analysis_results = {}
        self.plots = {}
        
        # 添加Ax优化器缓存，避免重复重建
        self._ax_optimizer_cache = {}
        
    def load_experiment_data(self, file_path: str) -> pd.DataFrame:
        """
        加载实验数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            加载的数据DataFrame
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            self.experiment_data = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            self.experiment_data = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.json':
            self.experiment_data = pd.read_json(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        print(f"📊 成功加载数据: {len(self.experiment_data)} 行, {len(self.experiment_data.columns)} 列")
        return self.experiment_data
    
    def get_parameter_types(self) -> Dict[str, str]:
        """
        获取参数类型信息
        
        Returns:
            参数类型字典
        """
        if self.experiment_data is None:
            return {}
        
        param_types = {}
        for col in self.experiment_data.columns:
            if self.experiment_data[col].dtype == 'object':
                param_types[col] = 'categorical'
            else:
                param_types[col] = 'numerical'
        
        return param_types
    
    def create_parallel_coordinates_plots(
        self,
        parameters: List[str],
        objectives: List[str],
        color_by: Optional[str] = None,
        title_prefix: str = "参数优化并行坐标图"
    ) -> Dict[str, go.Figure]:
        """
        创建并行坐标图
        
        Args:
            parameters: 要展示的参数列表
            objectives: 要展示的目标指标列表
            color_by: 用于着色的指标名称（可选）
            title_prefix: 标题前缀
            
        Returns:
            并行坐标图字典
        """
        if self.experiment_data is None:
            raise ValueError("没有实验数据")
        
        # 检查参数和目标是否存在
        all_columns = parameters + objectives
        missing_columns = [col for col in all_columns if col not in self.experiment_data.columns]
        if missing_columns:
            raise ValueError(f"以下列不存在于数据中: {missing_columns}")
        
        # 获取参数类型信息
        param_types = self.get_parameter_types()
        
        # 确定着色方式
        if color_by is None and objectives:
            color_by = objectives[0]
        elif color_by not in objectives:
            color_by = objectives[0]
        
        plots = {}
        
        # 创建包含所有目标的统一并行坐标图
        print(f"📊 生成包含所有目标的并行坐标图...")
        
        # 准备数据，包含所有参数和目标
        plot_data = self.experiment_data[parameters + objectives].copy()
        plot_data['trial_index'] = range(len(plot_data))
        
        # 创建维度列表
        dimensions = [
            # Trial Index
            dict(
                range=[0, len(plot_data) - 1],
                label='Trial Index',
                values=plot_data['trial_index'],
                ticktext=[f"Trial {i+1}" for i in range(len(plot_data))],
                tickvals=list(range(len(plot_data)))
            ),
            # Parameters (handle categorical variables)
            *[
                self._create_dimension_for_parameter(
                    param, plot_data[param], param_types.get(param, 'numerical')
                )
                for param in parameters
            ],
            # All Objectives
            *[
                dict(
                    range=[plot_data[objective].min(), plot_data[objective].max()],
                    label=f"{objective}",
                    values=plot_data[objective],
                    tickformat='.2f'
                )
                for objective in objectives
            ]
        ]
        
        # 创建并行坐标图
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=plot_data[color_by],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title=f"{color_by} (Color Scale)",
                        x=1.1,
                        len=0.8
                    )
                ),
                dimensions=dimensions,
                unselected=dict(line=dict(color='lightgray', opacity=0.3))
            )
        )
        
        # Update layout
        obj_names = ', '.join(objectives)
        fig.update_layout(
            title=dict(
                text=f"Parallel Coordinates Plot - Parameters & Objectives ({obj_names})",
                x=0.5,
                font=dict(size=16)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=11),
            # height=700,
            # width=max(1200, 150 * (len(parameters) + len(objectives) + 1)),  # 动态调整宽度
            autosize=True,
            margin=dict(l=50, r=150, t=80, b=50)
        )
        # 不显示图表到控制台，避免日志中输出大量HTML
        # fig.show(config={'responsive': True})
        plots["parallel_coords_combined"] = fig
        
        self.plots.update(plots)
        return plots
    
    def _create_dimension_for_parameter(
        self,
        param_name: str,
        param_values: pd.Series,
        param_type: str
    ) -> Dict[str, Any]:
        """
        为参数创建维度配置
        
        Args:
            param_name: 参数名称
            param_values: 参数值
            param_type: 参数类型
            
        Returns:
            维度配置字典
        """
        if param_type == 'categorical':
            # 类别变量处理
            unique_values = param_values.unique()
            value_to_index = {val: idx for idx, val in enumerate(unique_values)}
            numeric_values = [value_to_index[val] for val in param_values]
            
            return dict(
                range=[0, len(unique_values) - 1],
                label=f"{param_name} (Categorical)",
                values=numeric_values,
                ticktext=list(unique_values),
                tickvals=list(range(len(unique_values)))
            )
        else:
            # 数值变量处理
            return dict(
                range=[param_values.min(), param_values.max()],
                label=f"{param_name} (Numerical)",
                values=param_values,
                tickformat='.2f'
            )
    
    def generate_parallel_coordinates_analysis(
        self,
        parameters: List[str],
        objectives: List[str],
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        生成并行坐标图分析
        
        Args:
            parameters: 参数列表
            objectives: 目标指标列表
            save_results: 是否保存结果
            
        Returns:
            分析结果字典
        """
        print("🔍 开始并行坐标图分析...")
        
        # 生成并行坐标图
        print("📊 生成并行坐标图...")
        plots = self.create_parallel_coordinates_plots(
            parameters=parameters,
            objectives=objectives
        )
        
        # 保存结果
        if save_results:
            self.save_plots('html')
        
        print("✅ 并行坐标图分析完成!")
        print(f"📁 结果保存在: {self.output_dir}")
        
        return {
            'plots': plots,
            'total_plots': len(plots)
        }
    
    def create_feature_importance_analysis(
        self,
        parameters: List[str],
        objectives: List[str],
        model_type: str = 'random_forest',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        创建特征重要性分析（基于SHAP值）
        
        Args:
            parameters: 特征参数列表
            objectives: 目标变量列表
            model_type: 模型类型 ('random_forest', 'xgboost', 'lightgbm')
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            特征重要性分析结果
        """
        if self.experiment_data is None:
            raise ValueError("没有实验数据")
        
        print("🔍 开始特征重要性分析...")
        
        # 检查参数和目标是否存在
        all_columns = parameters + objectives
        missing_columns = [col for col in all_columns if col not in self.experiment_data.columns]
        if missing_columns:
            raise ValueError(f"以下列不存在于数据中: {missing_columns}")
        
        # 获取参数类型信息
        param_types = self.get_parameter_types()
        
        # 准备数据
        X = self.experiment_data[parameters].copy()
        y_dict = {obj: self.experiment_data[obj] for obj in objectives}
        
        # 处理类别变量
        label_encoders = {}
        for param in parameters:
            if param_types.get(param) == 'categorical':
                le = LabelEncoder()
                X[param] = le.fit_transform(X[param].astype(str))
                label_encoders[param] = le
        
        results = {}
        
        # 为每个目标创建特征重要性分析
        for objective in objectives:
            print(f"📊 为目标 '{objective}' 生成特征重要性分析...")
            
            y = y_dict[objective]
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # 训练模型
            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            model.fit(X_train, y_train)
            
            # 计算SHAP值
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # 创建特征重要性图
            feature_importance_fig = self._create_feature_importance_plot(
                X=X_test,
                y=y_test,
                shap_values=shap_values,
                feature_names=parameters,
                target_name=objective
            )
            
            # 保存到plots字典
            self.plots[f'feature_importance_{objective}'] = feature_importance_fig
            
            results[objective] = {
                'model': model,
                'explainer': explainer,
                'shap_values': shap_values,
                'feature_importance_plot': feature_importance_fig,
                'label_encoders': label_encoders
            }
        
        print("✅ 特征重要性分析完成!")
        return results
    
    def _create_feature_importance_plot(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        shap_values: np.ndarray,
        feature_names: List[str],
        target_name: str
    ) -> go.Figure:
        """
        创建特征重要性条形图
        """
        # 计算平均SHAP值
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # 创建条形图
        fig = go.Figure(data=[
            go.Bar(
                x=mean_shap_values,
                y=feature_names,
                orientation='h',
                marker_color='#1f77b4',  # 使用更好看的蓝色
                hovertemplate='<b>%{y}</b><br>' +
                            'Average SHAP Impact: %{x:.3f}<br>' +
                            '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=f"Feature Importance Analysis for {target_name}",
                x=0.5,  # 标题居中
                font=dict(size=16)
            ),
            xaxis_title=f"Average SHAP Impact on {target_name}",
            yaxis_title="Features",
            plot_bgcolor='white',
            paper_bgcolor='white',
            # height=500,
            # width=800,
            autosize=True,
            margin=dict(l=50, r=50, t=80, b=50),
            # 图表整体居中
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False
            )
        )
        # 不显示图表到控制台，避免日志中输出大量HTML
        # fig.show(config={'responsive': True})
        
        return fig
    
    def create_slice_plots(
        self,
        parameters: List[str],
        objectives: List[str],
        search_space: List[Dict[str, Any]],
        n_points: int = 100,
        confidence_level: float = 0.95,
        # 新增：自定义代理模型配置参数
        surrogate_model_class: Optional[Type] = None,
        kernel_class: Optional[Type] = None,
        kernel_options: Optional[Dict[str, Any]] = None,
        # 新增：指定要生成的参数列表（如果为None，则生成所有range参数的图表）
        target_parameters: Optional[List[str]] = None,
        # 新增：指定要生成的目标列表（如果为None，则生成所有目标的图表）
        target_objectives: Optional[List[str]] = None
    ) -> Dict[str, go.Figure]:
        """
        创建slice图，展示单一参数对目标的影响及置信区间
        基于Ax的SlicePlot实现，当且仅当所有参数都是range类型时生成
        
        Args:
            parameters: 参数列表（用于构建优化器，应包含所有参数）
            objectives: 目标指标列表
            search_space: 搜索空间配置（必须提供）
            n_points: 预测点的数量
            confidence_level: 置信区间水平
            target_parameters: 指定要生成的参数列表（如果为None，则生成所有range参数的图表）
            target_objectives: 指定要生成的目标列表（如果为None，则生成所有目标的图表）
            
        Returns:
            切片图字典
        """
        if self.experiment_data is None:
            raise ValueError("没有实验数据")
        
        # 检查参数和目标是否存在
        all_columns = parameters + objectives
        missing_columns = [col for col in all_columns if col not in self.experiment_data.columns]
        if missing_columns:
            raise ValueError(f"以下列不存在于数据中: {missing_columns}")
        
        # 获取参数类型信息
        param_types = self.get_parameter_types()
        
        # 使用convert_parameter_space_to_ax_format_for_analysis函数分析参数类型
        # 这里我们不需要全局检查所有参数，只需要检查用户指定的参数
        print("🔍 分析参数类型...")
        
        # 重建Ax优化器和代理模型
        print("🔧 重建Ax优化器和代理模型...")
        ax_optimizer = self._rebuild_ax_optimizer(
            parameters=parameters,
            objectives=objectives,
            search_space=search_space,
            # 传递自定义代理模型配置
            surrogate_model_class=surrogate_model_class,
            kernel_class=kernel_class,
            kernel_options=kernel_options
        )
        
        plots = {}
        
        # 确定要生成的参数列表
        if target_parameters is not None:
            # 只生成用户指定的参数图表
            # 首先检查用户指定的参数是否在数据中存在
            valid_target_params = [param for param in target_parameters if param in self.experiment_data.columns]
            if not valid_target_params:
                print(f"❌ 指定的参数 {target_parameters} 在数据中不存在")
                return {}
            
            # 使用参数空间配置来判断参数类型
            params_to_generate = []
            for param in valid_target_params:
                # 在search_space中查找参数配置
                param_config = None
                for config in search_space:
                    if config["name"] == param:
                        param_config = config
                        break
                
                if param_config is None:
                    print(f"⚠️ 参数 '{param}' 在参数空间配置中未找到，跳过")
                    continue
                
                # 根据参数空间配置判断类型
                if param_config["type"] == "range":
                    params_to_generate.append(param)
                    print(f"✅ 参数 '{param}' 是range类型，可以生成slice图")
                elif param_config["type"] == "choice":
                    print(f"⚠️ 参数 '{param}' 是choice类型，无法生成slice图")
                else:
                    print(f"⚠️ 参数 '{param}' 类型未知: {param_config['type']}，无法生成slice图")
            
            if not params_to_generate:
                print(f"❌ 指定的参数 {target_parameters} 中没有适合生成slice图的range类型参数")
                return {}
        else:
            # 生成所有range参数的图表（原有行为）
            # 从search_space中筛选range类型参数
            params_to_generate = []
            for config in search_space:
                if config["type"] == "range":
                    params_to_generate.append(config["name"])
            
            if not params_to_generate:
                print("❌ 参数空间中没有range类型的参数，无法生成slice图")
                return {}
        
        # 确定要生成的目标列表
        if target_objectives is not None:
            # 只生成用户指定的目标图表
            objectives_to_generate = [obj for obj in target_objectives if obj in objectives]
            if not objectives_to_generate:
                print(f"❌ 指定的目标 {target_objectives} 中没有有效目标")
                return {}
        else:
            # 生成所有目标的图表（原有行为）
            objectives_to_generate = objectives
        
        # 为每个目标创建slice图
        for objective in objectives_to_generate:
            print(f"📊 为目标 '{objective}' 生成slice图...")
            
            # 只为指定的range参数创建slice图
            for param in params_to_generate:
                print(f"  📈 生成参数 '{param}' 的slice图...")
                
                slice_fig = self._create_single_slice_plot_with_ax(
                    ax_optimizer=ax_optimizer,
                    param=param,
                    objective=objective,
                    param_types=param_types,
                    n_points=n_points,
                    confidence_level=confidence_level
                )
                
                # 使用JSON数组格式命名：slice_["目标","参数"]
                plot_key = f'slice_["{objective}","{param}"]'
                plots[plot_key] = slice_fig
                self.plots[plot_key] = slice_fig
                
                # 立即保存当前生成的切片图
                saved_path = self.save_single_plot(plot_key, slice_fig)
                
        
        return plots
    
    def create_contour_plots(
        self,
        parameters: List[str],
        objectives: List[str],
        search_space: List[Dict[str, Any]],
        n_points: int = 50,  
        confidence_level: float = 0.95,
        max_contour_pairs: int = None,  # 不限制等高线图数量
        # 新增：自定义代理模型配置参数
        surrogate_model_class: Optional[Type] = None,
        kernel_class: Optional[Type] = None,
        kernel_options: Optional[Dict[str, Any]] = None,
        # 新增：指定要生成的参数对列表（如果为None，则生成所有参数对的图表）
        target_parameter_pairs: Optional[List[Tuple[str, str]]] = None,
        # 新增：指定要生成的目标列表（如果为None，则生成所有目标的图表）
        target_objectives: Optional[List[str]] = None
    ) -> Dict[str, go.Figure]:
        """
        创建等高线图，展示两个参数对目标的影响
        基于Ax的ContourPlot实现，当且仅当所有参数都是range类型时生成
        
        Args:
            parameters: 参数列表（用于构建优化器，应包含所有参数）
            objectives: 目标指标列表
            search_space: 搜索空间配置（必须提供）
            n_points: 网格密度（n_points x n_points）
            confidence_level: 置信区间水平
            target_parameter_pairs: 指定要生成的参数对列表（如果为None，则生成所有参数对的图表）
            target_objectives: 指定要生成的目标列表（如果为None，则生成所有目标的图表）
            
        Returns:
            等高线图字典
        """
        if self.experiment_data is None:
            raise ValueError("没有实验数据")
        
        # 检查参数和目标是否存在
        all_columns = parameters + objectives
        missing_columns = [col for col in all_columns if col not in self.experiment_data.columns]
        if missing_columns:
            raise ValueError(f"以下列不存在于数据中: {missing_columns}")
        
        # 检查参数数量
        if len(parameters) < 2:
            print("❌ 等高线图需要至少2个参数")
            return {}
        
        # 获取参数类型信息
        param_types = self.get_parameter_types()
        
        # 使用参数空间配置来分析参数类型
        print("🔍 分析参数类型...")
        
        # 从search_space中筛选range类型参数
        range_params = []
        for config in search_space:
            if config["type"] == "range":
                range_params.append(config["name"])
        
        print(f"📊 参数空间中的range类型参数: {range_params}")
        
        # 重建Ax优化器和代理模型
        print("🔧 重建Ax优化器和代理模型...")
        ax_optimizer = self._rebuild_ax_optimizer(
            parameters=parameters,
            objectives=objectives,
            search_space=search_space,
            # 传递自定义代理模型配置
            surrogate_model_class=surrogate_model_class,
            kernel_class=kernel_class,
            kernel_options=kernel_options
        )
        
        plots = {}
        
        # 确定要生成的目标列表
        if target_objectives is not None:
            # 只生成用户指定的目标图表
            objectives_to_generate = [obj for obj in target_objectives if obj in objectives]
            if not objectives_to_generate:
                print(f"❌ 指定的目标 {target_objectives} 中没有有效目标")
                return {}
        else:
            # 生成所有目标的图表（原有行为）
            objectives_to_generate = objectives
        
        # 为每个目标创建等高线图
        for objective in objectives_to_generate:
            print(f"📊 为目标 '{objective}' 生成等高线图...")
            
            # 确定要生成的参数对列表
            if target_parameter_pairs is not None:
                # 只生成用户指定的参数对图表
                param_pairs = []
                for param1, param2 in target_parameter_pairs:
                    # 检查参数是否在数据中存在
                    if param1 not in self.experiment_data.columns:
                        print(f"⚠️ 参数 '{param1}' 在数据中不存在，跳过参数对 ({param1}, {param2})")
                        continue
                    if param2 not in self.experiment_data.columns:
                        print(f"⚠️ 参数 '{param2}' 在数据中不存在，跳过参数对 ({param1}, {param2})")
                        continue
                    
                    # 使用参数空间配置检查参数类型
                    param1_config = None
                    param2_config = None
                    
                    for config in search_space:
                        if config["name"] == param1:
                            param1_config = config
                        if config["name"] == param2:
                            param2_config = config
                    
                    if param1_config is None:
                        print(f"⚠️ 参数 '{param1}' 在参数空间配置中未找到，跳过参数对 ({param1}, {param2})")
                        continue
                    if param2_config is None:
                        print(f"⚠️ 参数 '{param2}' 在参数空间配置中未找到，跳过参数对 ({param1}, {param2})")
                        continue
                    
                    # 检查参数是否为range类型
                    if param1_config["type"] == "range" and param2_config["type"] == "range":
                        param_pairs.append((param1, param2))
                        print(f"✅ 参数对 ({param1}, {param2}) 都是range类型，可以生成等高线图")
                    else:
                        print(f"⚠️ 参数对 ({param1}, {param2}) 中至少有一个不是range类型，跳过")
                        if param1_config["type"] != "range":
                            print(f"   参数 '{param1}' 是 {param1_config['type']} 类型")
                        if param2_config["type"] != "range":
                            print(f"   参数 '{param2}' 是 {param2_config['type']} 类型")
                
                if not param_pairs:
                    print(f"❌ 指定的参数对 {target_parameter_pairs} 中没有有效的range类型参数对")
                    return {}
            else:
                # 计算所有可能的参数对（原有行为）
                param_pairs = []
                for i in range(len(range_params)):
                    for j in range(i + 1, len(range_params)):
                        param_pairs.append((range_params[i], range_params[j]))
                
                # 限制参数对数量（如果设置了限制）
                if max_contour_pairs is not None and len(param_pairs) > max_contour_pairs:
                    print(f"⚠️  参数对数量 ({len(param_pairs)}) 超过限制 ({max_contour_pairs})，只生成前 {max_contour_pairs} 个")
                    param_pairs = param_pairs[:max_contour_pairs]
            
            # 为选定的参数对创建等高线图
            for idx, (param1, param2) in enumerate(param_pairs, 1):
                print(f"  📈 生成参数 '{param1}' vs '{param2}' 的等高线图... ({idx}/{len(param_pairs)})")
                
                try:
                    contour_fig = self._create_single_contour_plot_with_ax(
                        ax_optimizer=ax_optimizer,
                        param1=param1,
                        param2=param2,
                        objective=objective,
                        param_types=param_types,
                        n_points=n_points,
                        confidence_level=confidence_level
                    )
                    
                    # 使用JSON数组格式命名：contour_["目标","参数1","参数2"]
                    plot_key = f'contour_["{objective}","{param1}","{param2}"]'
                    plots[plot_key] = contour_fig
                    self.plots[plot_key] = contour_fig
                    print(f"    ✅ 完成: {plot_key}")
                    
                except Exception as e:
                    print(f"    ❌ 失败: {param1} vs {param2} - {str(e)}")
                    continue
                    
        
        return plots
    
    def _create_single_contour_plot_with_ax(
        self,
        ax_optimizer,
        param1: str,
        param2: str,
        objective: str,
        param_types: Dict[str, str],
        n_points: int,
        confidence_level: float
    ) -> go.Figure:
        """
        使用Ax优化器的代理模型创建两个参数的等高线图
        基于Ax的ContourPlot实现
        """
        try:
            # 获取Ax的实验数据
            trials_df = ax_optimizer.get_optimization_history()
            
            # 获取参数值范围
            param1_values = np.linspace(
                trials_df[param1].min(),
                trials_df[param1].max(),
                n_points
            )
            param2_values = np.linspace(
                trials_df[param2].min(),
                trials_df[param2].max(),
                n_points
            )
            
            # 创建网格
            param1_grid, param2_grid = np.meshgrid(param1_values, param2_values)
            
            # 使用中位数和众数方法，确保等高线图平滑
            # 排除目标变量（因变量），只包含其他输入参数
            other_params = [p for p in trials_df.columns if p in param_types.keys() and p not in [param1, param2, objective]]
            status_quo = {}
            for p in other_params:
                if pd.api.types.is_numeric_dtype(trials_df[p]):
                    # 使用中位数，对异常值更鲁棒
                    status_quo[p] = trials_df[p].median()
                else:
                    # 对于类别参数，使用众数
                    status_quo[p] = trials_df[p].mode().iloc[0] if not trials_df[p].mode().empty else trials_df[p].iloc[0]
            
            status_quo = pd.Series(status_quo)
            
            # 生成预测网格
            predictions_grid = np.zeros((n_points, n_points))
            
            for i in range(n_points):
                for j in range(n_points):
                    try:
                        # 创建预测数据点（固定其他参数为中位数/众数值）
                        X_pred = status_quo.copy()
                        X_pred[param1] = param1_grid[i, j]
                        X_pred[param2] = param2_grid[i, j]
                        
                        # 使用Ax的预测方法
                        predictions_dict = ax_optimizer.ax_client.get_model_predictions(
                            metric_names=[objective],
                            parameterizations={0: X_pred.to_dict()}
                        )
                        
                        if 0 in predictions_dict and objective in predictions_dict[0]:
                            pred_mean, _ = predictions_dict[0][objective]
                            predictions_grid[i, j] = pred_mean
                        else:
                            predictions_grid[i, j] = 0.0
                            
                    except Exception as e:
                        predictions_grid[i, j] = 0.0
            
            # 创建等高线图
            fig = go.Figure()
            
            # 添加等高线
            fig.add_trace(go.Contour(
                x=param1_values,
                y=param2_values,
                z=predictions_grid,
                colorscale='Viridis',
                contours=dict(
                    coloring='heatmap',
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                ),
                colorbar=dict(
                    title=dict(text=objective)
                ),
                name='Predicted Outcome',
                hovertemplate=self._create_contour_hover_template(param1, param2, objective, status_quo, other_params) +
                             f'<b>{param1}</b>: %{{x}}<br>' +
                             f'<b>{param2}</b>: %{{y}}<br>' +
                             f'<b>{objective} (Predicted)</b>: %{{z:.3f}}<br>' +
                             '<extra></extra>'
            ))
            
            # 为实际观测点创建详细的hover信息
            # 每个观测点都应该显示该点的真实参数值
            observation_hover_templates = []
            for idx, row in trials_df.iterrows():
                # 为每个观测点创建完整的hover信息
                hover_info = f'<b>{param1}</b>: {row[param1]:.3f}<br>'
                hover_info += f'<b>{param2}</b>: {row[param2]:.3f}<br>'
                hover_info += f'<b>{objective} (Observed)</b>: {row[objective]:.3f}<br>'
                hover_info += '<b>Other Parameters:</b><br>'
                
                # 添加该样本点的其他参数值（每个点都不同）
                for other_param in other_params:
                    if other_param in row:
                        value = row[other_param]
                        if pd.api.types.is_numeric_dtype(trials_df[other_param]):
                            hover_info += f'  {other_param}: {value:.3f}<br>'
                        else:
                            hover_info += f'  {other_param}: {value}<br>'
                
                hover_info += '<extra></extra>'
                observation_hover_templates.append(hover_info)
            
            # 添加实际观测点
            fig.add_trace(go.Scatter(
                x=trials_df[param1],
                y=trials_df[param2],
                mode='markers',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='circle'
                ),
                name='Actual Observations',
                hovertemplate='%{customdata}<extra></extra>',
                customdata=observation_hover_templates,
                text=trials_df[objective]
            ))
            
            # 更新布局
            fig.update_layout(
                title=dict(
                    text=f"{param1} vs {param2} vs {objective} (Contour Plot)",
                    x=0.5,
                    font=dict(size=16)
                ),
                xaxis_title=param1,
                yaxis_title=param2,
                plot_bgcolor='white',
                paper_bgcolor='white',
                # height=600,
                # width=800,
                autosize=True,
                margin=dict(l=50, r=50, t=80, b=50),
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='lightgray',
                    borderwidth=1
                )
            )
            
            # 添加网格
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            # 不显示图表到控制台，避免日志中输出大量HTML
        # fig.show(config={'responsive': True})
            return fig
            
        except Exception as e:
            print(f"      ⚠️ 创建等高线图失败: {e}")
            # 返回空图表
            fig = go.Figure()
            fig.add_annotation(
                text=f"等高线图创建失败: {e}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def _create_single_slice_plot_with_ax(
        self,
        ax_optimizer,
        param: str,
        objective: str,
        param_types: Dict[str, str],
        n_points: int,
        confidence_level: float
    ) -> go.Figure:
        """
        使用Ax优化器的代理模型创建单个参数的slice图
        基于Ax的SlicePlot实现
        """
        try:
            # 获取Ax的实验数据
            trials_df = ax_optimizer.get_optimization_history()
            
            # 获取参数值范围
            param_values = self._get_parameter_values_for_slice(param, n_points)
            
            # 获取已采样的参数值
            sampled_values = sorted(trials_df[param].unique())
            
            # 使用中位数和众数方法，确保图表平滑
            # 排除目标变量（因变量），只包含其他输入参数
            other_params = [p for p in trials_df.columns if p in param_types.keys() and p not in [param, objective]]
            status_quo = {}
            for p in other_params:
                if pd.api.types.is_numeric_dtype(trials_df[p]):
                    # 使用中位数，对异常值更鲁棒
                    status_quo[p] = trials_df[p].median()
                else:
                    # 对于类别参数，使用众数
                    status_quo[p] = trials_df[p].mode().iloc[0] if not trials_df[p].mode().empty else trials_df[p].iloc[0]
            
            status_quo = pd.Series(status_quo)
            
            # 生成预测数据
            predictions = []
            confidence_intervals = []
            sampled_mask = []
            
            # 对每个参数值进行预测
            for param_val in param_values:
                try:
                    # 创建预测数据点（固定其他参数为中位数/众数值）
                    X_pred = status_quo.copy()
                    X_pred[param] = param_val
                    
                    # 使用Ax的预测方法
                    predictions_dict = ax_optimizer.ax_client.get_model_predictions(
                        metric_names=[objective],
                        parameterizations={0: X_pred.to_dict()}
                    )
                    
                    if 0 in predictions_dict and objective in predictions_dict[0]:
                        pred_mean, pred_std = predictions_dict[0][objective]
                        
                        # 计算置信区间
                        z_score = 1.96  # 95%置信区间
                        ci_lower = pred_mean - z_score * pred_std
                        ci_upper = pred_mean + z_score * pred_std
                        
                        predictions.append(pred_mean)
                        confidence_intervals.append([ci_lower, ci_upper])
                        
                        # 检查这个参数值是否在采样值中（使用容差比较）
                        is_sampled = False
                        for sampled_val in sampled_values:
                            if abs(param_val - sampled_val) < 1e-6:  # 使用小的容差
                                is_sampled = True
                                break
                        sampled_mask.append(is_sampled)
                    else:
                        print(f"      ⚠️ 预测失败，参数值={param_val}，预测结果无效")
                        predictions.append(0.0)
                        confidence_intervals.append([0.0, 0.0])
                        sampled_mask.append(False)
                        
                except Exception as e:
                    print(f"      ⚠️ 预测失败，参数值={param_val}，错误: {e}")
                    print(f"      🔍 错误详情: {type(e).__name__}: {str(e)}")
                    import traceback
                    print(f"      📋 错误堆栈: {traceback.format_exc()}")
                    predictions.append(0.0)
                    confidence_intervals.append([0.0, 0.0])
                    sampled_mask.append(False)
            
            # 创建图表
            fig = go.Figure()
            
            # 添加置信区间
            fig.add_trace(go.Scatter(
                x=param_values,
                y=[ci[1] for ci in confidence_intervals],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=param_values,
                y=[ci[0] for ci in confidence_intervals],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # 添加预测线
            fig.add_trace(go.Scatter(
                x=param_values,
                y=predictions,
                mode='lines',
                line=dict(color='#1f77b4', width=2),
                name='Predicted Outcome ',
                hovertemplate=self._create_slice_hover_template(param, objective, status_quo, other_params) +
                             f'<b>{param}</b>: %{{x}}<br>' +
                             f'<b>{objective} (Predicted)</b>: %{{y:.3f}}<br>' +
                             '<extra></extra>'
            ))
            
            # 添加已采样点标记
            sampled_indices = [i for i, sampled in enumerate(sampled_mask) if sampled]
            if sampled_indices:
                # 获取已采样点的实际观测值（而不是预测值）
                sampled_x = [param_values[i] for i in sampled_indices]
                sampled_y = []
                
                # 从原始数据中获取已采样点的实际观测值
                for x_val in sampled_x:
                    # 找到最接近的采样点
                    closest_idx = None
                    min_diff = float('inf')
                    for idx, row in trials_df.iterrows():
                        if abs(row[param] - x_val) < min_diff:
                            min_diff = abs(row[param] - x_val)
                            closest_idx = idx
                    
                    if closest_idx is not None:
                        sampled_y.append(trials_df.iloc[closest_idx][objective])
                    else:
                        sampled_y.append(predictions[sampled_indices[sampled_indices.index(x_val)]])
                
                # 为采样点创建详细的hover信息
                # 每个采样点都应该显示该点的真实参数值
                sampled_hover_templates = []
                for i, x_val in enumerate(sampled_x):
                    # 找到对应的原始数据行
                    closest_idx = None
                    min_diff = float('inf')
                    for idx, row in trials_df.iterrows():
                        if abs(row[param] - x_val) < min_diff:
                            min_diff = abs(row[param] - x_val)
                            closest_idx = idx
                    
                    if closest_idx is not None:
                        # 获取该样本点的所有参数值
                        sample_row = trials_df.iloc[closest_idx]
                        hover_info = f'<b>{param}</b>: {x_val:.3f}<br>'
                        hover_info += f'<b>{objective} (Observed)</b>: {sampled_y[i]:.3f}<br>'
                        hover_info += '<b>Other Parameters:</b><br>'
                        
                        # 添加该样本点的其他参数值（每个点都不同）
                        for other_param in other_params:
                            if other_param in sample_row:
                                value = sample_row[other_param]
                                if pd.api.types.is_numeric_dtype(trials_df[other_param]):
                                    hover_info += f'  {other_param}: {value:.3f}<br>'
                                else:
                                    hover_info += f'  {other_param}: {value}<br>'
                        
                        hover_info += '<extra></extra>'
                        sampled_hover_templates.append(hover_info)
                    else:
                        sampled_hover_templates.append(f'<b>{param}</b>: {x_val:.3f}<br><b>{objective} (Observed)</b>: {sampled_y[i]:.3f}<br><extra></extra>')
                
                fig.add_trace(go.Scatter(
                    x=sampled_x,
                    y=sampled_y,
                    mode='markers',
                    marker=dict(
                        color='black',
                        symbol='x',
                        size=8
                    ),
                    name='Sampled Points',
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=sampled_hover_templates
                ))
            
            # 更新布局
            fig.update_layout(
                title=dict(
                    text=f"{param} vs {objective} (Slice Plot)",
                    x=0.5,
                    font=dict(size=16)
                ),
                xaxis_title=param,
                yaxis_title=objective,
                plot_bgcolor='white',
                paper_bgcolor='white',
                # height=500,
                # width=800,
                autosize=True,
                margin=dict(l=50, r=50, t=80, b=50),
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='lightgray',
                    borderwidth=1
                )
            )
            
            # 添加网格
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False
            )
            # 不显示图表到控制台，避免日志中输出大量HTML
        # fig.show(config={'responsive': True})
            return fig
            
        except Exception as e:
            print(f"    ❌ 使用Ax模型创建slice图失败: {e}")
            print(f"    🔍 错误详情: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"    📋 错误堆栈: {traceback.format_exc()}")
            # 如果Ax模型失败，返回错误图表
            fig = go.Figure()
            fig.add_annotation(
                text=f"无法创建slice图<br>错误: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(
                title=f"{param} vs {objective} (Error)",
                plot_bgcolor='white',
                paper_bgcolor='white',
                # height=500,
                # width=800
                autosize=True
            )
            # 不显示图表到控制台，避免日志中输出大量HTML
        # fig.show(config={'responsive': True})
            return fig
    
    def _create_slice_hover_template(self, param: str, objective: str, status_quo: pd.Series, other_params: List[str]) -> str:
        """
        为slice图创建hover模板，显示所有参数的当前值
        
        Args:
            param: 当前分析的自变量参数
            objective: 目标变量
            status_quo: 其他参数的固定值（中位数/众数）
            other_params: 其他参数列表
            
        Returns:
            hover模板字符串
        """
        hover_template = '<b>Current Parameter State:</b><br>'
        
        # 添加其他参数的固定值
        for other_param in other_params:
            if other_param in status_quo:
                value = status_quo[other_param]
                if pd.api.types.is_numeric_dtype(type(value)) and not isinstance(value, bool):
                    hover_template += f'  {other_param}: {value:.3f}<br>'
                else:
                    hover_template += f'  {other_param}: {value}<br>'
        
        return hover_template
    
    def _create_contour_hover_template(self, param1: str, param2: str, objective: str, status_quo: pd.Series, other_params: List[str]) -> str:
        """
        为等高线图创建hover模板，显示所有参数的当前值
        
        Args:
            param1: 第一个自变量参数
            param2: 第二个自变量参数
            objective: 目标变量
            status_quo: 其他参数的固定值（中位数/众数）
            other_params: 其他参数列表
            
        Returns:
            hover模板字符串
        """
        hover_template = '<b>Current Parameter State:</b><br>'
        
        # 添加其他参数的固定值
        for other_param in other_params:
            if other_param in status_quo:
                value = status_quo[other_param]
                if pd.api.types.is_numeric_dtype(type(value)) and not isinstance(value, bool):
                    hover_template += f'  {other_param}: {value:.3f}<br>'
                else:
                    hover_template += f'  {other_param}: {value}<br>'
        
        return hover_template
    
    def _rebuild_ax_optimizer(
        self,
        parameters: List[str],
        objectives: List[str],
        search_space: List[Dict[str, Any]],
        # 新增：自定义代理模型配置参数
        surrogate_model_class: Optional[Type] = None,
        kernel_class: Optional[Type] = None,
        kernel_options: Optional[Dict[str, Any]] = None
    ) -> BayesianOptimizer:
        """
        基于实验数据重建Ax优化器（带缓存）
        
        Args:
            parameters: 参数列表
            objectives: 目标列表
            search_space: 搜索空间配置（必须提供）
            surrogate_model_class: 代理模型类（可选，如SingleTaskGP、MultiTaskGP等）
            kernel_class: 核函数类（可选，如MaternKernel、RBFKernel等）
            kernel_options: 核函数参数（可选，如{"nu": 2.5}用于MaternKernel）
            
        Returns:
            重建的Ax优化器实例
        """
        # 创建缓存键
        cache_key = (
            tuple(sorted(parameters)),
            tuple(sorted(objectives)),
            str(search_space),
            surrogate_model_class,
            kernel_class,
            str(kernel_options) if kernel_options else None
        )
        
        # 检查缓存
        if cache_key in self._ax_optimizer_cache:
            print(f"🔄 使用缓存的Ax优化器 (参数: {len(parameters)}, 目标: {len(objectives)})")
            return self._ax_optimizer_cache[cache_key]
        
        print(f"🔧 重建Ax优化器 (参数: {len(parameters)}, 目标: {len(objectives)})")
        # 从数据推断优化配置（用于创建实验，不影响预测）
        optimization_config = self._infer_optimization_config(objectives)
        
        # 创建优化器，支持自定义代理模型配置
        optimizer = BayesianOptimizer(
            search_space=search_space,
            optimization_config=optimization_config,
            experiment_name="experiment_rebuild",
            random_seed=42,
            # 传递自定义代理模型配置
            surrogate_model_class=surrogate_model_class,
            kernel_class=kernel_class,
            kernel_options=kernel_options
        )
        
        # 将实验数据转换为ExperimentResult格式
        experiments = []
        for _, row in self.experiment_data.iterrows():
            # 处理参数中的None值
            exp_params = {}
            for param in parameters:
                val = row[param]
                # 处理None值，将其转换为字符串"None"
                if pd.isna(val) or val is None:
                    exp_params[param] = "None"
                else:
                    # 确保数值类型参数转换为浮点数
                    if pd.api.types.is_numeric_dtype(self.experiment_data[param]):
                        exp_params[param] = float(val)
                    else:
                        exp_params[param] = val
            
            exp_metrics = {obj: row[obj] for obj in objectives}
            experiments.append(ExperimentResult(
                parameters=exp_params,
                metrics=exp_metrics
            ))
        
        # 添加先验实验数据
        optimizer.add_prior_experiments(experiments)
        
        # 缓存优化器
        self._ax_optimizer_cache[cache_key] = optimizer
        
        return optimizer
    
    def create_cross_validation_plots(
        self,
        parameters: List[str],
        objectives: List[str],
        search_space: List[Dict[str, Any]],
        untransform: bool = True,
        # 新增：自定义代理模型配置参数
        surrogate_model_class: Optional[Type] = None,
        kernel_class: Optional[Type] = None,
        kernel_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, go.Figure]:
        """
        创建留一法交叉验证图，展示模型预测准确性
        基于Ax代理模型的实现
        
        Args:
            parameters: 参数列表
            objectives: 目标指标列表
            search_space: 搜索空间配置（必须提供）
            untransform: 是否反变换预测结果
            
        Returns:
            交叉验证图字典
        """
        if self.experiment_data is None:
            raise ValueError("没有实验数据")
        
        # 检查参数和目标是否存在
        all_columns = parameters + objectives
        missing_columns = [col for col in all_columns if col not in self.experiment_data.columns]
        if missing_columns:
            raise ValueError(f"以下列不存在于数据中: {missing_columns}")
        
        print(f"🔍 开始留一法交叉验证分析...")
        
        # 重建Ax优化器和代理模型
        print("🔧 重建Ax优化器和代理模型...")
        ax_optimizer = self._rebuild_ax_optimizer(
            parameters=parameters,
            objectives=objectives,
            search_space=search_space,
            # 传递自定义代理模型配置
            surrogate_model_class=surrogate_model_class,
            kernel_class=kernel_class,
            kernel_options=kernel_options
        )
        
        plots = {}
        
        # 为每个目标创建交叉验证图
        for objective in objectives:
            print(f"📊 为目标 '{objective}' 生成交叉验证图...")
            
            cv_fig = self._create_single_cross_validation_plot(
                ax_optimizer=ax_optimizer,
                objective=objective,
                parameters=parameters,
                objectives=objectives,
                search_space=search_space,
                untransform=untransform
            )
            
            plot_key = f"cross_validation_{objective}"
            plots[plot_key] = cv_fig
            self.plots[plot_key] = cv_fig
        
        return plots
    
    def _create_single_cross_validation_plot(
        self,
        ax_optimizer,
        objective: str,
        parameters: List[str],
        objectives: List[str],
        search_space: List[Dict[str, Any]],
        untransform: bool
    ) -> go.Figure:
        """
        创建单个目标的交叉验证图
        基于Ax代理模型的留一法交叉验证
        """
        try:
            # 执行留一法交叉验证
            cv_results = self._perform_leave_one_out_cv(
                ax_optimizer=ax_optimizer,
                objective=objective,
                parameters=parameters,
                objectives=objectives,
                search_space=search_space,
                untransform=untransform
            )
            
            if not cv_results:
                print(f"      ⚠️ 交叉验证失败，目标: {objective}")
                return self._create_error_figure(f"交叉验证失败: {objective}")
            
            # 创建交叉验证图
            fig = go.Figure()
            
            # 准备详细的hover信息
            hover_texts = []
            for detail in cv_results['point_details']:
                hover_text = f"<b>Point {detail['point_id']}</b><br>"
                hover_text += f"<b>Observed {objective}:</b> {detail['observed']:.3f}<br>"
                hover_text += f"<b>Predicted {objective}:</b> {detail['predicted']:.3f} ± {detail['ci']:.3f}<br>"
                hover_text += "<b>Parameters:</b><br>"
                for param, value in detail['parameters'].items():
                    if isinstance(value, float):
                        hover_text += f"  {param}: {value:.3f}<br>"
                    else:
                        hover_text += f"  {param}: {value}<br>"
                hover_texts.append(hover_text)
            
            # 添加散点图，只有垂直误差棒（预测值的置信区间）
            fig.add_trace(go.Scatter(
                x=cv_results['observed'],
                y=cv_results['predicted'],
                mode='markers',
                marker=dict(
                    color='blue',
                    size=8,
                    opacity=0.7
                ),
                error_y=dict(
                    type='data',
                    array=cv_results['predicted_ci'],
                    visible=True,
                    color='blue',
                    thickness=1,
                    width=3
                ),
                name='Cross Validation Points',
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts
            ))
            
            # 添加对角线（完美预测线）
            min_val = min(
                cv_results['observed'].min(),
                (cv_results['predicted'] - cv_results['predicted_ci']).min()
            )
            max_val = max(
                cv_results['observed'].max(),
                (cv_results['predicted'] + cv_results['predicted_ci']).max()
            )
            
            fig.add_trace(go.Scatter(
                x=[min_val * 0.99, max_val * 1.01],
                y=[min_val * 0.99, max_val * 1.01],
                mode='lines',
                line=dict(
                    color='gray',
                    dash='dash',
                    width=2
                ),
                name='Perfect Prediction (y=x)',
                showlegend=True
            ))
            
            # 计算模型性能指标
            mse = np.mean((cv_results['observed'] - cv_results['predicted']) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(cv_results['observed'] - cv_results['predicted']))
            r2 = 1 - np.sum((cv_results['observed'] - cv_results['predicted']) ** 2) / np.sum((cv_results['observed'] - cv_results['observed'].mean()) ** 2)
            
            # 更新布局
            fig.update_layout(
                title=dict(
                    text=f"Cross Validation for {objective} (Leave-One-Out)",
                    x=0.5,
                    font=dict(size=16)
                ),
                xaxis_title=f"Actual {objective}",
                yaxis_title=f"Predicted {objective}",
                plot_bgcolor='white',
                paper_bgcolor='white',
                # height=600,
                # width=800,
                autosize=True,
                margin=dict(l=50, r=50, t=100, b=50),
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='lightgray',
                    borderwidth=1
                ),
                # 添加性能指标注释
                annotations=[
                    dict(
                        x=0.02,
                        y=0.3,
                        xref='paper',
                        yref='paper',
                        text=f"RMSE: {rmse:.3f}<br>MAE: {mae:.3f}<br>R²: {r2:.3f}",
                        showarrow=False,
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='lightgray',
                        borderwidth=1,
                        font=dict(size=12)
                    )
                ]
            )
            
            # 设置坐标轴为正方形
            fig.update_xaxes(
                range=[min_val * 0.99, max_val * 1.01],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            fig.update_yaxes(
                range=[min_val * 0.99, max_val * 1.01],
                scaleanchor='x',
                scaleratio=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            # 不显示图表到控制台，避免日志中输出大量HTML
        # fig.show(config={'responsive': True})
            return fig
            
        except Exception as e:
            print(f"      ⚠️ 创建交叉验证图失败: {e}")
            return self._create_error_figure(f"创建交叉验证图失败: {e}")
    
    def _perform_leave_one_out_cv(
        self,
        ax_optimizer,
        objective: str,
        parameters: List[str],
        objectives: List[str],
        search_space: List[Dict[str, Any]],
        untransform: bool
    ) -> Dict[str, Any]:
        """
        执行留一法交叉验证
        使用Ax代理模型进行预测
        """
        try:
            # 获取实验数据
            trials_df = ax_optimizer.get_optimization_history()
            n_samples = len(trials_df)
            
            if n_samples < 3:
                print(f"      ⚠️ 样本数量太少 ({n_samples})，无法进行交叉验证")
                return {}
            
            observed_values = []
            predicted_values = []
            predicted_cis = []
            point_names = []
            point_details = []  # 存储每个点的详细参数信息
            
            print(f"      🔄 执行留一法交叉验证，共 {n_samples} 个样本...")
            
            # 执行留一法交叉验证
            for i in range(n_samples):
                try:
                    # 获取当前样本的实际值
                    actual_value = trials_df.iloc[i][objective]
                    point_name = f"Point_{i+1}"
                    
                    # 创建临时训练数据（排除当前样本）
                    temp_data = trials_df.drop(index=trials_df.index[i]).copy()
                    
                    if len(temp_data) < 2:  # 至少需要2个样本进行训练
                        continue
                    
                    # 创建临时优化器进行预测
                    prediction_result = self._predict_with_temp_optimizer(
                        temp_data, trials_df.iloc[i], objective, parameters, objectives, search_space, untransform
                    )
                    
                    if prediction_result is not None:
                        # 收集当前点的详细参数信息
                        point_detail = {
                            'point_id': i + 1,
                            'observed': actual_value,
                            'predicted': prediction_result['predicted'],
                            'ci': prediction_result['ci'],
                            'parameters': {}
                        }
                        
                        # 添加所有参数的值
                        for param in parameters:
                            if param in trials_df.columns:
                                point_detail['parameters'][param] = trials_df.iloc[i][param]
                        
                        observed_values.append(actual_value)
                        predicted_values.append(prediction_result['predicted'])
                        predicted_cis.append(prediction_result['ci'])
                        point_names.append(point_name)
                        point_details.append(point_detail)
                        
                        print(f"        ✅ 样本 {i+1}/{n_samples}: 观测值={actual_value:.3f}, 预测值={prediction_result['predicted']:.3f}±{prediction_result['ci']:.3f}")
                    
                except Exception as e:
                    print(f"        ⚠️ 样本 {i+1} 预测失败: {e}")
                    continue
            
            if not observed_values:
                print(f"      ❌ 没有成功的预测结果")
                return {}
            
            print(f"      ✅ 成功完成 {len(observed_values)}/{n_samples} 个样本的交叉验证")
            
            return {
                'observed': np.array(observed_values),
                'predicted': np.array(predicted_values),
                'predicted_ci': np.array(predicted_cis),
                'point_names': point_names,
                'point_details': point_details
            }
            
        except Exception as e:
            print(f"      ⚠️ 交叉验证执行失败: {e}")
            return {}
    
    def _predict_with_temp_optimizer(
        self,
        temp_data: pd.DataFrame,
        test_point: pd.Series,
        objective: str,
        parameters: List[str],
        objectives: List[str],
        search_space: List[Dict[str, Any]],
        untransform: bool
    ) -> Optional[Dict[str, float]]:
        """
        使用临时优化器进行预测
        """
        try:
            # 使用用户指定的参数列表，排除所有目标变量
            user_params = [col for col in temp_data.columns 
                          if col in parameters]
            
            # 创建临时分析器
            temp_analyzer = ParameterOptimizationAnalysis(
                experiment_data=temp_data,
                output_dir="temp_cv"
            )
            
            # 重建临时Ax优化器
            temp_optimizer = temp_analyzer._rebuild_ax_optimizer(
                parameters=user_params,
                objectives=[objective],
                search_space=search_space
            )
            
            # 准备测试样本的参数
            test_params = {}
            for param in user_params:
                val = test_point[param]
                # 处理None值，将其转换为字符串"None"
                if pd.isna(val) or val is None:
                    test_params[param] = "None"
                else:
                    test_params[param] = val
            
            # 使用Ax的预测方法
            predictions_dict = temp_optimizer.ax_client.get_model_predictions(
                metric_names=[objective],
                parameterizations={0: test_params}
            )
            
            if 0 in predictions_dict and objective in predictions_dict[0]:
                pred_mean, pred_std = predictions_dict[0][objective]
                
                # 计算95%置信区间
                ci = pred_std * 1.96
                
                return {
                    'predicted': pred_mean,
                    'std': pred_std,
                    'ci': ci
                }
            else:
                print(f"        ⚠️ 预测返回空结果")
                return None
                
        except Exception as e:
            print(f"        ⚠️ 临时优化器预测失败: {str(e)}")
            import traceback
            print(f"        🔍 详细错误信息: {traceback.format_exc()}")
            return None
    
    def _create_error_figure(self, error_message: str) -> go.Figure:
        """
        创建错误图表
        """
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Cross Validation Error",
            plot_bgcolor='white',
            paper_bgcolor='white',
            # height=400,
            # width=600
            autosize=True
        )
        # 不显示图表到控制台，避免日志中输出大量HTML
        # fig.show(config={'responsive': True})
        return fig
    

    
    def _infer_optimization_config(self, objectives: List[str]) -> Dict[str, Any]:
        """
        从实验数据推断优化配置
        """
        # 默认所有目标都是最小化
        objectives_config = {obj: {"minimize": True} for obj in objectives}
        
        return {
            "objectives": objectives_config,
            "use_weights": False,
            "additional_metrics": []
        }
    
    def _get_parameter_values_for_slice(
        self,
        param: str,
        n_points: int
    ) -> List[float]:
        """
        获取参数值范围（用于slice图）
        基于Ax的get_parameter_values实现
        """
        param_data = self.experiment_data[param]
        param_min = param_data.min()
        param_max = param_data.max()
        
        # 生成均匀分布的点
        param_values = np.linspace(param_min, param_max, n_points).tolist()
        
        # 确保采样值也包含在预测点中
        sampled_values = sorted(param_data.unique())
        for sampled_val in sampled_values:
            if sampled_val not in param_values:
                param_values.append(sampled_val)
        
        # 重新排序
        param_values.sort()
        
        return param_values
    
    def _create_interactive_feature_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        shap_values: np.ndarray,
        feature_names: List[str],
        target_name: str
    ) -> go.Figure:
        """
        创建交互式特征分析图（包含特征重要性、相关性散点图和SHAP影响散点图）
        """
        # 计算平均SHAP值
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f"Feature Importance for {target_name}",
                "Feature vs Target Correlation",
                "SHAP Impact Analysis"
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}]],
            horizontal_spacing=0.1
        )
        
        # 1. 特征重要性条形图
        fig.add_trace(
            go.Bar(
                x=mean_shap_values,
                y=feature_names,
                orientation='h',
                marker_color='lightblue',
                name="Feature Importance",
                hovertemplate='<b>%{y}</b><br>' +
                            'Average SHAP Impact: %{x:.3f}<br>' +
                            '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. 默认显示第一个特征的相关性散点图
        default_feature = feature_names[0]
        fig.add_trace(
            go.Scatter(
                x=X[default_feature],
                y=y,
                mode='markers',
                marker=dict(color='lightblue', size=8),
                name=f"{default_feature} vs {target_name}",
                hovertemplate='<b>%{x}</b><br>' +
                            f'{target_name}: %{{y}}<br>' +
                            '<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 添加参考线
        fig.add_hline(
            y=y.mean(),
            line_dash="dash",
            line_color="red",
            row=1, col=2,
            annotation_text=f"Mean {target_name}: {y.mean():.2f}"
        )
        
        # 3. 默认显示第一个特征的SHAP影响散点图
        feature_idx = feature_names.index(default_feature)
        fig.add_trace(
            go.Scatter(
                x=X[default_feature],
                y=shap_values[:, feature_idx],
                mode='markers',
                marker=dict(color='lightblue', size=8),
                name=f"SHAP Impact of {default_feature}",
                hovertemplate='<b>%{x}</b><br>' +
                            'SHAP Impact: %{y:.3f}<br>' +
                            '<extra></extra>'
            ),
            row=1, col=3
        )
        
        # 添加参考线
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            row=1, col=3,
            annotation_text="Zero Impact"
        )
        
        # 更新布局
        fig.update_layout(
            title=f"Interactive Feature Analysis for {target_name}",
            plot_bgcolor='white',
            paper_bgcolor='white',
            # height=500,
            # width=1500,
            autosize=True,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=False
        )
        
        # 更新坐标轴标签
        fig.update_xaxes(title_text="Average SHAP Impact", row=1, col=1)
        fig.update_yaxes(title_text="Features", row=1, col=1)
        
        fig.update_xaxes(title_text=default_feature, row=1, col=2)
        fig.update_yaxes(title_text=target_name, row=1, col=2)
        
        fig.update_xaxes(title_text=default_feature, row=1, col=3)
        fig.update_yaxes(title_text="SHAP Impact", row=1, col=3)
        
        # 添加JavaScript回调用于交互
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    showactive=True,
                    x=0.1,
                    y=1.1,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label=feature,
                            method="update",
                            args=[
                                {
                                    "xaxis2.title": feature,
                                    "xaxis3.title": feature,
                                    "xaxis2.range": [X[feature].min(), X[feature].max()],
                                    "xaxis3.range": [X[feature].min(), X[feature].max()]
                                },
                                {
                                    "data": [
                                        # 更新相关性散点图
                                        dict(
                                            x=X[feature],
                                            y=y
                                        ),
                                        # 更新SHAP影响散点图
                                        dict(
                                            x=X[feature],
                                            y=shap_values[:, feature_names.index(feature)]
                                        )
                                    ]
                                }
                            ]
                        ) for feature in feature_names
                    ]
                )
            ]
        )
        # 不显示图表到控制台，避免日志中输出大量HTML
        # fig.show(config={'responsive': True})
        return fig
    
    def save_plots(self, format: str = 'html') -> List[str]:
        """
        保存所有生成的图表
        
        Args:
            format: 保存格式 ('html', 'png', 'jpg', 'svg')
            
        Returns:
            保存的文件路径列表
        """
        saved_files = []
        
        for plot_name, fig in self.plots.items():
            filename = f"{plot_name}.{format}"
            filepath = self.output_dir / filename
            
            if format == 'html':
                fig.write_html(str(filepath))
            else:
                fig.write_image(str(filepath))
            
            saved_files.append(str(filepath))
            print(f"💾 保存图表: {filename}")
        
        return saved_files
    
    def save_single_plot(self, plot_name: str, fig: go.Figure, format: str = 'html') -> str:
        """
        立即保存单个图表
        
        Args:
            plot_name: 图表名称
            fig: 图表对象
            format: 保存格式 ('html', 'png', 'jpg', 'svg')
            
        Returns:
            保存的文件路径
        """
        filename = f"{plot_name}.{format}"
        filepath = self.output_dir / filename
        
        if format == 'html':
            fig.write_html(str(filepath))
        else:
            fig.write_image(str(filepath))
        
        
        return str(filepath)
    
    def generate_feature_importance_analysis(
        self,
        parameters: List[str],
        objectives: List[str],
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        生成特征重要性分析
        
        Args:
            parameters: 参数列表
            objectives: 目标指标列表
            save_results: 是否保存结果
            
        Returns:
            分析结果字典
        """
        print("🔍 开始特征重要性分析...")
        
        # 生成特征重要性分析
        results = self.create_feature_importance_analysis(
            parameters=parameters,
            objectives=objectives
        )
        
        # 保存结果
        if save_results:
            self.save_plots('html')
        
        print("✅ 特征重要性分析完成!")
        print(f"📁 结果保存在: {self.output_dir}")
        
        return results


# 便捷函数
def analyze_parallel_coordinates(
    data_file: str,
    parameters: List[str],
    objectives: List[str],
    output_dir: str = "analysis_output"
) -> ParameterOptimizationAnalysis:
    """
    便捷函数：生成并行坐标图分析
    
    Args:
        data_file: 数据文件路径
        parameters: 参数列表
        objectives: 目标指标列表
        output_dir: 输出目录
        
    Returns:
        分析器实例
    """
    analyzer = ParameterOptimizationAnalysis(
        experiment_file=data_file,
        output_dir=output_dir
    )
    
    analyzer.generate_parallel_coordinates_analysis(parameters, objectives)
    return analyzer


def analyze_feature_importance(
    data_file: str,
    parameters: List[str],
    objectives: List[str],
    output_dir: str = "analysis_output"
) -> ParameterOptimizationAnalysis:
    """
    便捷函数：生成特征重要性分析
    
    Args:
        data_file: 数据文件路径
        parameters: 参数列表
        objectives: 目标指标列表
        output_dir: 输出目录
        
    Returns:
        分析器实例
    """
    analyzer = ParameterOptimizationAnalysis(
        experiment_file=data_file,
        output_dir=output_dir
    )
    
    analyzer.generate_feature_importance_analysis(parameters, objectives)
    return analyzer


def analyze_slice_plots(
    data_file: str,
    parameters: List[str],
    objectives: List[str],
    search_space: List[Dict[str, Any]] = None,
    optimization_config: Dict[str, Any] = None,
    output_dir: str = "analysis_output"
) -> ParameterOptimizationAnalysis:
    """
    便捷函数：生成slice图分析
    
    Args:
        data_file: 数据文件路径
        parameters: 参数列表
        objectives: 目标指标列表
        search_space: 搜索空间配置（可选，从数据推断）
        optimization_config: 优化配置（可选，从数据推断）
        output_dir: 输出目录
        
    Returns:
        分析器实例
    """
    analyzer = ParameterOptimizationAnalysis(
        experiment_file=data_file,
        output_dir=output_dir
    )
    
    analyzer.create_slice_plots(parameters, objectives, search_space, optimization_config)
    return analyzer


def analyze_cross_validation_plots(
    data_file: str,
    parameters: List[str],
    objectives: List[str],
    search_space: List[Dict[str, Any]],
    untransform: bool = True,
    output_dir: str = "analysis_output",
    # 新增：自定义代理模型配置参数
    surrogate_model_class: Optional[Type] = None,
    kernel_class: Optional[Type] = None,
    kernel_options: Optional[Dict[str, Any]] = None
) -> ParameterOptimizationAnalysis:
    """
    便捷函数：生成交叉验证分析
    
    Args:
        data_file: 数据文件路径
        parameters: 参数列表
        objectives: 目标指标列表
        search_space: 搜索空间配置（必须提供）
        untransform: 是否反变换预测结果
        output_dir: 输出目录
        
    Returns:
        分析器实例
    """
    analyzer = ParameterOptimizationAnalysis(
        experiment_file=data_file,
        output_dir=output_dir
    )
    
    # 生成交叉验证图
    plots = analyzer.create_cross_validation_plots(
        parameters, objectives, search_space, untransform,
        # 传递自定义代理模型配置
        surrogate_model_class=surrogate_model_class,
        kernel_class=kernel_class,
        kernel_options=kernel_options
    )
    
    # 保存生成的图表
    if plots:
        print(f"\n📊 生成的图表:")
        for plot_name in plots.keys():
            print(f"  - {plot_name}")
        
        print(f"\n💾 保存图表...")
        analyzer.save_plots()
        print(f"✅ 图表已保存!")
        print(f"📁 请查看 {output_dir}/ 目录中的HTML文件")
    
    return analyzer


def analyze_contour_plots(
    data_file: str,
    parameters: List[str],
    objectives: List[str],
    search_space: List[Dict[str, Any]],
    output_dir: str = "analysis_output",
    # 新增：自定义代理模型配置参数
    surrogate_model_class: Optional[Type] = None,
    kernel_class: Optional[Type] = None,
    kernel_options: Optional[Dict[str, Any]] = None
) -> ParameterOptimizationAnalysis:
    """
    便捷函数：生成等高线图分析
    
    Args:
        data_file: 数据文件路径
        parameters: 参数列表（需要至少2个参数）
        objectives: 目标指标列表
        search_space: 搜索空间配置（必须提供）
        output_dir: 输出目录
        
    Returns:
        分析器实例
    """
    analyzer = ParameterOptimizationAnalysis(
        experiment_file=data_file,
        output_dir=output_dir
    )
    
    analyzer.create_contour_plots(
        parameters, objectives, search_space,
        # 传递自定义代理模型配置
        surrogate_model_class=surrogate_model_class,
        kernel_class=kernel_class,
        kernel_options=kernel_options
    )
    return analyzer


if __name__ == "__main__":
    # 示例用法
    print("🚀 参数优化分析模块")
    print("=" * 50)
    
    # 使用示例数据进行分析
    data_file = "experiment_data.csv"
    if os.path.exists(data_file):
        print(f"📊 分析文件: {data_file}")
        
        # 定义参数和目标
        parameters = ['solvent', 'catalyst', 'temperature', 'concentration']  # 包含类别变量
        objectives = ['yield', 'side_product']  # 多目标
        
        # 用户指定的参数空间配置
        search_space = [
            {
                "name": "solvent",
                "type": "choice",
                "values": ["THF", "Toluene", "DMSO"]
            },
            {
                "name": "catalyst",
                "type": "choice", 
                "values": ["Pd/C", "CuO", "None"]
            },
            {
                "name": "temperature",
                "type": "range",
                "bounds": [-10, 25]
            },
            {
                "name": "concentration",
                "type": "range",
                "bounds": [0.1, 1.0]
            }
        ]
        
        # 优化配置
        optimization_config = {
            "objectives": {
                "yield": {"minimize": False},  # 最大化产率
                "side_product": {"minimize": True}  # 最小化副产物
            }
        }
        
        # 创建分析器实例
        analyzer = ParameterOptimizationAnalysis(
            experiment_file=data_file,
            output_dir="analysis_output"
        )
        
        # 生成并行坐标图分析
        print("\n📊 生成并行坐标图分析...")
        parallel_results = analyzer.generate_parallel_coordinates_analysis(
            parameters=parameters,
            objectives=objectives
        )
        
        # 生成特征重要性分析
        print("\n📊 生成特征重要性分析...")
        feature_results = analyzer.generate_feature_importance_analysis(
            parameters=parameters,
            objectives=objectives
        )
        
        # 生成交叉验证图分析
        print("\n📊 生成交叉验证图分析...")
        cv_results = analyzer.create_cross_validation_plots(
            parameters=parameters,
            objectives=objectives,
            search_space=search_space,
            untransform=True,
            # 示例：使用自定义代理模型配置（可选）
            surrogate_model_class=SingleTaskGP,
            kernel_class=MaternKernel,
            kernel_options={"nu": 2.5}
        )
        
        # 保存所有图表
        print("\n💾 保存所有图表...")
        analyzer.save_plots()
        
        print(f"\n✅ 分析完成！")
        print(f"📊 并行坐标图: {parallel_results['total_plots']} 个图表")
        print(f"📊 特征重要性分析: {len(objectives)} 个图表")
        print(f"📊 交叉验证图: {len(cv_results)} 个图表")
        print(f"📁 结果保存在: {analyzer.output_dir}")
    else:
        print("⚠️ 未找到示例数据文件，请提供实验数据文件进行分析。")
