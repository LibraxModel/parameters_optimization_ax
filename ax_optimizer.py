from typing import List, Dict, Any, Optional, Tuple, Type
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.adapter.registry import Generators
from ax.core.objective import ScalarizedObjective, MultiObjective, Objective
from ax.core.metric import Metric

# 新增导入以支持自定义代理模型、核函数和采集函数 (Ax 1.1.2 新路径)
from ax.generators.torch.botorch_modular.surrogate import SurrogateSpec, ModelConfig
from botorch.models import (
    SingleTaskGP, MultiTaskGP, KroneckerMultiTaskGP, 
    MixedSingleTaskGP, SingleTaskMultiFidelityGP,
    SaasFullyBayesianSingleTaskGP, SaasFullyBayesianMultiTaskGP,
    HigherOrderGP, SingleTaskVariationalGP
)
from gpytorch.kernels import (
    RBFKernel, MaternKernel, ScaleKernel, LinearKernel,
    PolynomialKernel, PeriodicKernel, SpectralMixtureKernel,
    RQKernel, PiecewisePolynomialKernel, AdditiveKernel,
    ProductKernel, CosineKernel
)
# 采集函数导入
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement, qNoisyExpectedImprovement, qUpperConfidenceBound
)
from botorch.acquisition.analytic import (
    ExpectedImprovement, UpperConfidenceBound, PosteriorMean
)
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient, qMultiFidelityKnowledgeGradient
)
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy, qMultiFidelityMaxValueEntropy
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement, qLogNoisyExpectedImprovement
)
# 多目标采集函数
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.utils.multi_objective.hypervolume import Hypervolume, infer_reference_point
from botorch.utils.multi_objective.pareto import is_non_dominated
        
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings
import torch
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from gpytorch.utils.warnings import NumericalWarning
    warnings.filterwarnings("ignore", category=NumericalWarning)
except ImportError:
    pass
# 抑制 botorch 的数值警告和优化警告

from botorch.exceptions.warnings import NumericsWarning, OptimizationWarning
warnings.filterwarnings("ignore", category=NumericsWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)

# 抑制 MapKeyToFloat 相关的 UserWarning
warnings.filterwarnings("ignore", message=".*MapKeyToFloat.*")

@dataclass
class ExperimentResult:
    """实验结果数据类"""
    parameters: Dict[str, Any]  # 参数配置
    metrics: Dict[str, float]   # 指标结果
    metadata: Optional[Dict[str, Any]] = None  # 额外元数据

class BayesianOptimizer:
    def __init__(
        self,
        search_space: List[Dict[str, Any]],
        optimization_config: Dict[str, Any],
        experiment_name: str = "bayesian_optimization",
        random_seed: Optional[int] = None,
        # 新增：可选的代理模型和核函数配置
        surrogate_model_class: Optional[Type] = None,
        kernel_class: Optional[Type] = None,
        kernel_options: Optional[Dict[str, Any]] = None,
        # 新增：可选的采集函数配置
        acquisition_function_class: Optional[Type[AcquisitionFunction]] = None,
        acquisition_function_options: Optional[Dict[str, Any]] = None
    ):
        """
        初始化贝叶斯优化器
        
        Args:
            search_space: 参数搜索空间配置列表，每个参数需包含：
                - name: 参数名
                - type: "range"（连续）或"choice"（离散）
                - bounds: [min, max] （type为range时）
                - values: list （type为choice时）
                - value_type: "int"或"float"或"str"
            optimization_config: 优化配置，包含：
                - objectives: 优化目标字典，格式为 {"metric_name": {"minimize": bool}}
                - use_weights: 是否使用权重（可选，默认False）
                - objective_weights: 目标权重字典，格式为 {"metric_name": weight}（可选）
                - additional_metrics: 其他需要记录的指标列表（可选）
            experiment_name: 实验名称
            random_seed: 随机种子
            surrogate_model_class: 代理模型类（可选，默认使用Ax默认模型）
            kernel_class: 核函数类（可选，如MaternKernel、RBFKernel等）
            kernel_options: 核函数参数（可选，如{"nu": 2.5}用于MaternKernel）
            acquisition_function_class: 采集函数类（可选，如qExpectedImprovement、qUpperConfidenceBound等）
            acquisition_function_options: 采集函数参数（可选，如{"beta": 0.1}用于UCB）
        """
        self.experiment_name = experiment_name
        self.random_seed = random_seed
        
        # 设置GPU设备
        import torch
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🚀 使用GPU加速: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("⚠️ GPU不可用，使用CPU")
        
        # 处理字符串到类的转换
        def convert_string_to_class(class_obj):
            """将字符串转换为类对象"""
            if isinstance(class_obj, str):
                import sys
                import os
                sys.path.append(os.path.dirname(__file__))
                from __init__ import get_class_from_string
                return get_class_from_string(class_obj)
            return class_obj
        
        surrogate_model_class = convert_string_to_class(surrogate_model_class)
        kernel_class = convert_string_to_class(kernel_class)
        acquisition_function_class = convert_string_to_class(acquisition_function_class)
        
        # 初始化AxClient，直接使用BOTORCH模型

        # 创建只使用BOTORCH_MODULAR的生成策略，支持自定义代理模型、核函数和采集函数
        model_kwargs = {}
        
        # 如果指定了自定义代理模型或核函数，创建SurrogateSpec
        if surrogate_model_class is not None or kernel_class is not None:
            # 设置默认值
            model_class = surrogate_model_class or None
            kernel_cls = kernel_class or None
            kernel_opts = kernel_options or {}
            
            # 特殊处理：为PolynomialKernel设置默认power参数
            if kernel_cls is not None and hasattr(kernel_cls, '__name__') and kernel_cls.__name__ == 'PolynomialKernel':
                if 'power' not in kernel_opts or kernel_opts['power'] == '' or kernel_opts['power'] is None:
                    kernel_opts['power'] = 2
                    print(f"🔧 PolynomialKernel: 设置默认power参数为2")
            
            # 特殊处理：某些模型不支持自定义 covar_module
            if model_class is not None and hasattr(model_class, '__name__'):
                model_name = model_class.__name__
                
                # 完全贝叶斯模型不支持自定义核函数
                if model_name in ['SaasFullyBayesianSingleTaskGP', 'SaasFullyBayesianMultiTaskGP']:
                    print(f"⚠️ 模型 {model_name} 不支持自定义核函数，将忽略 kernel_class 和 kernel_options 参数")
                    # 对于完全贝叶斯模型，只使用模型类，不设置核函数
                    model_config = ModelConfig(
                        botorch_model_class=model_class,
                    )
                else:
                    # 其他模型可以设置自定义核函数
                    # 特殊处理ScaleKernel的base_kernel配置
                    if kernel_cls is not None and hasattr(kernel_cls, '__name__') and kernel_cls.__name__ == 'ScaleKernel':
                        if 'base_kernel' in kernel_opts and isinstance(kernel_opts['base_kernel'], str):
                            # 将base_kernel字符串转换为实际的核函数类
                            base_kernel_class = convert_string_to_class(kernel_opts['base_kernel'])
                            # 创建核函数实例而不是类
                            base_kernel_instance = base_kernel_class()
                            kernel_opts['base_kernel'] = base_kernel_instance
                    
                    # 创建模型配置
                    model_config = ModelConfig(
                        botorch_model_class=model_class,
                        covar_module_class=kernel_cls,
                        covar_module_options=kernel_opts,
                    )
            else:
                # 如果没有模型类，但有核函数类，创建默认配置
                # 特殊处理：为PolynomialKernel设置默认power参数
                if kernel_cls is not None and hasattr(kernel_cls, '__name__') and kernel_cls.__name__ == 'PolynomialKernel':
                    if 'power' not in kernel_opts or kernel_opts['power'] == '' or kernel_opts['power'] is None:
                        kernel_opts['power'] = 2
                        print(f"🔧 PolynomialKernel: 设置默认power参数为2")
                
                # 特殊处理ScaleKernel的base_kernel配置
                if kernel_cls is not None and hasattr(kernel_cls, '__name__') and kernel_cls.__name__ == 'ScaleKernel':
                    if 'base_kernel' in kernel_opts and isinstance(kernel_opts['base_kernel'], str):
                        # 将base_kernel字符串转换为实际的核函数类
                        base_kernel_class = convert_string_to_class(kernel_opts['base_kernel'])
                        # 创建核函数实例而不是类
                        base_kernel_instance = base_kernel_class()
                        kernel_opts['base_kernel'] = base_kernel_instance
                
                # 创建模型配置
                model_config = ModelConfig(
                    botorch_model_class=model_class,
                    covar_module_class=kernel_cls,
                    covar_module_options=kernel_opts,
                )
            
            # 创建代理规格
            surrogate_spec = SurrogateSpec(model_configs=[model_config])
            model_kwargs["surrogate_spec"] = surrogate_spec
        
        # 如果指定了自定义采集函数，添加到model_kwargs中
        if acquisition_function_class is not None:
            model_kwargs["botorch_acqf_class"] = acquisition_function_class
        
        # 如果指定了采集函数选项，添加到model_kwargs中
        if acquisition_function_options is not None:
            if "acquisition_options" not in model_kwargs:
                model_kwargs["acquisition_options"] = {}
            model_kwargs["acquisition_options"].update(acquisition_function_options)
        
        # Ax 1.1.2 新 API: 使用 GenerationNode 和 GeneratorSpec
        generator_spec = GeneratorSpec(
            generator_enum=Generators.BOTORCH_MODULAR,
            model_kwargs=model_kwargs if model_kwargs else None
        )
        
        botorch_node = GenerationNode(
            node_name="BoTorchModular",
            generator_specs=[generator_spec],
        )
        
        gs = GenerationStrategy(
            name="BOTORCH_MODULAR",
            nodes=[botorch_node]
        )
        
        self.ax_client = AxClient(
            random_seed=random_seed,
            generation_strategy=gs
        )
        
        # 构建优化目标配置
        objectives = optimization_config.get("objectives", {})
        if not objectives:
            raise ValueError("optimization_config 必须包含 'objectives' 配置")
        
        # 检查权重设置
        use_weights = optimization_config.get("use_weights", False)
        objective_weights = optimization_config.get("objective_weights", {})
        
        # 构建ObjectiveProperties字典（Ax期望的格式）
        objective_properties = {}
        for metric_name, config in objectives.items():
            minimize = config.get("minimize", True)
            objective_properties[metric_name] = ObjectiveProperties(minimize=minimize)
        
        # 创建实验
        self.ax_client.create_experiment(
            name=experiment_name,
            parameters=search_space,
            objectives=objective_properties,
            tracking_metric_names=optimization_config.get("additional_metrics", [])
        )
        
        # 如果使用权重，在创建实验后设置ScalarizedObjective
        if use_weights and len(objectives) > 1:
            try:
                # 获取已创建的实验
                experiment = self.ax_client.experiment
                
                # 创建ScalarizedObjective
                metrics = []
                weights = []
                minimize = None
                
                for metric_name, config in objectives.items():
                    # 从实验中获取已创建的Metric
                    metric = experiment.metrics[metric_name]
                    metrics.append(metric)
                    
                    weight = objective_weights.get(metric_name, 1.0)
                    weights.append(weight)
                    
                    # 检查优化方向是否一致
                    current_minimize = config.get("minimize", True)
                    if minimize is None:
                        minimize = current_minimize
                    elif minimize != current_minimize:
                        raise ValueError(f"线性加权优化要求所有目标使用相同的优化方向，但 {metric_name} 的方向与其他目标不一致")
                
                # 创建ScalarizedObjective并设置到实验
                scalarized_objective = ScalarizedObjective(
                    metrics=metrics,
                    weights=weights,
                    minimize=minimize
                )
                
                # 更新实验的优化配置
                from ax.core.optimization_config import OptimizationConfig
                experiment.optimization_config = OptimizationConfig(objective=scalarized_objective)
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"设置ScalarizedObjective失败，将使用普通多目标优化: {e}")
        
        # 记录实验次数
        self.trial_count = 0
        
    def add_prior_experiments(self, experiments: List[ExperimentResult]) -> None:
        """
        添加先验实验数据
        
        Args:
            experiments: 先验实验结果列表
        """
        for exp in experiments:
            # 直接使用原始参数值，让Ax自己处理类型转换
            parameters = exp.parameters.copy()
            
            # 先创建试验
            _, trial_index = self.ax_client.attach_trial(parameters)
            
            # 更新试验结果
            raw_data = {
                metric_name: (metric_value, 0.0)  # (value, SEM)
                for metric_name, metric_value in exp.metrics.items()
            }
            self.ax_client.complete_trial(
                trial_index=trial_index,
                raw_data=raw_data,
                metadata=exp.metadata
            )
    
    def get_next_parameters(self, n: int = 1) -> List[Tuple[int, Dict[str, Any]]]:
        """
        获取下一组或多组建议的参数配置
        
        Args:
            n: 需要生成的参数组数，默认为1
            
        Returns:
            List of (trial_index, parameters) tuples:
            - trial_index: 实验索引
            - parameters: 参数配置字典
        """
        self.trial_count += n
        # Ax 1.1.2: get_next_trials 返回元组 (dict[int, dict], bool)
        trials_dict, _ = self.ax_client.get_next_trials(max_trials=n)
        results = [(trial_index, parameters) for trial_index, parameters in trials_dict.items()]
        return results
    
    def update_experiment(self, 
                         trial_index: int,
                         metrics: Dict[str, float],
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        更新实验结果
        
        Args:
            trial_index: 实验索引
            metrics: 指标结果字典
            metadata: 额外元数据（可选）
        """
        raw_data = {
            metric_name: (metric_value, 0.0)  # (value, SEM)
            for metric_name, metric_value in metrics.items()
        }
        self.ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=raw_data,
            metadata=metadata
        )
    
    def get_best_parameters(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        获取当前最优参数配置和对应的指标值
        自动处理三种优化类型：
        - 单目标优化 (Objective): 返回最优解
        - 帕累托多目标优化 (MultiObjective): 使用Ax返回的帕累托前沿集合，选择一个代表性解
        - 线性加权多目标优化 (ScalarizedObjective): 返回加权最优解
        
        Returns:
            best_parameters: 最优参数配置
            best_metrics: 最优指标值
        """
        try:
            # 如果是多目标（MOO），Ax要求使用get_pareto_optimal_parameters
            opt_config = self.ax_client.experiment.optimization_config
            if opt_config is not None and getattr(opt_config, "is_moo_problem", False):
                pareto = self.ax_client.get_pareto_optimal_parameters(use_model_predictions=True)
                if not pareto:
                    return {}, {}
                # 选择一个代表性解：按trial_index从小到大取第一个
                chosen_index = sorted(pareto.keys())[0]
                parameters, metric_values = pareto[chosen_index]
                means, _ = metric_values
                metrics = {name: (val, 0.0) for name, val in means.items()}
                return parameters, metrics

            # 直接使用Ax的get_best_trial方法，Ax会自动处理不同的优化类型
            best_trial = self.ax_client.get_best_trial()
            if best_trial is None:
                return {}, {}
                
            # 解包返回的三元组：(trial_index, parameters, metrics)
            _, parameters, metric_values = best_trial
            
            # 构建指标字典
            metrics = {}
            # metric_values 是一个元组：(means, covariances)
            means, _ = metric_values
            for metric_name, mean_value in means.items():
                metrics[metric_name] = (mean_value, 0.0)  # 使用0.0作为sem，因为我们没有使用协方差
                
            return parameters, metrics
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"获取最优参数失败: {e}")
            # 如果get_best_trial失败，尝试从历史数据中获取最优解
            try:
                # 再次优先处理MOO：直接返回帕累托集合中的一个代表性解
                opt_config = self.ax_client.experiment.optimization_config
                if opt_config is not None and getattr(opt_config, "is_moo_problem", False):
                    pareto = self.ax_client.get_pareto_optimal_parameters(use_model_predictions=True)
                    if pareto:
                        chosen_index = sorted(pareto.keys())[0]
                        parameters, metric_values = pareto[chosen_index]
                        means, _ = metric_values
                        metrics = {name: (val, 0.0) for name, val in means.items()}
                        return parameters, metrics

                trials_df = self.ax_client.get_trials_data_frame()
                if trials_df.empty:
                    return {}, {}
                
                # 获取优化目标类型
                objective = self.ax_client.experiment.optimization_config.objective
                
                # 找到最优的trial
                best_idx = None
                
                if isinstance(objective, ScalarizedObjective):
                    # 加权优化：直接找objective_mean列的最优值
                    if 'objective_mean' in trials_df.columns:
                        minimize = getattr(objective, 'minimize', True)
                        best_value = float('inf') if minimize else float('-inf')
                        
                        for idx, trial in trials_df.iterrows():
                            value = trial.get('objective_mean')
                            if pd.isna(value):
                                continue
                            if minimize:
                                if value < best_value:
                                    best_value = value
                                    best_idx = idx
                            else:
                                if value > best_value:
                                    best_value = value
                                    best_idx = idx
                    else:
                        return {}, {}  # 没有objective_mean列，无法处理
                else:
                    # 非加权优化：使用第一个目标
                    if hasattr(objective, 'metric_names'):
                        objective_names = list(objective.metric_names)
                    else:
                        objective_names = [col[:-5] for col in trials_df.columns if col.endswith('_mean') and col != 'objective_mean']
                    
                    if not objective_names:
                        return {}, {}
                    
                    target_col = f"{objective_names[0]}_mean"
                    minimize = getattr(objective, 'minimize', True)
                    best_value = float('inf') if minimize else float('-inf')
                    
                    for idx, trial in trials_df.iterrows():
                        value = trial.get(target_col)
                        if pd.isna(value):
                            continue
                        if minimize:
                            if value < best_value:
                                best_value = value
                                best_idx = idx
                        else:
                            if value > best_value:
                                best_value = value
                                best_idx = idx
                
                if best_idx is not None:
                    trial = trials_df.iloc[best_idx]
                    parameters = {}
                    metrics = {}
                    
                    # 提取参数
                    for param_name in self.ax_client.experiment.search_space.parameters.keys():
                        if f"{param_name}" in trial:
                            parameters[param_name] = trial[f"{param_name}"]
                    
                    # 提取指标
                    if isinstance(objective, ScalarizedObjective):
                        # 加权优化：提取原始指标值
                        for metric in getattr(objective, 'metrics', []):
                            metric_name = metric.name if hasattr(metric, 'name') else str(metric)
                            if f"{metric_name}_mean" in trial:
                                metrics[metric_name] = (trial[f"{metric_name}_mean"], trial.get(f"{metric_name}_sem", 0.0))
                    else:
                        # 非加权优化：提取目标指标值
                        if hasattr(objective, 'metric_names'):
                            objective_names = list(objective.metric_names)
                        else:
                            objective_names = [col[:-5] for col in trials_df.columns if col.endswith('_mean') and col != 'objective_mean']
                        
                        for name in objective_names:
                            if f"{name}_mean" in trial:
                                metrics[name] = (trial[f"{name}_mean"], trial.get(f"{name}_sem", 0.0))
                    
                    return parameters, metrics
                
            except Exception as e2:
                logger.error(f"从历史数据获取最优参数也失败: {e2}")
            
            return {}, {}
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        获取优化历史记录
        
        Returns:
            history_df: 包含所有实验记录的DataFrame
        """
        return self.ax_client.get_trials_data_frame()
    
    def compute_point_hypervolumes(self) -> List[float]:
        """
        计算每个实验点相对于参考点的 hypervolume 值（仅适用于多目标优化）
        
        处理流程：
        1. 读取全部实验目标值
        2. 根据优化方向将最小化目标取反，转成“越大越好”
        3. 对每个目标做 Min-Max 归一化，映射到 [0, 1] 区间，保证不同量纲可比
        4. 参考点设置为所有数据中最差的点（每个目标的最小值，归一化后为 0）
        5. 计算所有实验点（包含非帕累托点）相对于参考点的 hypervolume
        
        返回列表顺序与 `get_trials_data_frame()` 中的实验顺序一致。
        如果某个点不支配参考点，则该点的 hypervolume 值为 0。
        
        Raises:
            ValueError: 如果不是多目标优化问题
            ValueError: 如果没有先验数据
        """
        # 检查是否为多目标优化
        opt_config = self.ax_client.experiment.optimization_config
        if opt_config is None:
            raise ValueError("无法获取优化配置")
        
        # 检查是否为多目标优化
        is_moo = getattr(opt_config, "is_moo_problem", False)
        if not is_moo:
            raise ValueError("compute_hypervolume 仅适用于多目标优化问题")
        
        Y, minimize_flags = self._extract_objective_tensor(opt_config)
        
        # 获取优化方向（minimize 或 maximize）
        # 将最小化目标转换为最大化（BoTorch 的 Hypervolume 假设最大化）
        for i, minimize in enumerate(minimize_flags):
            if minimize:
                Y[:, i] = -Y[:, i]
        
        # 对每个目标做 Min-Max 归一化，映射到 [0, 1] 区间，避免不同单位或量纲影响 hypervolume
        Y_standardized = self._standardize_objectives(Y)
        
        # 验证归一化：归一化后的数据应该在 [0, 1] 区间内
        # 添加调试信息（可选，用于验证归一化是否正确）
        if False:  # 设置为 True 可以启用调试输出
            import logging
            logger = logging.getLogger(__name__)
            mins_after = Y_standardized.min(dim=0).values
            maxs_after = Y_standardized.max(dim=0).values
            means_after = Y_standardized.mean(dim=0)
            logger.info(f"归一化后统计: 最小值={mins_after.tolist()}, 最大值={maxs_after.tolist()}, "
                       f"均值={means_after.tolist()}")
        
        # 找到帕累托前沿
        pareto_mask = is_non_dominated(Y_standardized)
        pareto_Y = Y_standardized[pareto_mask]
        
        if pareto_Y.shape[0] == 0:
            return [0.0] * Y.shape[0]
        
        # 参考点：使用所有数据中最差的点（每个目标的最小值）
        # 归一化后，每个目标的最小值应该是 0，所以参考点应该是 [0, 0, ..., 0]
        ref_point = Y_standardized.min(dim=0).values.to(device=self.device, dtype=torch.float32)
        
        # 确保 ref_point 的维度正确
        if ref_point.shape[0] != pareto_Y.shape[1]:
            raise ValueError(
                f"参考点维度 ({ref_point.shape[0]}) 与目标数量 ({pareto_Y.shape[1]}) 不匹配"
            )
        
        # 计算 hypervolume（用于单点计算）
        hv = Hypervolume(ref_point=ref_point)
        point_hypervolumes = self._compute_point_hypervolumes(
            hypervolume=hv,
            Y=Y_standardized,
            ref_point=ref_point
        )
        return point_hypervolumes

    def _extract_objective_tensor(
        self,
        opt_config
    ) -> Tuple[torch.Tensor, List[bool]]:
        trials_df = self.ax_client.get_trials_data_frame()
        if trials_df.empty:
            raise ValueError("没有先验实验数据，无法计算 hypervolume")
        
        objective = opt_config.objective
        if isinstance(objective, MultiObjective):
            objective_names = [obj.metric.name for obj in objective.objectives]
            minimize_flags = [obj.minimize for obj in objective.objectives]
        elif isinstance(objective, ScalarizedObjective):
            objective_names = [metric.name for metric in objective.metrics]
            minimize = getattr(objective, "minimize", True)
            minimize_flags = [minimize] * len(objective_names)
        else:
            raise ValueError("无法识别优化目标类型")
        
        objective_values = []
        for _, row in trials_df.iterrows():
            values = []
            for obj_name in objective_names:
                mean_col = f"{obj_name}_mean"
                if mean_col in row:
                    value = row[mean_col]
                else:
                    value = row.get(obj_name, None)
                if pd.notna(value):
                    values.append(float(value))
                else:
                    break
            if len(values) == len(objective_names):
                objective_values.append(values)
        
        if not objective_values:
            raise ValueError("无法从实验数据中提取有效的目标值")
        
        Y = torch.tensor(objective_values, dtype=torch.float32, device=self.device)
        return Y, minimize_flags

    def _standardize_objectives(self, Y: torch.Tensor) -> torch.Tensor:
        """
        对每个目标做 Min-Max 归一化，将每个目标映射到 [0, 1] 区间。
        公式: (x - min) / (max - min)
        如果某个目标的值全部相同（max == min），则保持为 0.5，避免除零。
        """
        mins = Y.min(dim=0, keepdim=True).values
        maxs = Y.max(dim=0, keepdim=True).values
        ranges = maxs - mins
        # 如果范围太小（接近0），设为 1，避免除零，并将值设为 0.5
        ranges = torch.where(ranges < 1e-8, torch.ones_like(ranges), ranges)
        Y_normalized = (Y - mins) / ranges
        # 如果范围太小，将值设为 0.5（中间值）
        Y_normalized = torch.where((maxs - mins) < 1e-8, torch.full_like(Y_normalized, 0.5), Y_normalized)
        return Y_normalized

    def _compute_point_hypervolumes(
        self,
        hypervolume: Hypervolume,
        Y: torch.Tensor,
        ref_point: torch.Tensor
    ) -> List[float]:
        """
        计算每个点相对于参考点的 hypervolume 值
        
        对于每个点，计算该点与参考点围成的超体积。
        如果该点支配参考点（所有目标都优于参考点），则 hypervolume = ∏(yi - ri)
        如果该点不支配参考点，则 hypervolume = 0
        
        Args:
            hypervolume: Hypervolume 计算器（包含参考点）
            Y: 所有点的目标值张量 (n_points, n_objectives)，已转换为最大化空间
            ref_point: 参考点张量 (n_objectives,)，已转换为最大化空间
            
        Returns:
            每个点的 hypervolume 值列表（与 Y 的行顺序一致）
        """
        num_points = Y.shape[0]
        point_hvs = []
        
        for i in range(num_points):
            point = Y[i:i+1]  # 单个点，形状 (1, n_objectives)
            
            # 直接计算该点与参考点围成的超体积
            # Hypervolume.compute 内部会处理不支配参考点的情况（返回 0）
            hv_value = hypervolume.compute(point)
            point_hvs.append(float(hv_value))
        
        return point_hvs

def test_optimizer():
    """
    测试贝叶斯优化器的基本功能
    包括默认配置、自定义代理模型/核函数配置和自定义采集函数配置
    """
    # 定义搜索空间
    search_space = [
        {
            "name": "power",
            "type": "range",
            "bounds": [1000, 3000],
            "value_type": "int",
            "log_scale": False  # 线性尺度可能更适合功率参数
        },
        {
            "name": "speed",
            "type": "range",
            "bounds": [10, 50],
            "value_type": "float",
            "log_scale": False  # 线性尺度适合速度参数
        },
        {
            "name": "frequency",
            "type": "choice",
            "values": [500, 1000, 1500, 2000],
            "value_type": "int",
            "is_ordered": True,
            "sort_values": True
        }
    ]
    
    # 优化配置 - 帕累托多目标优化
    optimization_config = {
        "objectives": {
            "roughness": {"minimize": True},  # 最小化表面粗糙度
            "kerf_width": {"minimize": True}  # 最小化切缝宽度
        },
        "use_weights": False,  # 不启用权重，使用帕累托优化
        "additional_metrics": []  # 没有额外的跟踪指标
    }
    
    # 创建优化器
    optimizer = BayesianOptimizer(
        search_space=search_space,
        optimization_config=optimization_config,
        experiment_name="laser_cutting_optimization",
        random_seed=42
    )
    
    # 添加一些先验实验数据
    prior_experiments = [
        ExperimentResult(
            parameters={"power": 2000, "speed": 30.0, "frequency": 1000},  # 确保speed是float类型
            metrics={"roughness": 2.5, "kerf_width": 0.15}
        ),
        ExperimentResult(
            parameters={"power": 2500, "speed": 40.0, "frequency": 1500},  # 确保speed是float类型
            metrics={"roughness": 1.8, "kerf_width": 0.18}
        )
    ]
    optimizer.add_prior_experiments(prior_experiments)
    
    # 模拟优化过程
    # 一次性获取3组参数
    next_trials = optimizer.get_next_parameters(n=3)
    for i, (trial_index, parameters) in enumerate(next_trials):
        print(f"\n第 {i+1} 次实验参数：")
        print(parameters)
        
        # 模拟实验结果（这里用随机值代替实际实验）
        roughness = np.random.uniform(1.5, 3.0)
        kerf_width = np.random.uniform(0.1, 0.2)
        
        # 更新实验结果
        optimizer.update_experiment(
            trial_index=trial_index,
            metrics={"roughness": roughness, "kerf_width": kerf_width}
        )
    
    # 获取最优结果（帕累托优化 - Ax自动选择帕累托前沿上的一个解）
    best_parameters, best_metrics = optimizer.get_best_parameters()
    print(f"\n帕累托最优解（Ax自动选择）：")
    print("参数配置:")
    for param_name, value in best_parameters.items():
        print(f"  {param_name}: {value}")
    print("指标值:")
    for metric_name, (mean, sem) in best_metrics.items():
        print(f"  {metric_name}: {mean:.4f} ± {sem:.4f}")
    

    

    
    # 获取优化历史
    history = optimizer.get_optimization_history()
    print("\n优化历史：")
    print(history.to_string(index=False))
    
    print("\n" + "="*50)
    print("测试自定义代理模型和核函数配置")
    print("="*50)
    
    # 创建使用 Matérn-5/2 核函数的优化器
    # 注意：即使是多目标优化，Ax通常为每个目标使用独立的SingleTaskGP模型
    custom_optimizer = BayesianOptimizer(
        search_space=search_space,
        optimization_config=optimization_config,
        experiment_name="custom_kernel_optimization",
        random_seed=42,
        # 使用自定义配置 - 每个目标会使用独立的SingleTaskGP + Matérn核
        surrogate_model_class=SingleTaskGP,  # 多目标时Ax为每个目标创建独立的SingleTaskGP
        kernel_class=MaternKernel,           # 使用 Matérn 核函数
        kernel_options={"nu": 2.5}          # Matérn-5/2 核函数
    )
    
    print("\n" + "="*50)
    print("测试自定义采集函数配置")
    print("="*50)
    
    # 创建使用 qUpperConfidenceBound 采集函数的优化器
    acquisition_optimizer = BayesianOptimizer(
        search_space=search_space,
        optimization_config=optimization_config,
        experiment_name="custom_acquisition_optimization",
        random_seed=42,
        # 使用自定义采集函数
        acquisition_function_class=qExpectedHypervolumeImprovement,  # 多目标优化使用EHVI
        acquisition_function_options={}  # 可以添加采集函数参数
    )
    
    # 为自定义优化器也添加一些先验数据，避免变换错误
    custom_optimizer.add_prior_experiments(prior_experiments)
    
    # 运行几次试验
    next_trials = custom_optimizer.get_next_parameters(n=2)
    for i, (trial_index, parameters) in enumerate(next_trials):
        print(f"\n自定义配置 - 第 {i+1} 次实验参数：")
        print(parameters)
        
        # 模拟实验结果
        roughness = np.random.uniform(1.5, 3.0)
        kerf_width = np.random.uniform(0.1, 0.2)
        
        custom_optimizer.update_experiment(
            trial_index=trial_index,
            metrics={"roughness": roughness, "kerf_width": kerf_width}
        )
    
    print("\n自定义配置优化器创建成功！")
    print("配置详情:")
    print("- 多目标优化：roughness + kerf_width")
    print("- 代理模型：SingleTaskGP（每个目标独立建模）")
    print("- 核函数：Matérn-5/2 (nu=2.5)")
    print("- 对比默认配置：默认使用RBF核函数")

    # 为采集函数优化器也添加一些先验数据，避免变换错误
    acquisition_optimizer.add_prior_experiments(prior_experiments)
    
    # 运行几次试验
    next_trials = acquisition_optimizer.get_next_parameters(n=2)
    for i, (trial_index, parameters) in enumerate(next_trials):
        print(f"\n自定义采集函数 - 第 {i+1} 次实验参数：")
        print(parameters)
        
        # 模拟实验结果
        roughness = np.random.uniform(1.5, 3.0)
        kerf_width = np.random.uniform(0.1, 0.2)
        
        acquisition_optimizer.update_experiment(
            trial_index=trial_index,
            metrics={"roughness": roughness, "kerf_width": kerf_width}
        )
    
    print("\n自定义采集函数优化器创建成功！")
    print("配置详情:")
    print("- 多目标优化：roughness + kerf_width")
    print("- 采集函数：qExpectedHypervolumeImprovement (EHVI)")
    print("- 优势：适合多目标优化，直接优化超体积指标")
    print("- 对比默认配置：默认使用qNoisyExpectedImprovement")
    
    print("\n" + "="*60)
    print("综合测试：同时使用自定义模型、核函数和采集函数")
    print("="*60)
    
    # 创建一个同时使用所有自定义配置的优化器
    comprehensive_optimizer = BayesianOptimizer(
        search_space=search_space,
        optimization_config=optimization_config,
        experiment_name="comprehensive_custom_optimization",
        random_seed=42,
        # 同时使用所有自定义配置
        surrogate_model_class=SingleTaskGP,
        kernel_class=MaternKernel,
        kernel_options={"nu": 1.5},  # Matérn-3/2 核
        acquisition_function_class=qExpectedHypervolumeImprovement,
        acquisition_function_options={}  # 可根据需要添加参数
    )
    
    print("综合自定义优化器创建成功！")
    print("最终配置:")
    print("- 代理模型: SingleTaskGP")
    print("- 核函数: Matérn-3/2 (nu=1.5)")
    print("- 采集函数: qExpectedHypervolumeImprovement")
    print("- 优势: 综合了所有自定义配置，可实现高度个性化的优化")

def get_available_models_kernels_and_acquisitions():
    """
    返回所有可配置的代理模型、核函数和采集函数的详细信息
    
    Returns:
        dict: 包含代理模型、核函数和采集函数配置选项的字典
    """
    
    models_info = {
        "代理模型 (Surrogate Models)": {
            "SingleTaskGP": {
                "描述": "标准单任务高斯过程，适用于大多数优化问题",
                "适用场景": "单目标或多目标独立建模",
                "特点": "简单可靠，Ax默认推荐"
            },
            "MultiTaskGP": {
                "描述": "多任务高斯过程，能利用任务间相关性",
                "适用场景": "多个相关任务，需要任务特征",
                "特点": "可以共享信息，提高数据效率"
            },
            "KroneckerMultiTaskGP": {
                "描述": "Kronecker结构的多任务GP，适用于结构化多任务",
                "适用场景": "任务具有Kronecker积结构",
                "特点": "计算效率高，适合大规模多任务"
            },
            "MixedSingleTaskGP": {
                "描述": "支持混合变量类型（连续+离散）的GP",
                "适用场景": "同时包含连续和分类变量的优化",
                "特点": "处理混合变量类型"
            },
            "SingleTaskMultiFidelityGP": {
                "描述": "多保真度单任务GP，支持不同精度的评估",
                "适用场景": "有多个评估精度级别",
                "特点": "可以利用低成本的近似评估"
            },
            "SaasFullyBayesianSingleTaskGP": {
                "描述": "全贝叶斯单任务GP，使用Spike-and-Slab先验",
                "适用场景": "高维问题，需要特征选择",
                "特点": "自动特征选择，适合高维稀疏问题"
            },
            "SaasFullyBayesianMultiTaskGP": {
                "描述": "全贝叶斯多任务GP版本",
                "适用场景": "高维多任务问题",
                "特点": "结合多任务学习和特征选择"
            },
            "HigherOrderGP": {
                "描述": "高阶高斯过程，捕捉复杂的高阶相互作用",
                "适用场景": "存在复杂变量交互的问题",
                "特点": "能建模高阶交互效应"
            },
            "SingleTaskVariationalGP": {
                "描述": "变分推断的GP，适用于大数据集",
                "适用场景": "大规模数据集优化",
                "特点": "计算效率高，可扩展性好"
            }
        },
        
        "核函数 (Kernels)": {
            "RBFKernel": {
                "描述": "径向基函数核（高斯核），平滑且无限可微",
                "参数": "lengthscale (长度尺度)",
                "适用": "光滑函数，大多数工程问题",
                "示例": "kernel_options={'lengthscale': 1.0}"
            },
            "MaternKernel": {
                "描述": "Matérn核，通过nu参数控制平滑度",
                "参数": "nu (0.5, 1.5, 2.5, 无穷大), lengthscale",
                "适用": "不同平滑度需求，工程优化常用",
                "示例": "kernel_options={'nu': 2.5, 'lengthscale': 1.0}"
            },
            "LinearKernel": {
                "描述": "线性核，建模线性关系",
                "参数": "variance",
                "适用": "线性或近似线性问题",
                "示例": "kernel_options={'variance': 1.0}"
            },
            "PolynomialKernel": {
                "描述": "多项式核，建模多项式关系",
                "参数": "power (幂次), offset",
                "适用": "多项式关系的问题",
                "示例": "kernel_options={'power': 2, 'offset': 1.0}"
            },
            "PeriodicKernel": {
                "描述": "周期核，建模周期性模式",
                "参数": "period (周期), lengthscale",
                "适用": "具有周期性的优化问题",
                "示例": "kernel_options={'period': 1.0, 'lengthscale': 1.0}"
            },
            "SpectralMixtureKernel": {
                "描述": "谱混合核，可以近似任意平稳核",
                "参数": "num_mixtures (混合数量)",
                "适用": "复杂的频域特征",
                "示例": "kernel_options={'num_mixtures': 4}"
            },
            "RQKernel": {
                "描述": "有理二次核，结合RBF和多项式特性",
                "参数": "alpha (形状参数), lengthscale",
                "适用": "中等复杂度的平滑函数",
                "示例": "kernel_options={'alpha': 2.0, 'lengthscale': 1.0}"
            },
            "CosineKernel": {
                "描述": "余弦核，建模余弦相似性",
                "参数": "period",
                "适用": "余弦型周期模式",
                "示例": "kernel_options={'period': 1.0}"
            },
            "ScaleKernel": {
                "描述": "缩放核，为其他核添加输出缩放",
                "参数": "base_kernel, outputscale",
                "适用": "需要调整输出尺度的情况",
                "示例": "用作包装器核"
            },
            "AdditiveKernel": {
                "描述": "加性核，组合多个核函数",
                "参数": "kern1, kern2 (要组合的核)",
                "适用": "需要组合不同类型相关性",
                "示例": "RBF + Linear 组合"
            },
            "ProductKernel": {
                "描述": "乘积核，核函数的乘积",
                "参数": "kern1, kern2 (要相乘的核)",
                "适用": "需要核函数乘积的场景",
                "示例": "Periodic × RBF"
            }
        },
        
        "采集函数 (Acquisition Functions)": {
            "qExpectedImprovement": {
                "描述": "期望改进（批量版本），平衡开发与探索",
                "适用": "单目标优化，均衡的探索-开发策略",
                "参数": "eta (约束平滑度，默认1e-3)",
                "示例": "acquisition_function_options={'eta': 1e-3}"
            },
            "qNoisyExpectedImprovement": {
                "描述": "噪声期望改进，考虑观测噪声的EI",
                "适用": "单目标优化，存在观测噪声",
                "参数": "eta (约束平滑度，默认1e-3)",
                "示例": "acquisition_function_options={'eta': 1e-3}"
            },
            "qUpperConfidenceBound": {
                "描述": "上置信界，可调节探索程度",
                "适用": "单目标优化，需要控制探索-开发平衡",
                "参数": "beta (探索权重，默认0.2)",
                "示例": "acquisition_function_options={'beta': 0.1}"
            },
            "qKnowledgeGradient": {
                "描述": "知识梯度，考虑信息价值",
                "适用": "单目标优化，重视信息获取",
                "参数": "num_fantasies (幻想样本数)",
                "示例": "acquisition_function_options={'num_fantasies': 128}"
            },
            "qLogExpectedImprovement": {
                "描述": "对数期望改进，数值稳定性更好",
                "适用": "单目标优化，改进值较小时",
                "参数": "无特殊参数",
                "示例": "acquisition_function_class=qLogExpectedImprovement"
            },
            "qMaxValueEntropy": {
                "描述": "最大值熵搜索，优化最大值的不确定性",
                "适用": "单目标优化，高效的全局搜索",
                "参数": "num_mv_samples (最大值样本数)",
                "示例": "acquisition_function_options={'num_mv_samples': 10}"
            },
            "qExpectedHypervolumeImprovement": {
                "描述": "期望超体积改进，多目标优化经典方法",
                "适用": "多目标优化，直接优化帕累托前沿",
                "参数": "ref_point (参考点，可选)",
                "示例": "acquisition_function_class=qExpectedHypervolumeImprovement"
            },
            "qNoisyExpectedHypervolumeImprovement": {
                "描述": "噪声期望超体积改进",
                "适用": "多目标优化，存在观测噪声",
                "参数": "ref_point (参考点，可选)",
                "示例": "acquisition_function_class=qNoisyExpectedHypervolumeImprovement"
            },
            "qLogExpectedHypervolumeImprovement": {
                "描述": "对数期望超体积改进，数值稳定",
                "适用": "多目标优化，改进值较小时",
                "参数": "ref_point (参考点，可选)",
                "示例": "acquisition_function_class=qLogExpectedHypervolumeImprovement"
            },
            "qLogNParEGO": {
                "描述": "ParEGO的对数版本，将多目标转为单目标",
                "适用": "多目标优化，计算资源有限时",
                "参数": "无特殊参数",
                "示例": "acquisition_function_class=qLogNParEGO"
            },
            "PosteriorMean": {
                "描述": "后验均值，纯开发策略",
                "适用": "单目标优化，已知最优区域，精确搜索",
                "参数": "无特殊参数",
                "示例": "acquisition_function_class=PosteriorMean"
            },
            "ExpectedImprovement": {
                "描述": "经典期望改进（解析版本）",
                "适用": "单目标优化，计算高效",
                "参数": "无特殊参数", 
                "示例": "acquisition_function_class=ExpectedImprovement"
            },
            "UpperConfidenceBound": {
                "描述": "经典上置信界（解析版本）",
                "适用": "单目标优化，计算高效",
                "参数": "beta (探索权重)",
                "示例": "acquisition_function_options={'beta': 0.1}"
            }
        }
    }
    
    return models_info


def test_single_objective_acquisition_functions():
    """
    测试单目标优化的不同采集函数
    """
    print("🎯 单目标优化 - 采集函数对比测试")
    print("=" * 60)
    
    # 定义单目标搜索空间（激光切割优化）
    search_space = [
        {
            "name": "power",
            "type": "range", 
            "bounds": [1000, 3000],
            "value_type": "int"
        },
        {
            "name": "speed",
            "type": "range",
            "bounds": [10, 50], 
            "value_type": "float"
        },
        {
            "name": "frequency",
            "type": "choice",
            "values": [500, 1000, 1500, 2000],
            "value_type": "int"
        }
    ]
    
    # 单目标优化配置 - 只优化表面粗糙度
    optimization_config = {
        "objectives": {
            "roughness": {"minimize": True}
        },
        "additional_metrics": ["kerf_width"]  # 作为额外跟踪指标
    }
    
    # 先验实验数据
    prior_experiments = [
        ExperimentResult(
            parameters={"power": 2000, "speed": 30.0, "frequency": 1000},
            metrics={"roughness": 2.5, "kerf_width": 0.15}
        ),
        ExperimentResult(
            parameters={"power": 2500, "speed": 40.0, "frequency": 1500},
            metrics={"roughness": 1.8, "kerf_width": 0.18}
        )
    ]
    
    # 测试不同的采集函数
    acquisition_functions = [
        {
            "name": "qExpectedImprovement",
            "class": qExpectedImprovement,
            "options": {},
            "description": "期望改进 - 均衡探索与开发"
        },
        {
            "name": "qNoisyExpectedImprovement", 
            "class": qNoisyExpectedImprovement,
            "options": {},
            "description": "噪声期望改进 - 考虑观测噪声"
        },
        {
            "name": "qUpperConfidenceBound",
            "class": qUpperConfidenceBound,
            "options": {"beta": 0.1},
            "description": "上置信界 - 探索导向(beta=0.1)"
        },
        {
            "name": "qLogExpectedImprovement",
            "class": qLogExpectedImprovement, 
            "options": {},
            "description": "对数期望改进 - 数值稳定"
        }
    ]
    
    for i, acq_func in enumerate(acquisition_functions, 1):
        print(f"\n📋 测试 {i}: {acq_func['name']}")
        print(f"描述: {acq_func['description']}")
        print("-" * 50)
        
        try:
            # 创建优化器
            optimizer = BayesianOptimizer(
                search_space=search_space,
                optimization_config=optimization_config,
                experiment_name=f"single_obj_{acq_func['name']}",
                random_seed=42,
                surrogate_model_class=SingleTaskGP,
                kernel_class=MaternKernel,
                kernel_options={"nu": 2.5},
                acquisition_function_class=acq_func['class'],
                acquisition_function_options=acq_func['options']
            )
            
            # 添加先验数据
            optimizer.add_prior_experiments(prior_experiments)
            
            # 获取下一组参数
            next_trials = optimizer.get_next_parameters(n=2)
            
            for j, (trial_index, parameters) in enumerate(next_trials, 1):
                print(f"  试验 {j}: {parameters}")
                
                # 模拟实验结果
                roughness = np.random.uniform(1.5, 3.0)
                kerf_width = np.random.uniform(0.1, 0.2)
                
                optimizer.update_experiment(
                    trial_index=trial_index,
                    metrics={"roughness": roughness, "kerf_width": kerf_width}
                )
            
            # 获取最优结果
            best_params, best_metrics = optimizer.get_best_parameters()
            print(f"  最优粗糙度: {best_metrics.get('roughness', (0, 0))[0]:.4f}")
            print(f"  ✅ {acq_func['name']} 测试成功")
            
        except Exception as e:
            print(f"  ❌ {acq_func['name']} 测试失败: {e}")

def test_multi_objective_acquisition_functions():
    """
    测试多目标优化的不同采集函数
    """
    print("\n🎯 多目标优化 - 采集函数对比测试")
    print("=" * 60)
    
    # 定义搜索空间
    search_space = [
        {
            "name": "power",
            "type": "range",
            "bounds": [1000, 3000], 
            "value_type": "int"
        },
        {
            "name": "speed",
            "type": "range",
            "bounds": [10, 50],
            "value_type": "float"
        },
        {
            "name": "frequency",
            "type": "choice",
            "values": [500, 1000, 1500, 2000],
            "value_type": "int"
        }
    ]
    
    # 多目标优化配置
    optimization_config = {
        "objectives": {
            "roughness": {"minimize": True},
            "kerf_width": {"minimize": True}
        }
    }
    
    # 先验数据
    prior_experiments = [
        ExperimentResult(
            parameters={"power": 2000, "speed": 30.0, "frequency": 1000},
            metrics={"roughness": 2.5, "kerf_width": 0.15}
        ),
        ExperimentResult(
            parameters={"power": 2500, "speed": 40.0, "frequency": 1500},
            metrics={"roughness": 1.8, "kerf_width": 0.18}
        )
    ]
    
    # 测试多目标采集函数
    multi_obj_acquisition_functions = [
        {
            "name": "qExpectedHypervolumeImprovement",
            "class": qExpectedHypervolumeImprovement,
            "options": {},
            "description": "期望超体积改进 - 经典多目标方法"
        },
        {
            "name": "qNoisyExpectedHypervolumeImprovement",
            "class": qNoisyExpectedHypervolumeImprovement, 
            "options": {},
            "description": "噪声期望超体积改进 - 考虑噪声"
        },
        {
            "name": "qLogExpectedHypervolumeImprovement",
            "class": qLogExpectedHypervolumeImprovement,
            "options": {},
            "description": "对数期望超体积改进 - 数值稳定(推荐)"
        },
        {
            "name": "qLogNParEGO",
            "class": qLogNParEGO,
            "options": {},
            "description": "ParEGO对数版本 - 计算高效"
        }
    ]
    
    for i, acq_func in enumerate(multi_obj_acquisition_functions, 1):
        print(f"\n📋 测试 {i}: {acq_func['name']}")
        print(f"描述: {acq_func['description']}")
        print("-" * 50)
        
        try:
            # 创建优化器
            optimizer = BayesianOptimizer(
                search_space=search_space,
                optimization_config=optimization_config,
                experiment_name=f"multi_obj_{acq_func['name']}",
                random_seed=42,
                surrogate_model_class=SingleTaskGP,
                kernel_class=MaternKernel,
                kernel_options={"nu": 2.5},
                acquisition_function_class=acq_func['class'],
                acquisition_function_options=acq_func['options']
            )
            
            # 添加先验数据
            optimizer.add_prior_experiments(prior_experiments)
            
            # 获取下一组参数
            next_trials = optimizer.get_next_parameters(n=2)
            
            for j, (trial_index, parameters) in enumerate(next_trials, 1):
                print(f"  试验 {j}: {parameters}")
                
                # 模拟实验结果
                roughness = np.random.uniform(1.5, 3.0)
                kerf_width = np.random.uniform(0.1, 0.2)
                
                optimizer.update_experiment(
                    trial_index=trial_index,
                    metrics={"roughness": roughness, "kerf_width": kerf_width}
                )
            
            # 获取最优结果（帕累托前沿）
            best_params, best_metrics = optimizer.get_best_parameters()
            if best_metrics:
                roughness_val = best_metrics.get('roughness', (0, 0))[0]
                kerf_val = best_metrics.get('kerf_width', (0, 0))[0]
                print(f"  帕累托解: roughness={roughness_val:.4f}, kerf_width={kerf_val:.4f}")
            print(f"  ✅ {acq_func['name']} 测试成功")
            
        except Exception as e:
            print(f"  ❌ {acq_func['name']} 测试失败: {e}")

def test_acquisition_function_parameters():
    """
    测试采集函数参数对优化行为的影响
    """
    print("\n🎯 采集函数参数调优测试")
    print("=" * 60)
    
    search_space = [
        {
            "name": "x",
            "type": "range",
            "bounds": [-5, 5],
            "value_type": "float"
        },
        {
            "name": "y", 
            "type": "range",
            "bounds": [-5, 5],
            "value_type": "float"
        }
    ]
    
    # 简单的单目标优化
    optimization_config = {
        "objectives": {
            "objective": {"minimize": True}
        }
    }
    
    # 测试不同的beta值对UCB的影响
    beta_values = [0.01, 0.1, 0.5, 1.0]
    
    print("\n🔧 测试 qUpperConfidenceBound 的 beta 参数:")
    
    for beta in beta_values:
        print(f"\n  Beta = {beta} ({'低探索' if beta < 0.1 else '高探索' if beta > 0.5 else '中等探索'})")
        
        try:
            optimizer = BayesianOptimizer(
                search_space=search_space,
                optimization_config=optimization_config,
                experiment_name=f"ucb_beta_{beta}",
                random_seed=42,
                acquisition_function_class=qUpperConfidenceBound,
                acquisition_function_options={"beta": beta}
            )
            
            # 添加一些初始数据
            initial_data = [
                ExperimentResult(
                    parameters={"x": 0.0, "y": 0.0},
                    metrics={"objective": 1.0}
                )
            ]
            optimizer.add_prior_experiments(initial_data)
            
            # 获取下一个试验点
            # get_next_parameters 返回 [(trial_index, parameters), ...]
            trial_index, params = optimizer.get_next_parameters(n=1)[0]
            
            print(f"    下一个试验点: x={params['x']:.3f}, y={params['y']:.3f}")
            print(f"    ✅ Beta={beta} 测试成功")
            
        except Exception as e:
            print(f"    ❌ Beta={beta} 测试失败: {e}")
    
    # 测试 EI 采集函数的 eta 参数
    print("\n🔧 测试 qExpectedImprovement 的 eta 参数 (约束平滑度):")
    
    eta_values = [1e-4, 1e-3, 1e-2, 1e-1]
    
    for eta in eta_values:
        print(f"\n  Eta = {eta} ({'低平滑' if eta < 1e-3 else '高平滑' if eta > 1e-2 else '中等平滑'})")
        
        try:
            optimizer = BayesianOptimizer(
                search_space=search_space,
                optimization_config=optimization_config,
                experiment_name=f"ei_eta_{eta}",
                random_seed=42,
                acquisition_function_class=qExpectedImprovement,
                acquisition_function_options={"eta": eta}
            )
            
            # 添加一些初始数据
            initial_data = [
                ExperimentResult(
                    parameters={"x": 0.0, "y": 0.0},
                    metrics={"objective": 1.0}
                ),
                ExperimentResult(
                    parameters={"x": 1.0, "y": 1.0},
                    metrics={"objective": 0.5}
                )
            ]
            optimizer.add_prior_experiments(initial_data)
            
            # 获取下一个试验点
            # get_next_parameters 返回 [(trial_index, parameters), ...]
            trial_index, params = optimizer.get_next_parameters(n=1)[0]
            
            print(f"    下一个试验点: x={params['x']:.3f}, y={params['y']:.3f}")
            print(f"    ✅ Eta={eta} 测试成功")
            
        except Exception as e:
            print(f"    ❌ Eta={eta} 测试失败: {e}")
    
    # 测试 qKnowledgeGradient 的 num_fantasies 参数
    print("\n🔧 测试 qKnowledgeGradient 的 num_fantasies 参数:")
    
    fantasies_values = [16, 64, 128, 256]
    
    for num_fantasies in fantasies_values:
        print(f"\n  Num_fantasies = {num_fantasies} ({'低采样' if num_fantasies < 64 else '高采样' if num_fantasies > 128 else '中等采样'})")
        
        try:
            optimizer = BayesianOptimizer(
                search_space=search_space,
                optimization_config=optimization_config,
                experiment_name=f"kg_fantasies_{num_fantasies}",
                random_seed=42,
                acquisition_function_class=qKnowledgeGradient,
                acquisition_function_options={"num_fantasies": num_fantasies}
            )
            
            # 添加一些初始数据
            initial_data = [
                ExperimentResult(
                    parameters={"x": 0.0, "y": 0.0},
                    metrics={"objective": 1.0}
                )
            ]
            optimizer.add_prior_experiments(initial_data)
            
            # 获取下一个试验点
            # get_next_parameters 返回 [(trial_index, parameters), ...]
            trial_index, params = optimizer.get_next_parameters(n=1)[0]
            
            print(f"    下一个试验点: x={params['x']:.3f}, y={params['y']:.3f}")
            print(f"    ✅ Num_fantasies={num_fantasies} 测试成功")
            
        except Exception as e:
            print(f"    ❌ Num_fantasies={num_fantasies} 测试失败: {e}")

if __name__ == "__main__":

    # 运行不同的采集函数测试
    test_single_objective_acquisition_functions()
    test_multi_objective_acquisition_functions() 
    test_acquisition_function_parameters()
    
    print("\n" + "=" * 70)
    print("🎉 所有采集函数测试完成!")
    print("=" * 70)
    
    # 注释掉原来的综合测试
    # print("\n" + "=" * 60)
    # print("🚀 运行原始测试示例:")
    # print("=" * 60)
    # test_optimizer()
