"""
参数优化模块的初始化文件
提供类名到类对象的映射功能
"""

import importlib
from typing import Any, Dict, Tuple

# 定义类名到模块的映射
CLASS_MAPPING = {
    # 代理模型
    "SingleTaskGP": ("botorch.models", "SingleTaskGP"),
    "MultiTaskGP": ("botorch.models", "MultiTaskGP"),
    "KroneckerMultiTaskGP": ("botorch.models", "KroneckerMultiTaskGP"),
    "MixedSingleTaskGP": ("botorch.models", "MixedSingleTaskGP"),
    "SingleTaskMultiFidelityGP": ("botorch.models", "SingleTaskMultiFidelityGP"),
    "SaasFullyBayesianSingleTaskGP": ("botorch.models", "SaasFullyBayesianSingleTaskGP"),
    "SaasFullyBayesianMultiTaskGP": ("botorch.models", "SaasFullyBayesianMultiTaskGP"),
    "HigherOrderGP": ("botorch.models", "HigherOrderGP"),
    "SingleTaskVariationalGP": ("botorch.models", "SingleTaskVariationalGP"),
    
    # 核函数
    "RBFKernel": ("gpytorch.kernels", "RBFKernel"),
    "MaternKernel": ("gpytorch.kernels", "MaternKernel"),
    "ScaleKernel": ("gpytorch.kernels", "ScaleKernel"),
    "LinearKernel": ("gpytorch.kernels", "LinearKernel"),
    "PolynomialKernel": ("gpytorch.kernels", "PolynomialKernel"),
    "PeriodicKernel": ("gpytorch.kernels", "PeriodicKernel"),
    "SpectralMixtureKernel": ("gpytorch.kernels", "SpectralMixtureKernel"),
    "RQKernel": ("gpytorch.kernels", "RQKernel"),
    "PiecewisePolynomialKernel": ("gpytorch.kernels", "PiecewisePolynomialKernel"),
    "AdditiveKernel": ("gpytorch.kernels", "AdditiveKernel"),
    "ProductKernel": ("gpytorch.kernels", "ProductKernel"),
    "CosineKernel": ("gpytorch.kernels", "CosineKernel"),
    
    # 单目标采集函数
    "qExpectedImprovement": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    "qNoisyExpectedImprovement": ("botorch.acquisition.monte_carlo", "qNoisyExpectedImprovement"),
    "qUpperConfidenceBound": ("botorch.acquisition.monte_carlo", "qUpperConfidenceBound"),
    "qKnowledgeGradient": ("botorch.acquisition.knowledge_gradient", "qKnowledgeGradient"),
    "qLogExpectedImprovement": ("botorch.acquisition.logei", "qLogExpectedImprovement"),
    "qLogNoisyExpectedImprovement": ("botorch.acquisition.logei", "qLogNoisyExpectedImprovement"),
    "qMaxValueEntropy": ("botorch.acquisition.max_value_entropy_search", "qMaxValueEntropy"),
    "ExpectedImprovement": ("botorch.acquisition.analytic", "ExpectedImprovement"),
    "UpperConfidenceBound": ("botorch.acquisition.analytic", "UpperConfidenceBound"),
    "PosteriorMean": ("botorch.acquisition.analytic", "PosteriorMean"),
    
    # 多目标采集函数
    "qExpectedHypervolumeImprovement": ("botorch.acquisition.multi_objective.monte_carlo", "qExpectedHypervolumeImprovement"),
    "qNoisyExpectedHypervolumeImprovement": ("botorch.acquisition.multi_objective.monte_carlo", "qNoisyExpectedHypervolumeImprovement"),
    "qLogExpectedHypervolumeImprovement": ("botorch.acquisition.multi_objective.logei", "qLogExpectedHypervolumeImprovement"),
    "qLogNoisyExpectedHypervolumeImprovement": ("botorch.acquisition.multi_objective.logei", "qLogNoisyExpectedHypervolumeImprovement"),
    "qLogNParEGO": ("botorch.acquisition.multi_objective.parego", "qLogNParEGO"),
}

def get_class_from_string(class_name: str) -> Any:
    """
    根据字符串名称获取类对象
    
    Args:
        class_name: 类名字符串
        
    Returns:
        对应的类对象
        
    Raises:
        ValueError: 当类名不支持时抛出异常
    """
    if class_name not in CLASS_MAPPING:
        available_classes = list(CLASS_MAPPING.keys())
        raise ValueError(f"不支持的类名: {class_name}. 可用的类: {available_classes}")
    
    module_name, class_name_in_module = CLASS_MAPPING[class_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name_in_module)

def get_available_classes() -> Dict[str, list]:
    """
    获取可用的类列表，按类别分组
    
    Returns:
        包含各类别可用类的字典
    """
    categories = {
        "surrogate_models": [],
        "kernels": [],
        "single_objective_acquisition": [],
        "multi_objective_acquisition": []
    }
    
    # 代理模型
    surrogate_models = ["SingleTaskGP", "MultiTaskGP", "KroneckerMultiTaskGP", 
                       "MixedSingleTaskGP", "SingleTaskMultiFidelityGP",
                       "SaasFullyBayesianSingleTaskGP", "SaasFullyBayesianMultiTaskGP",
                       "HigherOrderGP", "SingleTaskVariationalGP"]
    
    # 核函数
    kernels = ["RBFKernel", "MaternKernel", "ScaleKernel", "LinearKernel",
               "PolynomialKernel", "PeriodicKernel", "SpectralMixtureKernel",
               "RQKernel", "PiecewisePolynomialKernel", "AdditiveKernel",
               "ProductKernel", "CosineKernel"]
    
    # 单目标采集函数
    single_objective = ["qExpectedImprovement", "qNoisyExpectedImprovement", 
                       "qUpperConfidenceBound", "qKnowledgeGradient",
                       "qLogExpectedImprovement", "qLogNoisyExpectedImprovement",
                       "qMaxValueEntropy", "ExpectedImprovement", 
                       "UpperConfidenceBound", "PosteriorMean"]
    
    # 多目标采集函数
    multi_objective = ["qExpectedHypervolumeImprovement", "qNoisyExpectedHypervolumeImprovement",
                      "qLogExpectedHypervolumeImprovement", "qLogNoisyExpectedHypervolumeImprovement",
                      "qLogNParEGO"]
    
    categories["surrogate_models"] = [cls for cls in surrogate_models if cls in CLASS_MAPPING]
    categories["kernels"] = [cls for cls in kernels if cls in CLASS_MAPPING]
    categories["single_objective_acquisition"] = [cls for cls in single_objective if cls in CLASS_MAPPING]
    categories["multi_objective_acquisition"] = [cls for cls in multi_objective if cls in CLASS_MAPPING]
    
    return categories

def get_class_parameters() -> Dict[str, Dict[str, str]]:
    """
    获取各类的常用参数说明
    
    Returns:
        包含各类参数说明的字典
    """
    return {
        "qUpperConfidenceBound": {"beta": "探索权重，默认0.2"},
        "UpperConfidenceBound": {"beta": "探索权重"},
        "qExpectedImprovement": {"eta": "约束平滑度，默认1e-3"},
        "qNoisyExpectedImprovement": {"eta": "约束平滑度，默认1e-3"},
        "qKnowledgeGradient": {"num_fantasies": "幻想样本数，默认64"},
        "qMaxValueEntropy": {"num_mv_samples": "最大值样本数，默认10"},
        "MaternKernel": {"nu": "平滑度参数，常用值：0.5, 1.5, 2.5"},
        "RBFKernel": {"lengthscale": "长度尺度参数"},
        "PolynomialKernel": {"power": "多项式幂次", "offset": "偏移量"},
        "PeriodicKernel": {"period": "周期长度", "lengthscale": "长度尺度"}
    }
