from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal
import uvicorn
import pandas as pd
import tempfile
import os
import json
from doe_init import generate_sobol_parameters, generate_lhs_parameters, generate_uniform_parameters
from ax_optimizer import BayesianOptimizer, ExperimentResult
from analysis import ParameterOptimizationAnalysis
from __init__ import get_class_from_string

app = FastAPI(
    title="参数优化API v3",
    description="支持先验实验数据的参数优化API，默认sobol采样，支持多种采样方式，新增贝叶斯优化支持",
    version="3.0.0"
)



# 定义参数空间模型
class ParameterSpace(BaseModel):
    name: str = Field(..., description="参数名称")
    type: Literal["choice", "range"] = Field(..., description="参数类型：choice（离散选择）或range（连续范围）")
    values: Union[List[Union[str, int, float]], List[float]] = Field(..., description="当type为choice时表示可选值列表，当type为range时表示[最小值, 最大值]")
    step: Optional[Union[int, float]] = Field(None, description="步长，用于将range参数转换为choice参数")

# 定义先验实验数据模型
class PriorExperiment(BaseModel):
    parameters: Dict[str, Union[str, int, float]] = Field(..., description="参数组合")
    metrics: Dict[str, Union[int, float]] = Field(..., description="实验结果指标")

# 定义初始化请求模型
class InitRequest(BaseModel):
    parameter_space: List[ParameterSpace] = Field(..., description="参数空间定义")
    objectives: List[str] = Field(..., description="优化目标列表")
    batch: int = Field(..., description="每批次参数数量", ge=1, le=20)
    seed: Optional[int] = Field(None, description="随机种子")
    prior_experiments: Optional[List[PriorExperiment]] = Field(None, description="先验实验数据")
    sampling_method: Optional[Literal["sobol", "lhs", "uniform"]] = Field("sobol", description="采样方法（仅在没有先验数据时生效）")

# 定义响应模型
class InitResponse(BaseModel):
    success: bool
    sampling_method: str
    results: List[Dict[str, Any]]
    message: str

# 定义更新请求模型
class UpdateRequest(BaseModel):
    parameter_space: List[ParameterSpace] = Field(..., description="参数空间定义")
    objectives: Dict[str, Dict[str, bool]] = Field(..., description="优化目标配置，格式为{'metric_name': {'minimize': bool}}")
    completed_experiments: List[PriorExperiment] = Field(..., description="已完成的实验结果")
    batch: int = Field(1, description="下一批次参数数量", ge=1, le=10)
    use_weights: Optional[bool] = Field(False, description="是否使用权重优化")
    objective_weights: Optional[Dict[str, float]] = Field(None, description="目标权重")
    additional_metrics: Optional[List[str]] = Field(None, description="额外跟踪指标")
    seed: Optional[int] = Field(None, description="随机种子")
    # 新增自定义模型配置
    surrogate_model_class: Optional[str] = Field(None, description="代理模型类名，如 'SingleTaskGP', 'MultiTaskGP' 等")
    kernel_class: Optional[str] = Field(None, description="核函数类名，如 'MaternKernel', 'RBFKernel' 等")
    kernel_options: Optional[Dict[str, Any]] = Field(None, description="核函数参数，如 {'nu': 2.5} for MaternKernel")
    # 新增采集函数配置
    acquisition_function_class: Optional[str] = Field(None, description="采集函数类名")
    acquisition_function_options: Optional[Dict[str, Any]] = Field(None, description="采集函数参数，如 {'beta': 0.1} for UCB")

# 定义更新响应模型
class UpdateResponse(BaseModel):
    success: bool
    next_parameters: List[Dict[str, Any]]
    message: str

# 定义分析请求模型
class AnalysisRequest(BaseModel):
    parameters: List[str] = Field(..., description="参数列名列表")
    objectives: List[str] = Field(..., description="目标列名列表")
    search_space: List[Dict[str, Any]] = Field(..., description="参数空间配置")
    # 可选的自定义代理模型配置
    surrogate_model_class: Optional[str] = Field(None, description="代理模型类名，如 'SingleTaskGP', 'MultiTaskGP' 等")
    kernel_class: Optional[str] = Field(None, description="核函数类名，如 'MaternKernel', 'RBFKernel' 等")
    kernel_options: Optional[Dict[str, Any]] = Field(None, description="核函数参数，如 {'nu': 2.5} for MaternKernel")

# 定义分析响应模型
class AnalysisResponse(BaseModel):
    success: bool
    message: str
    generated_plots: List[str] = Field(..., description="生成的图表列表")
    output_directory: str = Field(..., description="输出目录路径")
    has_categorical_data: bool = Field(..., description="是否包含类别数据")

def convert_parameter_space_to_ax_format(parameter_space: List[ParameterSpace]) -> List[Dict[str, Any]]:
    """将参数空间转换为Ax格式"""
    search_space = []
    for param in parameter_space:
        if param.type == "choice":
            # 确定value_type：字符串、浮点数、整数
            if all(isinstance(x, str) for x in param.values):
                value_type = "str"
            elif any(isinstance(x, float) for x in param.values):
                value_type = "float"
            else:
                value_type = "int"
            
            # 对于浮点数choice参数，确保值都是浮点数类型
            if value_type == "float":
                values = [float(v) for v in param.values]
            else:
                values = param.values
            
            search_space.append({
                "name": param.name,
                "type": "choice",
                "values": values,
                "value_type": value_type,
                "is_ordered": True,
                "sort_values": True
            })
        else:  # range类型
            if len(param.values) != 2:
                raise ValueError(f"参数 {param.name} 的range类型必须提供[最小值, 最大值]")
            
            # 检查是否有step参数，如果有则转换为choice
            if param.step is not None:
                min_val, max_val = param.values
                step = param.step
                
                # 生成步长序列
                values = []
                current_val = min_val
                while current_val <= max_val:
                    values.append(current_val)
                    current_val += step
                
                # 确保不超过最大值
                if values and values[-1] > max_val:
                    values[-1] = max_val
                
                # 确定value_type
                if any(isinstance(x, float) for x in [min_val, max_val, step]) or any(isinstance(x, float) for x in values):
                    value_type = "float"
                    values = [float(v) for v in values]
                else:
                    value_type = "int"
                    values = [int(v) for v in values]
                
                search_space.append({
                    "name": param.name,
                    "type": "choice",
                    "values": values,
                    "value_type": value_type,
                    "is_ordered": True,
                    "sort_values": True
                })
            else:
                # 没有step参数，保持为range类型
                search_space.append({
                    "name": param.name,
                    "type": "range",
                    "bounds": param.values,
                    "value_type": "float" if any(isinstance(x, float) for x in param.values) else "int"
                })
    return search_space

def convert_prior_experiments_to_dict(prior_experiments: List[PriorExperiment]) -> List[Dict[str, Any]]:
    """将先验实验数据转换为字典格式"""
    experiments = []
    for exp in prior_experiments:
        # 合并参数和指标
        combined_data = {}
        combined_data.update(exp.parameters)
        combined_data.update(exp.metrics)
        experiments.append(combined_data)
    return experiments

def fix_float_precision(params_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """修复浮点数精度问题"""
    fixed_params_list = []
    for params in params_list:
        fixed_params = {}
        for key, value in params.items():
            if isinstance(value, float):
                # 对于浮点数，进行四舍五入到合理的小数位数
                # 如果看起来像是整数（如8.0），则转换为整数
                if abs(value - round(value)) < 1e-10:
                    fixed_params[key] = int(round(value))
                else:
                    # 对于小数，进行更精确的处理
                    # 尝试找到最接近的"干净"的小数值
                    rounded = round(value, 10)
                    # 如果四舍五入后的小数位数很少，直接使用
                    if len(str(rounded).split('.')[-1]) <= 2:
                        fixed_params[key] = rounded
                    else:
                        # 否则尝试找到最接近的简单分数
                        # 例如 0.7999999999999999 -> 0.8
                        for precision in [1, 2, 3, 4, 5]:
                            test_value = round(value, precision)
                            if abs(value - test_value) < 1e-10:
                                fixed_params[key] = test_value
                                break
                        else:
                            # 如果都找不到，使用原始值
                            fixed_params[key] = value
            else:
                fixed_params[key] = value
        fixed_params_list.append(fixed_params)
    return fixed_params_list







@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "参数优化API v3.0",
        "version": "3.0.0",
        "features": {
            "prior_experiments": "支持先验实验数据",
            "sampling_methods": "支持sobol/lhs/uniform三种采样方式",
            "smart_sampling": "有先验数据时默认sobol，无先验数据时可选择采样方式",
            "bayesian_optimization": "支持贝叶斯优化，基于历史数据推荐参数",
            "custom_surrogate_models": "支持自定义代理模型，如 SingleTaskGP, MultiTaskGP 等",
            "custom_kernels": "支持自定义核函数，如 MaternKernel, RBFKernel 等",
            "custom_acquisition_functions": "支持自定义采集函数，包括单目标和多目标采集函数",
            "experiment_analysis": "支持实验数据分析，生成可视化图表"
        },
        "endpoints": {
            "POST /init": "初始化优化，支持传统采样",
            "POST /update": "贝叶斯优化接口，支持自定义代理模型、核函数和采集函数",
            "POST /analysis": "实验数据分析接口，生成可视化图表",
            "GET /available_classes": "获取可用的代理模型、核函数和采集函数列表"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "parameter_optimization_v3"}

@app.get("/available_classes")
async def get_available_classes():
    """获取可用的类列表"""
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from __init__ import get_available_classes, get_class_parameters
        return {
            "categories": get_available_classes(),
            "parameters": get_class_parameters()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取可用类列表失败: {str(e)}")



@app.post("/init", response_model=InitResponse)
async def initialize_optimization(request: InitRequest):
    """初始化优化，支持先验实验数据"""
    try:
        # 转换参数空间格式
        search_space = convert_parameter_space_to_ax_format(request.parameter_space)
        
        # 确定采样方法
        if request.prior_experiments:
            # 有先验数据时，强制使用sobol采样
            sampling_method = "sobol"
            
            # 转换先验实验数据
            prior_experiments = convert_prior_experiments_to_dict(request.prior_experiments)
            
            # 使用sobol采样（带先验数据）
            params_list = generate_sobol_parameters(
                search_space, 
                num_points=request.batch, 
                seed=request.seed,
                prior_experiments=prior_experiments
            )
        else:
            # 没有先验数据时，使用用户指定的采样方法
            sampling_method = request.sampling_method or "sobol"
            
            if sampling_method == "sobol":
                params_list = generate_sobol_parameters(search_space, num_points=request.batch, seed=request.seed)
            elif sampling_method == "lhs":
                params_list = generate_lhs_parameters(search_space, num_points=request.batch, seed=request.seed)
            elif sampling_method == "uniform":
                params_list = generate_uniform_parameters(search_space, num_points=request.batch, seed=request.seed)
            else:
                raise HTTPException(status_code=400, detail=f"不支持的采样方法: {sampling_method}")
        
        # 检查生成的参数列表
        if not params_list:
            raise HTTPException(status_code=500, detail="采样函数未生成任何参数组合")
        
        # 修复浮点数精度问题
        results = fix_float_precision(params_list)
        
        return InitResponse(
            success=True,
            sampling_method=sampling_method,
            results=results,
            message=f"初始化成功，使用{sampling_method}采样生成{len(results)}个参数组合"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")

@app.post("/update", response_model=UpdateResponse)
async def update_optimization(request: UpdateRequest):
    """贝叶斯优化接口：基于历史数据推荐下一组参数"""
    try:
        # 转换参数空间格式
        search_space = convert_parameter_space_to_ax_format(request.parameter_space)
        
        # 构建优化配置
        optimization_config = {
            "objectives": request.objectives,
            "use_weights": request.use_weights or False,
            "objective_weights": request.objective_weights or {},
            "additional_metrics": request.additional_metrics or []
        }
        
        # 创建贝叶斯优化器
        optimizer = BayesianOptimizer(
            search_space=search_space,
            optimization_config=optimization_config,
            random_seed=request.seed,
            surrogate_model_class=request.surrogate_model_class,
            kernel_class=request.kernel_class,
            kernel_options=request.kernel_options,
            acquisition_function_class=request.acquisition_function_class,
            acquisition_function_options=request.acquisition_function_options
        )
        
        # 添加历史实验数据
        for exp in request.completed_experiments:
            experiment_result = ExperimentResult(
                parameters=exp.parameters,
                metrics=exp.metrics
            )
            optimizer.add_prior_experiments([experiment_result])
        
        # 获取下一组推荐参数
        next_trials = optimizer.get_next_parameters(n=request.batch)
        next_parameters = [params for params, _ in next_trials]
        
        # 修复浮点数精度问题
        next_parameters = fix_float_precision(next_parameters)
        
        # 构建响应消息
        message = f"成功推荐{len(next_parameters)}个参数组合"
        
        custom_components = []
        if request.surrogate_model_class:
            custom_components.append("代理模型:"+request.surrogate_model_class)
        if request.kernel_class:
            custom_components.append("核函数:"+request.kernel_class)
        if request.acquisition_function_class:
            custom_components.append("采集函数:"+request.acquisition_function_class)
        
        if custom_components:
            message += f"，使用{'+'.join(custom_components)}"
            
        parameter_info = []
        if request.kernel_options:
            parameter_info.append(f"核函数参数: {request.kernel_options}")
        if request.acquisition_function_options:
            parameter_info.append(f"采集函数参数: {request.acquisition_function_options}")     
        if parameter_info:
            message += f"，{', '.join(parameter_info)}"
        else:
            message += "，使用默认配置"
        
        return UpdateResponse(
            success=True,
            next_parameters=next_parameters,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"优化失败: {str(e)}")

def check_categorical_data(data: pd.DataFrame, parameters: List[str]) -> bool:
    """检查数据中是否包含类别数据"""
    for param in parameters:
        if param in data.columns:
            # 检查是否为非数值类型
            if not pd.api.types.is_numeric_dtype(data[param]):
                return True
            # 检查数值类型但唯一值数量较少（可能是离散数值）
            unique_count = data[param].nunique()
            if unique_count <= 10:  # 如果唯一值数量少于等于10，认为是类别数据
                return True
    return False



@app.post("/analysis", response_model=AnalysisResponse)
async def analyze_experiment_data(
    file: UploadFile = File(..., description="实验数据CSV文件"),
    parameters: str = Field(..., description="参数列名，用逗号分隔"),
    objectives: str = Field(..., description="目标列名，用逗号分隔"),
    search_space: str = Field(..., description="参数空间配置，JSON格式字符串"),
    surrogate_model_class: Optional[str] = Field(None, description="代理模型类名"),
    kernel_class: Optional[str] = Field(None, description="核函数类名"),
    kernel_options: Optional[str] = Field(None, description="核函数参数，JSON格式字符串")
):
    """分析实验数据，生成可视化图表"""
    try:
        # 解析参数
        param_list = [p.strip() for p in parameters.split(',')]
        objective_list = [o.strip() for o in objectives.split(',')]
        
        # 解析搜索空间
        search_space_dict = json.loads(search_space)
        
        # 解析核函数参数
        kernel_options_dict = None
        if kernel_options:
            kernel_options_dict = json.loads(kernel_options)
        
        # 获取模型类（使用项目现有的类获取机制）
        surrogate_model_cls = None
        kernel_cls = None
        if surrogate_model_class:
            surrogate_model_cls = get_class_from_string(surrogate_model_class)
        if kernel_class:
            kernel_cls = get_class_from_string(kernel_class)
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 读取数据
            data = pd.read_csv(tmp_file_path)
            
            # 检查数据中是否包含类别数据
            has_categorical = check_categorical_data(data, param_list)
            
            # 创建输出目录
            output_dir = f"api_analysis_output_{tempfile.mktemp()}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建分析器
            analyzer = ParameterOptimizationAnalysis(
                experiment_file=tmp_file_path,
                output_dir=output_dir
            )
            
            generated_plots = []
            
            # 生成并行坐标图
            print("📊 生成并行坐标图...")
            parallel_plots = analyzer.create_parallel_coordinates_plots(
                parameters=param_list,
                objectives=objective_list
            )
            generated_plots.extend([f"parallel_coords_{obj}" for obj in objective_list])
            
            # 生成特征重要性图
            print("📊 生成特征重要性图...")
            shap_plots = analyzer.create_feature_importance_plots(
                parameters=param_list,
                objectives=objective_list
            )
            generated_plots.extend([f"feature_importance_{obj}" for obj in objective_list])
            
            # 生成交叉验证图
            print("📊 生成交叉验证图...")
            cv_plots = analyzer.create_cross_validation_plots(
                parameters=param_list,
                objectives=objective_list,
                search_space=search_space_dict,
                untransform=True,
                surrogate_model_class=surrogate_model_cls,
                kernel_class=kernel_cls,
                kernel_options=kernel_options_dict
            )
            generated_plots.extend([f"cross_validation_{obj}" for obj in objective_list])
            
            # 如果没有类别数据，生成额外的图表
            if not has_categorical:
                print("📊 生成切片图...")
                slice_plots = analyzer.create_slice_plots(
                    parameters=param_list,
                    objectives=objective_list,
                    search_space=search_space_dict,
                    surrogate_model_class=surrogate_model_cls,
                    kernel_class=kernel_cls,
                    kernel_options=kernel_options_dict
                )
                generated_plots.extend([f"slice_{obj}_{param}" for obj in objective_list for param in param_list])
                
                print("📊 生成等高线图...")
                contour_plots = analyzer.create_contour_plots(
                    parameters=param_list,
                    objectives=objective_list,
                    search_space=search_space_dict,
                    surrogate_model_class=surrogate_model_cls,
                    kernel_class=kernel_cls,
                    kernel_options=kernel_options_dict
                )
                generated_plots.extend([f"contour_{obj}_{param1}_{param2}" for obj in objective_list for param1 in param_list for param2 in param_list if param1 != param2])
            
            # 保存所有图表
            analyzer.save_plots()
            
            # 构建响应消息
            if has_categorical:
                message = f"检测到类别数据，生成了3种图表：并行坐标图、特征重要性图、交叉验证图"
            else:
                message = f"未检测到类别数据，生成了5种图表：并行坐标图、特征重要性图、交叉验证图、切片图、等高线图"
            
            if surrogate_model_class or kernel_class:
                custom_components = []
                if surrogate_model_class:
                    custom_components.append(f"代理模型:{surrogate_model_class}")
                if kernel_class:
                    custom_components.append(f"核函数:{kernel_class}")
                message += f"，使用{'+'.join(custom_components)}"
            
            return AnalysisResponse(
                success=True,
                message=message,
                generated_plots=generated_plots,
                output_directory=output_dir,
                has_categorical_data=has_categorical
            )
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3320)
