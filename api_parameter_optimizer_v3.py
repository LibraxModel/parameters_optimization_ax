from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from starlette.responses import Response
from starlette.types import Scope
from starlette.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal, Tuple
import uvicorn
import pandas as pd
import tempfile
import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from doe_init import generate_sobol_parameters, generate_lhs_parameters, generate_uniform_parameters
from ax_optimizer import BayesianOptimizer, ExperimentResult
from analysis import ParameterOptimizationAnalysis
from __init__ import get_class_from_string

class CacheControlledStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "public, max-age=184600, immutable"
        return response

app = FastAPI(
    title="参数优化API v3",
    description="支持先验实验数据的参数优化API，默认sobol采样，支持多种采样方式，新增贝叶斯优化支持",
    version="3.0.0"
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.mount("/static", CacheControlledStaticFiles(directory=Path("static"), html=True), name="static")


# 持久化存储配置 - 使用当前工作目录，兼容非root用户和Windows
PERSISTENT_OUTPUT_DIR = Path.cwd() / "analysis_outputs"
CHART_METADATA_DIR = PERSISTENT_OUTPUT_DIR / "chart_metadata"
# 确保持久化目录存在
PERSISTENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_METADATA_DIR.mkdir(parents=True, exist_ok=True)

# 旧的加载函数已移除，现在使用小文件存储

def save_single_chart_metadata(chart_id: str, metadata: Dict[str, Any]) -> bool:
    """保存单个图表的元数据到小文件"""
    try:
        metadata_file = CHART_METADATA_DIR / f"{chart_id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ 保存图表元数据失败 {chart_id}: {e}")
        return False

def load_single_chart_metadata(chart_id: str) -> Optional[Dict[str, Any]]:
    """加载单个图表的元数据"""
    try:
        metadata_file = CHART_METADATA_DIR / f"{chart_id}.json"
        if not metadata_file.exists():
            return None
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载图表元数据失败 {chart_id}: {e}")
        return None

def list_all_chart_metadata() -> Dict[str, Dict[str, Any]]:
    """列出所有图表的元数据"""
    charts = {}
    try:
        for metadata_file in CHART_METADATA_DIR.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    chart_id = metadata_file.stem  # 文件名作为chart_id
                    charts[chart_id] = metadata
            except Exception as e:
                print(f"⚠️ 读取元数据文件失败 {metadata_file}: {e}")
                continue
    except Exception as e:
        print(f"❌ 遍历元数据目录失败: {e}")
    return charts

# 旧的保存函数已移除，现在直接使用小文件存储

def cleanup_expired_files(days=30):
    """清理超过指定天数的过期文件"""
    try:
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 清理过期的图表文件
        expired_count = 0
        for metadata_file in CHART_METADATA_DIR.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                created_at_str = metadata.get('created_at', '')
                if created_at_str:
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at < cutoff_date:
                        # 删除图表文件
                        file_path = metadata.get('path', '')
                        if file_path and os.path.exists(file_path):
                            try:
                                os.unlink(file_path)
                                print(f"🗑️ 删除过期文件: {file_path}")
                            except Exception as e:
                                print(f"⚠️ 删除文件失败 {file_path}: {e}")
                        
                        # 删除元数据文件
                        metadata_file.unlink()
                        expired_count += 1
                        
            except Exception as e:
                print(f"⚠️ 处理文件失败 {metadata_file}: {e}")
                continue
        
        if expired_count > 0:
            print(f"✅ 清理了 {expired_count} 个过期文件")
        
        # 清理空的输出目录
        for output_dir in PERSISTENT_OUTPUT_DIR.iterdir():
            if output_dir.is_dir() and not any(output_dir.iterdir()):
                try:
                    output_dir.rmdir()
                    print(f"🗑️ 删除空目录: {output_dir}")
                except Exception as e:
                    print(f"⚠️ 删除目录失败 {output_dir}: {e}")
                    
    except Exception as e:
        print(f"⚠️ 清理过期文件失败: {e}")

# 启动时清理过期文件（可选）
# cleanup_expired_files(days=30)



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
    batch: int = Field(..., description="每批次参数数量", ge=1)
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
    batch: int = Field(1, description="下一批次参数数量", ge=1)
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
    results: List[Dict[str, Any]]
    message: str

# 定义分析请求模型
class AnalysisRequest(BaseModel):
    parameter_space: List[ParameterSpace] = Field(..., description="参数空间定义")
    objectives: Union[List[str], Dict[str, Dict[str, bool]]] = Field(..., description="优化目标配置，可以是字符串列表或字典格式")
    completed_experiments: List[PriorExperiment] = Field(..., description="已完成的实验结果")
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
    view_links: List[Dict[str, str]] = Field(default=[], description="图表查看链接列表，包含name、url、type字段")

# 定义切片图请求模型
class SliceAnalysisRequest(BaseModel):
    parameter_space: List[ParameterSpace] = Field(..., description="参数空间定义")
    objectives: Union[List[str], Dict[str, Dict[str, bool]]] = Field(..., description="优化目标配置，可以是字符串列表或字典格式")
    completed_experiments: List[PriorExperiment] = Field(..., description="已完成的实验结果")
    parameter: str = Field(..., description="要分析的参数名称")
    objective: str = Field(..., description="要分析的目标名称")
    surrogate_model_class: Optional[str] = Field(None, description="代理模型类名")
    kernel_class: Optional[str] = Field(None, description="核函数类名")
    kernel_options: Optional[Dict[str, Any]] = Field(None, description="核函数参数")

# 定义等高线图请求模型
class ContourAnalysisRequest(BaseModel):
    parameter_space: List[ParameterSpace] = Field(..., description="参数空间定义")
    objectives: Union[List[str], Dict[str, Dict[str, bool]]] = Field(..., description="优化目标配置，可以是字符串列表或字典格式")
    completed_experiments: List[PriorExperiment] = Field(..., description="已完成的实验结果")
    parameter1: str = Field(..., description="第一个参数名称")
    parameter2: str = Field(..., description="第二个参数名称")
    objective: str = Field(..., description="要分析的目标名称")
    surrogate_model_class: Optional[str] = Field(None, description="代理模型类名")
    kernel_class: Optional[str] = Field(None, description="核函数类名")
    kernel_options: Optional[Dict[str, Any]] = Field(None, description="核函数参数")

# 定义单个图表响应模型
class SinglePlotResponse(BaseModel):
    success: bool
    message: str
    plot_name: str = Field(..., description="生成的图表名称")
    view_link: Dict[str, str] = Field(..., description="图表查看链接，包含name、url、type字段")

def convert_parameter_space_to_ax_format(parameter_space: List[ParameterSpace]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """将参数空间转换为Ax格式，返回(search_space, step_info)"""
    search_space = []
    step_info = {}  # 存储step信息，用于后续舍入
    
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
            
            # 保持为range类型，无论是否有step参数
            # step参数将在后处理中用于舍入推荐值
            param_config = {
                "name": param.name,
                "type": "range",
                "bounds": param.values,
                "value_type": "float" if any(isinstance(x, float) for x in param.values) else "int"
            }
            
            # 如果有step参数，存储到step_info中，而不是添加到Ax配置中
            if param.step is not None:
                step_info[param.name] = {
                    "step": param.step,
                    "bounds": param.values
                }
            
            search_space.append(param_config)
    return search_space, step_info

def convert_parameter_space_to_ax_format_for_analysis(parameter_space: List[ParameterSpace]) -> List[Dict[str, Any]]:
    """将参数空间转换为Ax格式（专门用于analysis接口，忽略step参数）"""
    search_space = []
    for param in parameter_space:
        if param.type == "choice":
            # 确定value_type：字符串、浮点数、整数
            if all(isinstance(x, str) for x in param.values):
                value_type = "str"
                values = param.values
            else:
                # 对于数值类型，统一使用float以避免类型转换问题
                value_type = "float"
                values = [float(v) for v in param.values]
            
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
            
            # 在analysis接口中，忽略step参数，始终保持为range类型
            # 为了兼容CSV数据中的numpy.float64类型，统一使用float类型
            value_type = "float"
            
            # 确保bounds中的值也是浮点数类型
            float_bounds = [float(val) for val in param.values]
            
            search_space.append({
                "name": param.name,
                "type": "range",
                "bounds": float_bounds,
                "value_type": value_type
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

def round_to_step(value: float, step: float, bounds: List[float]) -> float:
    """
    将值舍入到最近的step值
    
    Args:
        value: 原始值
        step: 步长
        bounds: [min, max] 边界
        
    Returns:
        舍入后的值
    """
    min_val, max_val = bounds
    
    # 计算最接近的step值
    num_steps = round((value - min_val) / step)
    rounded_value = min_val + num_steps * step
    
    # 确保在边界内
    rounded_value = max(min_val, min(max_val, rounded_value))
    
    # 计算小数位数
    decimal_places = 0
    if isinstance(step, float):
        step_str = f"{step:.10f}".rstrip('0').rstrip('.')
        if '.' in step_str:
            decimal_places = len(step_str.split('.')[1])
    
    # 使用相同的小数位数进行舍入
    rounded_value = round(rounded_value, decimal_places)
    
    return rounded_value

def apply_step_constraints(params_list: List[Dict[str, Any]], step_info: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    应用step约束，将推荐值舍入到最近的step值
    
    Args:
        params_list: 推荐的参数列表
        step_info: 包含step信息的字典
        
    Returns:
        应用step约束后的参数列表
    """
    # 应用step约束
    constrained_params_list = []
    for params in params_list:
        constrained_params = {}
        for key, value in params.items():
            if key in step_info and isinstance(value, (int, float)):
                # 应用step舍入
                info = step_info[key]
                constrained_value = round_to_step(value, info["step"], info["bounds"])
                constrained_params[key] = constrained_value
            else:
                constrained_params[key] = value
        constrained_params_list.append(constrained_params)
    
    return constrained_params_list

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
            "POST /analysis": "实验数据分析接口，生成并行坐标图、特征重要性图、交叉验证图",
            "POST /analysis/slice": "生成单个切片图，用户指定参数和目标",
            "POST /analysis/contour": "生成单个等高线图，用户指定参数对和目标",
            "GET /available_classes": "获取可用的代理模型、核函数和采集函数列表"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "parameter_optimization_v3"}

@app.get("/chart/{file_id}", response_class=HTMLResponse)
async def view_chart(file_id: str):
    """查看图表（在浏览器中渲染）"""
    # 使用新的小文件存储方式，直接读取单个图表的元数据
    chart_metadata = load_single_chart_metadata(file_id)
    
    if not chart_metadata:
        raise HTTPException(status_code=404, detail="图表不存在")
    
    file_path = chart_metadata.get("path", "")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="图表文件已被删除")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        response = HTMLResponse(content=html_content)
        response.headers["Access-Control-Allow-Origin"] = "*"  # 允许所有域访问
        response.headers["Access-Control-Allow-Methods"] = "GET"  # 允许的HTTP方法
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"  # 允许的请求头
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取图表文件失败: {str(e)}")

@app.get("/charts")
async def list_charts():
    """列出所有可用的图表文件"""
    try:
        # 使用新的小文件存储方式，遍历所有小文件
        current_chart_files = list_all_chart_metadata()
        
        charts_info = []
        for file_id, file_info in current_chart_files.items():
            chart_info = {
                "file_id": file_id,
                "filename": file_info.get("filename", "unknown"),
                "type": file_info.get("type", "unknown"),
                "created_at": file_info.get("created_at", "unknown"),
                "url": f"/chart/{file_id}",
                "exists": os.path.exists(file_info.get("path", ""))
            }
            charts_info.append(chart_info)
        
        # 按创建时间排序（最新的在前）
        charts_info.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return {
            "success": True,
            "total_charts": len(charts_info),
            "charts": charts_info,
            "output_directory": str(PERSISTENT_OUTPUT_DIR)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图表列表失败: {str(e)}")

@app.delete("/charts/{file_id}")
async def delete_chart(file_id: str):
    """删除指定的图表文件"""
    # 使用新的小文件存储方式
    chart_metadata = load_single_chart_metadata(file_id)
    if not chart_metadata:
        raise HTTPException(status_code=404, detail="图表不存在")
    
    try:
        file_path = chart_metadata.get("path", "")
        
        # 删除实际文件
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
        
        # 删除元数据文件
        metadata_file = CHART_METADATA_DIR / f"{file_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        return {
            "success": True,
            "message": f"成功删除图表文件: {chart_metadata.get('filename', 'unknown')}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除图表文件失败: {str(e)}")

@app.post("/charts/cleanup")
async def cleanup_charts(days: int = 30):
    """清理超过指定天数的过期图表文件"""
    try:
        cleanup_expired_files(days)
        return {
            "success": True,
            "message": f"清理完成，删除了超过 {days} 天的过期文件"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理文件失败: {str(e)}")




@app.post("/init", response_model=InitResponse)
async def initialize_optimization(request: InitRequest):
    """初始化优化，支持先验实验数据"""
    try:
        # 转换参数空间格式
        search_space, step_info = convert_parameter_space_to_ax_format(request.parameter_space)
        
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
        
        # 应用step约束（如果有）
        results = apply_step_constraints(params_list, step_info)
        
        # 修复浮点数精度问题
        results = fix_float_precision(results)
        
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
        search_space, step_info = convert_parameter_space_to_ax_format(request.parameter_space)
        
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
        
        # 应用step约束（如果有）
        next_parameters = apply_step_constraints(next_parameters, step_info)
        
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
            results=next_parameters,
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
async def analyze_experiment_data(request: AnalysisRequest):
    """分析实验数据，生成可视化图表"""
    try:
        # 从请求中提取参数和目标
        param_list = [param.name for param in request.parameter_space]
        
        # 处理两种格式的objectives
        if isinstance(request.objectives, list):
            # 如果是字符串列表格式
            objective_list = request.objectives
        else:
            # 如果是字典格式
            objective_list = list(request.objectives.keys())
        
        # 转换参数空间为Ax格式
        search_space_dict, _ = convert_parameter_space_to_ax_format(request.parameter_space)
        
        # 获取模型类
        surrogate_model_cls = None
        kernel_cls = None
        if request.surrogate_model_class:
            surrogate_model_cls = get_class_from_string(request.surrogate_model_class)
        if request.kernel_class:
            kernel_cls = get_class_from_string(request.kernel_class)
        
        # 将实验数据转换为DataFrame格式（用于检查类别数据）
        data_rows = []
        for exp in request.completed_experiments:
            row = {}
            # 添加参数
            for param_name, param_value in exp.parameters.items():
                row[param_name] = param_value
            # 添加目标
            for obj_name, obj_value in exp.metrics.items():
                row[obj_name] = obj_value
            data_rows.append(row)
        
        data = pd.DataFrame(data_rows)
        
        # 检查数据中是否包含类别数据
        has_categorical = check_categorical_data(data, param_list)
        
        # 创建持久化输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PERSISTENT_OUTPUT_DIR / f"analysis_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 将实验数据转换为字典格式
        completed_experiments_dict = []
        for exp in request.completed_experiments:
            exp_dict = {
                "parameters": exp.parameters,
                "metrics": exp.metrics
            }
            completed_experiments_dict.append(exp_dict)
        
        # 创建分析器，直接使用JSON数据
        analyzer = ParameterOptimizationAnalysis(
            completed_experiments=completed_experiments_dict,
            output_dir=output_dir
        )
        
        generated_plots = []
        view_links = []  # 存储查看链接信息
        
        # 生成并行坐标图
        print("📊 生成并行坐标图...")
        parallel_plots = analyzer.create_parallel_coordinates_plots(
            parameters=param_list,
            objectives=objective_list
        )
        # 立即保存并行坐标图
        if "parallel_coords_combined" in analyzer.plots:
            saved_path = analyzer.save_single_plot("parallel_coords_combined", analyzer.plots["parallel_coords_combined"])
            
            # 生成唯一的文件ID
            file_id = str(uuid.uuid4())
            filename = f"parallel_coords_combined.html"
            
            # 存储文件映射到小文件
            metadata = {
                "path": str(saved_path),
                "filename": filename,
                "type": "parallel_coordinates",
                "created_at": datetime.now().isoformat()
            }
            save_single_chart_metadata(file_id, metadata)
            
            # 添加查看链接
            view_links.append({
                "name": "parallel_coords_combined",
                "url": f"/chart/{file_id}",
                "type": "parallel_coordinates"
            })
            
        generated_plots.append("parallel_coords_combined")
        
        # 生成特征重要性图
        print("📊 生成特征重要性图...")
        shap_plots = analyzer.create_feature_importance_analysis(
            parameters=param_list,
            objectives=objective_list
        )
        # 立即保存特征重要性图
        for obj in objective_list:
            plot_name = f"feature_importance_{obj}"
            if plot_name in analyzer.plots:
                saved_path = analyzer.save_single_plot(plot_name, analyzer.plots[plot_name])
                
                # 生成唯一的文件ID
                file_id = str(uuid.uuid4())
                filename = f"{plot_name}.html"
                
                # 存储文件映射到小文件
                metadata = {
                    "path": str(saved_path),
                    "filename": filename,
                    "type": "feature_importance",
                    "created_at": datetime.now().isoformat()
                }
                save_single_chart_metadata(file_id, metadata)
                
                # 添加查看链接
                view_links.append({
                    "name": plot_name,
                    "url": f"/chart/{file_id}",
                    "type": "feature_importance"
                })
                
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
            kernel_options=request.kernel_options
        )
        # 立即保存交叉验证图
        for obj in objective_list:
            plot_name = f"cross_validation_{obj}"
            if plot_name in analyzer.plots:
                saved_path = analyzer.save_single_plot(plot_name, analyzer.plots[plot_name])
                
                # 生成唯一的文件ID
                file_id = str(uuid.uuid4())
                filename = f"{plot_name}.html"
                
                # 存储文件映射到小文件
                metadata = {
                    "path": str(saved_path),
                    "filename": filename,
                    "type": "cross_validation",
                    "created_at": datetime.now().isoformat()
                }
                save_single_chart_metadata(file_id, metadata)
                
                # 添加查看链接
                view_links.append({
                    "name": plot_name,
                    "url": f"/chart/{file_id}",
                    "type": "cross_validation"
                })
                
        generated_plots.extend([f"cross_validation_{obj}" for obj in objective_list])
        
        # 注意：slice图和contour图已移至单独的接口
        # 使用 /analysis/slice 接口生成单个切片图
        # 使用 /analysis/contour 接口生成单个等高线图
        
        # 构建响应消息
        plot_count = len(generated_plots)
        message = f"生成了3种类型共{plot_count}个图表：并行坐标图（1个）、特征重要性图（{len(objective_list)}个）、交叉验证图（{len(objective_list)}个）"
        
        if request.surrogate_model_class or request.kernel_class:
            custom_components = []
            if request.surrogate_model_class:
                custom_components.append(f"代理模型:{request.surrogate_model_class}")
            if request.kernel_class:
                custom_components.append(f"核函数:{request.kernel_class}")
            message += f"，使用{'+'.join(custom_components)}"
        
        return AnalysisResponse(
            success=True,
            message=message,
            generated_plots=generated_plots,
            output_directory=str(output_dir),
            has_categorical_data=has_categorical,
            view_links=view_links
        )
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

@app.post("/analysis/slice", response_model=SinglePlotResponse)
async def analyze_slice_plot(request: SliceAnalysisRequest):
    """生成单个切片图"""
    try:
        # 转换参数空间为Ax格式（分析接口忽略step参数）
        search_space_dict = convert_parameter_space_to_ax_format_for_analysis(request.parameter_space)
        
        # 获取模型类
        surrogate_model_cls = None
        kernel_cls = None
        if request.surrogate_model_class:
            surrogate_model_cls = get_class_from_string(request.surrogate_model_class)
        if request.kernel_class:
            kernel_cls = get_class_from_string(request.kernel_class)
        
        # 创建持久化输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PERSISTENT_OUTPUT_DIR / f"slice_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 将实验数据转换为字典格式
        completed_experiments_dict = []
        for exp in request.completed_experiments:
            exp_dict = {
                "parameters": exp.parameters,
                "metrics": exp.metrics
            }
            completed_experiments_dict.append(exp_dict)
        
        # 创建分析器，直接使用JSON数据
        analyzer = ParameterOptimizationAnalysis(
            completed_experiments=completed_experiments_dict,
            output_dir=output_dir
        )
        
        # 生成单个切片图
        print(f"📊 生成切片图: {request.parameter} vs {request.objective}")
        # 传入完整的参数列表构建优化器，但只生成用户指定参数的切片图
        all_parameters = [param["name"] for param in search_space_dict]
        slice_plots = analyzer.create_slice_plots(
            parameters=all_parameters,  # 传入所有参数，确保优化器有完整数据
            objectives=[request.objective],
            search_space=search_space_dict,
            surrogate_model_class=surrogate_model_cls,
            kernel_class=kernel_cls,
            kernel_options=request.kernel_options,
            target_parameters=[request.parameter],  # 只生成用户指定参数的切片图
            target_objectives=[request.objective]   # 只生成用户指定目标的切片图
        )
        
        # 使用JSON数组格式命名：slice_["目标","参数"]
        plot_name = f'slice_["{request.objective}","{request.parameter}"]'
        # 检查是否生成了用户指定的切片图
        if plot_name in analyzer.plots:
            # 保存图表并获取实际保存路径
            # 注：analysis.py内部已保存一次，这里再次保存以确保获得准确的文件路径
            saved_path = analyzer.save_single_plot(plot_name, analyzer.plots[plot_name])
            
            # 生成唯一的文件ID
            file_id = str(uuid.uuid4())
            filename = f"{plot_name}.html"
            
            # 存储文件映射到小文件
            metadata = {
                "path": str(saved_path),
                "filename": filename,
                "type": "slice",
                "created_at": datetime.now().isoformat()
            }
            save_single_chart_metadata(file_id, metadata)
            
            # 构建查看链接
            view_link = {
                "name": plot_name,
                "url": f"/chart/{file_id}",
                "type": "slice"
            }
            
            # 构建响应消息
            message = f"成功生成切片图: {request.parameter} vs {request.objective}"
            if request.surrogate_model_class or request.kernel_class:
                custom_components = []
                if request.surrogate_model_class:
                    custom_components.append(f"代理模型:{request.surrogate_model_class}")
                if request.kernel_class:
                    custom_components.append(f"核函数:{request.kernel_class}")
                message += f"，使用{'+'.join(custom_components)}"
            
            return SinglePlotResponse(
                success=True,
                message=message,
                plot_name=plot_name,
                view_link=view_link
            )
        else:
            return SinglePlotResponse(
                success=False,
                message=f"未能生成指定的切片图: {request.parameter} vs {request.objective}",
                plot_name="",
                view_link={}
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"切片图分析失败: {str(e)}")

@app.post("/analysis/contour", response_model=SinglePlotResponse)
async def analyze_contour_plot(request: ContourAnalysisRequest):
    """生成单个等高线图"""
    try:
        # 验证参数
        if request.parameter1 == request.parameter2:
            raise HTTPException(status_code=400, detail="两个参数不能相同")
        
        # 转换参数空间为Ax格式（分析接口忽略step参数）
        search_space_dict = convert_parameter_space_to_ax_format_for_analysis(request.parameter_space)
        
        # 获取模型类
        surrogate_model_cls = None
        kernel_cls = None
        if request.surrogate_model_class:
            surrogate_model_cls = get_class_from_string(request.surrogate_model_class)
        if request.kernel_class:
            kernel_cls = get_class_from_string(request.kernel_class)
        
        # 创建持久化输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PERSISTENT_OUTPUT_DIR / f"contour_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 将实验数据转换为字典格式
        completed_experiments_dict = []
        for exp in request.completed_experiments:
            exp_dict = {
                "parameters": exp.parameters,
                "metrics": exp.metrics
            }
            completed_experiments_dict.append(exp_dict)
        
        # 创建分析器，直接使用JSON数据
        analyzer = ParameterOptimizationAnalysis(
            completed_experiments=completed_experiments_dict,
            output_dir=output_dir
        )
        
        # 生成单个等高线图
        print(f"📊 生成等高线图: {request.parameter1} vs {request.parameter2} for {request.objective}")
        # 传入完整的参数列表构建优化器，但只生成用户指定参数对的等高线图
        all_parameters = [param["name"] for param in search_space_dict]
        contour_plots = analyzer.create_contour_plots(
            parameters=all_parameters,  # 传入所有参数，确保优化器有完整数据
            objectives=[request.objective],
            search_space=search_space_dict,
            surrogate_model_class=surrogate_model_cls,
            kernel_class=kernel_cls,
            kernel_options=request.kernel_options,
            target_parameter_pairs=[(request.parameter1, request.parameter2)],  # 只生成用户指定参数对的等高线图
            target_objectives=[request.objective]  # 只生成用户指定目标的等高线图
        )
        
        # 使用JSON数组格式命名：contour_["目标","参数1","参数2"]
        plot_name = f'contour_["{request.objective}","{request.parameter1}","{request.parameter2}"]'
        # 检查是否生成了用户指定的等高线图
        if plot_name in analyzer.plots:
            # 保存图表并获取实际保存路径
            # 注：analysis.py内部已保存一次，这里再次保存以确保获得准确的文件路径
            saved_path = analyzer.save_single_plot(plot_name, analyzer.plots[plot_name])
            
            # 生成唯一的文件ID
            file_id = str(uuid.uuid4())
            filename = f"{plot_name}.html"
            
            # 存储文件映射到小文件
            metadata = {
                "path": str(saved_path),
                "filename": filename,
                "type": "contour",
                "created_at": datetime.now().isoformat()
            }
            save_single_chart_metadata(file_id, metadata)
            
            # 构建查看链接
            view_link = {
                "name": plot_name,
                "url": f"/chart/{file_id}",
                "type": "contour"
            }
            
            # 构建响应消息
            message = f"成功生成等高线图: {request.parameter1} vs {request.parameter2} for {request.objective}"
            if request.surrogate_model_class or request.kernel_class:
                custom_components = []
                if request.surrogate_model_class:
                    custom_components.append(f"代理模型:{request.surrogate_model_class}")
                if request.kernel_class:
                    custom_components.append(f"核函数:{request.kernel_class}")
                message += f"，使用{'+'.join(custom_components)}"
            
            return SinglePlotResponse(
                success=True,
                message=message,
                plot_name=plot_name,
                view_link=view_link
            )
        else:
            return SinglePlotResponse(
                success=False,
                message=f"未能生成指定的等高线图: {request.parameter1} vs {request.parameter2} for {request.objective}",
                plot_name="",
                view_link={}
            )
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ 等高线图分析失败详细错误:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"等高线图分析失败: {str(e)}")

if __name__ == "__main__":
    # 单进程启动（用于开发测试）
    # 生产环境建议使用命令行启动多进程：uvicorn api_parameter_optimizer_v3:app --host 0.0.0.0 --port 3320 --workers 4
    print("🚀 启动API服务（单进程模式）")
    print("💡 提示：生产环境建议使用以下命令启动多进程：")
    print("   uvicorn api_parameter_optimizer_v3:app --host 0.0.0.0 --port 3320 --workers 4")
    uvicorn.run(app, host="0.0.0.0", port=3320)
