# 参数优化 API - 基于 Ax 框架的贝叶斯优化

一个基于 Ax 框架的贝叶斯参数优化 API，支持自定义代理模型、核函数和采集函数，适用于各种机器学习超参数优化和实验设计场景。

## 🚀 主要特性

- **多种初始化采样方式**: 支持 Sobol、LHS、Uniform 采样
- **贝叶斯优化**: 基于历史数据的智能参数推荐
- **自定义代理模型**: 支持 SingleTaskGP、MultiTaskGP 等多种高斯过程模型
- **自定义核函数**: 支持 MaternKernel、RBFKernel 等多种核函数
- **自定义采集函数**: 支持单目标和多目标优化的各种采集函数
- **先验数据支持**: 可以集成历史实验数据
- **多目标优化**: 支持帕累托优化和权重优化
- **实验数据分析**: 生成多种可视化图表（并行坐标图、特征重要性图、交叉验证图、切片图、等高线图）
- **RESTful API**: 提供简洁的 HTTP 接口

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/LibraxModel/parameters_optimization_ax.git
cd parameters_optimization_ax

# 创建 conda 环境
conda create -n ax_env python=3.12
conda activate ax_env

# 安装依赖
pip install -r requirements.txt

# 或者手动安装核心依赖
# pip install ax-platform botorch gpytorch fastapi uvicorn pandas numpy
```

## 🏃‍♂️ 快速开始

### 启动 API 服务器

```bash
python api_parameter_optimizer_v3.py
```

服务器将在 `http://localhost:3320` 启动。

### 基础使用示例

#### 1. 初始化优化（传统采样）

```python
import requests

# 初始化请求
init_request = {
    "parameter_space": [
        {
            "name": "learning_rate",
            "type": "range",
            "values": [0.001, 0.1]
        },
        {
            "name": "batch_size",
            "type": "choice",
            "values": [32, 64, 128, 256]
        }
    ],
    "objectives": ["accuracy"],
    "batch": 5,
    "seed": 42,
    "sampling_method": "sobol"
}

response = requests.post("http://localhost:3320/init", json=init_request)
print(response.json())
```

#### 2. 贝叶斯优化（自定义配置）

```python
# 贝叶斯优化请求
update_request = {
    "parameter_space": [
        {
            "name": "learning_rate",
            "type": "range",
            "values": [0.001, 0.1]
        },
        {
            "name": "batch_size",
            "type": "choice",
            "values": [32, 64, 128, 256]
        }
    ],
    "objectives": {
        "accuracy": {"minimize": False},
        "training_time": {"minimize": True}
    },
    "completed_experiments": [
        {
            "parameters": {
                "learning_rate": 0.01,
                "batch_size": 64
            },
            "metrics": {
                "accuracy": 0.85,
                "training_time": 120
            }
        }
    ],
    "batch": 3,
    "seed": 42,
    "surrogate_model_class": "SingleTaskGP",
    "kernel_class": "MaternKernel",
    "kernel_options": {"nu": 2.5},
    "acquisition_function_class": "qExpectedHypervolumeImprovement",
    "acquisition_function_options": {}
}

response = requests.post("http://localhost:3320/update", json=update_request)
print(response.json())
```

## 📚 API 接口文档

### 基础端点

#### GET `/`
获取 API 信息和使用说明

#### GET `/health`
健康检查端点

#### GET `/available_classes`
获取所有可用的代理模型、核函数和采集函数列表

### 核心接口

#### POST `/init`
初始化优化，使用传统采样方法

**请求参数:**
- `parameter_space`: 参数空间定义
- `objectives`: 优化目标列表
- `batch`: 每批次参数数量
- `seed`: 随机种子（可选）
- `prior_experiments`: 先验实验数据（可选）
- `sampling_method`: 采样方法（"sobol", "lhs", "uniform"）

#### POST `/update`
贝叶斯优化接口，基于历史数据推荐参数

**请求参数:**
- `parameter_space`: 参数空间定义
- `objectives`: 优化目标配置
- `completed_experiments`: 已完成的实验结果
- `batch`: 下一批次参数数量
- `use_weights`: 是否使用权重优化（可选）
- `objective_weights`: 目标权重（可选）
- `additional_metrics`: 额外跟踪指标（可选）
- `seed`: 随机种子（可选）
- `surrogate_model_class`: 代理模型类名（可选）
- `kernel_class`: 核函数类名（可选）
- `kernel_options`: 核函数参数（可选）
- `acquisition_function_class`: 采集函数类名（可选）
- `acquisition_function_options`: 采集函数参数（可选）

#### POST `/analysis`
实验数据分析接口，生成基础可视化图表

**请求参数:**
- `file`: 实验数据CSV文件
- `parameters`: 参数列名，用逗号分隔
- `objectives`: 目标列名，用逗号分隔
- `parameter_space`: 参数空间配置，JSON格式字符串
- `surrogate_model_class`: 代理模型类名（可选）
- `kernel_class`: 核函数类名（可选）
- `kernel_options`: 核函数参数，JSON格式字符串（可选）

**生成图表:**
- 并行坐标图（1个）
- 特征重要性图（每个目标1个）
- 交叉验证图（每个目标1个）

#### POST `/analysis/slice`
生成单个切片图，展示指定参数对指定目标的影响

**请求参数:**
- `file`: 实验数据CSV文件
- `parameter`: 要分析的参数名称
- `objective`: 要分析的目标名称
- `parameter_space`: 参数空间配置，JSON格式字符串
- `surrogate_model_class`: 代理模型类名（可选）
- `kernel_class`: 核函数类名（可选）
- `kernel_options`: 核函数参数，JSON格式字符串（可选）

**返回:**
- 单个切片图的查看链接
- 只生成用户指定的参数图表

#### POST `/analysis/contour`
生成单个等高线图，展示指定参数对组合对指定目标的影响

**请求参数:**
- `file`: 实验数据CSV文件
- `parameter1`: 第一个参数名称
- `parameter2`: 第二个参数名称
- `objective`: 要分析的目标名称
- `parameter_space`: 参数空间配置，JSON格式字符串
- `surrogate_model_class`: 代理模型类名（可选）
- `kernel_class`: 核函数类名（可选）
- `kernel_options`: 核函数参数，JSON格式字符串（可选）

**返回:**
- 单个等高线图的查看链接
- 只生成用户指定的参数对图表

#### GET `/chart/{file_id}`
查看生成的图表（在浏览器中渲染）

**参数:**
- `file_id`: 图表文件ID（从分析接口返回）

**返回:**
- HTML格式的图表内容，可直接在浏览器中查看

## 🔧 可配置组件详解

### 代理模型 (Surrogate Models)

| 模型名称 | 描述 | 适用场景 |
|---------|------|----------|
| `SingleTaskGP` | 单任务高斯过程 | 单目标优化，Ax 默认推荐 |
| `MultiTaskGP` | 多任务高斯过程 | 多个相关任务，需要任务特征 |
| `KroneckerMultiTaskGP` | Kronecker 结构多任务 GP | 结构化多任务，计算效率高 |
| `MixedSingleTaskGP` | 混合变量类型 GP | 同时包含连续和分类变量 |
| `SingleTaskMultiFidelityGP` | 多保真度单任务 GP | 有多个评估精度级别 |
| `SaasFullyBayesianSingleTaskGP` | 全贝叶斯单任务 GP | 高维问题，需要特征选择 |
| `SaasFullyBayesianMultiTaskGP` | 全贝叶斯多任务 GP | 高维多任务问题 |
| `HigherOrderGP` | 高阶高斯过程 | 存在复杂变量交互的问题 |
| `SingleTaskVariationalGP` | 变分推断 GP | 大规模数据集优化 |

### 核函数 (Kernels)

| 核函数名称 | 描述 | 参数 | 适用场景 |
|-----------|------|------|----------|
| `RBFKernel` | 径向基函数核（高斯核） | `lengthscale` | 光滑函数，大多数工程问题 |
| `MaternKernel` | Matérn 核 | `nu` (0.5, 1.5, 2.5) | 不同平滑度需求，工程优化常用 |
| `LinearKernel` | 线性核 | `variance` (0.1,0.5,1.0,2.0) | 线性或近似线性问题 |
| `PolynomialKernel` | 多项式核 | `power` (1,2,3,4) | 多项式关系的问题 |
| `PeriodicKernel` | 周期核 | `period`, `lengthscale` | 具有周期性的优化问题 |
| `SpectralMixtureKernel` | 谱混合核 | `num_mixtures` | 复杂的频域特征 |
| `RQKernel` | 有理二次核 | `alpha`, `lengthscale` | 中等复杂度的平滑函数 |
| `CosineKernel` | 余弦核 | `period` | 余弦型周期模式 |
| `ScaleKernel` | 缩放核 | `base_kernel`, `outputscale` | 需要调整输出尺度的情况 |
| `AdditiveKernel` | 加性核 | `kern1`, `kern2` | 需要组合不同类型相关性 |
| `ProductKernel` | 乘积核 | `kern1`, `kern2` | 需要核函数乘积的场景 |

### 采集函数 (Acquisition Functions)

#### 单目标采集函数

| 采集函数名称 | 描述 | 参数 | 适用场景 |
|-------------|------|------|----------|
| `qExpectedImprovement` | 期望改进（批量版本） | `eta` (约束平滑度，默认1e-3) | 单目标优化，均衡的探索-开发策略 |
| `qNoisyExpectedImprovement` | 噪声期望改进 | `eta` (约束平滑度，默认1e-3) | 单目标优化，存在观测噪声 |
| `qUpperConfidenceBound` | 上置信界 | `beta` (探索权重，默认0.2) | 单目标优化，需要控制探索-开发平衡 |
| `qKnowledgeGradient` | 知识梯度 | `num_fantasies` (幻想样本数，默认64) | 单目标优化，重视信息获取 |
| `qLogExpectedImprovement` | 对数期望改进 | 无特殊参数 | 单目标优化，数值稳定性更好 |
| `qMaxValueEntropy` | 最大值熵搜索 | `num_mv_samples` (最大值样本数，默认10) | 单目标优化，高效的全局搜索 |
| `ExpectedImprovement` | 经典期望改进（解析版本） | 无特殊参数 | 单目标优化，计算高效 |
| `UpperConfidenceBound` | 经典上置信界（解析版本） | `beta` (探索权重) | 单目标优化，计算高效 |
| `PosteriorMean` | 后验均值 | 无特殊参数 | 单目标优化，纯开发策略 |

#### 多目标采集函数

| 采集函数名称 | 描述 | 参数 | 适用场景 |
|-------------|------|------|----------|
| `qExpectedHypervolumeImprovement` | 期望超体积改进 | `ref_point` (参考点，可选) | 多目标优化，直接优化帕累托前沿 |
| `qNoisyExpectedHypervolumeImprovement` | 噪声期望超体积改进 | `ref_point` (参考点，可选) | 多目标优化，存在观测噪声 |
| `qLogExpectedHypervolumeImprovement` | 对数期望超体积改进 | `ref_point` (参考点，可选) | 多目标优化，数值稳定（推荐） |
| `qLogNoisyExpectedHypervolumeImprovement` | 对数噪声期望超体积改进 | `ref_point` (参考点，可选) | 多目标优化，数值稳定 |
| `qLogNParEGO` | ParEGO 的对数版本 | 无特殊参数 | 多目标优化，计算资源有限时 |

## 🎯 使用示例

### 示例 1: 机器学习超参数优化

```python
import requests

# 定义超参数空间
parameter_space = [
    {
        "name": "learning_rate",
        "type": "range",
        "values": [0.0001, 0.1]
    },
    {
        "name": "batch_size",
        "type": "choice",
        "values": [16, 32, 64, 128]
    },
    {
        "name": "dropout",
        "type": "range",
        "values": [0.1, 0.5]
    },
    {
        "name": "optimizer",
        "type": "choice",
        "values": ["adam", "sgd", "rmsprop"]
    }
]

# 贝叶斯优化配置
optimization_request = {
    "parameter_space": parameter_space,
    "objectives": {
        "validation_accuracy": {"minimize": False},
        "training_time": {"minimize": True}
    },
    "completed_experiments": [
        {
            "parameters": {
                "learning_rate": 0.01,
                "batch_size": 32,
                "dropout": 0.2,
                "optimizer": "adam"
            },
            "metrics": {
                "validation_accuracy": 0.85,
                "training_time": 120
            }
        }
    ],
    "batch": 3,
    "seed": 42,
    "surrogate_model_class": "SingleTaskGP",
    "kernel_class": "MaternKernel",
    "kernel_options": {"nu": 2.5},
    "acquisition_function_class": "qLogExpectedHypervolumeImprovement"
}

response = requests.post("http://localhost:3320/update", json=optimization_request)
print(response.json())
```

### 示例 2: 化学反应条件优化

```python
# 化学反应优化
reaction_optimization = {
    "parameter_space": [
        {
            "name": "temperature",
            "type": "range",
            "values": [25, 100]
        },
        {
            "name": "pressure",
            "type": "range",
            "values": [1, 10]
        },
        {
            "name": "catalyst",
            "type": "choice",
            "values": ["A", "B", "C", "D"]
        },
        {
            "name": "reaction_time",
            "type": "range",
            "values": [30, 180],
            "step": 15
        }
    ],
    "objectives": {
        "yield": {"minimize": False},
        "purity": {"minimize": False},
        "cost": {"minimize": True}
    },
    "completed_experiments": [
        {
            "parameters": {
                "temperature": 50,
                "pressure": 5,
                "catalyst": "A",
                "reaction_time": 60
            },
            "metrics": {
                "yield": 75,
                "purity": 85,
                "cost": 100
            }
        }
    ],
    "batch": 2,
    "surrogate_model_class": "MultiTaskGP",
    "kernel_class": "RBFKernel",
    "acquisition_function_class": "qExpectedHypervolumeImprovement"
}

response = requests.post("http://localhost:3320/update", json=reaction_optimization)
print(response.json())
```

### 示例 3: 高探索性配置

```python
# 高探索性配置
exploration_config = {
    "parameter_space": [
        {
            "name": "x",
            "type": "range",
            "values": [-5, 5]
        },
        {
            "name": "y",
            "type": "range",
            "values": [-5, 5]
        }
    ],
    "objectives": {
        "objective": {"minimize": True}
    },
    "completed_experiments": [
        {
            "parameters": {"x": 0, "y": 0},
            "metrics": {"objective": 1.0}
        }
    ],
    "batch": 3,
    "kernel_class": "MaternKernel",
    "kernel_options": {"nu": 0.5},  # 低平滑度，增加探索
    "acquisition_function_class": "qUpperConfidenceBound",
    "acquisition_function_options": {"beta": 0.5}  # 高探索权重
}

response = requests.post("http://localhost:3320/update", json=exploration_config)
print(response.json())
```

### 示例 4: 实验数据分析

#### 基础分析（生成所有基础图表）

```python
import requests
import json

# 准备分析请求
analysis_request = {
    'file': open('experiment_data.csv', 'rb'),
    'parameters': 'solvent,catalyst,temperature,concentration',
    'objectives': 'yield,side_product',
    'parameter_space': json.dumps([
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
            "values": [-10, 40]
        },
        {
            "name": "concentration",
            "type": "range",
            "values": [0.1, 1.0]
        }
    ]),
    'surrogate_model_class': 'SingleTaskGP',
    'kernel_class': 'MaternKernel',
    'kernel_options': json.dumps({"nu": 2.5})
}

# 发送分析请求
files = {'file': analysis_request['file']}
data = {k: v for k, v in analysis_request.items() if k != 'file'}

response = requests.post('http://localhost:3320/analysis', files=files, data=data)
result = response.json()

print(f"分析结果: {result['message']}")
print(f"生成的图表: {result['generated_plots']}")
print(f"查看链接: {result['view_links']}")
```

#### 生成单个切片图

```python
# 生成温度对产率的切片图
slice_request = {
    'file': open('experiment_data.csv', 'rb'),
    'parameter': 'temperature',
    'objective': 'yield',
    'parameter_space': json.dumps([
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
            "values": [-10, 40]
        },
        {
            "name": "concentration",
            "type": "range",
            "values": [0.1, 1.0]
        }
    ]),
    'surrogate_model_class': 'SingleTaskGP',
    'kernel_class': 'MaternKernel',
    'kernel_options': json.dumps({"nu": 2.5})
}

files = {'file': slice_request['file']}
data = {k: v for k, v in slice_request.items() if k != 'file'}

response = requests.post('http://localhost:3320/analysis/slice', files=files, data=data)
result = response.json()

print(f"切片图生成结果: {result['message']}")
print(f"图表名称: {result['plot_name']}")
print(f"查看链接: {result['view_link']['url']}")
```

#### 生成单个等高线图

```python
# 生成温度和浓度对产率的等高线图
contour_request = {
    'file': open('experiment_data.csv', 'rb'),
    'parameter1': 'temperature',
    'parameter2': 'concentration',
    'objective': 'yield',
    'parameter_space': json.dumps([
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
            "values": [-10, 40]
        },
        {
            "name": "concentration",
            "type": "range",
            "values": [0.1, 1.0]
        }
    ]),
    'surrogate_model_class': 'SingleTaskGP',
    'kernel_class': 'MaternKernel',
    'kernel_options': json.dumps({"nu": 2.5})
}

files = {'file': contour_request['file']}
data = {k: v for k, v in contour_request.items() if k != 'file'}

response = requests.post('http://localhost:3320/analysis/contour', files=files, data=data)
result = response.json()

print(f"等高线图生成结果: {result['message']}")
print(f"图表名称: {result['plot_name']}")
print(f"查看链接: {result['view_link']['url']}")
```

#### 查看图表

```python
# 获取图表查看链接后，可以直接在浏览器中打开
chart_url = f"http://localhost:3320{result['view_link']['url']}"
print(f"在浏览器中打开: {chart_url}")

# 或者使用requests获取图表内容
response = requests.get(chart_url)
if response.status_code == 200:
    # 保存为HTML文件
    with open('chart.html', 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("图表已保存为 chart.html")
```

## 🔍 常用配置组合推荐

### 1. 单目标通用优化
```python
{
    "surrogate_model_class": "SingleTaskGP",
    "kernel_class": "MaternKernel",
    "kernel_options": {"nu": 2.5},
    "acquisition_function_class": "qExpectedImprovement"
}
```

### 2. 单目标噪声环境
```python
{
    "surrogate_model_class": "SingleTaskGP",
    "kernel_class": "RBFKernel",
    "acquisition_function_class": "qNoisyExpectedImprovement"
}
```

### 3. 单目标探索重点
```python
{
    "surrogate_model_class": "SingleTaskGP",
    "kernel_class": "MaternKernel",
    "acquisition_function_class": "qUpperConfidenceBound",
    "acquisition_function_options": {"beta": 0.1}
}
```

### 4. 多目标优化
```python
{
    "surrogate_model_class": "SingleTaskGP",
    "kernel_class": "MaternKernel",
    "acquisition_function_class": "qLogExpectedHypervolumeImprovement"
}
```

### 5. 高维稀疏问题
```python
{
    "surrogate_model_class": "SaasFullyBayesianSingleTaskGP",
    "kernel_class": "MaternKernel",
    "acquisition_function_class": "qLogExpectedImprovement"
}
```

### 6. 实验数据分析（基础图表）
```python
{
    "surrogate_model_class": "SingleTaskGP",
    "kernel_class": "MaternKernel",
    "kernel_options": {"nu": 2.5}
}
```

### 7. 切片图和等高线图生成
```python
{
    "surrogate_model_class": "SingleTaskGP",
    "kernel_class": "RBFKernel"
}
```

## 📊 性能优化建议

### 贝叶斯优化配置
1. **数据量较少时** (< 20 个实验): 使用 `MaternKernel` 和 `qExpectedImprovement`
2. **数据量中等时** (20-100 个实验): 使用 `RBFKernel` 和 `qNoisyExpectedImprovement`
3. **数据量较大时** (> 100 个实验): 使用 `SingleTaskVariationalGP` 和 `qLogExpectedImprovement`
4. **高维问题** (> 10 个参数): 使用 `SaasFullyBayesianSingleTaskGP`
5. **多目标优化**: 优先使用 `qLogExpectedHypervolumeImprovement`

### 图表生成优化
1. **切片图生成**: 只生成需要的参数，避免不必要的计算
2. **等高线图生成**: 使用 `RBFKernel` 获得更平滑的等高线
3. **交叉验证图**: 包含详细hover信息，便于分析模型表现
4. **参数类型判断**: 基于参数空间配置而非数据统计，更准确
5. **图表查看**: 使用 `/chart/{file_id}` 接口直接查看HTML图表

### 服务器配置
1. **单进程模式**: 开发测试时使用 `python api_parameter_optimizer_v3.py`
2. **多进程模式**: 生产环境使用 `uvicorn api_parameter_optimizer_v3:app --host 0.0.0.0 --port 3320 --workers 4`
3. **内存优化**: 大量图表生成时注意清理临时文件

## 🆕 新功能特性

### 细粒度图表生成
- **按需生成**: 只生成用户指定的图表，提高效率
- **参数精确控制**: 基于参数空间配置判断参数类型，更准确
- **独立接口**: 切片图和等高线图有独立的API接口

### 增强的交互体验
- **详细hover信息**: 交叉验证图显示每个点的完整参数信息
- **直接查看**: 图表可直接在浏览器中查看，无需下载
- **实时生成**: 图表按需生成，减少存储空间

### 智能参数处理
- **类型智能判断**: 基于参数空间配置而非数据统计
- **中位数/众数固定**: 切片图中其他参数使用统计值固定
- **缓存优化**: Ax优化器缓存机制，避免重复重建

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请通过 GitHub Issues 联系。
