# 参数优化API分析功能说明

## 概述

基于Ax官方文档和analysis模块，我们为参数优化API添加了强大的实验分析功能。这些功能可以帮助用户深入理解优化过程，评估模型性能，并生成可视化的分析报告。

## 新增的分析接口

### 1. 生成分析报告
```
GET /analysis/{uuid}
```

**功能**: 为指定的实验生成完整的分析报告，包括图片和表格

**参数**:
- `uuid`: 实验的唯一标识符

**返回**: 
```json
{
  "success": true,
  "uuid": "实验UUID",
  "analysis_dir": "分析文件目录路径",
  "files": [
    {
      "type": "cross_validation",
      "html": "cross_validation.html",
      "png": "cross_validation.png",
      "description": "模型交叉验证图，显示预测值与实际值的对比"
    },
    {
      "type": "scatter_plot",
      "html": "scatter_plot.html", 
      "png": "scatter_plot.png",
      "description": "目标accuracy与latency的散点图"
    },
    {
      "type": "experiment_summary",
      "csv": "experiment_summary.csv",
      "excel": "experiment_summary.xlsx",
      "description": "实验摘要表"
    },
    {
      "type": "optimization_history",
      "csv": "optimization_history.csv",
      "excel": "optimization_history.xlsx", 
      "description": "优化历史数据表"
    },
    {
      "type": "parameter_importance",
      "json": "parameter_importance.json",
      "description": "参数重要性分析和最佳参数"
    }
  ],
  "summary": {
    "total_trials": 6,
    "completed_trials": 6,
    "failed_trials": 0,
    "running_trials": 0,
    "columns": ["trial_index", "arm_name", "trial_status", "accuracy", "latency", ...],
    "csv": "experiment_summary.csv",
    "excel": "experiment_summary.xlsx"
  },
  "message": "实验分析报告已生成，包含 5 个分析文件"
}
```

### 2. 列出分析文件
```
GET /analysis/{uuid}/files
```

**功能**: 列出指定实验的所有分析文件

**返回**:
```json
{
  "success": true,
  "uuid": "实验UUID",
  "analysis_dir": "分析文件目录路径",
  "total_files": 8,
  "files": [
    {
      "filename": "cross_validation.html",
      "size_bytes": 245760,
      "size_mb": 0.23,
      "modified": "2024-01-15 10:30:45",
      "type": "html"
    },
    {
      "filename": "cross_validation.png",
      "size_bytes": 156432,
      "size_mb": 0.15,
      "modified": "2024-01-15 10:30:45", 
      "type": "png"
    }
  ]
}
```

### 3. 下载分析文件
```
GET /analysis/{uuid}/download/{filename}
```

**功能**: 获取指定分析文件的下载信息

**参数**:
- `uuid`: 实验UUID
- `filename`: 文件名

**返回**:
```json
{
  "success": true,
  "uuid": "实验UUID",
  "filename": "cross_validation.html",
  "file_path": "/path/to/file",
  "size_bytes": 245760,
  "size_mb": 0.23,
  "message": "文件 cross_validation.html 已准备下载"
}
```

## 生成的分析内容

### 1. 交叉验证图 (Cross Validation Plot)
- **文件**: `cross_validation.html`, `cross_validation.png`
- **描述**: 评估模型预测质量，显示预测值与实际值的对比
- **用途**: 判断模型是否过拟合或欠拟合
- **Ax模块**: `CrossValidationPlot`

### 2. 散点图 (Scatter Plot)
- **文件**: `scatter_plot.html`, `scatter_plot.png`
- **描述**: 显示两个目标指标之间的关系
- **用途**: 分析目标之间的权衡关系，识别帕累托前沿
- **Ax模块**: `ScatterPlot`
- **条件**: 仅在有多个目标时生成

### 3. 实验摘要表 (Experiment Summary)
- **文件**: `experiment_summary.csv`, `experiment_summary.xlsx`
- **描述**: 实验的总体概览，包含所有试验的详细信息
- **内容**: 试验索引、参数值、指标值、试验状态等
- **Ax模块**: `Summary`

### 4. 优化历史数据表 (Optimization History)
- **文件**: `optimization_history.csv`, `optimization_history.xlsx`
- **描述**: 完整的优化过程历史记录
- **内容**: 所有试验的详细数据，包括参数、指标、时间戳等

### 5. 参数重要性分析 (Parameter Importance)
- **文件**: `parameter_importance.json`
- **描述**: 最佳参数配置和实验统计信息
- **内容**: 
  - 最佳参数组合
  - 对应的指标值
  - 参数空间定义
  - 实验总数

### 6. 分析元数据 (Analysis Metadata)
- **文件**: `analysis_metadata.json`
- **描述**: 分析报告的元数据信息
- **内容**: 生成时间、文件列表、摘要统计等

## 文件存储结构

```
data/
├── {uuid}.json                    # 实验状态文件
├── {uuid}_history.json           # 历史实验结果
└── {uuid}_analysis/              # 分析结果目录
    ├── analysis_metadata.json    # 分析元数据
    ├── cross_validation.html     # 交叉验证图(HTML)
    ├── cross_validation.png      # 交叉验证图(PNG)
    ├── scatter_plot.html         # 散点图(HTML)
    ├── scatter_plot.png          # 散点图(PNG)
    ├── experiment_summary.csv    # 实验摘要(CSV)
    ├── experiment_summary.xlsx   # 实验摘要(Excel)
    ├── optimization_history.csv  # 优化历史(CSV)
    ├── optimization_history.xlsx # 优化历史(Excel)
    └── parameter_importance.json # 参数重要性分析
```

## 使用示例

### 1. 完整的分析流程

```bash
# 1. 初始化实验
curl -X POST "http://localhost:3004/init" \
  -H "Content-Type: application/json" \
  -d '{
    "round": 1,
    "batch": 3,
    "objectives": ["accuracy", "latency"],
    "objective_mode": ["max", "min"],
    "uuid": "test-uuid",
    "parameter_space": [...]
  }'

# 2. 更新实验结果
curl -X POST "http://localhost:3004/update" \
  -H "Content-Type: application/json" \
  -d '{
    "round": 1,
    "batch": 3,
    "objectives": ["accuracy", "latency"],
    "objective_mode": ["max", "min"],
    "uuid": "test-uuid",
    "results": [...]
  }'

# 3. 生成分析报告
curl -X GET "http://localhost:3004/analysis/test-uuid"

# 4. 查看分析文件列表
curl -X GET "http://localhost:3004/analysis/test-uuid/files"
```

### 2. 使用测试脚本

```bash
# 运行完整的测试
./test_analysis_api.sh
```

## 技术实现

### 基于Ax官方模块

1. **CrossValidationPlot**: 交叉验证分析
2. **ScatterPlot**: 散点图分析  
3. **Summary**: 实验摘要分析
4. **PlotlyAnalysis**: 图表生成基础

### 文件格式支持

- **HTML**: 交互式图表，可在浏览器中查看
- **PNG**: 静态图片，适合报告和文档
- **CSV**: 结构化数据，适合数据分析
- **Excel**: 表格数据，适合人工查看
- **JSON**: 结构化数据，适合程序处理

### 错误处理

- 实验不存在时返回404错误
- 数据不足时返回400错误
- 分析生成失败时记录警告日志但不中断流程
- 每个分析组件独立处理，单个失败不影响其他分析

## 依赖要求和安装

### 必需的依赖包

```bash
# 基础依赖
pip install plotly pandas numpy

# Excel文件支持
pip install openpyxl

# PNG图片生成支持
pip install kaleido
```

### 可选依赖

- **Chrome浏览器**: 用于生成PNG图片，如果不可用，只会生成HTML文件
- **Torch/Botorch**: 用于敏感性分析，如果不可用，敏感性分析会跳过

### 快速安装

运行安装脚本：
```bash
chmod +x install_analysis_dependencies.sh
./install_analysis_dependencies.sh
```

### 验证安装

```bash
python check_analysis_dependencies.py
```

## 注意事项

1. **数据要求**: 需要至少有一些历史实验结果才能生成分析
2. **文件大小**: 图片文件可能较大，建议定期清理
3. **权限**: 确保API有写入文件系统的权限
4. **Chrome支持**: 如果没有Chrome，PNG图片生成会失败，但HTML文件仍可正常生成
5. **性能**: 分析生成可能需要一些时间，特别是数据量大的时候
6. **错误处理**: 单个分析组件失败不会影响其他分析功能

## 扩展功能

未来可以考虑添加的分析功能：

1. **敏感性分析**: 分析参数对目标的影响程度
2. **收敛性分析**: 评估优化过程的收敛性
3. **置信区间**: 显示预测的置信区间
4. **参数相关性**: 分析参数之间的相关性
5. **自定义图表**: 支持用户自定义的分析图表 