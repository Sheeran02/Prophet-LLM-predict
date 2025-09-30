# Prophet-LLM-predict
# 基于多模态大模型的建筑能源预测系统

## 项目简介

本项目实现了一个基于多模态大模型的建筑能源预测系统，旨在提高建筑能源预测的准确性。系统结合了时间序列数据、建筑结构信息、环境参数和历史工况数据，利用大语言模型进行多源信息融合和预测校准。

## 核心功能

### 1. 数据预处理
- 加载和清洗建筑能源数据
- 按时间顺序划分训练集和测试集
- 处理环境参数（温度、湿度、湿球温度等）

### 2. Prophet基线模型
- 实现Facebook Prophet时间序列预测模型
- 集成环境参数作为额外回归因子
- 提供基线预测结果用于对比

### 3. 多模态大模型模块
- 使用Qwen-VL-Max分析建筑平面图，提取建筑结构信息
- 识别建筑功能区域、机械设备和被动式节能设计
- 利用Qwen-Max进行能耗预测校准

### 4. RAG检索模块
- 将历史工况数据向量化存储
- 基于当前环境参数检索相似历史工况
- 提供参考信息用于预测校准

### 5. 大语言模型推理
- 设计专业Prompt指导大语言模型进行预测
- 融合多源信息生成校准后的能耗预测
- 输出预测值、置信区间和调整原因

### 6. 实验运行与评估
- 实现完整的实验流程
- 提供基线方法和完整系统的对比实验
- 使用MAE、RMSE、MAPE等指标评估预测性能

## 项目结构

```
.
├── baseline_methods.py          # 基线方法实现
├── data_preprocessing.py        # 数据预处理脚本
├── demonstrate_metrics.py       # 评估指标演示
├── experiment_runner.py         # 实验运行器
├── main.py                      # 主程序入口
├── Prophet.py                   # Prophet模型实现
├── qwen.py                      # Qwen模型调用
├── rag_utils.py                 # RAG工具函数
├── run_model_comparison.py      # 大模型选型对比实验
├── simple_model_comparison.py   # 简化版模型对比实验
├── test_metrics.py              # 评估指标测试
├── train_data.csv              # 训练数据
├── test_data.csv               # 测试数据
└── cleaned_202310_wetbulb_filled.xlsx  # 原始数据文件
```

## 安装依赖

```bash
pip install pandas numpy scikit-learn prophet dashscope openai langchain langchain-openai
```

## 使用方法

### 1. 数据预处理
```bash
python data_preprocessing.py
```

### 2. 运行完整实验
```bash
python experiment_runner.py
```

### 3. 大模型选型对比实验
```bash
python run_model_comparison.py
```


## 许可证

本项目仅供学术研究使用。
