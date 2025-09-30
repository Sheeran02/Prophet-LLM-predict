#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基线方法实现，包括数据加载、模型训练和预测等功能
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(train_file='train_data.csv', test_file='test_data.csv'):
    """
    加载数据
    
    参数:
    train_file (str): 训练集文件名
    test_file (str): 测试集文件名
    
    返回:
    DataFrame: 包含所有数据的DataFrame
    """
    # 读取训练集和测试集文件
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # 转换时间戳列
    train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
    test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
    
    # 合并数据
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 按时间排序
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    return df

def select_test_timestamps(df, num_tests=50, train_ratio=0.8):
    """
    选择测试时间点，并划分训练集和验证集
    
    参数:
    df: 数据DataFrame
    num_tests: 测试样本数量
    train_ratio: 训练集比例
    
    返回:
    tuple: (测试时间点列表, 训练数据DataFrame, 验证数据DataFrame)
    """
    # 先划分训练集和验证集
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:].copy()
    
    # 只选择验证集中Total_kW > 0的数据作为测试集
    non_zero_val_df = val_df[val_df['Total_kW'] > 0].copy()
    
    # 按照一定间隔选择测试点
    step = len(non_zero_val_df) // num_tests
    if step == 0:
        step = 1
    
    test_indices = range(0, len(non_zero_val_df), step)[:num_tests]
    test_df = non_zero_val_df.iloc[test_indices]
    
    test_timestamps = test_df['Timestamp'].tolist()
    
    return test_timestamps, train_df, val_df

def train_prophet_baseline(df):
    """
    训练纯Prophet基线模型
    
    参数:
    df: 训练数据DataFrame
    
    返回:
    Prophet模型
    """
    from Prophet import load_and_prepare_data, train_prophet_model
    
    # 准备数据
    prophet_df = load_and_prepare_data_from_df(df)
    
    # 训练模型
    model = train_prophet_model(prophet_df)
    
    return model

def load_and_prepare_data_from_df(df, target_column='Total_kW', exclude_zeros=False):
    """
    从DataFrame准备Prophet模型所需的数据格式
    
    参数:
    df: 输入DataFrame
    target_column: 目标列名
    exclude_zeros: 是否排除零值
    
    返回:
    为Prophet准备好的DataFrame
    """
    # 如果需要，过滤掉零值
    if exclude_zeros:
        original_shape = df.shape[0]
        df = df[df[target_column] > 0]
        filtered_shape = df.shape[0]
        print(f"过滤掉 {original_shape - filtered_shape} 行 {target_column} = 0 的数据")
    
    # 为Prophet准备数据
    prophet_df = df[['Timestamp', target_column]].copy()
    prophet_df.columns = ['ds', 'y']
    
    # 确保ds是datetime类型
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    return prophet_df

def predict_with_prophet(model, timestamps):
    """
    使用Prophet模型进行预测
    
    参数:
    model: Prophet模型
    timestamps: 时间戳列表
    
    返回:
    预测结果字典
    """
    from Prophet import predict_for_timestamps
    
    # 进行预测
    forecast = predict_for_timestamps(model, timestamps)
    
    # 返回第一个预测结果
    result = {
        "timestamp": str(forecast['ds'].iloc[0]),
        "predicted_total_kw": round(forecast['yhat'].iloc[0], 2),
        "confidence_interval": {
            "lower": round(forecast['yhat_lower'].iloc[0], 2),
            "upper": round(forecast['yhat_upper'].iloc[0], 2)
        }
    }
    
    return result

def prophet_with_env_params(df, timestamps, temp=None, humidity=None, wet_bulb_temp=None):
    """
    使用带环境参数的Prophet模型进行预测
    
    参数:
    df: 训练数据DataFrame
    timestamps: 时间戳列表
    temp: 温度
    humidity: 湿度
    wet_bulb_temp: 湿球温度
    
    返回:
    预测结果字典
    """
    # 对于这个简化版本，我们直接使用普通Prophet预测
    # 在更复杂的实现中，可以添加环境参数作为额外回归因子
    model = train_prophet_baseline(df)
    return predict_with_prophet(model, timestamps)

def mean_absolute_percentage_error(y_true, y_pred):
    """
    计算平均绝对百分比误差(MAPE)
    
    参数:
    y_true: 真实值
    y_pred: 预测值
    
    返回:
    MAPE值
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零的情况
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_predictions(actual_values, predicted_values):
    """
    评估预测结果，计算MAE、RMSE和MAPE指标
    
    参数:
    actual_values: 真实值列表
    predicted_values: 预测值列表
    
    返回:
    包含评估指标的字典
    """
    # 确保输入是numpy数组
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    
    # 确保两个数组长度相同
    min_len = min(len(actual_values), len(predicted_values))
    actual_values = actual_values[:min_len]
    predicted_values = predicted_values[:min_len]
    
    # 计算MAE
    mae = mean_absolute_error(actual_values, predicted_values)
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    
    # 计算MAPE
    mape = mean_absolute_percentage_error(actual_values, predicted_values)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
