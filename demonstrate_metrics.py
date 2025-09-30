#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
演示如何计算MAE、RMSE和MAPE三个评估指标
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

def main():
    """
    主函数：演示评估指标的计算
    """
    # 示例数据
    actual_values = [100, 150, 200, 250, 300]
    predicted_values = [90, 160, 190, 240, 310]
    
    print("=== 评估指标计算演示 ===")
    print(f"实际值: {actual_values}")
    print(f"预测值: {predicted_values}")
    
    # 计算评估指标
    metrics = evaluate_predictions(actual_values, predicted_values)
    
    print("\n=== 评估结果 ===")
    print(f"MAE (平均绝对误差): {metrics['MAE']:.2f}")
    print(f"RMSE (均方根误差): {metrics['RMSE']:.2f}")
    print(f"MAPE (平均绝对百分比误差): {metrics['MAPE']:.2f}%")
    
    # 解释每个指标的含义
    print("\n=== 指标解释 ===")
    print("MAE (Mean Absolute Error): 平均绝对误差，表示预测值与真实值之间平均相差多少。值越小表示预测越准确。")
    print("RMSE (Root Mean Square Error): 均方根误差，对大误差更加敏感。值越小表示预测越准确。")
    print("MAPE (Mean Absolute Percentage Error): 平均绝对百分比误差，表示预测误差相对于真实值的百分比。值越小表示预测越准确。")

if __name__ == "__main__":
    main()
