#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试评估指标计算函数的正确性
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from demonstrate_metrics import evaluate_predictions, mean_absolute_percentage_error

def test_mean_absolute_percentage_error():
    """
    测试MAPE计算函数
    """
    print("=== 测试MAPE计算函数 ===")
    
    # 测试用例1: 正常情况
    y_true = [100, 200, 300]
    y_pred = [90, 210, 310]
    expected_mape = 6.11  # 手动计算的结果
    
    calculated_mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"测试用例1 - 正常情况:")
    print(f"  真实值: {y_true}")
    print(f"  预测值: {y_pred}")
    print(f"  期望MAPE: {expected_mape:.2f}%")
    print(f"  计算MAPE: {calculated_mape:.2f}%")
    print(f"  结果: {'通过' if abs(calculated_mape - expected_mape) < 0.01 else '失败'}")
    
    # 测试用例2: 包含零值
    y_true = [0, 200, 300]
    y_pred = [10, 210, 310]
    expected_mape = 4.17  # 手动计算的结果（忽略零值）
    
    calculated_mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"\n测试用例2 - 包含零值:")
    print(f"  真实值: {y_true}")
    print(f"  预测值: {y_pred}")
    print(f"  期望MAPE: {expected_mape:.2f}%")
    print(f"  计算MAPE: {calculated_mape:.2f}%")
    print(f"  结果: {'通过' if abs(calculated_mape - expected_mape) < 0.01 else '失败'}")
    
    # 测试用例3: 完全准确预测
    y_true = [100, 200, 300]
    y_pred = [100, 200, 300]
    expected_mape = 0.0
    
    calculated_mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"\n测试用例3 - 完全准确预测:")
    print(f"  真实值: {y_true}")
    print(f"  预测值: {y_pred}")
    print(f"  期望MAPE: {expected_mape:.2f}%")
    print(f"  计算MAPE: {calculated_mape:.2f}%")
    print(f"  结果: {'通过' if abs(calculated_mape - expected_mape) < 0.01 else '失败'}")

def test_evaluate_predictions():
    """
    测试完整的评估函数
    """
    print("\n=== 测试完整评估函数 ===")
    
    # 测试用例: 正常情况
    actual_values = [100, 150, 200, 250, 300]
    predicted_values = [90, 160, 190, 240, 310]
    
    # 使用sklearn计算期望值
    expected_mae = mean_absolute_error(actual_values, predicted_values)
    expected_rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    
    # 计算MAPE
    expected_mape = mean_absolute_percentage_error(actual_values, predicted_values)
    
    # 使用我们的函数计算
    metrics = evaluate_predictions(actual_values, predicted_values)
    
    print(f"测试用例 - 正常情况:")
    print(f"  真实值: {actual_values}")
    print(f"  预测值: {predicted_values}")
    print(f"  期望MAE: {expected_mae:.2f}, 计算MAE: {metrics['MAE']:.2f}, 结果: {'通过' if abs(metrics['MAE'] - expected_mae) < 0.01 else '失败'}")
    print(f"  期望RMSE: {expected_rmse:.2f}, 计算RMSE: {metrics['RMSE']:.2f}, 结果: {'通过' if abs(metrics['RMSE'] - expected_rmse) < 0.01 else '失败'}")
    print(f"  期望MAPE: {expected_mape:.2f}%, 计算MAPE: {metrics['MAPE']:.2f}%, 结果: {'通过' if abs(metrics['MAPE'] - expected_mape) < 0.01 else '失败'}")

def main():
    """
    主函数：运行所有测试
    """
    print("=== 评估指标计算函数测试 ===")
    test_mean_absolute_percentage_error()
    test_evaluate_predictions()
    print("\n=== 所有测试完成 ===")

if __name__ == "__main__":
    main()
