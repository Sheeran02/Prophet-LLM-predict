#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实验运行脚本，用于执行对比实验
"""

import pandas as pd
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from baseline_methods import load_data, train_prophet_baseline, predict_with_prophet, prophet_with_env_params, evaluate_predictions, select_test_timestamps
from qwen import analyze_energy_usage

def run_complete_system_experiment(df, test_timestamps):
    """运行完整系统实验"""
    # 获取测试时间点的环境参数
    test_df = df[df['Timestamp'].isin(test_timestamps)]
    
    predictions = []
    actual_values = []
    
    # 只训练一次Prophet模型
    print("训练Prophet模型...")
    from baseline_methods import train_prophet_baseline
    prophet_model = train_prophet_baseline(df)
    
    # 只调用一次VL模型获取建筑结构信息
    print("获取建筑结构信息...")
    try:
        from VL import analyze_building_structure
        space_info_json = analyze_building_structure()
        print("建筑结构信息获取成功")
    except Exception as e:
        print(f"获取建筑结构信息失败: {e}")
        space_info_json = "{}"  # 使用空的JSON作为默认值
    
    # 导入RAG工具
    print("加载RAG检索器...")
    try:
        from rag_utils import build_or_load_vectorstore
        RETRIEVER = build_or_load_vectorstore()
        print("RAG检索器加载成功")
    except Exception as e:
        print(f"加载RAG检索器失败: {e}")
        RETRIEVER = None
    
    for _, row in test_df.iterrows():
        timestamp = row['Timestamp']
        temp = row['Temp']
        humidity = row['Humidity']
        wet_bulb_temp = row['WetBulbTemp']
        actual_kwh = row['Total_kW']
        
        try:
            # 使用预训练的Prophet模型进行预测
            from baseline_methods import predict_with_prophet
            prophet_forecast = predict_with_prophet(prophet_model, [str(timestamp)])
            
            # 构造查询并检索历史相似工况
            query_env = f"Time: {timestamp}, Outdoor temp: {temp}°C, Humidity: {humidity}%, Wet-bulb: {wet_bulb_temp}°C"
            rag_context = "无相关历史记录"
            if RETRIEVER:
                print("检索历史相似工况...")
                retrieved_docs = RETRIEVER.invoke(query_env)
                rag_context = "\n".join([doc.page_content for doc in retrieved_docs])
                print("历史工况检索完成")
            
            # 直接调用完整系统进行预测，而不是重新实现
            from qwen import analyze_energy_usage
            result = analyze_energy_usage(
                timestamps=[str(timestamp)],
                temp=temp,
                humidity=humidity,
                wet_bulb_temp=wet_bulb_temp
            )
            
            predicted_kwh = result['analysis_report']['corrected_total_kw']
            
            predictions.append(predicted_kwh)
            actual_values.append(actual_kwh)
            
            print(f"时间: {timestamp}, 实际值: {actual_kwh:.2f}, 预测值: {predicted_kwh:.2f}")
        except Exception as e:
            print(f"预测失败 {timestamp}: {e}")
            # 即使某个时间点预测失败，也要确保actual_values和predictions长度一致
            # 使用Prophet预测值作为替代
            try:
                from baseline_methods import predict_with_prophet
                if 'prophet_model' in locals():
                    fallback_pred = predict_with_prophet(prophet_model, [str(timestamp)])
                    predictions.append(fallback_pred['predicted_total_kw'])
                    actual_values.append(actual_kwh)
                    print(f"使用Prophet预测值作为替代: {fallback_pred['predicted_total_kw']:.2f}")
            except:
                print(f"无法生成替代预测值")
            continue
    
    # 确保返回的数组长度一致
    min_len = min(len(actual_values), len(predictions))
    if min_len == 0:
        print("警告：没有成功预测任何样本")
        return [], []
    
    return actual_values[:min_len], predictions[:min_len]

def run_baselines_experiment(train_df, val_df, test_timestamps):
    """运行基线方法实验"""
    # 1. 纯Prophet基线
    print("训练纯Prophet基线模型...")
    prophet_model = train_prophet_baseline(train_df)
    
    print("使用纯Prophet模型进行预测...")
    prophet_predictions = []
    actual_values = []
    
    test_df = val_df[val_df['Timestamp'].isin(test_timestamps)]
    
    for _, row in test_df.iterrows():
        timestamp = row['Timestamp']
        actual_kwh = row['Total_kW']
        
        try:
            pred = predict_with_prophet(prophet_model, [timestamp])
            predicted_kwh = pred['predicted_total_kw']
            
            prophet_predictions.append(predicted_kwh)
            actual_values.append(actual_kwh)
            
            print(f"时间: {timestamp}, 实际值: {actual_kwh:.2f}, Prophet预测值: {predicted_kwh:.2f}")
        except Exception as e:
            print(f"Prophet预测失败 {timestamp}: {e}")
            continue
    
    # 2. 带环境参数的Prophet基线
    print("使用带环境参数的Prophet模型进行预测...")
    prophet_env_predictions = []
    
    for _, row in test_df.iterrows():
        timestamp = row['Timestamp']
        temp = row['Temp']
        humidity = row['Humidity']
        wet_bulb_temp = row['WetBulbTemp']
        
        try:
            pred = prophet_with_env_params(train_df, [timestamp], 
                                         temp=temp, 
                                         humidity=humidity, 
                                         wet_bulb_temp=wet_bulb_temp)
            predicted_kwh = pred['predicted_total_kw']
            
            prophet_env_predictions.append(predicted_kwh)
            
            print(f"时间: {timestamp}, 带环境参数Prophet预测值: {predicted_kwh:.2f}")
        except Exception as e:
            print(f"带环境参数Prophet预测失败 {timestamp}: {e}")
            # 如果失败，使用纯Prophet预测值
            prophet_env_predictions.append(prophet_predictions[len(prophet_env_predictions)])
    
    return actual_values, prophet_predictions, prophet_env_predictions

def statistical_significance_test(actual_values, method1_predictions, method2_predictions):
    """进行统计显著性检验"""
    # 计算误差
    method1_errors = np.abs(np.array(actual_values) - np.array(method1_predictions))
    method2_errors = np.abs(np.array(actual_values) - np.array(method2_predictions))
    
    # 简单的t检验（这里使用简化版本）
    mean_diff = np.mean(method1_errors - method2_errors)
    std_diff = np.std(method1_errors - method2_errors)
    n = len(method1_errors)
    
    # 计算t统计量
    t_stat = mean_diff / (std_diff / np.sqrt(n)) if std_diff > 0 else 0
    
    return t_stat

def main():
    """主函数：执行完整的对比实验"""
    print("=== 建筑能源预测对比实验 ===")
    
    # 加载数据
    print("1. 加载数据...")
    df = load_data('train_data.csv', 'test_data.csv')
    print(f"   数据加载完成，共{len(df)}条记录")
    
    # 选择测试时间点，并划分训练集和验证集
    print("2. 选择测试时间点并划分数据集...")
    test_timestamps, train_df, val_df = select_test_timestamps(df, num_tests=2, train_ratio=0.8)
    print(f"   选择了{len(test_timestamps)}个测试时间点")
    print(f"   训练集大小: {len(train_df)}")
    print(f"   验证集大小: {len(val_df)}")
    
    # 运行基线方法实验
    print("3. 运行基线方法实验...")
    actual_values, prophet_predictions, prophet_env_predictions = run_baselines_experiment(train_df, val_df, test_timestamps)
    
    # 运行完整系统实验
    print("4. 运行完整系统实验...")
    complete_actual_values, complete_predictions = run_complete_system_experiment(df, test_timestamps)
    
    # 确保使用相同的实际值
    actual_values = complete_actual_values
    
    # 评估结果
    print("5. 评估实验结果...")
    
    # 评估纯Prophet基线
    prophet_metrics = evaluate_predictions(actual_values, prophet_predictions)
    print(f"   纯Prophet模型评估结果: MAE={prophet_metrics['MAE']:.2f}, RMSE={prophet_metrics['RMSE']:.2f}, MAPE={prophet_metrics['MAPE']:.2f}%")
    
    # 评估带环境参数的Prophet基线
    prophet_env_metrics = evaluate_predictions(actual_values, prophet_env_predictions)
    print(f"   带环境参数Prophet模型评估结果: MAE={prophet_env_metrics['MAE']:.2f}, RMSE={prophet_env_metrics['RMSE']:.2f}, MAPE={prophet_env_metrics['MAPE']:.2f}%")
    
    # 评估完整系统
    complete_metrics = evaluate_predictions(actual_values, complete_predictions)
    print(f"   完整系统评估结果: MAE={complete_metrics['MAE']:.2f}, RMSE={complete_metrics['RMSE']:.2f}, MAPE={complete_metrics['MAPE']:.2f}%")
    
    # 统计显著性检验
    print("6. 进行统计显著性检验...")
    
    # Prophet vs 带环境参数Prophet
    t_stat1 = statistical_significance_test(actual_values, prophet_predictions, prophet_env_predictions)
    print(f"   Prophet vs 带环境参数Prophet t统计量: {t_stat1:.2f}")
    
    # Prophet vs 完整系统
    t_stat2 = statistical_significance_test(actual_values, prophet_predictions, complete_predictions)
    print(f"   Prophet vs 完整系统 t统计量: {t_stat2:.2f}")
    
    # 带环境参数Prophet vs 完整系统
    t_stat3 = statistical_significance_test(actual_values, prophet_env_predictions, complete_predictions)
    print(f"   带环境参数Prophet vs 完整系统 t统计量: {t_stat3:.2f}")
    
    # 保存结果
    print("7. 保存实验结果...")
    results = {
        "experiment_time": datetime.now().isoformat(),
        "test_samples": len(actual_values),
        "baseline_prophet": prophet_metrics,
        "baseline_prophet_with_env": prophet_env_metrics,
        "complete_system": complete_metrics,
        "statistical_tests": {
            "prophet_vs_prophet_env": t_stat1,
            "prophet_vs_complete": t_stat2,
            "prophet_env_vs_complete": t_stat3
        }
    }
    
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("实验完成！结果已保存到 experiment_results.json")
    
    # 打印总结
    print("\n=== 实验结果总结 ===")
    print(f"测试样本数: {len(actual_values)}")
    print(f"纯Prophet模型 MAE: {prophet_metrics['MAE']:.2f} kW")
    print(f"带环境参数Prophet模型 MAE: {prophet_env_metrics['MAE']:.2f} kW")
    print(f"完整系统 MAE: {complete_metrics['MAE']:.2f} kW")
    
    # 计算改进百分比
    prophet_improvement = (prophet_metrics['MAE'] - complete_metrics['MAE']) / prophet_metrics['MAE'] * 100
    print(f"完整系统相比纯Prophet模型的改进: {prophet_improvement:.2f}%")

if __name__ == "__main__":
    main()
