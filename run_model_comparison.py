#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大模型选型对比实验运行脚本
用于对比不同大语言模型在建筑能源预测任务中的性能
"""

import os
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import dashscope
from openai import OpenAI

# 导入现有的评估函数和数据加载函数
from baseline_methods import load_data, select_test_timestamps

# 设置阿里云百炼API Key
# 请确保已设置环境变量DASHSCOPE_API_KEY
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量DASHSCOPE_API_KEY")

def call_llama4(prompt):
    """调用LLaMA4模型"""
    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt}
            ]
        }
    ]
    
    response = dashscope.MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model='llama-4-maverick-17b-128e-instruct',
        messages=messages,
    )
    
    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    else:
        raise Exception(f'LLaMA4调用失败: {response.message}')

def call_deepseek(prompt):
    """调用DeepSeek模型"""
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="deepseek-v3.2-exp",
        messages=messages,
        extra_body={"enable_thinking": True},
    )
    
    return completion.choices[0].message.content

def call_qwen(prompt):
    """调用Qwen模型"""
    # 这里应该使用您已有的Qwen调用方式
    # 为了简化，我们直接返回一个示例响应
    # 在实际使用中，请替换为您的Qwen调用代码
    from qwen import analyze_energy_usage
    
    # 解析prompt中的信息
    lines = prompt.strip().split('\n')
    timestamp = lines[1].split(': ')[1]
    temp = float(lines[2].split(': ')[1].replace('°C', ''))
    humidity = float(lines[3].split(': ')[1].replace('%', ''))
    wet_bulb_temp = float(lines[4].split(': ')[1].replace('°C', ''))
    
    # 调用Qwen模型
    result = analyze_energy_usage(
        timestamps=[timestamp],
        temp=temp,
        humidity=humidity,
        wet_bulb_temp=wet_bulb_temp
    )
    
    return str(result['analysis_report']['corrected_total_kw'])

def run_model_experiment(model_name, model_function, test_data):
    """运行单个大模型实验"""
    print(f"运行{model_name}模型实验...")
    
    predictions = []
    actual_values = []
    
    for data_point in test_data:
        timestamp = data_point['timestamp']
        temp = data_point['temp']
        humidity = data_point['humidity']
        wet_bulb_temp = data_point['wet_bulb_temp']
        actual_kwh = data_point['actual_kwh']
        
        # 构造Prompt
        prompt = f"""
请根据以下信息预测建筑能耗：
时间: {timestamp}
室外温度: {temp}°C
湿度: {humidity}%
湿球温度: {wet_bulb_temp}°C

请只输出预测的能耗值(kW)，不要包含其他文字。
"""
        
        try:
            # 调用模型进行预测
            response = model_function(prompt)
            
            # 解析预测值
            # 尝试多种解析方式
            predicted_kwh = None
            try:
                # 首先尝试直接转换为浮点数
                predicted_kwh = float(response.strip())
            except ValueError:
                # 如果直接转换失败，尝试从文本中提取数字
                import re
                numbers = re.findall(r'\d+\.?\d*', response)
                if numbers:
                    predicted_kwh = float(numbers[0])
            
            if predicted_kwh is not None:
                predictions.append(predicted_kwh)
                actual_values.append(actual_kwh)
                
                print(f"时间: {timestamp}, 实际值: {actual_kwh:.2f}, {model_name}预测值: {predicted_kwh:.2f}")
            else:
                print(f"{model_name}预测失败 {timestamp}: 无法解析预测值 '{response}'")
                continue
        except Exception as e:
            print(f"{model_name}预测失败 {timestamp}: {e}")
            continue
    
    return actual_values, predictions

def calculate_mape(actual_values, predicted_values):
    """计算MAPE指标"""
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    
    # 避免除以零的情况
    non_zero_actual = actual_values != 0
    if np.sum(non_zero_actual) == 0:
        return np.inf
    
    mape = np.mean(np.abs((actual_values[non_zero_actual] - predicted_values[non_zero_actual]) / actual_values[non_zero_actual])) * 100
    return mape

def prepare_test_data(df, test_timestamps, num_samples=5):
    """准备测试数据"""
    test_df = df[df['Timestamp'].isin(test_timestamps)]
    
    test_data = []
    for _, row in test_df.head(num_samples).iterrows():
        test_data.append({
            'timestamp': str(row['Timestamp']),
            'temp': row['Temp'],
            'humidity': row['Humidity'],
            'wet_bulb_temp': row['WetBulbTemp'],
            'actual_kwh': row['Total_kW']
        })
    
    return test_data

def main():
    """主函数：执行大模型选型对比实验"""
    print("=== 大模型选型对比实验 ===")
    
    # 加载数据
    print("1. 加载数据...")
    df = load_data()
    print(f"   数据加载完成，共{len(df)}条记录")
    
    # 选择测试时间点
    print("2. 选择测试时间点...")
    test_timestamps, train_df, val_df = select_test_timestamps(df, num_tests=10)
    print(f"   选择了{len(test_timestamps)}个测试时间点")
    
    # 准备测试数据
    test_data = prepare_test_data(df, test_timestamps, num_samples=5)
    print(f"   准备了{len(test_data)}个测试样本")
    
    # 定义要测试的模型
    models = {
        "Qwen": call_qwen,
        "LLaMA4": call_llama4,
        "DeepSeek": call_deepseek
    }
    
    # 存储各模型的实验结果
    results = {}
    
    # 运行各模型实验
    for model_name, model_function in models.items():
        try:
            actual_values, predictions = run_model_experiment(model_name, model_function, test_data)
            if len(actual_values) > 0 and len(predictions) > 0:
                # 计算评估指标
                mae = mean_absolute_error(actual_values, predictions)
                rmse = np.sqrt(mean_squared_error(actual_values, predictions))
                mape = calculate_mape(actual_values, predictions)
                
                results[model_name] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "actual_values": actual_values,
                    "predictions": predictions
                }
                print(f"   {model_name}模型评估结果: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
            else:
                print(f"   {model_name}模型实验失败：没有成功预测任何样本")
        except Exception as e:
            print(f"   {model_name}模型实验失败: {e}")
    
    # 保存结果
    print("3. 保存实验结果...")
    experiment_results = {
        "experiment_time": datetime.now().isoformat(),
        "test_samples": len(test_data),
        "model_results": results
    }
    
    with open("model_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)
    
    print("大模型选型对比实验完成！结果已保存到 model_comparison_results.json")
    
    # 打印总结
    print("\n=== 实验结果总结 ===")
    for model_name, metrics in results.items():
        print(f"{model_name}模型:")
        print(f"  测试样本数: {len(metrics['actual_values'])}")
        print(f"  MAE: {metrics['MAE']:.2f} kW")
        print(f"  RMSE: {metrics['RMSE']:.2f} kW")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")


if __name__ == "__main__":
    main()
