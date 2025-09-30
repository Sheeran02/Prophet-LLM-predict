#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主程序文件，用于调用 qwen.py 中的函数进行建筑能源使用情况分析
"""

import json
from qwen import analyze_energy_usage

def main():
    """主函数：执行建筑能源使用情况分析"""
    print("开始建筑能源使用情况分析...")
    
    # 设置要预测的时间戳（可以根据需要修改）
    # 支持多种时间戳格式，包括 '2023/10/12 19:08' 和 '2023-10-15 14:00:00'
    timestamps = ['2023/10/19 19:24']  # 用户提供的格式示例
    
    # 设置环境参数（硬编码）
    temp = 26.3  # 外界温度 (°C)
    humidity = 93.6  # 湿度 (%)
    wet_bulb_temp = 25.47  # 湿球温度 (°C)
    
    # 也可以设置多个时间戳，支持混合格式
    # timestamps = ['2023/10/12 19:08', '2023-10-15 14:00:00']
    
    # 调用分析函数，传递额外的环境参数
    result = analyze_energy_usage(timestamps, temp=temp, humidity=humidity, wet_bulb_temp=wet_bulb_temp)
    
    # 显示结果
    print("\n分析完成！")
    print("=" * 50)
    print("时间序列预测结果:")
    # print(json.dumps(result["prophet_forecast"], ensure_ascii=False, indent=2))
    print("=" * 50)
    # print("建筑结构信息:")
    # print(result["space_info"])
    print("=" * 50)
    print("分析报告:")
    print(result["analysis_report"])

if __name__ == "__main__":
    main()
