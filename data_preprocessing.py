#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据预处理脚本
用于去除能耗为0的数据，并按照8:2的比例划分训练集和测试集
"""

import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path='cleaned_202310_wetbulb_filled.xlsx'):
    """
    加载并预处理数据
    
    参数:
    file_path (str): Excel文件路径
    
    返回:
    DataFrame: 预处理后的数据
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 转换时间戳列
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # 按时间排序
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    print(f"原始数据形状: {df.shape}")
    print(f"原始数据时间范围: {df['Timestamp'].min()} 到 {df['Timestamp'].max()}")
    
    # 去除Total_kW为0的数据
    original_shape = df.shape[0]
    df = df[df['Total_kW'] > 0]
    filtered_shape = df.shape[0]
    
    print(f"\n过滤掉 {original_shape - filtered_shape} 行 Total_kW = 0 的数据")
    print(f"过滤后数据形状: {df.shape}")
    print(f"过滤后数据时间范围: {df['Timestamp'].min()} 到 {df['Timestamp'].max()}")
    
    return df

def split_train_test_data(df, train_ratio=0.8):
    """
    按照指定比例划分训练集和测试集
    
    参数:
    df (DataFrame): 输入数据
    train_ratio (float): 训练集比例
    
    返回:
    tuple: (训练集DataFrame, 测试集DataFrame)
    """
    # 按时间顺序划分
    train_size = int(len(df) * train_ratio)
    
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    print(f"\n数据集划分:")
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    print(f"训练集时间范围: {train_df['Timestamp'].min()} 到 {train_df['Timestamp'].max()}")
    print(f"测试集时间范围: {test_df['Timestamp'].min()} 到 {test_df['Timestamp'].max()}")
    
    return train_df, test_df

def save_datasets(train_df, test_df, train_file='train_data.csv', test_file='test_data.csv'):
    """
    保存训练集和测试集到CSV文件
    
    参数:
    train_df (DataFrame): 训练集数据
    test_df (DataFrame): 测试集数据
    train_file (str): 训练集文件名
    test_file (str): 测试集文件名
    """
    # 保存训练集
    train_df.to_csv(train_file, index=False)
    print(f"\n训练集已保存到: {train_file}")
    
    # 保存测试集
    test_df.to_csv(test_file, index=False)
    print(f"测试集已保存到: {test_file}")

def load_train_test_data(train_file='train_data.csv', test_file='test_data.csv'):
    """
    加载训练集和测试集数据
    
    参数:
    train_file (str): 训练集文件名
    test_file (str): 测试集文件名
    
    返回:
    tuple: (训练集DataFrame, 测试集DataFrame)
    """
    # 加载训练集
    train_df = pd.read_csv(train_file)
    train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
    
    # 加载测试集
    test_df = pd.read_csv(test_file)
    test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
    
    print(f"\n加载数据:")
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    return train_df, test_df

def main():
    """
    主函数：执行完整的数据预处理流程
    """
    print("=== 数据预处理 ===")
    
    # 加载并预处理数据
    df = load_and_preprocess_data()
    
    # 划分训练集和测试集
    train_df, test_df = split_train_test_data(df, train_ratio=0.8)
    
    # 保存数据集
    save_datasets(train_df, test_df, 'train_data.csv', 'test_data.csv')
    
    print("\n数据预处理完成！")

if __name__ == "__main__":
    main()
