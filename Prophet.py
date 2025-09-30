import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def load_and_prepare_data(file_path, target_column='Total_kW', exclude_zeros=True):
    """
    从文件加载数据并为Prophet准备数据
    
    参数:
    file_path (str): 文件路径（支持Excel和CSV格式）
    target_column (str): 要预测的列名
    exclude_zeros (bool): 是否排除零值（空调关闭时段）
    
    返回:
    为Prophet准备好的DataFrame
    """
    # 根据文件扩展名选择读取方法
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        # 读取Excel文件
        df = pd.read_excel(file_path)
    
    # 显示数据基本信息
    # print("原始数据形状:", df.shape)
    # print("列名:", df.columns.tolist())
    # print("\n前几行数据:")
    # print(df.head())
    
    # 如果需要，过滤掉零值（空调关闭时段）
    if exclude_zeros:
        original_shape = df.shape[0]
        df = df[df[target_column] > 0]
        filtered_shape = df.shape[0]
        print(f"\n过滤掉 {original_shape - filtered_shape} 行 {target_column} = 0 的数据")
        print(f"剩余数据形状: {df.shape}")
    
    # 为Prophet准备数据 (ds表示日期，y表示值)
    # 使用指定的目标列
    prophet_df = df[['Timestamp', 'Temp', 'Humidity', 'WetBulbTemp', target_column]].copy()
    prophet_df.columns = ['ds', 'temp', 'humidity', 'wet_bulb_temp', 'y']
    
    # 确保ds是datetime类型
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # 显示用于训练的数据样本
    # print("\n用于训练的数据样本:")
    # print(prophet_df.head())
    # print("数据形状:", prophet_df.shape)
    
    return prophet_df

def train_prophet_model(df):
    """
    在数据上训练Prophet模型
    
    参数:
    df: 包含'ds'和'y'列的DataFrame
    
    返回:
    训练好的Prophet模型
    """
    # 初始化并拟合模型
    # yearly_seasonality: 年季节性
    # weekly_seasonality: 周季节性
    # daily_seasonality: 日季节性
    # changepoint_prior_scale: 趋势变化点的正则化参数
    # seasonality_prior_scale: 季节性正则化参数
    # seasonality_mode: 季节性模式 ('additive' or 'multiplicative')
    model = Prophet(
        yearly_seasonality=True,         # 启用年季节性
        weekly_seasonality=True,         # 启用周季节性
        daily_seasonality=True,          # 启用日季节性
        changepoint_prior_scale=0.1,     # 增加趋势变化点的灵活性
        seasonality_prior_scale=10.0,    # 增加季节性效应
        seasonality_mode='multiplicative', # 使用乘法季节性模式
        interval_width=0.95              # 置信区间宽度
    )
    
    # 不添加额外的回归因子，仅使用时间序列数据
    pass
    
    # 拟合模型
    model.fit(df)
    
    print("模型训练成功!")
    print("模型的额外回归因子:", list(model.extra_regressors.keys()) if model.extra_regressors else "无")
    return model

def make_predictions(model, periods=30):
    """
    进行未来预测
    
    参数:
    model: 训练好的Prophet模型
    periods: 预测的周期数
    
    返回:
    预测结果DataFrame
    """
    # 创建未来数据框
    future = model.make_future_dataframe(periods=periods, freq='H')  # 每小时预测
    
    # 进行预测
    forecast = model.predict(future)
    
    return forecast

def plot_results(model, forecast, target_column='Total_kW'):
    """
    绘制预测结果
    
    参数:
    model: 训练好的Prophet模型
    forecast: 预测结果
    target_column: 目标列名
    """
    # 绘制预测结果图
    fig = model.plot(forecast)
    plt.title(f'{target_column} 预测结果')
    plt.xlabel('日期')
    plt.ylabel(target_column)
    plt.show()
    
    # 绘制组件图（趋势、季节性等）
    fig2 = model.plot_components(forecast)
    plt.show()

def predict_for_timestamps(model, timestamps, historical_data=None, temp=None, humidity=None, wet_bulb_temp=None):
    """
    预测指定时间戳的Total_kW值
    
    参数:
    model: 训练好的Prophet模型
    timestamps: 时间戳字符串列表或datetime对象
    historical_data: 历史数据DataFrame，用于获取回归因子的实际值
    temp (float): 外界温度 (°C)
    humidity (float): 湿度 (%)
    wet_bulb_temp (float): 湿球温度 (°C)
    
    返回:
    包含预测结果的DataFrame
    """
    # 如果不是列表，转换为列表
    if isinstance(timestamps, str):
        timestamps = [timestamps]
    
    # 自动解析多种时间戳格式
    parsed_timestamps = []
    for ts in timestamps:
        # 尝试解析不同的时间戳格式
        try:
            # 尝试解析 '2023/10/12 19:08' 格式
            if '/' in ts and ':' in ts:
                parsed_ts = pd.to_datetime(ts, format='%Y/%m/%d %H:%M')
            # 尝试解析 '2023-10-15 14:00:00' 格式
            elif '-' in ts and ':' in ts:
                parsed_ts = pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S')
            # 其他格式使用pandas自动解析
            else:
                parsed_ts = pd.to_datetime(ts)
            parsed_timestamps.append(parsed_ts)
        except Exception as e:
            # 如果解析失败，使用pandas自动解析作为后备方案
            parsed_timestamps.append(pd.to_datetime(ts))
    
    # 创建包含时间戳的数据框
    timestamp_df = pd.DataFrame({
        'ds': parsed_timestamps
    })
    
    # 进行预测
    # print("预测用的数据框:")
    # print(timestamp_df)
    forecast = model.predict(timestamp_df)
    
    # 返回相关列（预测值和置信区间）
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def get_prophet_forecast(file_path='cleaned_202310_wetbulb_filled.xlsx', target_column='Total_kW', 
                         timestamps=None, temp=None, humidity=None, wet_bulb_temp=None):
    """
    封装的Prophet预测函数
    
    参数:
    file_path (str): Excel文件路径
    target_column (str): 要预测的列名
    timestamps (list): 要预测的时间戳列表，默认为None
    temp (float): 外界温度 (°C)
    humidity (float): 湿度 (%)
    wet_bulb_temp (float): 湿球温度 (°C)
    
    返回:
    dict: 包含预测结果的字典
    """
    # 如果没有提供时间戳，使用默认值
    if timestamps is None:
        timestamps = ['2023-10-15 14:00:00']
    # 加载和准备数据
    print("正在加载和准备数据...")
    data = load_and_prepare_data(file_path, target_column)
    
    # 训练模型
    print("\n正在训练Prophet模型...")
    model = train_prophet_model(data)
    
    # 指定时间戳预测
    # print("\n指定时间戳预测:")
    # specific_forecast = predict_for_timestamps(model, timestamps, data, temp, humidity, wet_bulb_temp)
    # print(specific_forecast)
    specific_forecast = predict_for_timestamps(model, timestamps, data, temp, humidity, wet_bulb_temp)
    
    # 构造预测结果字典
    forecast_results = []
    for _, row in specific_forecast.iterrows():
        forecast_results.append({
            "timestamp": str(row['ds']),  # 将 Timestamp 转换为字符串
            "predicted_total_kw": round(row['yhat'], 2),
            "confidence_interval": {
                "lower": round(row['yhat_lower'], 2),
                "upper": round(row['yhat_upper'], 2)
            }
        })
    
    return forecast_results[0] if len(forecast_results) == 1 else forecast_results

def main():
    """
    主函数 - 执行完整的预测流程（非交互式）
    """
    # 使用封装的函数进行预测
    result = get_prophet_forecast()
    
    # 显示预测结果
    print("\n预测结果:")
    if isinstance(result, dict):
        print(f"时间戳: {result['timestamp']}")
        print(f"  预期 Total_kW: {result['predicted_total_kw']:.2f}")
        print(f"  下限: {result['confidence_interval']['lower']:.2f}")
        print(f"  上限: {result['confidence_interval']['upper']:.2f}")
    else:
        for item in result:
            print(f"时间戳: {item['timestamp']}")
            print(f"  预期 Total_kW: {item['predicted_total_kw']:.2f}")
            print(f"  下限: {item['confidence_interval']['lower']:.2f}")
            print(f"  上限: {item['confidence_interval']['upper']:.2f}")
            print()
    
    print("\n处理完成!")

if __name__ == "__main__":
    main()
