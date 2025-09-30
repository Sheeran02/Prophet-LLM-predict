import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 导入自定义模块
from VL import analyze_building_structure
from Prophet import get_prophet_forecast

# 导入 RAG 工具
from rag_utils import build_or_load_vectorstore

# 全局加载 retriever（避免重复加载）
RETRIEVER = build_or_load_vectorstore()

class EnergyCorrectionResult(BaseModel):
    corrected_total_kw: float = Field(..., description="校准后的总能耗 (kW)")
    confidence_interval: List[float] = Field(..., description="置信区间 [下限, 上限]")
    adjustment_reason: str = Field(..., description="简短调整原因")

def analyze_energy_usage(
    timestamps: Optional[List[str]] = None,
    temp: Optional[float] = None,
    humidity: Optional[float] = None,
    wet_bulb_temp: Optional[float] = None
) -> Dict[str, Any]:
    # ... [前面的 Prophet 和 VL 调用保持不变] ...
    prophet_forecast = get_prophet_forecast(timestamps=timestamps, temp=temp, humidity=humidity, wet_bulb_temp=wet_bulb_temp)
    predicted_total_kw = prophet_forecast.get("predicted_total_kw", 0.0)

    if timestamps is None:
        timestamps = ["2023-10-15 14:00:00"]
    
    space_info_json = analyze_building_structure()

    # === RAG 检索 ===
    query_time = timestamps[0]  # 使用第一个时间戳作为查询依据
    query_env = f"Time: {query_time}, Outdoor temp: {temp}°C, Humidity: {humidity}%, Wet-bulb: {wet_bulb_temp}°C"
    
    print("🔍 正在检索历史相似工况...")
    retrieved_docs = RETRIEVER.invoke(query_env)
    rag_context = "\n".join([doc.page_content for doc in retrieved_docs])
    print("📚 检索到的历史记录：\n", rag_context)

    # === LLM 调用 ===
    chat_llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-max",
        temperature=0.0,
    )

    system_prompt = """
    # 角色
    你是一位专业的智能能源分析师，专注于基于严格限定的输入数据对 NTU 南洋理工大学 SADM（艺术、设计与媒体学院）楼宇的 HVAC 总能耗进行时空校准。你的核心任务是通过整合以下五类输入信息，输出高精度的 `Total_kW` 校准值：

    1. **时间序列预测**（Prophet 模型输出的原始预测）
    2. **建筑空间结构**（含 AHU 分区映射、被动设计特征等）
    3. **预测环境参数**（实时温度、湿度、湿球温度）
    4. **预测时间戳**（目标校准时间范围）
    5. **RAG 检索到的历史相似工况**（仅限 2023 年 10 月真实运行数据，`Total_kW = 0` 表示 HVAC 未运行，必须忽略）

    ## 核心原则
    - **严格数据边界**：仅使用上述五类输入数据，禁止引用外部知识或自行杜撰楼宇细节。
    - **时区规范**：所有时间戳均视为 Asia/Singapore 时区（UTC+8），需据此判断工作日/周末及运行时段。
    - **输出纯净性**：仅生成指定 JSON 结果，禁止输出解释性文本、分析步骤或建议。

    ## 技能与执行规范

    ### 技能 1: 数据解析与校准逻辑
    - **空间结构解析**
    - **AHU 负荷调整**：如果 `ahu_zone_map` 可用，对应区域不运行时，下调其所服务的 AHU 负荷。若同一区域由多台 AHU 重叠服务，可能存在冗余，适度下调。
    - **被动设计调整**：如果 `passive_features` 可用，根据其信息进行比例调整。

    - **RAG 历史工况处理**
    - **仅当记录中 `Total_kW > 0` 时参考**（`Total_kW = 0` 表示 HVAC 未运行，必须忽略）。
    - 比对当前环境参数（温度、湿度）与历史工况的相似性：
        - 无可参考记录（或全部 `Total_kW = 0`）→ 不触发此调整。
    - 禁止假设历史数据细节，仅基于输入值计算。

    - **环境参数与时维度融合**
    - 将当前环境参数（温度、湿度、湿球温度）与时间维度（工作日/周末、运行时段）结合，进行综合校准。

    ### 技能 2: 时间序列预测调整
    - **时间序列预测校准**：基于 Prophet 模型输出的原始预测，结合当前环境参数和历史相似工况，进行动态调整，确保预测值更加准确。

    ### 技能 3: 输出结果生成
        强制 JSON 格式（仅包含以下三个字段）：
        （示例格式，不要直接复制）
        corrected_total_kw: 校准后数值，保留两位小数（例：123.45）
        confidence_interval: [置信下限，保留两位小数, 置信上限，保留两位小数]
        adjustment_reason: ≤50 字简短说明，必须引用关键因素，给出合理分析
        
        字段生成规则：
        corrected_total_kw：基于校准流程计算的最终值
        confidence_interval：必须为数值数组，格式 [low, high]
        adjustment_reason：仅说明实际应用的调整项
    请基于以上信息，输出校准后的 Total_kW。
    """

    human_prompt = """
    【当前时间戳】
    {timestamps}

    【当前环境参数】
    外界温度: {temp}°C, 湿度: {humidity}%, 湿球温度: {wet_bulb_temp}°C

    【时间序列预测】
    {prophet_forecast}

    【建筑结构】
    {space_info}

    【RAG 检索到的历史相似工况】
    {rag_context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    structured_llm = chat_llm.with_structured_output(EnergyCorrectionResult, method="function_calling")

    input_data = {
        "timestamps": json.dumps(timestamps, ensure_ascii=False),
        "temp": temp or "N/A",
        "humidity": humidity or "N/A",
        "wet_bulb_temp": wet_bulb_temp or "N/A",
        "prophet_forecast": json.dumps(prophet_forecast, ensure_ascii=False, indent=2),
        "space_info": json.dumps(space_info_json, ensure_ascii=False, indent=2),
        "rag_context": rag_context if rag_context.strip() else "无相关历史记录"
    }

    try:
        result = structured_llm.invoke(prompt.format(**input_data))
        analysis_report = result.dict()
    except Exception as e:
        print(f"⚠️ LLM 调用失败: {e}")
        analysis_report = {
            "corrected_total_kw": predicted_total_kw,
            "confidence_interval": [predicted_total_kw * 0.9, predicted_total_kw * 1.1],
            "adjustment_reason": "LLM 校准失败，使用原始预测值"
        }

    return {
        "prophet_forecast": prophet_forecast,
        "space_info": space_info_json,
        "rag_context": rag_context,
        "analysis_report": analysis_report
    }

# 注意：此文件已重构为函数库，不再直接运行。
# 请通过 main.py 或其他脚本调用 analyze_energy_usage() 函数。
