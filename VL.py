import os
from dashscope import MultiModalConversation
import dashscope 

def analyze_building_structure(image_path="/Users/sheeranshi/Desktop/SADM.png"):
    """
    分析建筑平面图并输出结构化信息
    
    参数:
    image_path (str): 建筑平面图的路径
    
    返回:
    str: 建筑结构的JSON字符串
    """
    # 构造图像路径
    image_file_path = f"file://{image_path}"
    
    # 构造消息
    messages = [
        {"role": "system",
         "content": [{"text": "严格只输出json格式，不要多余的描述。"}]},
        {'role': 'user',
         'content': [{'image': image_file_path},
                     {'text': '''你是一个专业的建筑智能分析助手。请仔细观察提供的建筑平面图或系统图，完成以下任务：

                                1. **建筑总体特征**：
                                - 识别总楼层数、估算总面积（单位：平方米）、外形特点（是否曲面）、主要建材、屋顶类型。
                                - 若无明确标注，请基于图形比例和常见建筑类型合理推断。

                                2. **功能区域识别**：
                                - 识别所有房间或功能区，包括编号、用途、面积估算（基于网格比例尺）、热负荷等级（low/medium/high）、使用模式（如 office, 24x7, weekend_only）。
                                - 若区域未标注用途，请根据布局推测（如靠近服务器机房则可能是控制室）。
                                - 所有面积需注明"estimated"，并在 inference_notes 中说明依据。

                                3. **机械设备识别**：
                                - 识别所有 HVAC 设备（如 AHU、冷却塔、风机），记录其 ID、容量（如有）、位置、服务楼层或区域。
                                - 根据风管走向判断其服务范围，若无连接则标记为 "unknown_service_area"。
                                - 记录冷源、热源、控制系统类型（如 BMS）。

                                4. **被动式节能设计分析**：
                                - 识别庭院、反射池、遮阳板、天窗等被动设计元素。
                                - 分析其对自然通风、降温、采光的潜在贡献。
                                - 估算其冷却贡献（kW），若无法估算则写 "unknown"。

                                5. **输出格式要求**：
                                - 必须返回标准 JSON 对象，不得包含任何 Markdown 代码块（如 ```json）。
                                - 所有数值估算必须标注 "estimated"。
                                - 所有未知信息填 "unknown" 或留空，并在 inference_notes 中说明原因。

                                6. **JSON Schema**：
                                {
                                "building_metadata": {
                                    "total_area_sqm": 15000,
                                    "floor_count": 3,
                                    "has_curved_exterior": true,
                                    "primary_construction_material": "reinforced_concrete",
                                    "roof_type": "flat_green_roof"
                                },
                                "functional_zones": [
                                    {
                                    "zone_id": "ST-8A",
                                    "area_type": "office",
                                    "area_sqm": 30,
                                    "heat_load_level": "medium",
                                    "occupancy_pattern": "weekdays_9am_6pm",
                                    "natural_ventilation_potential": 0.3,
                                    "solar_gain_impact": "moderate",
                                    "notes": "near glass wall, moderate heat gain"
                                    }
                                ],
                                "mechanical_systems": {
                                    "ahu_units": [
                                    {
                                        "unit_id": "AHU 1-6",
                                        "capacity_tons": 60,
                                        "location": "AHU ROOM",
                                        "serves_floors": [1, 2],
                                        "service_areas": ["ST-8A", "CTRL_RM"],
                                        "status": "active"
                                    }
                                    ],
                                    "cooling_source": "chilled_water",
                                    "heating_source": "electric_reheat",
                                    "controls_system": "BMS_Digital"
                                },
                                "passive_design_features": [
                                    {
                                    "feature_type": "reflection_pond",
                                    "location": "near_courtyard",
                                    "purpose": "reduce_solar_gain",
                                    "estimated_cooling_contribution_kW": 15
                                    }
                                ],
                                "inference_notes": [
                                    "Area estimated using grid scale (each cell ≈ 5m x 5m).",
                                    "AHU service area inferred from duct routing.",
                                    "No explicit labeling for room functions; assumed based on layout."
                                ]
                                }

                                现在开始分析图像内容，并输出纯 JSON 对象。'''
                     }]}
    ]
    
    # 调用多模态模型
    response = MultiModalConversation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model='qwen-vl-max-latest',  
        messages=messages)
    
    # 返回结果
    return response["output"]["choices"][0]["message"].content[0]["text"]

if __name__ == "__main__":
    # 默认调用
    result = analyze_building_structure()
    print(result)
