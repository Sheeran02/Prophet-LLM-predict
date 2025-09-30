import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

def load_and_prepare_knowledge_base(file_path: str = "cleaned_202310_wetbulb_filled.xlsx"):
    """将 Excel 数据转换为文本段落，用于嵌入和检索"""
    df = pd.read_excel(file_path)
    # 确保 timestamp 是字符串
    df['timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    documents = []
    for _, row in df.iterrows():
        doc = (
            f"On {row['timestamp']}, the outdoor dry-bulb temperature was {row['Temp']:.1f}°C, "
            f"relative humidity was {row['Humidity']:.1f}%, wet-bulb temperature was {row['WetBulbTemp']:.1f}°C, "
            f"and the total HVAC energy consumption was {row['Total_kW']:.2f} kW."
        )
        documents.append(doc)
    return documents

def build_or_load_vectorstore(persist_dir: str = "chroma_rag_kb", excel_path: str = "cleaned_202310_wetbulb_filled.xlsx"):
    """构建或加载 Chroma 向量数据库"""
    # 使用阿里云百炼embedding模型
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v4"
    )
    
    if os.path.exists(persist_dir):
        # print("✅ 加载已有向量数据库...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        # print("🔄 构建新向量数据库...")
        docs = load_and_prepare_knowledge_base(excel_path)
        
        # 直接使用文档列表创建向量数据库
        vectorstore = Chroma.from_texts(
            texts=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    return vectorstore.as_retriever(search_kwargs={"k": 3})  # 返回 top-3 相似记录
