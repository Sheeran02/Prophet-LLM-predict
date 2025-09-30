import os
from openai import OpenAI

# Try to list models using OpenAI compatible API
def check_models():
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    try:
        models = client.models.list()
        print("✅ Available models:")
        for model in models.data:
            print(f"  - {model.id}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_models()
