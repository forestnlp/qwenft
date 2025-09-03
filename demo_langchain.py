from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

def run_ollama_qwen_langchain():
    try:
        print("\n与 Ollama (LangChain 格式) 对话中...")
        
        # 初始化 Ollama
        llm = Ollama(model="qwen3:8b")
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("human", "你好，请问有什么可以帮助你的吗？")
        ])
        
        # 创建链
        chain = prompt | llm
        
        # 调用模型
        response = chain.invoke({})
        print("Ollama 回复 (LangChain 格式):\n", response)
    except Exception as e:
        print(f"调用 Ollama (LangChain 格式) 时出错: {e}")

if __name__ == "__main__":
    run_ollama_qwen_langchain()