from openai import OpenAI

# Point to the local Ollama instance
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

def run_ollama_qwen_openai_format():
    try:
        print("\n与 Ollama (OpenAI 格式) 对话中...")
        response = client.chat.completions.create(
            model="qwen3:8b",
            messages=[
                {"role": "user", "content": "你好，请问有什么可以帮助你的吗？"}
            ],
            stream=False
        )
        print("Ollama 回复 (OpenAI 格式):\n", response.choices[0].message.content)
    except Exception as e:
        print(f"调用 Ollama (OpenAI 格式) 时出错: {e}")

if __name__ == "__main__":
    run_ollama_qwen_openai_format()