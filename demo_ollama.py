import ollama

def run_ollama_qwen():
    try:
        response = ollama.chat(model='qwen3:8b', messages=[
            {
                'role': 'user',
                'content': '你好，请问有什么可以帮助你的吗？',
            },
        ])
        print("Ollama 回复 (原始格式):\n", response['message']['content'])
    except Exception as e:
        print(f"调用 Ollama (原始格式) 时出错: {e}")

if __name__ == "__main__":
    run_ollama_qwen()