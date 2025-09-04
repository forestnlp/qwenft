import torch
import subprocess
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set proxy for Notebook environment
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

def download_model():
    """
    下载Qwen3-8B-Instruct模型用于微调训练
    """
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    output_dir = "/autodl-tmp/Qwen/Qwen3-4B-Instruct-2507"
    
    print(f"Downloading model {model_name}...")
    
    # 下载tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 下载模型
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 保存模型和tokenizer
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model download completed!")

if __name__ == "__main__":
    download_model()
