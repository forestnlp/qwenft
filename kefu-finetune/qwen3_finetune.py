from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

MODEL = "unsloth/Qwen3-0.6B"
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 模型加载和LoRA配置保持不变（原始代码可运行，不修改）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# -------------------------- 仅修改数据加载和处理部分 --------------------------
# 加载清洗好的Qwen3格式数据集（替换原数学推理数据集）
dataset = load_dataset("json", data_files="../data/qwen3_finetune_data.jsonl")["train"]
print("一条处理前的数据样本:", dataset[0])  # 查看加载的原始数据

# 应用Qwen3聊天模板（将system和conversations转换为模型可识别的文本）
def format_dataset(sample):
    # 调用tokenizer的聊天模板，自动添加<|im_start|>/<|im_end|>标记
    formatted_text = tokenizer.apply_chat_template(
        sample["conversations"],
        system=sample["system"],  # 传入system提示词
        tokenize=False,
        add_generation_prompt=False  # 微调时不需要生成提示
    )
    return {"text": formatted_text}  # 返回SFTTrainer需要的"text"字段

# 批量处理数据集
final_dataset = dataset.map(
    format_dataset,
    remove_columns=dataset.column_names  # 只保留格式化后的"text"字段
)

print("一条处理后的数据样本:", final_dataset[0]["text"])  # 查看格式化后的文本
print("final_dataset 数据量:", len(final_dataset))
# ------------------------------------------------------------------------------

# 训练配置保持不变（仅根据数据量调整max_steps，避免训练不充分）
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=final_dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,  # 根据数据量调整（示例：1万条数据建议500-1000步）
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        output_dir="outputs",  # 模型保存路径
    ),
)

# 开始训练
print("\n开始训练...")
trainer_stats = trainer.train()

print("\n训练完成！")
print("训练统计信息:", trainer_stats)
