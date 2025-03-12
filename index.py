from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

import os

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 检查硬件是否支持 BFloat16
bf16_supported = torch.cuda.is_bf16_supported()

max_seq_length = 2048+2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 根据检查结果设置数据类型
if bf16_supported:
    dtype = torch.bfloat16
else:
    dtype = torch.float16

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./model/unsloth/llama-3-8b-bnb-4bit",  # 模型路径
    max_seq_length=max_seq_length,  # 可以设置为任何值内部做了自适应处理
    dtype=dtype,
    load_in_4bit=True,
    load_in_8bit=False
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # 选择任何大于0的数字！建议使用8、16、32、64、128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # 支持任何值，但等于0时经过优化
    bias="none",  # 支持任何值，但等于"none"时经过优化
    use_gradient_checkpointing='unsloth',  # True或"unsloth"适用于非常长的上下文
    random_state=3407,
    use_rslora=False,  # 支持排名稳定的LoRA
    loftq_config=None,  # 和LoftQ
)

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_sft_prompts_func(examples):
    instructions = examples["instruction"]

    instances = examples['instances']
    inputs = []
    outputs = []

    for instance in instances:
        inputs.append(instance[0]['input'])
        outputs.append(instance[0]['output'])
    texts = []

    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }


dataset = load_dataset("json", data_files="./data/output/sft_data_1500.jsonl",
                       split="train")

dataset = dataset.map(formatting_sft_prompts_func, batched=True, )

# 根据检查结果设置训练参数
training_args = TrainingArguments(
    output_dir="models/lora/llama",  # 输出目录
    per_device_train_batch_size=2,  # 每个设备的训练批量大小
    gradient_accumulation_steps=4,  # 梯度累积步数
    warmup_steps=40,  #
    max_steps=400,  # 最大训练步数，测试时设置
    num_train_epochs=3,  # 训练轮数
    logging_steps=10,  # 日志记录频率
    save_strategy="steps",  # 模型保存策略
    save_steps=100,  # 模型保存步数
    learning_rate=2e-4,  # 学习率
    fp16=not bf16_supported,  # 如果不支持 BFloat16，则使用 FP16
    bf16=bf16_supported,  # 如果支持 BFloat16，则使用 BFloat16
    optim="adamw_8bit",  # 优化器
    seed=3407,  # 随机种子
    weight_decay=0.01,  # 正则化技术，通过在损失函数中添加一个正则化项来减小权重的大小
    lr_scheduler_type="linear",  # 学习率衰减策略
)

trainer = SFTTrainer(
    model=model,  # 模型
    tokenizer=tokenizer,  # 分词器
    args=training_args,  # 训练参数
    train_dataset=dataset,  # 训练数据集
)

# 当前GPU信息
gpu_stats = torch.cuda.get_device_properties(0)
# 当前模型内存占用
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# GPU最大内存
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# 计算总的GPU使用内存（单位：GB）
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# 计算LoRA模型使用的GPU内存（单位：GB）
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# 计算总的GPU内存使用百分比
used_percentage = round(used_memory / max_memory * 100, 3)
# 计算LoRA模型的GPU内存使用百分比
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

FastLanguageModel.for_inference(model)  # 启用原生推理速度快2倍

inputs = tokenizer(
    [
        alpaca_prompt.format(
            "描述原子的结构。",  # instruction 描述原生原生原生原生原生原生原生
            "",  # input
            "",  # output
        )
    ], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=False)

print(tokenizer.batch_decode(outputs))

lora_model = "output/llama_lora/Llama3"
model.save_pretrained(lora_model)
tokenizer.save_pretrained(lora_model)