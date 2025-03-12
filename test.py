import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name="./model/unsloth/llama-3-8b-bnb-4bit",
    model_name='./output/llama_lora/Llama3',
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

alpaca_prompt = """
下面是一项描述任务的说明，配有提供进一步背景信息的输入。写出一个适当完成请求的回应。

### Instruction:
{}

### Input:
{}

### Response:
{}
"""

while True:
    # 获取用户输入
    instruction = input("请输入你的问题（输入 'exit' 退出）：")
    if instruction.lower() == 'exit':
        print("程序已停止。0")
        break

    if not instruction:
        continue

    has_input = input("是否有额外输入信息？(y/n)：").lower()
    if has_input == 'y':
        input_info = input("请输入额外输入信息：")
    else:
        input_info = ""

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction,  # instruction
                input_info,  # input
                "",  # output
            )
        ], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)

    response = tokenizer.batch_decode(outputs)
    for item in tokenizer.batch_decode(outputs):
        if "### Response:" in item:
            start = item.index("### Response:") + len("### Response:")
            response = item[start:].strip()
            break

    print("模型回复:", response)