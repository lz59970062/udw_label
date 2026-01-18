import json
import torch
from datasets import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 配置参数
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR = "./output_qwen3_cot"

# 2. 模拟 CoT 数据加载 (建议实际使用时从 jsonl 加载)
# CoT 的核心在于 assistant 的回答中包含推理过程 (Thought)
def load_cot_data():
    data = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                        {"type": "text", "text": "描述这张图片并分析场景。"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": "思维链推理：\n1. **视觉观察**：图片中显示一个女子带着一只狗在沙滩上。\n2. **细节提取**：女子穿着运动装，狗在奔跑，背景是落日余晖。\n3. **逻辑联系**：从光线强度和影子看，这是傍晚时分。从环境看，这是一次典型的户外休闲活动。\n结论：这张图片展示了日落时分人与宠物在海滩互动的温馨场景。"
                }
            ]
        }
    ]
    return Dataset.from_list(data)

# 3. 数据处理函数
def process_data(examples, processor):
    input_ids = []
    labels = []
    
    for i in range(len(examples["messages"])):
        messages = examples["messages"][i]
        
        # 使用 apply_chat_template 构造输入
        # tokenize=True 会返回 input_ids
        texts = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False, # 训练时不需要 generation prompt
            return_dict=True,
        )
        
        input_id = texts["input_ids"]
        # 在简单实现中，我们可以直接拿 input_id 作为 label (模型会自动 shift)
        # 更好的做法是把 user 部分的 label 设为 -100 以屏蔽损失
        label = input_id.clone()
        
        # 找到 assistant 回复的起始点 (这里简化处理，实际需要根据 template 寻找分割点)
        # TODO: 精确计算 assistant 之前的长度并将其 label 设为 -100
        
        input_ids.append(input_id)
        labels.append(label)
        
    return {"input_ids": input_ids, "labels": labels}

# 4. 主程序
def train():
    # 加载模型和处理器
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2", # 如果显存够且环境支持建议开启
    )

    # 配置 LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 准备数据
    dataset = load_cot_data()
    # 注意：这里需要根据 Qwen3-VL 的处理器特性进行处理
    # 简单的实现直接映射
    
    # 5. 训练配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        remove_unused_columns=False, # 必须为 False 否则会删掉图像张量
    )

    # 这里的 DataCollator 需要能处理 image tokens，transformers 官方库在最新版中已支持
    collator = DataCollatorForSeq2Seq(processor.tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset, # 这里需要预处理后的 dataset
        data_collator=collator,
    )

    # trainer.train() 

if __name__ == "__main__":
    print("这是一个简单的 Qwen3-VL CoT 微调参考实现框架。")
    # train()
