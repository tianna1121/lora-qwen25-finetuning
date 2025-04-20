import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer
import logging
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_generation(model, tokenizer, prompt, **gen_kwargs):
    """测试生成函数"""
    logger.info(f"测试提示词: '{prompt}'")
    messages = [{"role": "user", "content": prompt}]
    
    # 应用聊天模板并进行显式标记化
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)
    
    # 确保输入长度在模型限制内
    max_length = model.config.max_position_embeddings
    if inputs.shape[1] > max_length:
        logger.warning(f"输入长度 {inputs.shape[1]} 超过模型上下文长度 {max_length}。进行截断。")
        inputs = inputs[:, :max_length]
    
    # 创建注意力掩码
    attention_mask = torch.ones_like(inputs)
    
    # 生成设置（增强防重复设置）
    default_gen_kwargs = {
        "max_new_tokens": 256,          # 增加生成长度
        "temperature": 0.7,             # 稍微提高温度增加多样性
        "top_p": 0.92,                  # 略微提高多样性
        "repetition_penalty": 1.3,      # 适中的重复惩罚
        "no_repeat_ngram_size": 4,      # 防止4-gram重复
        "do_sample": True,
        "use_cache": True,
    }
    # 覆盖默认设置
    default_gen_kwargs.update(gen_kwargs)
    
    # 创建文本流
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    outputs = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        streamer=text_streamer,
        **default_gen_kwargs
    )
    
    # 返回完整生成输出
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

class SimpleLMDataCollator:
    """简单数据整理器，只返回张量字典"""
    def __call__(self, examples):
        batch = {
            "input_ids": torch.stack([torch.tensor(ex["input_ids"]) for ex in examples]),
            "attention_mask": torch.stack([torch.tensor(ex["attention_mask"]) for ex in examples]),
            "labels": torch.stack([torch.tensor(ex["labels"]) for ex in examples])
        }
        return batch

def main():
    # CUDA可用性检查
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        logger.info(f"当前GPU: {torch.cuda.current_device()}")

    # 设置参数
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    load_in_4bit = True
    
    # 关键参数：最大序列长度
    # 增加到1536，接近模型最大值但保留安全余量
    max_seq_length = 2000  
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    logger.info("加载 Qwen 2.5 模型...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-7B-Instruct",
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            max_seq_length=max_seq_length,
        )
        logger.info(f"模型加载成功，max_seq_length: {max_seq_length}")
    except Exception as e:
        logger.error(f"加载模型出错: {e}", exc_info=True)
        raise

    # 添加 LoRA 适配器（提高参数）
    logger.info("添加 LoRA 适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # 提高到32，增强模型学习能力
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=64,  # 提高到64
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    # 应用聊天模板
    logger.info("设置聊天模板...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",  # 使用 chatml
        map_eos_token=True,
    )
    
    # 加载数据集（增加比例）
    logger.info("加载和处理数据集...")
    dataset = load_dataset("mlabonne/FineTome-100k", split="train[:60%]")  # 使用更多数据
    
    try:
        dataset = standardize_sharegpt(dataset)
        logger.info("数据集标准化成功。")
    except Exception as e:
        logger.warning(f"无法标准化数据集 (可能不需要): {e}")
        if "conversations" not in dataset.column_names:
            raise ValueError(f"数据集缺少 'conversations' 列。列: {dataset.column_names}")
    
    # 检查数据集结构
    logger.info(f"数据集列: {dataset.column_names}")
    logger.info(f"样本对话: {dataset[0]['conversations'][:2]}")
    
    # 手动预处理 - 预计算所有标记化示例，确保统一形状
    # 使用更高效的批处理方式
    def process_batch(examples):
        conversations_list = examples["conversations"]
        result = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for conversations in conversations_list:
            # 格式化为纯文本
            text = tokenizer.apply_chat_template(
                conversations, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # 进行标记化，带有填充和截断
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors="np"
            )
            
            result["input_ids"].append(encoded.input_ids[0].tolist())
            result["attention_mask"].append(encoded.attention_mask[0].tolist())
            result["labels"].append(encoded.input_ids[0].tolist())
            
        return result
    
    # 使用批处理方式处理数据集
    logger.info("批量处理数据集...")
    tokenized_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=64,  # 批量大小
        remove_columns=["conversations"],
        num_proc=os.cpu_count() // 2,  # 并行处理
        desc="预处理数据集"
    )
    
    logger.info(f"创建了包含 {len(tokenized_dataset)} 个示例的数据集")
    logger.info(f"示例 input_ids 长度: {len(tokenized_dataset[0]['input_ids'])}")
    
    # 创建数据整理器
    data_collator = SimpleLMDataCollator()
    
    # 配置训练参数（提高训练步数和学习率）
    training_args = TrainingArguments(
        per_device_train_batch_size=4,  # 提高到2
        gradient_accumulation_steps=4,  # 调整梯度累积
        warmup_steps=100,               # 增加预热步数
        max_steps=1000,                 # 增加训练步数
        learning_rate=1e-4,             # 提高学习率
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",     # 余弦调度
        seed=3407,
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=200,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=0.5,              # 提高梯度裁剪阈值
        gradient_checkpointing=True,    # 启用梯度检查点
        ddp_find_unused_parameters=False,
    )

    # 初始化标准 Trainer
    logger.info("配置训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 训练模型
    logger.info("开始训练过程...")
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    try:
        trainer_stats = trainer.train()
        logger.info(f"训练完成! 用时: {trainer_stats.metrics['train_runtime']:.2f} 秒")
        
        # 保存模型
        logger.info("保存 LoRA 适配器...")
        model.save_pretrained(os.path.join(output_dir, "lora_model"), safe_serialization=True)
        tokenizer.save_pretrained(os.path.join(output_dir, "lora_model"))
        logger.info(f"模型保存在: {os.path.join(output_dir, 'lora_model')}")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}", exc_info=True)
        raise e
    finally:
        torch.cuda.empty_cache()

    # 导出为 GGUF 格式
    logger.info("导出为 GGUF 格式 (q4_k_m)...")
    try:
        model.save_pretrained_gguf(
            os.path.join(output_dir, "gguf_model"),
            tokenizer,
            quantization_method="q4_k_m"
        )
        logger.info(f"GGUF 模型保存在: {os.path.join(output_dir, 'gguf_model')}")
    except Exception as e:
        logger.error(f"导出 GGUF 出错: {e}", exc_info=True)

    # 运行测试推理
    logger.info("运行推理测试...")
    FastLanguageModel.for_inference(model)
    
    # 使用各种提示测试
    test_prompts = [
        "什么是Kubernetes (k8s)以及它与Docker Compose的区别?",
        "用通俗的语言解释机器学习",
        "Python的主要特点是什么?",
        "如何设计一个高可用的微服务架构?",
        "解释一下量子计算的基本原理"
    ]
    
    for prompt in test_prompts:
        logger.info("-" * 40)
        logger.info(f"测试提示词: {prompt}")
        
        # 使用调整后的生成参数测试
        test_generation(model, tokenizer, prompt)
    
    logger.info("所有过程完成!")

if __name__ == "__main__":
    main()