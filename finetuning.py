import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
import logging
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from transformers import TextStreamer # Added import

# (Keep logging setup as before)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    # (Keep CUDA check and basic parameters as before)
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        logger.info(f"当前GPU: {torch.cuda.current_device()}")

    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16 # Safer dtype setting
    load_in_4bit = True
    max_seq_length = 2000
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("正在加载Qwen 2.5模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct",
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    logger.info("正在添加LoRA适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", # Make sure this is correct for 4-bit
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # --- Data Preparation ---
    logger.info("正在设置对话模板...")
    # Important: Use the *exact* template name if known, or let Unsloth auto-detect.
    # Check Unsloth/HF Hub for the precise template name for "unsloth/Qwen2.5-7B-Instruct"
    # It might be 'qwen' or 'chatml' or similar depending on how Unsloth configured it.
    # Let's assume 'qwen-2.5' is correct based on your code.
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5", # Double-check this template name is correct for the Unsloth version
        map_eos_token=True, # Add EOS token
    )

    # Formatting function remains the same
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    logger.info("正在加载并处理数据集...")
    # Consider using a smaller split for debugging: split="train[:1%]"
    dataset = load_dataset("mlabonne/FineTome-100k", split="train") # Using full dataset

    # Standardize should happen *before* applying the template in this case
    # Check if standardize_sharegpt is needed with FineTome format
    try:
        dataset = standardize_sharegpt(dataset)
        logger.info("Dataset standardized.")
    except Exception as e:
        logger.warning(f"Could not standardize dataset (may not be needed for FineTome): {e}")
        # Ensure the column name matches what formatting_prompts_func expects
        if "conversations" not in dataset.column_names:
             logger.warning("Dataset does not have 'conversations' column after potential standardization attempt. Adjusting.")
             # Add logic here if the column name changes (e.g., rename if necessary)
             # dataset = dataset.rename_column("old_name", "conversations")


    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=os.cpu_count() // 2) # Use half CPU cores
    logger.info(f"Dataset formatted. Example text: {dataset[0]['text'][:500]}") # Log an example

    logger.info(f"Original dataset size: {len(dataset)}")
    # Define a function to check token length
    def is_sequence_length_valid(example):
        # Tokenize the text field to check its length
        return len(tokenizer(example['text']).input_ids) <= max_seq_length

    # Filter the dataset
    # This can be slow if run sequentially, use num_proc
    dataset = dataset.filter(is_sequence_length_valid, num_proc=os.cpu_count() // 2)
    logger.info(f"Filtered dataset size (<= {max_seq_length} tokens): {len(dataset)}")

    # Add a check to ensure the dataset is not empty after filtering
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty after filtering sequences longer than {max_seq_length} tokens. "
                        "Check max_seq_length or the source dataset.")


    logger.info("正在配置训练器...")

    # Data Collator: Still useful for padding strategy
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest", # Pad to longest sequence in batch
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    # SFTTrainer setup
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset, # Pass the dataset with the 'text' column
        args=TrainingArguments(
            per_device_train_batch_size=2, # Keep small batch size
            gradient_accumulation_steps=4, # Effective batch size 8
            warmup_steps=10, # Slightly more warmup
            max_steps = 100, # Use max_steps for quick testing, comment out for full run
            num_train_epochs=1, # Use num_train_epochs for full run
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5, # Log more frequently during debug
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_strategy="steps", # Save checkpoints during training
            save_steps=200,       # Adjust frequency as needed
            report_to="none",    # Or "tensorboard", "wandb"
            remove_unused_columns=True, # Let Trainer remove columns not used by model forward
            max_grad_norm=1.0, # Keep gradient clipping
        ),
        data_collator=data_collator, # Pass the collator
        # packing=False, # Consider setting packing=False if issues persist, might use more memory
    )

    # Check if the dataset has the required column BEFORE applying train_on_responses_only
    if "text" not in trainer.train_dataset.column_names:
       raise ValueError(f"Dataset missing 'text' field before applying train_on_responses_only. Columns: {trainer.train_dataset.column_names}")

    # --- Apply train_on_responses_only ---
    # This modifies the trainer's internal dataset processing to mask labels.
    # Ensure the parts match the *exact* output of the tokenizer's chat template application.
    # Log the template parts for verification.
    logger.info("Applying train_on_responses_only...")
    # Example for Qwen format:
    # User: <|im_start|>user\n{prompt}<|im_end|>\n
    # Assistant: <|im_start|>assistant\n{response}<|im_end|>
    # The parts should be the tokens marking the start of user and assistant turns.
    instruction_part = "<|im_start|>user" # Check if '\n' is included by tokenizer
    response_part = "<|im_start|>assistant"
    logger.info(f"Using instruction part: '{instruction_part}'")
    logger.info(f"Using response part: '{response_part}'")

    # It's safer to apply this *after* SFTTrainer initialization but *before* training.
    # Note: This function might internally re-process the dataset or set up label masking.
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part, # Or use chatml_tags=True if it's a standard ChatML template
        response_part=response_part,
    )
    logger.info("Applied train_on_responses_only.")


    # --- Execute Training ---
    # REMOVED the outer try...except block around train() for clearer debugging
    # REMOVED the fallback Trainer logic

    # Check dataset columns *after* train_on_responses_only might modify it
    logger.info(f"Columns in trainer's dataset before training: {trainer.train_dataset.column_names}")
    # It should now contain input_ids, attention_mask, labels (potentially modified)

    logger.info("开始训练过程...")
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1' # Keep this if needed by Unsloth optimizations

    # Use a try-finally to ensure model cleanup or state reset if needed,
    # but let the original error propagate for debugging.
    try:
        trainer_stats = trainer.train()
        logger.info(f"训练完成! 用时: {trainer_stats.metrics['train_runtime']:.2f}秒")
        logger.info(f"训练指标: {trainer_stats.metrics}")

        # --- Save Model ---
        logger.info("保存LoRA适配器...")
        # Use safe_serialization=True for compatibility
        model.save_pretrained(os.path.join(output_dir, "lora_model"), safe_serialization=True)
        tokenizer.save_pretrained(os.path.join(output_dir, "lora_model"))
        logger.info(f"模型保存在: {os.path.join(output_dir, 'lora_model')}")

    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True) # Log full traceback
        # Rethrow the exception if you want the script to terminate on error
        raise e
    finally:
        # Optional: Clean up GPU memory if necessary
        # del model
        # del trainer
        # torch.cuda.empty_cache()
        pass


    # --- GGUF Export and Inference (Keep as before, but ensure model is loaded correctly) ---
    logger.info("正在导出为GGUF格式（q4_k_m）...")
    try:
        # 可选: 如果 save_pretrained_gguf 需要合并后的模型，先执行 merge_and_unload
        # logger.info("Merging adapters before GGUF conversion...")
        # model = model.merge_and_unload() # 取消注释以启用合并（如果需要）

        model.save_pretrained_gguf(
            os.path.join(output_dir, "gguf_model"),
            tokenizer,
            quantization_method="q4_k_m" # 确认这是你想要的量化方法
        )
        logger.info(f"GGUF模型保存在: {os.path.join(output_dir, 'gguf_model')}") # 提供保存路径
        logger.info("GGUF导出完成!")
    except Exception as e:
        logger.error(f"导出GGUF时出错: {e}", exc_info=True) # 打印完整错误信息

    logger.info("进行推理测试...")
    # Ensure model is in inference mode (Unsloth might handle this)
    FastLanguageModel.for_inference(model)

    # Re-apply chat template for inference if necessary (tokenizer state might have changed)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5", # Ensure consistency
        map_eos_token=True,
    )

    messages = [
        {"role": "user", "content": "What is the difference between k8s and docker-compose?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True, # Important for inference
        return_tensors="pt",
    ).to("cuda")

    logger.info("生成回答...")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.7,
        # top_p=0.9, # Consider adding top_p
        do_sample=True, # Sample for non-deterministic output
    )

    logger.info("全部流程完成！")


if __name__ == "__main__":
    main()