"""
Qwen 2.5 模型 GGUF 推理脚本
使用 ctransformers 库加载并使用训练好的 GGUF 模型进行推理
"""

import os
import argparse
import time
import json
from typing import List, Dict, Any, Optional
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen 2.5 GGUF 模型推理")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="outputs/gguf_model/model-unsloth-Q4_K_M.gguf",
        help="GGUF模型路径"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default=None,
        help="单次推理的提示词"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="是否使用交互模式"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=512,
        help="生成的最大token数量"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="生成的温度，越高越随机"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Top-p采样参数"
    )
    parser.add_argument(
        "--log_file", 
        type=str, 
        default="inference_logs.txt",
        help="对话日志文件"
    )
    return parser.parse_args()

def init_model(model_path: str):
    """初始化 ctransformers 模型"""
    try:
        from ctransformers import AutoModelForCausalLM
        
        logger.info(f"正在加载模型: {model_path}")
        
        # 尝试自动检测GPU支持
        try:
            # 尝试使用GPU (CUDA)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="llama",
                gpu_layers=99,  # 尝试使用尽可能多的GPU层
                context_length=2048,  # 上下文窗口大小，必须与训练时一致
            )
            logger.info("已成功使用GPU加载模型")
        except Exception as e:
            logger.warning(f"无法使用GPU加载模型: {e}")
            logger.info("尝试使用CPU加载模型...")
            # 退回到CPU
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="llama",  # 使用llama模型类型，适用于多数GGUF模型
                context_length=2048,
            )
            logger.info("已成功使用CPU加载模型")
        
        return model
    
    except ImportError:
        logger.error("请先安装ctransformers: pip install ctransformers")
        exit(1)

def format_prompt(user_message: str, system_prompt: Optional[str] = None) -> str:
    """格式化对话提示词，使用Qwen 2.5的对话格式"""
    # 默认的系统提示词
    if system_prompt is None:
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    
    # 使用Qwen 2.5的对话格式
    formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    
    return formatted_prompt

def generate_response(model, prompt: str, max_tokens: int = 512, 
                     temperature: float = 0.7, top_p: float = 0.9) -> str:
    """生成回复"""
    logger.debug(f"正在生成回复，最大token: {max_tokens}，温度: {temperature}")
    
    # 记录开始时间以计算生成速度
    start_time = time.time()
    
    # 使用ctransformers生成
    response_text = model(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|im_end|>", "<|im_start|>"],  # 停止标记
    )
    
    # 计算生成时间和速度
    end_time = time.time()
    generation_time = end_time - start_time
    tokens_generated = len(response_text.split())
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    logger.debug(f"生成完成，用时: {generation_time:.2f}秒，速度: {tokens_per_second:.2f} tokens/s")
    
    # 去掉可能的结束标记
    if "<|im_end|>" in response_text:
        response_text = response_text.split("<|im_end|>")[0]
    
    return response_text.strip()

def stream_response(model, prompt: str, max_tokens: int = 512, 
                   temperature: float = 0.7, top_p: float = 0.9):
    """流式生成回复，逐字显示"""
    full_response = ""
    
    print("\n助手: ", end="", flush=True)
    
    # 使用迭代流式生成
    for word in model.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|im_end|>", "<|im_start|>"],
        stream=True
    ):
        print(word, end="", flush=True)
        full_response += word
        
        # 检查是否到达结束标记
        if "<|im_end|>" in full_response:
            break
    
    print()  # 最后换行
    
    # 去掉可能的结束标记
    if "<|im_end|>" in full_response:
        full_response = full_response.split("<|im_end|>")[0]
    
    return full_response.strip()

def interactive_mode(model, max_tokens: int, temperature: float, top_p: float, log_file: str):
    """交互模式，持续对话"""
    logger.info("进入交互模式，输入 'exit' 或 'quit' 退出")
    
    # 对话历史
    conversation_history = []
    system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    
    try:
        # 询问用户是否要自定义系统提示词
        print("\n是否要自定义系统提示词？[y/N]: ", end="")
        if input().lower() == 'y':
            print("请输入自定义系统提示词: ")
            system_prompt = input()
            print(f"系统提示词已设置为: '{system_prompt}'")
        
        # 主对话循环
        while True:
            # 获取用户输入
            print("\n用户: ", end="")
            user_input = input()
            
            # 检查退出命令
            if user_input.lower() in ['exit', 'quit']:
                print("退出对话")
                break
            
            # 添加到对话历史
            conversation_history.append({"role": "user", "content": user_input})
            
            # 构建完整提示词
            full_prompt = format_prompt(user_input, system_prompt)
            
            # 流式生成回复
            try:
                response = stream_response(
                    model, 
                    full_prompt, 
                    max_tokens=max_tokens, 
                    temperature=temperature, 
                    top_p=top_p
                )
                
                # 添加回复到对话历史
                conversation_history.append({"role": "assistant", "content": response})
                
                # 记录对话
                log_conversation(log_file, user_input, response)
                
            except Exception as e:
                logger.error(f"生成回复时出错: {e}")
                print("\n生成回复时发生错误，请重试")
    
    except KeyboardInterrupt:
        print("\n用户中断，退出对话")
    
    # 保存完整对话历史
    save_full_conversation(log_file, conversation_history, system_prompt)

def log_conversation(log_file: str, user_input: str, assistant_response: str):
    """记录单次对话到日志文件"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] 用户: {user_input}\n")
        f.write(f"[{timestamp}] 助手: {assistant_response}\n")
        f.write("-" * 50 + "\n")

def save_full_conversation(log_file: str, conversation_history: List[Dict[str, str]], system_prompt: str):
    """保存完整对话历史为JSON格式"""
    # 构造文件名，使用原始日志文件名，但改为JSON扩展名
    json_filename = log_file.rsplit(".", 1)[0] + ".json"
    
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": system_prompt,
        "conversation": conversation_history
    }
    
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"对话历史已保存到 {json_filename}")

def single_query(model, prompt: str, max_tokens: int, temperature: float, top_p: float, log_file: str):
    """处理单次查询"""
    logger.info("处理单次查询")
    
    # 格式化提示词
    formatted_prompt = format_prompt(prompt)
    
    # 生成回复
    try:
        response = generate_response(
            model, 
            formatted_prompt, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p
        )
        
        # 显示回复
        print(f"\n助手: {response}")
        
        # 记录对话
        log_conversation(log_file, prompt, response)
        
    except Exception as e:
        logger.error(f"生成回复时出错: {e}")
        print("\n生成回复时发生错误")

def main():
    # 解析命令行参数
    args = setup_args()
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        exit(1)
    
    # 初始化模型
    model = init_model(args.model_path)
    
    # 根据模式执行操作
    if args.interactive:
        interactive_mode(model, args.max_tokens, args.temperature, args.top_p, args.log_file)
    elif args.prompt:
        single_query(model, args.prompt, args.max_tokens, args.temperature, args.top_p, args.log_file)
    else:
        logger.error("请提供提示词或使用交互模式 (--prompt 或 --interactive)")
        exit(1)

if __name__ == "__main__":
    main()
