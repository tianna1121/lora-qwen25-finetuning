# 交互模式
python inference.py --model_path outputs/gguf_model/unsloth.Q4_K_M.gguf --interactive

# 单次查询模式
python inference.py --model_path outputs/gguf_model/unsloth.Q4_K_M.gguf --prompt "请解释人工智能的工作原理"

# 自定义参数
python inference.py --model_path outputs/gguf_model/unsloth.Q4_K_M.gguf --interactive --max_tokens 1024 --temperature 0.8 --top_p 0.95