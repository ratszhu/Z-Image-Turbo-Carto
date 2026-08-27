# -*- coding: utf-8 -*-
"""
全局配置文件
用于统一管理模型路径、默认参数及系统常量。
"""
import os

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 基础模型路径 (请确保该路径下包含完整的 diffusers 模型文件)
MODEL_PATH = os.path.join(BASE_DIR, "Z-Image-Model")

# 用户导入的 LoRA 模型库。内置色彩增强暂不再自动加载。
LORA_DIR = os.path.join(BASE_DIR, "loras")
MAX_LORA_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

# 输出与数据库路径
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DB_PATH = os.path.join(BASE_DIR, "database", "history.db")
WEB_DIR = os.path.join(BASE_DIR, "web")

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

# --- 默认生成参数 ---
DEFAULT_PROMPT = "A cinematic shot of a cyberpunk city, neon lights, rain, high detail, 8k"
DEFAULT_NEGATIVE_PROMPT = "cartoon, painting, 3d render, low poly, blurry, low quality, distorted, ugly, watermark"

# 尺寸与步数
DEFAULT_STEPS = 9      # Turbo 模型推荐步数
DEFAULT_CFG = 0.0      # Turbo 模型推荐 CFG 为 0
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_SEED = -1      # -1 代表随机种子

# LoRA 默认设置
DEFAULT_LORA_SCALE = 1.0
DEFAULT_LORA_ENABLE = False

# CUDA 显存策略: auto / model / sequential / none
# auto 会在小于 16GB 显存时使用 sequential offload，避免先把完整模型搬入显存。
CUDA_OFFLOAD_MODE = os.getenv("ZIMAGE_CUDA_OFFLOAD", "auto").lower()

# 开发时可显式开启热重载。默认关闭，避免重载进程干扰大模型生命周期。
UVICORN_RELOAD = os.getenv("ZIMAGE_RELOAD", "0") == "1"

# --- 系统配置 ---
# 代理设置 (解决本地连接问题)
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
