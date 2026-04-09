# -*- coding: utf-8 -*-
"""
工具模块
包含硬件设备检测、精度选择等通用辅助函数。
"""
import torch
import platform

def detect_device():
    """
    智能检测当前系统最佳的推理设备。
    优先级: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    
    Returns:
        str: 设备名称字符串
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_torch_dtype(device: str):
    """
    根据设备类型自动匹配最佳计算精度。
    
    Args:
        device (str): 设备名称
        
    Returns:
        torch.dtype: 推荐的张量数据类型
    """
    if device == "cuda":
        # NVIDIA 显卡: FP16 性能最佳且显存占用低
        return torch.bfloat16 #强制修改为 bfloat16 ，但不支持RTX 20系 (Turing), GTX 10系 (Pascal) 及更老架构的显卡
    elif device == "mps":
        # Apple Silicon: 必须使用 Bfloat16
        # 原因1: 避免 FP16 的溢出(黑屏)问题
        # 原因2: 相比 FP32 节省一半显存，速度翻倍
        return torch.bfloat16
    else:
        # CPU: 兜底使用 FP32，兼容性最好
        return torch.float32
