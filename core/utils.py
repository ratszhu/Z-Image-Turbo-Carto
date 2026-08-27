# -*- coding: utf-8 -*-
"""
工具模块
包含硬件设备检测、精度选择等通用辅助函数。
"""
import torch
from typing import Any


def is_mps_available() -> bool:
    """Windows/Linux 的 PyTorch 构建不保证暴露 MPS 后端。"""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def detect_device():
    """
    智能检测当前系统最佳的推理设备。
    优先级: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    
    Returns:
        str: 设备名称字符串
    """
    if torch.cuda.is_available():
        return "cuda"
    elif is_mps_available():
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
        # BF16 可以解决部分新显卡上的 FP16 黑图，但旧架构并不支持。
        # 必须按运行时能力判断，不能按显卡名称或系列硬编码。
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == "mps":
        # Apple Silicon 使用 BF16，VAE 在引擎中单独保持 FP32。
        return torch.bfloat16
    else:
        # CPU: 兜底使用 FP32，兼容性最好
        return torch.float32


def get_hardware_info() -> dict[str, Any]:
    """返回可直接暴露给状态接口的硬件诊断信息。"""
    device = detect_device()
    info: dict[str, Any] = {
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": is_mps_available(),
    }

    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        info.update({
            "device_name": properties.name,
            "vram_gb": round(properties.total_memory / 1024**3, 2),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    elif device == "mps":
        info["device_name"] = "Apple Silicon (MPS)"
        info["bf16_supported"] = True
    else:
        info["device_name"] = "CPU"
        info["bf16_supported"] = False

    return info
