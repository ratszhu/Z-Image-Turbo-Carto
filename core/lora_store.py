# -*- coding: utf-8 -*-
"""本地 LoRA 模型库。

文件名使用 `<uuid>--<原始文件名>`，无需额外数据库也能保留展示名，
同时 API 只接受 UUID，避免路径穿越。
"""
from __future__ import annotations

import os
import re
import uuid

from safetensors import safe_open

import config


_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SAFE_NAME_RE = re.compile(r"[^\w.()\-\u4e00-\u9fff]+", re.UNICODE)


def sanitize_display_name(filename: str) -> str:
    name = os.path.basename(filename).strip()
    if not name.lower().endswith(".safetensors"):
        raise ValueError("仅支持 .safetensors 格式")
    name = _SAFE_NAME_RE.sub("_", name)
    return name[:180] or "custom_lora.safetensors"


def allocate_path(original_filename: str) -> tuple[str, str]:
    lora_id = uuid.uuid4().hex
    safe_name = sanitize_display_name(original_filename)
    return lora_id, os.path.join(config.LORA_DIR, f"{lora_id}--{safe_name}")


def resolve_path(lora_id: str) -> str:
    if not _ID_RE.fullmatch(lora_id):
        raise ValueError("无效的 LoRA ID")
    prefix = f"{lora_id}--"
    matches = [name for name in os.listdir(config.LORA_DIR) if name.startswith(prefix)]
    if len(matches) != 1:
        raise FileNotFoundError("LoRA 不存在")
    return os.path.join(config.LORA_DIR, matches[0])


def inspect_safetensors(path: str) -> int:
    """只读取 safetensors 索引，不将大权重整体载入内存。"""
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    a_count = sum(key.endswith(".lora_A.weight") for key in keys)
    b_count = sum(key.endswith(".lora_B.weight") for key in keys)
    if a_count == 0 or a_count != b_count:
        raise ValueError(
            f"未找到完整的 PEFT LoRA A/B 权重 (A={a_count}, B={b_count})。"
            "请确认训练时的基础模型为 Z-Image，并导出 Diffusers/PEFT 格式。"
        )
    return a_count


def list_loras() -> list[dict]:
    items = []
    for filename in os.listdir(config.LORA_DIR):
        if "--" not in filename or not filename.lower().endswith(".safetensors"):
            continue
        lora_id, display_name = filename.split("--", 1)
        if not _ID_RE.fullmatch(lora_id):
            continue
        path = os.path.join(config.LORA_DIR, filename)
        items.append({
            "id": lora_id,
            "name": display_name,
            "size": os.path.getsize(path),
            "modified_at": os.path.getmtime(path),
        })
    return sorted(items, key=lambda item: item["modified_at"], reverse=True)
