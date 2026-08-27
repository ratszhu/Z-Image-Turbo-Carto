# -*- coding: utf-8 -*-
"""Z-Image LoRA adapter 管理。

LoRA 以 PEFT adapter 形式挂载到 Transformer，避免直接、不可逆地修改基础权重。
这样开关和调整强度都不需要重新加载 6B 基础模型。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import safetensors.torch


@dataclass(frozen=True)
class LoRAUpdateResult:
    success: bool
    message: str
    matched_layers: int = 0


class LoRAManager:
    ADAPTER_NAME = "custom_lora"

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.loaded_path: str | None = None
        self.enabled = False
        self.scale = 0.0
        self.matched_layers = 0

    def _delete_loaded_adapter(self) -> None:
        """删除旧 adapter，保证切换 LoRA 时不叠加、不重名。"""
        if not self.loaded_path:
            return
        self.transformer.disable_adapters()
        self.transformer.delete_adapters(self.ADAPTER_NAME)
        self.loaded_path = None
        self.enabled = False
        self.scale = 0.0
        self.matched_layers = 0

    @property
    def transformer(self):
        transformer = getattr(self.pipeline, "transformer", None)
        if transformer is None:
            raise RuntimeError("当前 pipeline 不包含 transformer，无法加载 LoRA")
        return transformer

    @staticmethod
    def _normalize_state_dict(state_dict):
        """把训练文件的 diffusion_model.* 键转换成 Diffusers/PEFT 格式。"""
        converted = {}
        for key, value in state_dict.items():
            if not (key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")):
                continue
            normalized = key.removeprefix("diffusion_model.")
            converted[f"transformer.{normalized}"] = value
        return converted

    def _validate_targets(self, state_dict) -> int:
        """在注入前确认 LoRA 指向的基础层真实存在。"""
        matched = 0
        transformer = self.transformer
        for key in state_dict:
            if not key.endswith(".lora_A.weight"):
                continue
            module_path = key.removeprefix("transformer.").removesuffix(".lora_A.weight")
            try:
                module = transformer.get_submodule(module_path)
            except (AttributeError, KeyError):
                continue
            if hasattr(module, "weight"):
                matched += 1
        return matched

    def load(self, lora_path: str, scale: float = 1.0) -> LoRAUpdateResult:
        if not os.path.isfile(lora_path):
            return LoRAUpdateResult(False, f"LoRA 文件未找到: {lora_path}")

        try:
            import peft  # noqa: F401  # PEFT 是 Diffusers adapter 的运行时依赖
            from diffusers.utils.peft_utils import set_weights_and_activate_adapters
        except ImportError:
            return LoRAUpdateResult(False, "缺少 PEFT 依赖，请执行 pip install -r requirements.txt")

        try:
            raw_state_dict = safetensors.torch.load_file(lora_path, device="cpu")
            state_dict = self._normalize_state_dict(raw_state_dict)
            del raw_state_dict

            a_count = sum(key.endswith(".lora_A.weight") for key in state_dict)
            b_count = sum(key.endswith(".lora_B.weight") for key in state_dict)
            if a_count == 0 or a_count != b_count:
                raise ValueError(f"LoRA A/B 权重不完整: A={a_count}, B={b_count}")

            matched = self._validate_targets(state_dict)
            if matched == 0:
                raise ValueError("LoRA 未匹配到任何 Transformer 层")
            if matched != a_count:
                raise ValueError(f"LoRA 仅匹配 {matched}/{a_count} 层，已拒绝部分注入")

            # 校验全部通过后才卸载旧 LoRA，避免一个无效文件影响当前状态。
            if self.loaded_path:
                self._delete_loaded_adapter()

            self.transformer.load_lora_adapter(
                state_dict,
                prefix="transformer",
                adapter_name=self.ADAPTER_NAME,
                low_cpu_mem_usage=True,
                _pipeline=self.pipeline,
            )
            set_weights_and_activate_adapters(
                self.transformer,
                [self.ADAPTER_NAME],
                [float(scale)],
            )
            self.transformer.enable_adapters()

            self.loaded_path = lora_path
            self.enabled = True
            self.scale = float(scale)
            self.matched_layers = matched
            return LoRAUpdateResult(True, f"LoRA 已加载，共匹配 {matched} 层", matched)
        except Exception as exc:
            return LoRAUpdateResult(False, f"LoRA 加载失败: {exc}")

    def unload(self) -> LoRAUpdateResult:
        try:
            self._delete_loaded_adapter()
            return LoRAUpdateResult(True, "LoRA 已卸载")
        except Exception as exc:
            return LoRAUpdateResult(False, f"LoRA 卸载失败: {exc}")

    def update(self, enable: bool, scale: float) -> LoRAUpdateResult:
        if not self.loaded_path:
            return LoRAUpdateResult(False, "LoRA 尚未加载")

        try:
            if enable:
                from diffusers.utils.peft_utils import set_weights_and_activate_adapters

                set_weights_and_activate_adapters(
                    self.transformer,
                    [self.ADAPTER_NAME],
                    [float(scale)],
                )
                self.transformer.enable_adapters()
                self.enabled = True
                self.scale = float(scale)
                return LoRAUpdateResult(True, f"LoRA 已启用，强度 {self.scale}", self.matched_layers)

            self.transformer.disable_adapters()
            self.enabled = False
            return LoRAUpdateResult(True, "LoRA 已停用", self.matched_layers)
        except Exception as exc:
            return LoRAUpdateResult(False, f"LoRA 状态更新失败: {exc}", self.matched_layers)


# 保留旧名称，避免第三方代码导入后立即失败。
LoRAMerger = LoRAManager
