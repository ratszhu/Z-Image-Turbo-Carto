# -*- coding: utf-8 -*-
"""Z-Image 推理引擎及其生命周期管理。"""
from __future__ import annotations

import gc
import os
import threading
import time
from typing import Any

import torch
from diffusers import DiffusionPipeline

import config
from core.lora_manager import LoRAManager, LoRAUpdateResult
from core.utils import detect_device, get_hardware_info, get_torch_dtype, is_mps_available


class GenerationCancelled(Exception):
    """用户主动停止本次采样。"""


class ZImageEngine:
    def __init__(self):
        self.pipe = None
        self.device = detect_device()
        self.dtype = get_torch_dtype(self.device)
        self.hardware_info = get_hardware_info()

        self.state = "idle"
        self.status_message = "等待加载模型"
        self.error: str | None = None
        self.offload_mode = "none"
        self.generation_active = False
        self.cancel_event = threading.Event()

        self.lora_manager: LoRAManager | None = None
        self.current_lora_applied = False
        self.current_lora_scale: float | None = None
        self.current_lora_path: str | None = None
        self.lora_error: str | None = None

        # update_lora + generate 必须作为一个原子操作执行，避免多请求修改同一模型。
        self.inference_lock = threading.RLock()

    def is_loaded(self) -> bool:
        return self.state == "ready" and self.pipe is not None

    def get_status(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded(),
            "state": self.state,
            "message": self.status_message,
            "error": self.error,
            "device": self.device,
            "dtype": str(self.dtype),
            "offload_mode": self.offload_mode,
            "busy": self.generation_active,
            "cancelling": self.generation_active and self.cancel_event.is_set(),
            "lora_enabled": self.current_lora_applied,
            "lora_scale": self.current_lora_scale,
            "lora_id": os.path.basename(self.current_lora_path).split("--", 1)[0] if self.current_lora_path else None,
            "lora_error": self.lora_error,
            "hardware": self.hardware_info,
        }

    @staticmethod
    def _clear_device_cache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_mps_available():
            torch.mps.empty_cache()

    def _select_cuda_offload_mode(self) -> str:
        configured = config.CUDA_OFFLOAD_MODE
        allowed = {"auto", "model", "sequential", "none"}
        if configured not in allowed:
            raise ValueError(f"无效的 ZIMAGE_CUDA_OFFLOAD={configured}，可选值: {sorted(allowed)}")
        if configured != "auto":
            return configured

        vram_gb = float(self.hardware_info.get("vram_gb") or 0)
        return "sequential" if vram_gb < 16 else "model"

    def _prepare_pipeline_device(self, pipe) -> str:
        """在不制造整模型 GPU 峰值的前提下设置运行设备。"""
        if self.device == "mps":
            pipe.to("mps")
            return "none"
        if self.device == "cpu":
            pipe.to("cpu")
            return "none"

        mode = self._select_cuda_offload_mode()
        if mode == "none":
            pipe.to("cuda")
        elif mode == "model":
            pipe.enable_model_cpu_offload(device="cuda")
        else:
            pipe.enable_sequential_cpu_offload(device="cuda")
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        return mode

    @staticmethod
    def _prepare_vae(pipe) -> None:
        vae = getattr(pipe, "vae", None)
        if vae is None:
            return
        # 在 pipeline 仍位于 CPU 时转换，避免显存已满后再把 VAE 扩成 FP32。
        vae.to(dtype=torch.float32)
        if hasattr(vae.config, "force_upcast"):
            vae.config.force_upcast = True

    def load_model(self) -> tuple[bool, str]:
        with self.inference_lock:
            self.state = "loading"
            self.status_message = "正在读取模型权重"
            self.error = None
            self.lora_error = None
            self.current_lora_applied = False
            self.current_lora_scale = None
            self.current_lora_path = None
            self.offload_mode = "none"
            local_pipe = None

            try:
                self.device = detect_device()
                self.dtype = get_torch_dtype(self.device)
                self.hardware_info = get_hardware_info()
                print(f"🚀 [Engine] 正在加载模型... 设备: {self.device.upper()}, 精度: {self.dtype}")

                local_pipe = DiffusionPipeline.from_pretrained(
                    config.MODEL_PATH,
                    # 兼容仓库现有的 0.36.0.dev0；新版 Diffusers 仍保留该别名。
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
                self.status_message = "正在配置 VAE 与 LoRA"
                self._prepare_vae(local_pipe)

                local_lora_manager = LoRAManager(local_pipe)
                self.status_message = "正在配置显存策略"
                offload_mode = self._prepare_pipeline_device(local_pipe)

                # 只有所有关键步骤成功后，才发布新的 pipeline。
                previous_pipe = self.pipe
                self.pipe = local_pipe
                self.lora_manager = local_lora_manager
                self.offload_mode = offload_mode
                self.current_lora_applied = local_lora_manager.enabled
                self.current_lora_scale = local_lora_manager.scale if local_lora_manager.enabled else None
                self.current_lora_path = local_lora_manager.loaded_path
                self.state = "ready"
                self.status_message = f"模型就绪 ({self.device.upper()})"
                local_pipe = None

                if previous_pipe is not None:
                    del previous_pipe
                    self._clear_device_cache()

                print(f"✅ [Engine] 模型加载完毕，offload={self.offload_mode}")
                return True, self.status_message
            except Exception as exc:
                self.pipe = None
                self.lora_manager = None
                self.state = "error"
                self.error = str(exc)
                self.status_message = "模型加载失败"
                if local_pipe is not None:
                    del local_pipe
                self._clear_device_cache()
                print(f"❌ [Engine] 加载失败: {exc}")
                return False, str(exc)

    def update_lora(self, enable: bool, scale: float, lora_path: str | None = None) -> LoRAUpdateResult:
        if not self.is_loaded() or self.lora_manager is None:
            return LoRAUpdateResult(False, "模型尚未就绪")

        if enable and not lora_path:
            return LoRAUpdateResult(False, "请先选择一个自定义 LoRA")

        if enable and os.path.realpath(lora_path) != os.path.realpath(self.lora_manager.loaded_path or ""):
            result = self.lora_manager.load(lora_path, scale)
        else:
            result = self.lora_manager.update(enable, scale)

        if result.success:
            self.current_lora_applied = self.lora_manager.enabled
            self.current_lora_scale = self.lora_manager.scale if self.lora_manager.enabled else None
            self.current_lora_path = self.lora_manager.loaded_path
            self.lora_error = None
        else:
            self.lora_error = result.message
        return result

    def generate(
        self,
        prompt: str,
        neg_prompt: str,
        steps: int,
        cfg: float,
        width: int,
        height: int,
        seed: int,
        seed_mode: str,
    ) -> dict[str, Any]:
        if not self.is_loaded():
            return {"success": False, "error": self.error or "模型尚未就绪"}

        start_time = time.time()
        actual_seed = (
            torch.randint(0, 2**32 - 1, (1,)).item()
            if seed_mode == "random" or seed == -1
            else int(seed)
        )
        generator_device = "cpu" if self.device == "mps" else self.device
        generator = torch.Generator(generator_device).manual_seed(actual_seed)

        print(f"🎨 [Generate] 尺寸: {width}x{height} | 步数: {steps} | 种子: {actual_seed}")
        self.cancel_event.clear()
        self.generation_active = True

        def check_cancellation(_pipeline, _step_index, _timestep, callback_kwargs):
            if self.cancel_event.is_set():
                raise GenerationCancelled("生成已停止")
            return callback_kwargs

        try:
            with torch.inference_mode():
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=check_cancellation,
                ).images[0]
            return {
                "success": True,
                "image": image,
                "seed": actual_seed,
                "duration": round(time.time() - start_time, 2),
            }
        except GenerationCancelled:
            return {"success": False, "cancelled": True, "error": "生成已停止"}
        except Exception as exc:
            self._clear_device_cache()
            return {"success": False, "error": str(exc)}
        finally:
            self.generation_active = False
            self.cancel_event.clear()

    def request_stop(self) -> bool:
        """请求在当前扩散步结束后中断，不等待推理锁。"""
        if not self.generation_active:
            return False
        self.cancel_event.set()
        return True
