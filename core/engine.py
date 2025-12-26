# -*- coding: utf-8 -*-
"""
推理引擎 (API适配版)
负责模型的加载、显存优化及图片生成。
"""
import torch
from diffusers import DiffusionPipeline # type: ignore
import gc
import time
from core.utils import detect_device, get_torch_dtype
from core.lora_manager import LoRAMerger
import config

class ZImageEngine:
    # ... (前面的 __init__, is_loaded, load_model 保持不变) ...
    def __init__(self):
        self.pipe = None
        self.device = None
        self.dtype = None
        self.lora_merger = None
        self.current_lora_applied = False

    def is_loaded(self):
        return self.pipe is not None

    def load_model(self):
        # ... (保持不变) ...
        self.device = detect_device()
        self.dtype = get_torch_dtype(self.device)
        
        print(f"🚀 [Engine] 正在加载模型... 设备: {self.device.upper()}, 精度: {self.dtype}")
        
        # 清理旧显存
        if self.pipe:
            del self.pipe
            self.pipe = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()

        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                config.MODEL_PATH,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            self.pipe.to(self.device)
            
            self.lora_merger = LoRAMerger(self.pipe)
            self.current_lora_applied = False
            
            self._apply_optimizations()
            
            print("✅ [Engine] 模型加载完毕。")
            return True, f"就绪 ({self.device.upper()})"
            
        except Exception as e:
            print(f"❌ [Engine] 加载失败: {e}")
            return False, str(e)

    def _apply_optimizations(self):
        """应用优化策略"""
        # [关键修复] VAE 强制 FP32
        # 无论是在 MPS 还是 CUDA 上，VAE 在 FP16/BF16 下都极易溢出导致黑图或模糊
        # 因此，必须强制 VAE 运行在 float32 模式
        if hasattr(self.pipe, "vae"):
            self.pipe.vae.to(dtype=torch.float32) # type: ignore
            # 某些版本的 diffusers 需要显式设置 force_upcast
            if hasattr(self.pipe.vae.config, "force_upcast"): # type: ignore
                self.pipe.vae.config.force_upcast = True # type: ignore
            print("👁️ [Optim] VAE 已切换至 FP32 (防止黑图/模糊)")

        # 硬件特定优化
        if self.device == "mps":
            # MPS 显存足够时关闭 Tiling 以获得最佳画质
            pass 
        elif self.device == "cuda":
            # 12GB 显存用户必须开启 CPU Offload
            # 这会将暂时不用的模型层移到内存，腾出显存给 VAE 解码
            self.pipe.enable_model_cpu_offload() # type: ignore
            
            # [建议开启] VAE Tiling 也是防止 12G 显存解码爆显存/黑屏的关键
            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling() # type: ignore
                print("🧠 [Optim] CUDA: VAE Tiling 已开启 (节省显存)")
            
            print("🧠 [Optim] CUDA: CPU Offload 已开启")

    # ... (后面的 update_lora 和 generate 保持不变) ...
    def update_lora(self, enable, scale):
        if not self.is_loaded(): return
        if (not enable and self.current_lora_applied) or (enable and self.current_lora_applied):
            print("🔄 [Engine] LoRA 变更，重载模型...")
            self.load_model()
            if enable:
                self.lora_merger.load_lora_weights(config.LORA_PATH, scale) # type: ignore
                self.current_lora_applied = True
        elif enable and not self.current_lora_applied:
            self.lora_merger.load_lora_weights(config.LORA_PATH, scale) # type: ignore
            self.current_lora_applied = True

    def generate(self, prompt, neg_prompt, steps, cfg, width, height, seed, seed_mode):
        start_time = time.time()
        gc.collect()
        if self.device == "mps": torch.mps.empty_cache()
        if self.device == "cuda": torch.cuda.empty_cache()

        if seed_mode == "random" or seed == -1:
            actual_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            actual_seed = int(seed)
            
        gen_device = "cpu" if self.device == "mps" else self.device
        generator = torch.Generator(gen_device).manual_seed(actual_seed) # type: ignore

        print(f"🎨 [Generate] 尺寸: {width}x{height} | 步数: {steps} | 种子: {actual_seed}")

        try:
            image = self.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                generator=generator
            ).images[0] # type: ignore
            
            duration = time.time() - start_time
            
            return {
                "success": True,
                "image": image,
                "seed": actual_seed,
                "duration": round(duration, 2)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }