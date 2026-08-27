import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from pydantic import ValidationError
from fastapi.testclient import TestClient

from core.engine import ZImageEngine
from core.lora_manager import LoRAManager
from core.utils import detect_device, get_torch_dtype, is_mps_available
from main import GenerateRequest, app, engine


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(4, 4, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attention()


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


class _Pipeline:
    def __init__(self):
        self.transformer = _Transformer()


class _DevicePipeline:
    def __init__(self):
        self.calls = []

    def to(self, device):
        self.calls.append(("to", device))

    def enable_model_cpu_offload(self, device=None):
        self.calls.append(("model", device))

    def enable_sequential_cpu_offload(self, device=None):
        self.calls.append(("sequential", device))

    def enable_vae_tiling(self):
        self.calls.append(("vae_tiling", None))


class LoRAManagerTests(unittest.TestCase):
    def test_normalizes_and_matches_base_module(self):
        raw = {
            "diffusion_model.layers.0.attention.to_q.lora_A.weight": torch.zeros(2, 4),
            "diffusion_model.layers.0.attention.to_q.lora_B.weight": torch.zeros(4, 2),
        }
        manager = LoRAManager(_Pipeline())
        converted = manager._normalize_state_dict(raw)

        self.assertIn("transformer.layers.0.attention.to_q.lora_A.weight", converted)
        self.assertEqual(manager._validate_targets(converted), 1)


class RequestValidationTests(unittest.TestCase):
    def test_rejects_dangerous_dimensions(self):
        with self.assertRaises(ValidationError):
            GenerateRequest(prompt="test", width=10_000, height=10_000)

    def test_rejects_blank_prompt(self):
        with self.assertRaises(ValidationError):
            GenerateRequest(prompt="   ")


class EngineStateTests(unittest.TestCase):
    def test_failed_load_does_not_report_ready(self):
        engine = ZImageEngine()
        with (
            patch("core.engine.DiffusionPipeline.from_pretrained", side_effect=RuntimeError("broken")),
            patch.object(engine, "_clear_device_cache"),
        ):
            success, _ = engine.load_model()

        self.assertFalse(success)
        self.assertFalse(engine.is_loaded())
        self.assertEqual(engine.state, "error")
        self.assertIsNone(engine.pipe)

    def test_stop_request_only_applies_to_active_generation(self):
        engine = ZImageEngine()
        self.assertFalse(engine.request_stop())
        engine.generation_active = True
        self.assertTrue(engine.request_stop())
        self.assertTrue(engine.cancel_event.is_set())


class WindowsCudaCompatibilityTests(unittest.TestCase):
    def test_windows_build_without_mps_backend_falls_back_safely(self):
        with (
            patch("core.utils.torch.cuda.is_available", return_value=False),
            patch("core.utils.torch.backends", SimpleNamespace()),
        ):
            self.assertFalse(is_mps_available())
            self.assertEqual(detect_device(), "cpu")

    def test_cuda_dtype_uses_runtime_bf16_capability(self):
        with patch("core.utils.torch.cuda.is_bf16_supported", return_value=True):
            self.assertEqual(get_torch_dtype("cuda"), torch.bfloat16)
        with patch("core.utils.torch.cuda.is_bf16_supported", return_value=False):
            self.assertEqual(get_torch_dtype("cuda"), torch.float16)

    def test_12gb_cuda_uses_sequential_offload_without_full_cuda_move(self):
        engine = ZImageEngine()
        engine.device = "cuda"
        engine.hardware_info = {"vram_gb": 12}
        pipe = _DevicePipeline()
        with patch("config.CUDA_OFFLOAD_MODE", "auto"):
            mode = engine._prepare_pipeline_device(pipe)
        self.assertEqual(mode, "sequential")
        self.assertIn(("sequential", "cuda"), pipe.calls)
        self.assertNotIn(("to", "cuda"), pipe.calls)

    def test_24gb_cuda_uses_model_offload_without_full_cuda_move(self):
        engine = ZImageEngine()
        engine.device = "cuda"
        engine.hardware_info = {"vram_gb": 24}
        pipe = _DevicePipeline()
        with patch("config.CUDA_OFFLOAD_MODE", "auto"):
            mode = engine._prepare_pipeline_device(pipe)
        self.assertEqual(mode, "model")
        self.assertIn(("model", "cuda"), pipe.calls)
        self.assertNotIn(("to", "cuda"), pipe.calls)


class ApiValidationTests(unittest.TestCase):
    def test_history_rejects_unbounded_limit(self):
        with patch.object(engine, "load_model", return_value=(True, "mocked")):
            with TestClient(app) as client:
                response = client.get("/api/history?limit=-1")
        self.assertEqual(response.status_code, 422)

    def test_status_exposes_loading_state(self):
        with patch.object(engine, "load_model", return_value=(True, "mocked")):
            with TestClient(app) as client:
                response = client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("state", response.json())
        self.assertIn("hardware", response.json())

    def test_static_app_uses_absolute_web_path(self):
        previous_cwd = os.getcwd()
        try:
            os.chdir("/")
            with patch.object(engine, "load_model", return_value=(True, "mocked")):
                with TestClient(app) as client:
                    response = client.get("/")
        finally:
            os.chdir(previous_cwd)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Z-Image Carto", response.text)
        self.assertIn("window.location.protocol === 'file:'", response.text)

    def test_custom_lora_upload_and_list(self):
        tensors = {
            "diffusion_model.layers.0.attention.to_q.lora_A.weight": torch.zeros(2, 4),
            "diffusion_model.layers.0.attention.to_q.lora_B.weight": torch.zeros(4, 2),
        }
        with tempfile.TemporaryDirectory() as lora_dir:
            source = os.path.join(lora_dir, "source.safetensors")
            save_file(tensors, source)
            with open(source, "rb") as handle:
                payload = handle.read()
            os.remove(source)

            with (
                patch("config.LORA_DIR", lora_dir),
                patch.object(engine, "load_model", return_value=(True, "mocked")),
                TestClient(app) as client,
            ):
                uploaded = client.post(
                    "/api/loras",
                    files={"file": ("my-face.safetensors", payload, "application/octet-stream")},
                )
                listing = client.get("/api/loras")

        self.assertEqual(uploaded.status_code, 201)
        self.assertEqual(uploaded.json()["name"], "my-face.safetensors")
        self.assertEqual(uploaded.json()["layers"], 1)
        self.assertEqual(len(listing.json()), 1)

    def test_custom_lora_rejects_non_safetensors(self):
        with tempfile.TemporaryDirectory() as lora_dir:
            with (
                patch("config.LORA_DIR", lora_dir),
                patch.object(engine, "load_model", return_value=(True, "mocked")),
                TestClient(app) as client,
            ):
                response = client.post(
                    "/api/loras",
                    files={"file": ("weights.bin", b"not-safe", "application/octet-stream")},
                )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
