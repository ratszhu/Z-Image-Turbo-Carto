# -*- coding: utf-8 -*-
"""Z-Image Carto API 与静态页面入口。"""
from __future__ import annotations

import os
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import config
from core.engine import ZImageEngine
from core.lora_store import allocate_path, inspect_safetensors, list_loras, resolve_path
from database.db_manager import DatabaseManager


engine = ZImageEngine()
db = DatabaseManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """先启动 Web 服务，再在线程中加载大模型。"""
    print("🌟 Web 服务已启动，正在后台加载模型...")
    threading.Thread(target=engine.load_model, name="model-loader", daemon=True).start()
    yield


app = FastAPI(title="Z-Image Carto", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8888", "http://localhost:8888"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000)
    steps: int = Field(default=config.DEFAULT_STEPS, ge=1, le=50)
    cfg: float = Field(default=config.DEFAULT_CFG, ge=0.0, le=10.0)
    width: int = Field(default=config.DEFAULT_WIDTH, ge=512, le=2048, multiple_of=64)
    height: int = Field(default=config.DEFAULT_HEIGHT, ge=512, le=2048, multiple_of=64)
    seed: int = Field(default=config.DEFAULT_SEED, ge=-1, le=2**32 - 1)
    seed_mode: Literal["fixed", "random"] = "fixed"
    lora_enabled: bool = config.DEFAULT_LORA_ENABLE
    lora_scale: float = Field(default=config.DEFAULT_LORA_SCALE, ge=0.0, le=2.0)
    lora_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt 不能为空")
        return value


@app.get("/api/status")
def get_status():
    return engine.get_status()


@app.get("/api/loras")
def get_loras():
    current_id = engine.get_status().get("lora_id")
    return [
        {**item, "active": item["id"] == current_id and engine.current_lora_applied}
        for item in list_loras()
    ]


@app.post("/api/loras", status_code=201)
async def upload_lora(file: UploadFile = File(...)):
    try:
        lora_id, destination = allocate_path(file.filename or "")
    except ValueError as exc:
        await file.close()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    temporary = f"{destination}.uploading"
    total = 0
    try:
        with open(temporary, "xb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > config.MAX_LORA_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="LoRA 文件超过 2 GB 上限")
                output.write(chunk)
        layer_count = inspect_safetensors(temporary)
        os.replace(temporary, destination)
    except HTTPException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    except Exception as exc:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise HTTPException(status_code=400, detail=f"LoRA 文件校验失败: {exc}") from exc
    finally:
        await file.close()

    item = next(item for item in list_loras() if item["id"] == lora_id)
    return {**item, "layers": layer_count, "active": False}


@app.delete("/api/loras/{lora_id}")
def delete_lora(lora_id: str):
    try:
        path = resolve_path(lora_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    with engine.inference_lock:
        if engine.current_lora_path and os.path.realpath(engine.current_lora_path) == os.path.realpath(path):
            result = engine.lora_manager.unload() if engine.lora_manager else None
            if result and not result.success:
                raise HTTPException(status_code=500, detail=result.message)
            engine.current_lora_applied = False
            engine.current_lora_scale = None
            engine.current_lora_path = None
        os.remove(path)
    return {"status": "deleted"}


@app.post("/api/generate")
def generate_image(req: GenerateRequest):
    if not engine.is_loaded():
        detail = engine.error or engine.status_message
        raise HTTPException(status_code=503, detail=detail)

    # 同一时刻只允许一个请求修改 adapter 或调用 pipeline。
    with engine.inference_lock:
        selected_path = None
        if req.lora_enabled:
            if not req.lora_id:
                raise HTTPException(status_code=400, detail="请先选择一个自定义 LoRA")
            try:
                selected_path = resolve_path(req.lora_id)
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        selected_changed = bool(
            req.lora_enabled
            and selected_path
            and os.path.realpath(selected_path) != os.path.realpath(engine.current_lora_path or "")
        )
        scale_changed = (
            req.lora_enabled
            and engine.current_lora_scale is not None
            and abs(req.lora_scale - engine.current_lora_scale) > 1e-6
        )
        if req.lora_enabled != engine.current_lora_applied or scale_changed or selected_changed:
            lora_result = engine.update_lora(req.lora_enabled, req.lora_scale, selected_path)
            if not lora_result.success:
                raise HTTPException(status_code=500, detail=lora_result.message)

        result = engine.generate(
            prompt=req.prompt,
            neg_prompt=req.negative_prompt,
            steps=req.steps,
            cfg=req.cfg,
            width=req.width,
            height=req.height,
            seed=req.seed,
            seed_mode=req.seed_mode,
        )

        if not result["success"]:
            if result.get("cancelled"):
                raise HTTPException(status_code=409, detail="生成已停止")
            raise HTTPException(status_code=500, detail=result["error"])

        filename = f"{uuid.uuid4().hex}.png"
        save_path = os.path.join(config.OUTPUT_DIR, filename)
        result["image"].save(save_path, format="PNG")

        record = {
            "filename": filename,
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "steps": req.steps,
            "cfg": req.cfg,
            "seed": result["seed"],
            "width": req.width,
            "height": req.height,
            "lora_enabled": engine.current_lora_applied,
            "lora_scale": engine.current_lora_scale or 0.0,
            "lora_id": req.lora_id if engine.current_lora_applied else None,
            "lora_name": (
                os.path.basename(engine.current_lora_path).split("--", 1)[-1]
                if engine.current_lora_applied and engine.current_lora_path
                else None
            ),
            "device": engine.device,
            "duration": result["duration"],
        }
        try:
            new_id = db.add_record(record)
        except Exception:
            # 数据库写入失败时回滚刚保存的图片，避免孤立文件。
            if os.path.exists(save_path):
                os.remove(save_path)
            raise

    return {
        "id": new_id,
        "url": f"/outputs/{filename}",
        "seed": result["seed"],
        "duration": result["duration"],
        "meta": record,
    }


@app.post("/api/generate/stop")
def stop_generation():
    # 不获取 inference_lock：生成请求正持有该锁。
    accepted = engine.request_stop()
    return {
        "accepted": accepted,
        "message": "正在停止生成" if accepted else "当前没有正在进行的生成",
    }


@app.get("/api/history")
def get_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    records = db.get_history(limit, offset)
    for record in records:
        record["url"] = f"/outputs/{record['filename']}"
        file_path = os.path.join(config.OUTPUT_DIR, record["filename"])
        record["file_size"] = os.path.getsize(file_path) if os.path.isfile(file_path) else None
    return records


@app.delete("/api/history/{record_id}")
def delete_history(record_id: int):
    success = db.delete_record(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"status": "deleted"}


app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    print("🚀 Z-Image Carto 全栈版已启动!")
    print("👉 请访问: http://127.0.0.1:8888")
    uvicorn.run(
        "main:app" if config.UVICORN_RELOAD else app,
        host="127.0.0.1",
        port=8888,
        reload=config.UVICORN_RELOAD,
    )
