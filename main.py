# -*- coding: utf-8 -*-
"""
Z-Image Carto 全栈入口
同时负责 API 服务和 静态页面托管。
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid

from core.engine import ZImageEngine
from database.db_manager import DatabaseManager
import config

# --- 1. 初始化 ---
app = FastAPI(title="Z-Image Carto")

# 允许跨域 (保留作为保险)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ZImageEngine()
db = DatabaseManager()

# --- 2. 数据模型 ---
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 9
    cfg: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: int = -1
    seed_mode: str = "fixed"
    lora_enabled: bool = True
    lora_scale: float = 1.3

# --- 3. API 接口 (先定义 API，优先级最高) ---

@app.on_event("startup")
async def startup_event():
    print("🌟 系统启动中，正在加载模型...")
    engine.load_model()
    engine.update_lora(config.DEFAULT_LORA_ENABLE, config.DEFAULT_LORA_SCALE)

@app.get("/api/status")
def get_status():
    return {
        "loaded": engine.is_loaded(),
        "device": engine.device,
        "dtype": str(engine.dtype),
        "lora_enabled": engine.current_lora_applied
    }

@app.post("/api/generate")
def generate_image(req: GenerateRequest):
    if not engine.is_loaded():
        raise HTTPException(status_code=503, detail="模型未加载")

    if req.lora_enabled != engine.current_lora_applied:
        engine.update_lora(req.lora_enabled, req.lora_scale)
    
    result = engine.generate(
        prompt=req.prompt,
        neg_prompt=req.negative_prompt,
        steps=req.steps,
        cfg=req.cfg,
        width=req.width,
        height=req.height,
        seed=req.seed,
        seed_mode=req.seed_mode
    )
    
    if not result["success"]:
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
        "lora_enabled": req.lora_enabled,
        "lora_scale": req.lora_scale,
        "device": engine.device,
        "duration": result["duration"]
    }
    new_id = db.add_record(record)
    
    return {
        "id": new_id,
        "url": f"/outputs/{filename}",
        "seed": result["seed"],
        "duration": result["duration"],
        "meta": record
    }

@app.get("/api/history")
def get_history(limit: int = 20, offset: int = 0):
    records = db.get_history(limit, offset)
    for r in records:
        r["url"] = f"/outputs/{r['filename']}"
    return records

@app.delete("/api/history/{record_id}")
def delete_history(record_id: int):
    success = db.delete_record(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"status": "deleted"}

# --- 4. 静态文件托管 (最后定义，作为兜底) ---

# 挂载 outputs 目录，用于访问生成的图片
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")

# [关键修改] 挂载 web 目录到根路径 '/'，实现“打开网址即由后端提供页面”
# 注意：html=True 表示访问 / 会自动寻找 index.html
app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    print("🚀 Z-Image Carto 全栈版已启动!")
    print("👉 请访问: http://127.0.0.1:8888")
    uvicorn.run("main:app", host="127.0.0.1", port=8888, reload=True)