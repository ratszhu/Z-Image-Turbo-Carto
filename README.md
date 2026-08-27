# Z-Image Carto

> 面向本地部署的 Z-Image-Turbo 文生图 WebUI，支持 Apple Silicon、NVIDIA CUDA、自定义 LoRA 和生成历史管理。

[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Z--Image--Turbo-orange)](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Windows tests](https://github.com/ratszhu/Z-Image-Turbo-Carto/actions/workflows/windows-tests.yml/badge.svg)](https://github.com/ratszhu/Z-Image-Turbo-Carto/actions/workflows/windows-tests.yml)

Z-Image Carto 基于 [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)，使用 FastAPI、Vue 3 和 SQLite 构建。项目重点解决本地推理中的显存策略、Apple Silicon 精度适配、LoRA 动态加载以及生成记录管理问题。

![Z-Image Carto 界面](./examples/UI.jpg)

## 生成示例

| | |
| :---: | :---: |
| ![汉服古风写真](./examples/image1.png) | ![Ins 风格少女自拍](./examples/image2.png) |
| ![卧室宠物生活照](./examples/image3.png) | ![户外雪山风景](./examples/image4.png) |
| ![赛博朋克科技风](./examples/image6.png) | ![Clay Render 陆家嘴](./examples/image7.png) |
| ![跑车抓拍](./examples/image5.png) | ![人像近景](./examples/image8.png) |
| ![农村冬季雪景夕阳](./examples/image9.png) | ![二次元插画](./examples/image10.png) |
| ![JK 少女写真 1](./examples/image11.png) | ![JK 少女写真 2](./examples/image12.png) |
| ![JK 少女写真 3](./examples/image13.png) | |

## 这一版有什么新功能

- 新增本地 LoRA 模型库：可在 WebUI 中导入、选择、停用和删除 `.safetensors`。
- LoRA 权重改为百分比显示，支持滑杆与手动输入；`100%` 对应推理权重 `1.0`。
- 画布比例按钮具有选中状态，同时支持直接输入宽度和高度。
- 新增“停止生成”，可在当前扩散步结束后中断采样，不保存未完成图片。
- 模型在后台加载，Web 服务会先启动，页面可直接查看加载状态和错误信息。
- 完善 CUDA 显存策略、推理串行化、API 参数校验与失败回滚。

## 功能概览

### 本地生成工作台

- 正面提示词、负面提示词、Steps、CFG 和 Seed 设置。
- `1:1`、`2:3`、`3:2`、`9:16` 比例预设与自定义宽高。
- 自定义尺寸范围为 `512–2048`，自动对齐到 64 像素。
- 生成中可主动停止，停止后保留上一张预览图。
- 大图预览、原图下载、参数复用和历史记录分页。

### 自定义 LoRA

- 支持从浏览器导入本地 `.safetensors` 文件。
- 使用 PEFT Adapter 动态加载，切换强度时无需重新读取 6B 基础模型。
- 加载前检查 LoRA A/B 权重是否完整，并验证所有目标层是否匹配 Z-Image Transformer。
- 不允许“部分层匹配”后继续生成，避免无声产生错误结果。
- 当前一次只启用一个 LoRA，切换时会卸载旧 Adapter，不会意外叠加。
- 历史记录会保存 LoRA 名称、ID 和权重，用于后续复用参数。

> LoRA 必须基于 Z-Image 训练，并使用 Diffusers/PEFT 的 `lora_A.weight` / `lora_B.weight` 键名。SD 1.5、SDXL、Flux 等其他底模的 LoRA 不通用。

### 硬件与精度策略

- Apple Silicon：使用 MPS 和 BF16，VAE 保持 FP32 解码。
- NVIDIA CUDA：支持 BF16 时优先 BF16，否则使用 FP16。
- CUDA 显存小于 16 GB 时，`auto` 策略默认使用 Sequential CPU Offload。
- CUDA 显存不小于 16 GB 时，`auto` 策略默认使用 Model CPU Offload。
- 无 CUDA/MPS 时可回退到 CPU FP32，但不建议用于日常生成。

## 系统要求

| 项目 | 建议 |
| :--- | :--- |
| 操作系统 | macOS（Apple Silicon）、Windows/Linux（NVIDIA GPU） |
| Python | 3.10–3.12 |
| Git | 必需，`requirements.txt` 会安装指定提交的 Diffusers |
| Apple Silicon | 建议 16 GB 或更多统一内存 |
| NVIDIA | 建议 12 GB 或更多显存；低显存可用 Sequential Offload，但会更慢 |
| 硬盘 | 需为基础模型、LoRA 和生成图片预留空间 |

实际内存和显存占用会随分辨率、精度、Offload 策略和 LoRA 大小变化。

## 快速开始

### 1. 获取代码

```bash
git clone https://github.com/ratszhu/Z-Image-Turbo-Carto.git
cd Z-Image-Turbo-Carto
```

### 2. 创建虚拟环境

macOS / Linux：

```bash
python3 -m venv z_image_env
source z_image_env/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell：

```powershell
py -3.11 -m venv z_image_env
.\z_image_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. 安装依赖

Apple Silicon 可直接安装：

```bash
pip install -r requirements.txt
```

NVIDIA 用户建议先按 [PyTorch 官方安装向导](https://pytorch.org/get-started/locally/) 安装与本机 CUDA 匹配的 PyTorch，再执行：

```bash
pip install -r requirements.txt
```

> 若使用镜像源，请注意 `requirements.txt` 中的 Diffusers 来自 GitHub 指定提交，仍需要能够访问 GitHub。

### 4. 下载基础模型

将 [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 的完整 Diffusers 模型放到项目根目录的 `Z-Image-Model/`。安装依赖后也可使用：

```bash
hf download Tongyi-MAI/Z-Image-Turbo --local-dir ./Z-Image-Model
```

正常的核心目录结构应类似：

```text
Z-Image-Model/
├── model_index.json
├── scheduler/
├── text_encoder/
├── tokenizer/
├── transformer/
└── vae/
```

### 5. 启动服务

```bash
python main.py
```

浏览器访问：

```text
http://127.0.0.1:8888
```

Web 页面会先启动，基础模型在后台加载。右上角显示 `LOADING` 时请继续等待，状态变为 `MPS`、`CUDA` 或 `CPU` 后即可生成。

> 不要直接双击 `web/index.html`。`file://` 页面无法正常访问 `/api/*`，必须通过 FastAPI 提供的 `http://127.0.0.1:8888` 访问。

## 使用自定义 LoRA

1. 确认 LoRA 的训练底模为 Z-Image，并导出为 `.safetensors`。
2. 在 WebUI 的“自定义 LoRA”区域点击 `+`。
3. 选择本地文件；校验成功后它会出现在下拉列表中。
4. 选择 LoRA 后设置权重。建议先从训练页面给出的推荐值开始；界面中 `100% = 1.0`。
5. 若训练时使用了触发词，需在正面提示词中加入该触发词。

导入文件保存在 `loras/`，单个上传文件上限为 2 GB。LoRA 默认不启用，基础模型不会再自动加载旧版的色彩增强 LoRA。

## 推荐参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| Steps | `9` | Z-Image-Turbo 的建议起点；增加步数会增加耗时，不一定改善结果 |
| CFG | `0.0` | Turbo 模型的建议起点 |
| Width / Height | `1024 × 1024` | 可使用比例预设或手动输入，需为 64 的倍数 |
| Seed | `-1` | `-1` 或随机模式会生成新种子；固定 Seed 便于对比参数 |
| LoRA | 关闭 | 不选择 LoRA 时使用原始基础模型 |
| LoRA 权重 | `100%` | 对应推理权重 `1.0`，可在 `0%–200%` 之间调节 |

Z-Image-Turbo 在 `CFG ≤ 1` 时不会使用负面提示词，WebUI 会显示对应提示。

## CUDA 显存策略

默认值为 `auto`，可在启动前设置 `ZIMAGE_CUDA_OFFLOAD`：

| 值 | 行为 | 适用场景 |
| :--- | :--- | :--- |
| `auto` | 按显存大小选择 `model` 或 `sequential` | 推荐 |
| `model` | Model CPU Offload | 显存较充足，希望兼顾速度和占用 |
| `sequential` | Sequential CPU Offload | 小显存显卡，以速度换显存 |
| `none` | 将管线直接放到 CUDA | 仅适合确认显存充足的环境 |

macOS / Linux：

```bash
ZIMAGE_CUDA_OFFLOAD=sequential python main.py
```

Windows PowerShell：

```powershell
$env:ZIMAGE_CUDA_OFFLOAD = "sequential"
python main.py
```

开发时可设置 `ZIMAGE_RELOAD=1` 启用 Uvicorn 热重载。日常生成建议保持默认关闭，避免重载进程重复加载大模型。

## 数据与目录

| 路径 | 内容 |
| :--- | :--- |
| `Z-Image-Model/` | Z-Image-Turbo 基础模型，不纳入 Git |
| `loras/` | WebUI 导入的 LoRA，不纳入 Git |
| `outputs/` | 生成的 PNG 图片，不纳入 Git |
| `database/history.db` | SQLite 生成历史 |

删除历史记录时，对应的图片文件也会被删除。升级前如需保留个人数据，建议备份 `outputs/`、`loras/` 和 `database/history.db`。

## API

FastAPI 交互文档：[http://127.0.0.1:8888/docs](http://127.0.0.1:8888/docs)

| 方法 | 路径 | 用途 |
| :--- | :--- | :--- |
| `GET` | `/api/status` | 模型、硬件、生成和 LoRA 状态 |
| `POST` | `/api/generate` | 生成图片 |
| `POST` | `/api/generate/stop` | 请求停止当前生成 |
| `GET` | `/api/loras` | 获取本地 LoRA 列表 |
| `POST` | `/api/loras` | 上传 LoRA |
| `DELETE` | `/api/loras/{id}` | 删除 LoRA |
| `GET` | `/api/history` | 分页获取生成历史 |
| `DELETE` | `/api/history/{id}` | 删除历史及对应图片 |

## 项目结构

```text
Z-Image-Turbo-Carto/
├── main.py                    # FastAPI 入口与 API
├── config.py                  # 路径、默认参数和运行策略
├── requirements.txt           # Python 依赖
├── core/
│   ├── engine.py              # 模型生命周期、显存策略与推理
│   ├── lora_manager.py        # PEFT Adapter 加载、切换与层校验
│   ├── lora_store.py          # LoRA 上传与本地模型库
│   └── utils.py               # 设备、精度与硬件检测
├── database/db_manager.py      # SQLite 历史记录
├── web/index.html             # Vue 3 单页应用
├── tests/test_core.py         # 核心逻辑与 API 测试
├── examples/                  # 界面与生成效果示例
├── Z-Image-Model/             # 本地基础模型（忽略）
├── loras/                     # 导入的 LoRA（忽略）
└── outputs/                   # 生成图片（忽略）
```

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py config.py core database tests
pip check
```

## 常见问题

### 页面提示 CORS，请求地址变成 `file:///api/...`

这是直接打开 `web/index.html` 导致的。请先运行 `python main.py`，然后访问 `http://127.0.0.1:8888`。

### NVIDIA 显卡被识别为 CPU

当前虚拟环境很可能安装了 CPU 版 PyTorch。请按 [PyTorch 官方向导](https://pytorch.org/get-started/locally/) 重新安装与 CUDA 匹配的 wheel，再查看启动日志或 `/api/status`。

### CUDA 生成时显存不足

先降低宽高，然后尝试 `ZIMAGE_CUDA_OFFLOAD=sequential`。Sequential Offload 能降低峰值显存，但会显著增加生成时间并占用系统内存。

### LoRA 上传成功，生成时却显示不兼容

上传阶段只能验证 safetensors 结构和 A/B 权重完整性；首次启用时才会与当前 Z-Image Transformer 逐层对比。请确认 LoRA 使用 Z-Image 作为底模，并导出为 Diffusers/PEFT 格式。

### 同样的提示词为什么结果差异很大

请同时确认完整提示词、Seed、宽高、Steps、CFG、LoRA 文件和 LoRA 权重。其中任意一项变化都可能导致构图和人物差异。

### 点击停止后为什么不是立即结束

为了不破坏当前管线状态，停止请求在当前扩散步结束时生效。单步耗时较长时，按钮可能会短暂显示“正在停止”。

### 页面样式或图标加载失败

当前 WebUI 通过 CDN 加载 Vue 3、Tailwind CSS 和 Lucide。首次打开页面需要可访问相应 CDN；后续版本可考虑将前端依赖改为本地打包。

## 已知限制

- 当前一次只支持一个 LoRA，暂不支持多 LoRA 叠加。
- 同一进程内只执行一个生成任务，用于避免 Adapter 状态和 GPU/MPS 内存冲突。
- 服务默认仅绑定 `127.0.0.1`，没有用户认证，不应直接暴露到公网。
- 基础模型和用户 LoRA 不包含在本仓库中，需按各自许可协议下载和使用。

## 参与贡献

欢迎提交 Issue 和 Pull Request。报告问题时，建议附上：

- 操作系统、Python 和 PyTorch 版本。
- 显卡型号或 Apple Silicon 芯片与统一内存容量。
- 启动日志和 `/api/status` 返回内容。
- 可复现问题的尺寸、Steps、CFG 和 Offload 设置。
- LoRA 问题请说明训练底模和导出工具，不要上传无授权的模型文件。

## 许可证与致谢

- 本项目代码使用 [MIT License](./LICENSE)。
- Z-Image-Turbo 模型权重遵循其 [模型页面](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 声明的许可条款。
- LoRA 的下载、训练和发布需遵循底模、训练数据和 LoRA 发布者的相关许可条款。
- 感谢 Tongyi-MAI、Hugging Face Diffusers、PEFT 以及开源社区。

Made with ❤️ by [ratszhu](https://github.com/ratszhu)
