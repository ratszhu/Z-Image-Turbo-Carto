# 🎨 Z-Image Carto

**Z-Image Carto** 是一个专为高性能本地部署设计的 AI 绘画工作站（WebUI）。

本项目基于阿里通义 [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo "null") 模型，针对 **Apple Silicon (M1/M2/M3)** 和 **NVIDIA RTX (Windows)** 进行了深度的底层工程优化。它抛弃了传统的 Gradio 界面，采用 **FastAPI + Vue 3** 的现代化前后端分离架构，提供极速、高清、沉浸式的创作体验。

![界面截图](./examples/UI.jpg "界面截图")

## ✨ 核心特性 (Core Features)

### 🍎 Mac (Apple Silicon) 深度优化

*   **Bfloat16 原生加速**：摒弃FP32，利用 M 系列芯片原生的 `bfloat16` 指令集，**推理速度提升 200%**，显存占用减半（仅需 12G 左右），解决了 FP16 溢出导致的黑屏问题。
*   **VAE 混合精度修复**：独家优化的推理管线。主模型跑在 BF16，强制 VAE 解码器运行在 FP32，**彻底根除**了 Bfloat16 下画面模糊的通病，实现像素级锐利。
*   **显存智能管理**：默认关闭 Tiling 以追求极致画质，M1 Max/Ultra 可轻松直出 2K 大图。

### 💻 Windows (NVIDIA) 完美兼容

*   **CUDA + FP16**：自动检测 NVIDIA 显卡，启用最成熟的 FP16 半精度推理。
*   **CPU Offload**：针对 8GB/12GB 显存的显卡（如 RTX 4060/4070 Laptop），自动开启 CPU 卸载功能，防止爆显存。

### 🎨 独家画质增强系统

*   **LoRA 手动注入算法**：重写了 LoRA 加载逻辑，能够精准识别并注入 Z-Image 特有的 `context_refiner` 和 `noise_refiner` 层。
*   **内置色彩增强**：默认集成 `Technically-Color` LoRA，修复了原版模型色彩发灰的问题，开箱即享电影级胶片质感。

### 🛠 专业级工作室体验 (Studio UI)

*   **现代化架构**：FastAPI 后端 + Vue 3 前端 + SQLite 数据库。
*   **历史回溯系统**：自动记录每一次生成的图片及其完整元数据（Prompt, Seed, CFG, 耗时, 体积等），支持一键复用参数。
*   **沉浸式预览**：支持全屏 Lightbox 查看，纯净无遮挡，支持原图下载。
    

## 🤩 生成效果图

![丝袜近景特写](./examples/image8.png "丝袜近景特写")
![汉服古风写真](./examples/image1.png "汉服古风写真")
![ins风格少女自拍](./examples/image2.png "ins风格少女自拍")
![JK少女写真](./examples/image11.png "JK少女写真")
![JK少女写真](./examples/image12.png "JK少女写真")
![JK少女写真](./examples/image13.png "JK少女写真")
![神里绫华二次元插画](./examples/image10.png "神里绫华二次元插画")
![卧室温馨宠物生活照](./examples/image3.png "卧室温馨宠物生活照")
![户外雪山风景](./examples/image4.png "户外雪山风景")
![宝马跑车抓拍](./examples/image5.png "宝马跑车抓拍")
![赛博朋克科技风](./examples/image6.png "赛博朋克科技风")
![Clay Render陆家嘴](./examples/image7.png "Clay Render陆家嘴")
![农村冬季雪景夕阳](./examples/image9.png "农村冬季雪景夕阳")



## 📂 项目结构

```
Z-Image-Carto/
├── main.py                 # [入口] FastAPI 服务器启动文件
├── config.py               # [配置] 全局参数与路径配置
├── requirements.txt        # [依赖] 项目依赖列表
├── core/                   # [核心] 算法核心包
│   ├── engine.py           # 推理引擎 (模型加载/显存优化/生成逻辑)
│   ├── lora_manager.py     # LoRA 手动注入管理器
│   └── utils.py            # 硬件检测工具
├── database/               # [数据] SQLite 数据库管理
├── web/                    # [前端] 静态资源
│   └── index.html          # Vue 3 单页应用
└── outputs/                # [存储] 图片保存目录

```

## 🚀 快速开始 (Quick Start)

### 1. 环境准备

*   **操作系统**：macOS (Apple Silicon) 或 Windows 10/11 (NVIDIA GPU)。
*   **Python**：推荐 **Python 3.10** 或 **3.11**。

    *   ⚠️ **警告**：请勿使用 Python 3.13，目前 PyTorch 对其支持尚不稳定。

### 2. 安装依赖

在项目根目录下打开终端：

```
# 1. 创建虚拟环境 (推荐)
python -m venv venv
# Mac/Linux 激活:
source venv/bin/activate
# Windows 激活:
.\venv\Scripts\activate

# 2. 安装依赖 (使用清华源加速)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

```

### 3. 模型准备

请确保项目根目录下存在以下文件（或在 `config.py` 中修改路径）：

1.  **基础模型**：`./Z-Image-Model`

    *   从 Hugging Face 下载 `Tongyi-MAI/Z-Image-Turbo` 完整文件夹。
2.  **LoRA 文件**：`./Technically_Color_Z_Image_Turbo_v1_renderartist_2000.safetensors`

    *   用于画质增强。

### 4. 启动应用

```
python main.py
```

等待终端显示： `🚀 Z-Image Studio 全栈版已启动! 👉 请访问: http://127.0.0.1:8888`

在浏览器打开该地址即可开始创作！

## ⚙️ 最佳实践指南

| 参数               | 推荐值   | 说明                                                     |
| :----------------- | :------- | :------------------------------------------------------- |
| **Steps (步数)**   | **9**    | Turbo 模型特化步数，9步即可获得最佳细节与速度平衡。      |
| **CFG (引导系数)** | **0.0**  | Z-Image-Turbo 推荐设为 0。过高会导致画面纹理劣化。       |
| **LoRA Scale**     | **1.3**  | 配合 Technically-Color LoRA 的最佳甜点值，色彩最通透。   |
| **Width/Height**   | **1024** | 训练时的原生分辨率。M1 Max 用户可尝试挑战 1536 或 2048。 |

## ❓ 常见问题 (FAQ)

### Q1: 为什么我的 Mac 上生成的图片很模糊？

**A**: 这通常是 VAE 在半精度下解码导致的。请确保您使用的是本项目最新代码，我们的 `core/engine.py` 中包含了强制 `VAE FP32` 的修复逻辑。

### Q2: 报错 `ModuleNotFoundError: No module named 'fastapi'`？

**A**: 您可能没有安装 Web 相关的依赖。请重新运行 `pip install -r requirements.txt`。

### Q3: 生成图片时显示“内存不足”？

**A**:

*   **Mac 用户**：尝试降低分辨率到 1024x1024。如果您是 16G 内存机型，请在 `core/engine.py` 中开启 `enable_vae_tiling()`。
*   **Windows 用户**：程序会自动开启 `enable_model_cpu_offload()`，如果依然爆显存，请关闭浏览器和其他占用显存的应用。

## 📝 许可证与致谢

*   本项目代码遵循 MIT 许可证。
*   模型权重遵循 [Tongyi-MAI](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo "null") 开源协议。
*   感谢开源社区提供的 LoRA 训练思路与技术支持。

*Made with ❤️ by \[ratszhu]*
