# 更易用的 AI Toolkit（中文 README）

本项目是 AI Toolkit 的中文与易用性优化版本（`ai-toolkit-easy2use`）。在保留原功能的基础上，聚焦「更易安装、更易上手、更易维护」。本 README 全文为中文，并对安装、运行与 UI 使用进行汉化说明。

> 原项目作者：Ostris；本仓库维护：DocWorkBox。

---

## 项目简介

- 面向扩散模型（Diffusion Models）的训练与推理一体化工具。
- 支持图像、视频、音频模型，以及编辑 / instruction 类模型。
- 提供命令行（CLI）与 Web 用户界面（UI），上手门槛低同时功能完备。
- 本仓库持续跟进上游更新，同时保留中文文档、中文 UI 和更贴近中文用户的使用体验。
### 图像模型（上游支持）
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (FLUX.1)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (FLUX.2)
- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) (FLUX.2-klein-base-4B)
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (FLUX.2-klein-base-9B)
- [ostris/Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) (Flex.1)
- [ostris/Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview) (Flex.2)
- [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) (Chroma)
- [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) (Lumina2)
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (Qwen-Image)
- [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (Qwen-Image-2512)
- [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) (HiDream I1)
- [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) (OmniGen2)
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Z-Image Turbo)
- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Z-Image)
- [ostris/Z-Image-De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) (Z-Image De-Turbo)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL)
- [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (SD 1.5)
- [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) (ERNIE-Image)
- [NucleusAI/Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (Nucleus-Image)
- [HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) (HiDream O1)
- [Photoroom/prxpixel-t2i](https://huggingface.co/Photoroom/prxpixel-t2i) (PRXPixel)

## 当前支持概览

- 图像：FLUX.1 / FLUX.2 / FLUX.2 Klein / Flex.1 / Flex.2 / Chroma / Lumina2 / Qwen-Image / Qwen-Image-2512 / HiDream I1 / HiDream-O1 / Z-Image / Z-Image Turbo / Z-Image De-Turbo / Z-Image L2P / SDXL / SD1.5 / ERNIE-Image / Nucleus-Image / PRX Pixel / Ideogram 4 等
- 编辑：Qwen-Image-Edit / Qwen-Image-Edit-2509 / Qwen-Image-Edit-2511 / HiDream E1 / FireRed-Image-Edit-1.1 / Krea 2 Edit / Boogu Image Edit 等
- 视频：Wan 2.x / LTX-2 / LTX-2.3 / Sulphur-2（复用 LTX-2.3 架构预设）等
- 音频：ACE-Step 1.5 / ACE-Step 1.5 XL
- 实验性：Zeta-Chroma / Ideogram 4 / Boogu Image / Krea 2 / PRX Pixel 等

### 视频模型（上游支持）
- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (Wan 2.1 1.3B)
- [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) (Wan 2.1 I2V 14B-480P)
- [Wan-AI/Wan2.1-I2V-14B-720P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) (Wan 2.1 I2V 14B-720P)
- [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (Wan 2.1 14B)
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (Wan 2.2 14B)
- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (Wan 2.2 I2V 14B)
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (Wan 2.2 TI2V 5B)
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) (LTX-2)
- [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (LTX-2.3)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2)

## 环境要求

- Python >= 3.10（推荐 3.12）
- Git
- Python 虚拟环境
- Node.js >= 20（运行 Web UI）
- NVIDIA GPU（按训练任务准备足够显存）

### 实验性模型
- [lodestones/Zeta-Chroma](https://huggingface.co/lodestones/Zeta-Chroma) (Zeta Chroma)
- [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8) (Ideogram 4 FP8)

### Mac Apple Silicon

上游已加入 Apple Silicon 的实验性支持。若你在 macOS 上使用，可直接尝试：

```bash
chmod +x run_mac.zsh
./run_mac.zsh
```

如果你在 Mac 上遇到兼容性问题，欢迎优先参考上游最新 issue 或在本仓库反馈。

## 安装（Linux / Windows）

### 1）克隆仓库

```bash
git clone https://github.com/DocWorkBox/ai-toolkit-easy2use.git
cd ai-toolkit-easy2use
```

### 2）创建并激活虚拟环境

Linux / macOS：

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows（PowerShell）：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3）安装 PyTorch（示例：CUDA 12.8）

请根据你的 CUDA / 驱动环境调整版本。以下为当前仓库推荐示例：

```bash
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

### 4）安装项目依赖

```bash
pip install -r requirements.txt
```

### 5）DGX OS

DGX OS 设备请参考仓库内的 DGX 说明，并使用：

```bash
pip install -r dgx_requirements.txt
```

## 运行 UI（中文界面）

UI 为基于 Next.js 的 Web 应用。训练任务本身不依赖 UI 持续前台运行，UI 主要用于创建、启动、停止和监控任务。

### 旧版本升级到新版（重要）

如果你是从旧版本直接 `git pull` 到新版本，而不是全新安装，请务必在启动 UI 前同步 Prisma Client 和本地数据库结构。

这是因为上游更新会不定期修改 [`ui/prisma/schema.prisma`](ui/prisma/schema.prisma)。如果代码已经更新，但还没有重新生成 Prisma Client / 执行 `db push`，就可能出现这类问题：

- 页面加载时报 `Application error` 或 client-side exception
- 创建 / 保存任务失败
- 接口报 Prisma 字段不存在，例如 `Unknown argument job_ref`

推荐升级步骤：

```bash
cd ui
npm install
npm run update_db
npm run build
```

其中：

- `npm run update_db` 会执行 `npx prisma generate && npx prisma db push`
- `prisma generate` 用于重建 Prisma Client
- `prisma db push` 用于把本地 SQLite 数据库结构同步到最新 schema

如果你当前已经在运行 UI，请先停止正在运行的 `node` / `next` 进程，再执行上面的命令；否则在 Windows 下可能会遇到 Prisma DLL 被占用、无法更新的问题。

### 开发模式

```bash
cd ui
npm install
npm run dev
```

访问：

- `http://localhost:3000/`
- `http://localhost:3000/dashboard`
- `http://localhost:3000/jobs/new`

### 生产模式

```bash
cd ui
npm run build_and_start
```

默认端口：

- `http://localhost:8675`
- `http://<your-ip>:8675`

### 安全建议

如果你把 UI 暴露在公网，建议设置认证令牌：

```bash
# Linux
AI_TOOLKIT_AUTH=your_token npm run build_and_start

# Windows PowerShell
$env:AI_TOOLKIT_AUTH="your_token"; npm run build_and_start
```

## 本仓库新增特性

### ModelScope 模型下载源

设置页（`/settings`）新增「模型下载源」选项，支持切换 HuggingFace 官方源和 ModelScope 国内镜像源。选择 ModelScope 后，训练和打标任务的模型下载会自动通过 `modelscope` 包重定向到 `modelscope.cn`，适合国内网络环境。

> 需要 `pip install modelscope`，模型下载源切换后新建的任务会自动生效。

### 自定义 API 打标增强

数据集自动打标选择「自定义 API」时，填写 Base URL 和 API Key 后点击「测试 API」：
- 使用 `GET /health` 检查连通性
- 自动调用 `GET /v1/models` 拉取可用模型列表
- 模型名称以下拉框选取，无需手动输入

### 打标草稿自动保存

打标弹窗中的配置会自动保存到数据库，关闭后重新打开可恢复上次填写的内容，避免重复输入。

### Ostris Cloud

上游 README 新增了 [Ostris Cloud](https://cloud.ostris.com) 说明：这是 Ostris 运营的云 GPU 服务，可用于租用 GPU 运行 AI Toolkit；相关收入会反哺上游项目开发。

<a href="https://cloud.ostris.com" target="_blank"><img src="https://cloud.ostris.com/api/og" alt="Ostris Cloud" style="max-width:100%;width:600px;height:auto;"></a>

## 近期已并入的重要上游能力

- **新增 HiDream-O1 微调支持**（`hidream_o1`）：集成官方流水线、训练配置与 LoRA 加载，适配 Comfy 兼容格式。
- **DFE7（Diffusion Feature Extractor 7）损失加权**：引入基于 `velocity_equiv_weight` 的加权策略，提升训练稳定性与特征误差分布合理性。
- **恢复训练时清理未来步的 Loss 日志**：当从较早 step 恢复时，自动删除后续步的冗余记录，避免历史日志误导。
- **DFE 相关模型梯度检查点增强**：兼容更多 checkpointing 接口，降低显存占用。
- **损失异常值抑制（`max_loss` / `max_loss_debug`）**：支持对单步损失上限截断，防止极端异常值导致训练震荡。
- Qwen Image 原生增加 `1328` 分辨率支持
- 新增 `ERNIE-Image` 与 `Nucleus-Image` 模型支持
- Flux2 / Flux2 Klein 低显存加载、量化顺序和控制图编码修复
- `compile: true` 真正生效，并修复 compile 时序
- 数据集页支持 duplicate dataset、改进拖拽 / 上传模型、隐藏文件过滤
- 数据集自动 caption、音频数据集 captioning、音频样本下载
- ACE-Step 1.5 / XL 音频模型支持
- `flac` / `ogg` 支持
- Light Mode 支持
- `AdvancedPromptEmbeds` 引入与增强：新增 `frozen_dtype_keys` 机制，兼容需要部分张量锁定精度的模型。

## 中文版 UI 截图

![仪表盘（中文）](ui/public/screenshots/dashboard_zh.png)
![新建任务（中文）](ui/public/screenshots/jobs_new_zh.png)

### 手机端 UI 截图

![手机端：适配器页面（1）](ui/public/screenshots/adapter-UI0.png)
![手机端：适配器页面（2）](ui/public/screenshots/adapter-UI1.png)

## FireRed-Image-Edit-1.1 预设说明

- 训练 UI 已新增 `FireRed-Image-Edit-1.1` 预设，适用于基于该模型的 LoRA 训练起步配置。
- 该预设复用当前仓库中的 `qwen_image_edit_plus` 兼容链路，底层按 `QwenImageEditPlusPipeline` 方式加载。
- 这是一份 ai-toolkit 风格预设，不是 FireRed 原仓 `train_lora.sh` 的逐项复刻；未直接映射的 FireRed 专属训练参数继续沿用 ai-toolkit 现有机制。
- 推荐优先从示例配置 `config/examples/train_lora_firered_image_edit_1_1_32gb.yaml` 启动，再按你的显存和数据集情况微调。

## 常见问题（FAQ）

- 显存不足如何处理？
  - 训练大型模型时，如遇显存限制，可在配置中开启低显存选项（如 `low_vram: true`），或对部分模块量化 / CPU 卸载。
- Windows 安装遇到困难？
  - 优先确认 Python、CUDA、驱动版本匹配；必要时建议使用 WSL 获得更稳定的依赖环境。
- UI 无法访问或接口报错？
  - 请确认 Node.js 版本（>=20）、依赖已安装完成，并检查 `npm run dev` / `npm run build_and_start` 是否正常启动。
- 旧版本升级后页面报错、数据集页打不开、保存任务失败？
  - 很多时候是 Prisma Client 或数据库结构没有同步。请进入 `ui/` 目录后执行：`npm install && npm run update_db && npm run build`。
- 想启用 Hugging Face 高速下载？
  - 可在启动前设置 `HF_HUB_ENABLE_HF_TRANSFER=1`。
- 音频 / 视频数据集读取失败？
  - 请检查 `ffmpeg`、`torchaudio` 与容器格式支持，必要时重新安装依赖并确认系统里可用的解码后端。

## 目录指南（简要）

- `config/`：训练或推理配置示例
- `ui/`：Next.js 中文 UI 源码
- `docs/docker/`：本仓库额外整理的 Docker 说明
- `requirements.txt` / `requirements_base.txt`：Python 依赖
- `dgx_requirements.txt`：DGX OS 专用依赖

## 致谢与说明

- 本仓库以更易用为目标进行中文化与体验优化，基于原 AI Toolkit 项目实现。
- 本仓库欢迎 issue、bug report 和改进建议。
- 自动化生成的 PR 请至少补充清晰的人类说明与验证结果，否则不建议直接合并。

## 许可证

本仓库遵循原项目的许可证政策。请在商用或分发前，额外确认模型、数据集和第三方依赖各自的许可证要求。

## 上游补充说明

### 数据集准备

数据集通常是图片与同名 caption 文本文件组成的文件夹，例如 `image2.jpg` 与 `image2.txt`。caption 文件可以包含 `[trigger]` 占位符，若训练配置中设置了 `trigger_word`，训练时会自动替换。图片无需手动裁剪或统一尺寸，loader 会按 bucket 规则自动缩放和分桶。

### 指定层训练

LoRA 网络可通过 `network_kwargs.only_if_contains` 只训练指定层，也可通过 `network_kwargs.ignore_if_contains` 排除指定层；两者同时命中时，`ignore_if_contains` 优先。层名遵循 diffusers state dict 命名。

```yaml
network:
  type: "lora"
  linear: 128
  linear_alpha: 128
  network_kwargs:
    only_if_contains:
      - "transformer.single_transformer_blocks."
```

### LoKr 训练

LoKr 仍沿用常规训练配置，只需要把 network type 改为 `lokr` 并设置对应参数：

```yaml
network:
  type: "lokr"
  lokr_full_rank: true
  lokr_factor: 8
```

### 支持上游

如需支持原项目持续开发，可参考上游赞助页：[Ostris Sponsors](https://ostris.com/sponsors)。
