# MiniMax-H3 人物 LoRA 训练调研报告

> 调研日期：基于 MiniMax-H3 开源权重（HuggingFace 发布，33B 全模态视频+音频 DiT）与当前社区训练工具链。
> 本仓库（ai-toolkit-easy2use）已将 MiniMax-H3 列为"上游支持"模型，但**没有提供 minimax_h3 的训练示例配置**（只有生成/推理路径），因此本文重点介绍社区当前可用的训练路线。

---

## 0. 背景澄清：H3 与"海螺人物 LoRA"

- **MiniMax-H3 = 海螺 H3 / Hailuo 03**，是 MiniMax 开源的 33B 全模态（视频 + 同步立体声音频）DiT 模型，权重在 [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)（2.2M+ 下载）。
- H3 权重分为两个分区，**DiT 与文本编码器权重不同**：
  - `FL2VA`：文生音视频 + 首尾帧（first/last frame）引导
  - `Ref2VA`：参考图/参考视频/参考音频驱动
  - LoRA 训练在哪个分区上训练，就必须在哪个分区上使用。
- **MiniMax 官方没有发布训练器**，HuggingFace Diffusers 集成仅支持推理；训练由社区补齐。
- 注意：MiniMax-H2O（海螺 02）**没有官方 HuggingFace 权重**（HF 搜索无结果），其人物一致性走官方产品/API 路线；所以"minimax 人物 LoRA"目前实际指 H3。

---

## 1. 核心问题：只用照片 + 标注能训吗？

**直接训练：不能。**

H3 是视频 DiT，训练管线（无论哪套工具）都要求训练样本是**视频文件**：样本要经过视频 VAE 编码 + 时间维打包（packing），metadata 里的数据字段是 `video`（.mp4）。一张静态照片不是合法的训练样本，数据加载器会直接报错。

但"只用照片做人物一致性"有 3 条可行路线：

| 路线 | 是否训练 | 说明 |
|---|---|---|
| **A. 参考（Ref2VA / 官方"参考"功能）** ⭐推荐 | 不训练 | H3 Ref2VA 原生支持参考图/参考视频/参考音频作为条件；海螺官方产品的"主体参考/参考"就是这个方案，**直接传照片即可保持人物一致**，不需要 LoRA |
| **B. 照片转短视频再训练** | 训练 | 把每张照片做成 5 秒左右的静态片段（或加运镜/轻微运动，或用 I2V 生成动图）进数据集。能学到"长相/身份"，但动作单一、多样性差，LoRA 的泛化（姿态、动态、表情）会明显打折 |
| **C. 官方平台定制功能** | 平台内部训练 | 例如[可灵 Custom Models / 视频模型定制](https://klingai.com/release-note/release-notes/u3o4p73f2h)接受纯照片训练——但那是平台侧处理，与本地 H3 无关 |

**结论**：要训出真正好用的 H3 人物 LoRA，数据集主体应该是**人物视频片段**（多角度、多表情、有动作/说话）；照片只适合作为参考条件，或者临时补足静态镜头。

---

## 2. 数据集怎么准备

以 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)（目前最完整的社区训练器，含 NF4 量化单卡方案）为例：

### 2.1 目录与 metadata.json

数据集根目录下放视频文件，配一个 `metadata.json`：

```json
[
  {
    "video": "train_video.mp4",
    "prompt": "A young woman with long black hair and brown eyes, wearing a denim jacket, smiling and talking: \"Hello, nice to meet you.\"",
    "input_audio": "train_video.mp4",
    "references": [
      {"type": "image", "image": "ref_0.png"}
    ],
    "frame_rate": 24
  }
]
```

- `video`：必填，训练目标片段（mp4）。
- `prompt`：标注，描述人物外貌 + 场景 + 动作 + 台词。H3 用 Qwen3-VL 编码（layer-50 hidden states），**自然语言整句描述效果最好**；中英文都可以，英文一般更稳。需要稳定触发时可在句首放一个 trigger word（如 `<char>` 或 `sks` 之类罕见词），不要滥用。
- `input_audio`：可选。要学**声音/口型同步**就带上与视频对齐的声轨；不带则音频损失权重为 0（不影响训练，只是不学音频）。
- `references`：可选（Ref2VA 训练用）。`image` / `video` / `audio` / `video_audio` 四种类型。
- `frame_rate`：固定 24。

### 2.2 每个片段的技术要求（H3 硬约束）

- **帧数必须满足 `num_frames % 17 == 5`**（VAE 时间压缩的 clip 结构），训练画布常用 **124 帧**（124 = 17×7+5，24fps 下约 5.2 秒）。124 帧是最常见的训练长度；短片段可用 22 / 39 / 56 / 73 / 90 / 107 帧。
- 宽高必须是 **32 的倍数**（16× VAE 空间压缩 × 2×2 patch）。示例脚本用 **480×832**。
- 帧率固定 **24fps**；音频 **32kHz 立体声**（如带音频）。
- 参考视频按 24fps 采样、裁剪到训练画布；参考图以原生分辨率交给管线（内部按参考短边缩放）。

### 2.3 数量与内容建议（人物 LoRA 通用经验）

- **片段数量：20–50 个**，每个 3–10 秒（训练脚本默认 `dataset_repeat 100`、`num_epochs 5`，总步数自动放大，数据多不会跑不够）。
- 同一人物、**多角度**（正面/侧面/背面）、**多表情、多光照、多场景**、全身 + 半身 + 特写都要有。
- 画面清晰、人物主体完整、少遮挡；**避免同框多人和画面里出现"另一个更像目标特征的人"**（会稀释身份）。
- 标注风格统一：外貌特征（发型/发色/眼睛/脸型/服装）+ 正在做的事 + 台词。

---

## 3. 特殊要求（坑点清单）

1. **训练目标约定与常见模型不同（社区踩坑重灾区）**：H3 是 guidance-distilled 的 rectified-flow 模型——**没有负向提示词、不用 CFG**；时间输入是 `t = 1 − σ`（t=1 为干净），模型预测的是数据向速度 `v = x₀ − ε`。训练损失要把这两个约定写对，否则 loss 会不降反升（IAmIronMan42 实测：约定反了 loss 7.2→9.5）。
2. **视频/音频用两条独立的 σ 调度**：视频 shift 12.0、音频 shift 3.0，同一时刻各自采样——音频和视频在推理时是锁步推进的，训练也要照做。
3. **显存（最大门槛）**：
   - 全精度（bf16）LoRA + 全注意力：实测 **8×A800-80GB** 级别，单样本 token 上限约 **70k**（448×768 @ 27s ≈ 65k token ≈ 76GB 稳态；576×1024 @ 30s ≈ 127k 直接 OOM）。
   - **NF4 量化版**（[DiffSynth-Studio/MiniMax-H3-NF4](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)）：所有组件可同时载入**单卡**，是消费级硬件的唯一现实路径；但仍建议从短片段（如 22–56 帧）+ 低分辨率（如 256–384 边长、32 倍数）冒烟测试起步，再按显存放大。
   - 工具普遍要求**两阶段**：先 `data_process` 把视频/音频/文本编码成 latent 缓存到磁盘（因为 DiT 与 Qwen3-VL-32B 文本编码器无法同卡共存），再正式训练。NF4 量化版可以单阶段（全部同时载入）。
4. **LoRA 超参参考**：rank 32（DiffSynth 默认）/ 16（IAmIronMan42，约 10.7M 参数），lr 1e-4，作用在 `attn.qkv_proj / attn.out_proj / mlp.fc1 / mlp.fc2`；开启 gradient checkpointing。训练完的 LoRA 是标准 H3 key 布局（`diffusion_model.blocks.N.attn.qkv_proj`），**ComfyUI 直接 `Load LoRA` 可用**。
5. **许可证（重要）**：[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)：
   - **排除领土：欧盟、英国、韩国、美国**——这些地区不可使用/分发（中国大陆在许可范围内）。
   - **LoRA 属于 Model Derivative**：分发你的 LoRA 需要附带许可副本 + NOTICE 文本，改动文件要标注。
   - 商业产品年收入 > 2000 万美元需向 MiniMax 单独申请授权（api@minimax.io）；商业 UI 需展示 "MiniMax H3"。
   - 不得用 H3 或其输出改进其他 AI 模型。
   - 文本编码器 Qwen3-VL-32B 是 Apache 2.0，单独适用。
6. **训练数据不要用 H3 自己的输出**（违反许可 V.3，也容易学进伪影）。

---

## 4. 三条训练路线对比

| 路线 | 硬件门槛 | 数据集格式 | 备注 |
|---|---|---|---|
| **DiffSynth-Studio**（推荐） | 单卡可训（NF4）~ 多卡 | `metadata.json`（video/prompt/audio/references） | 最完整：支持 FL2VA / Ref2VA / 全量 / LoRA；[LoRA 脚本](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Ref2VA.sh)；中文文档 |
| **IAmIronMan42/MiniMax-H3-FineTuning** | 生产级 8×A800 | jsonl manifest + `prepare_cache.py` 两阶段 | 参数最讲究（修正了 9 个坑），LoRA r16，含真实音频损失；但参考资料条件暂未接入 |
| **本仓库 ai-toolkit-easy2use** | 多卡（33B 全精度） | ai-toolkit 惯例（视频 + .txt 标注） | 代码层 `target_lora_modules = MiniMaxH3Transformer`、17n+5 帧对齐、音频同步都做好了，但**无现成 config 示例**，且推理权重是预量化 int8（训练需换 bf16 权重、自行写配置），不建议作为首选 |

社区已出现的 H3 LoRA 佐证可行性：
- [fal/MiniMax-H3-Realism-People-LoRA](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA)（人物写实风格 LoRA，约 1.2 万下载）
- [DiffSynth-Studio/MiniMax-H3-LoRA-LineartAnime](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-LoRA-LineartAnime)（线稿上色 LoRA）
- [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)（4–8 步推理加速 LoRA，约 600 步训练，证明小规模 LoRA 训练在普通硬件上可行）
- 参考：[HF 社区讨论 "How easy is it to train LoRAs"](https://huggingface.co/MiniMaxAI/MiniMax-H3/discussions/43)

---

## 5. 实操步骤（DiffSynth NF4 路线）

```bash
# 1. 安装
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio && pip install -e .

# 2. 准备数据集（见第 2 节）：视频 + metadata.json，全部 24fps

# 3. 训练（NF4 量化，单卡可跑；参数参考官方示例脚本）
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./my_dataset \
  --dataset_metadata_path ./my_dataset/metadata.json \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 --width 832 --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-text-encoder-nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-ref2va-nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:video_vae_nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:audio_vae_nf4.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --learning_rate 1e-4 --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path ./models/train/my-character \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing

# 4. 单卡显存不足时：降分辨率（保持 32 倍数）、降帧数（17n+5：22/39/56...）、
#    减小 --dataset_repeat、加 --gradient_accumulation_steps

# 5. 验证：固定 prompt + seed，对比 加载/不加载 LoRA 的输出
```

**上手建议**：
1. 先别急着训练——**H3 Ref2VA 的参考图能力可能已经满足"人物一致性"需求**（照片直传，零训练成本），先用它做一轮验证。
2. 真要训：先用 5–10 个短片、22 帧、低分辨率把管线跑通（冒烟测试），再放大到 124 帧 / 480×832 / 20–50 个片段。
3. 训练出来的 LoRA 在 ComfyUI 里直接加载（标准 H3 key），配合 FL2VA/Ref2VA 分区使用。

---

## 参考链接

- MiniMax-H3 权重：https://huggingface.co/MiniMaxAI/MiniMax-H3
- MiniMax-H3 Community License：https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
- DiffSynth-Studio H3 中文文档（推理+训练）：https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/zh/Model_Details/MiniMax-H3.md
- DiffSynth H3 LoRA 训练脚本：https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Ref2VA.sh
- IAmIronMan42/MiniMax-H3-FineTuning：https://github.com/IAmIronMan42/MiniMax-H3-FineTuning
- HF 社区讨论（LoRA 训练难度）：https://huggingface.co/MiniMaxAI/MiniMax-H3/discussions/43
- fal 人物写实 LoRA：https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA
- larryvrh Turbo LoRA：https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora
- DiffSynth 线稿 LoRA：https://huggingface.co/DiffSynth-Studio/MiniMax-H3-LoRA-LineartAnime
- ComfyUI MiniMax-H3 教程：https://docs.comfy.org/tutorials/video/minimax/minimax-h3
- 可灵 Custom Models（照片可训的官方平台路线）：https://klingai.com/release-note/release-notes/u3o4p73f2h
- 海螺"参考"一致性功能解读（卡兹克）：https://cloud.tencent.com.cn/developer/article/2513808
