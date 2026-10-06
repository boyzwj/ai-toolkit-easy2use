# Qwen-Image 2.1 训练

训练主路径参考 [Fizgig](https://github.com/shootthesound/Fizgig/tree/1c8ec88edc1a4f99f73585aeae00fec740b12ea6)
的 2026-10-05 实现。配置示例：`config/examples/train_lora_qwen_image_21.yaml`。

新建 Web UI 任务选择 Qwen-Image-2.1 后，默认采用「人物快速训练」。旧配置没有
`train.qwen_image_21` 时保留原来的训练逻辑；已有任务恢复时不要更换预设、rank、身份层
或辅助 LoRA。要对旧数据采用新训练方式，使用新的任务名称。
新预设目前用于单进程训练。`identity` 是 Fizgig 的实验性少层方案，建议用同一数据集与
`fast` 对照；不能把个别人物的测试结果当作通用质量保证。

| 预设 | rank | 分辨率 | 学习率 | 训练层 |
| --- | ---: | ---: | --- | --- |
| fast | 8 | 704 | 自适应 2e-4～4e-4 | 全部 block |
| standard | 16 | 704 | 自适应 1e-4～2e-4 | 全部 block |
| identity | 8 | 512 | 自适应 2e-4～4e-4 | block 索引 10～14 |
| style | 16 | 704 | 固定 1.5e-4 | 全部 block |
| edit | 8 | 704 | 自适应 2e-4～4e-4 | 全部 block |
| edit_standard | 16 | 704 | 自适应 1e-4～2e-4 | 全部 block |

预设提供默认值；YAML 中显式设置的 rank、分辨率、缓存等配置优先生效。启用自适应
学习率时，初始学习率为上下限的几何平均，`lr` 输入不控制该模式。
UI 更换预设会同时更新相应设置，之后仍可调整。

`identity` 根据 Fizgig 两个人物实验挑选层，适合优先学习人物相似度。它不保证每个人物、
姿态或画风都比全部层更好。可以用相同图片、提示词和种子对比 `fast` 与 `identity`。
训练目标仍为普通 flow-matching MSE，**没有把 ArcFace 加入反向传播**。

## 训练行为

- 冻结 Fizgig 训练辅助 LoRA，默认强度 1；训练开启，预览关闭，导出的用户 LoRA 不含辅助权重。
  `training_adapter: false` 关闭默认辅助文件；自定义路径使用 `model.assistant_lora_path`。
- 时间步为 `sigmoid(N(0,1) + mu)`，`mu` 从 256 token 的 0.5 到 8192 token 的 0.9
  线性插值。训练不使用推理调度器的 terminal stretching。
  `min_timestep`、`max_timestep` 最后对采样值做线性缩放。
- VAE 使用 posterior mode。目标缓存和参考缓存包含版本、参考像素、透明度和缩放信息，
  不复用旧的随机采样缓存。
- MLP gate/up 分别训练独立 A/B，底模仍保留原生融合 `gate_up` GEMM 与量化数据。
  导出使用 Fizgig/diffusers 的 `transformer.*` split MLP 格式，ComfyUI 支持该格式。
  旧融合 gate/up LoRA 可无损拆成独立分支加载。工具箱推理引擎使用默认 `lora_mode: hook`。
- 默认缓存字幕并释放 Qwen3-VL；文本条件直接读取最后一层归一化前的隐藏状态，不计算
  词表 logits。纯文生图不保留视觉塔，编辑训练保留视觉塔。
- 编辑参考 latent 预缓存至数据目录的 `_qwen21_reference_cache`，按最终像素寻址；
  字幕 dropout 不会把参考条件错误地带入无参考提示。RGBA 通路继续可用。
  不同参考数量、比例及 dropout 的编辑 batch 分行计算；替换同路径参考图会重新生成文本缓存。
- EMA 默认 0.98。自适应学习率持续下降两轮后 ×1.25，停滞时 ×0.5；权重范数增长异常时，
  按 70% 上轮 / 30% 当前回退并恢复优化器状态，同时对齐 EMA。
- 每张图片的真实损失在 batch 归约前记录到 `loss_log/per_image_loss.jsonl`，问题诊断按噪声
  区间归一化；报告提供学习停滞、训练饱和及建议检查的 epoch。`per_image_lr` 默认关闭。
- `auto_recaption` 默认关闭。开启后，轮次之间用本地 Qwen3-VL-4B-Instruct 修复停滞图片字幕，
  原字幕备份到训练输出的 `caption_repairs`，更新全部相关文本缓存；最多两次修复后仍停滞则
  本次训练排除。使用文本缓存、梯度累积 1 和单进程训练；`captioner_model` 可指定本地模型。
  排除记录保存在任务输出的 `qwen21_excluded.json`。
- `context_lora_path` 作为冻结基础风格，训练和预览均启用。
  `preview_lora_path` 仅预览开启，可配合 turbo LoRA；对应预览调度器不做 terminal stretching，
  可通过 `preview_shift_terminal` 覆盖。预览步数仍由 `sample.sample_steps` 决定。
- 预览为每个正/负条件单独缓存 prefix KV。异常或显存不足后恢复辅助 LoRA、网络和 EMA
  状态，预览 OOM 不终止训练。

训练检查点旁的 `.safetensors.training-state` 文件保存原始训练权重、优化器、EMA、自适应
学习率和图片诊断状态。导出 `.safetensors` 是 EMA 权重；原始权重必须与对应的优化器一起
恢复。状态文件包含检查点摘要和配置指纹，不会配错检查点。保留检查点时也保留其状态文件。
跨步累积时还保存未提交的梯度和累积计数。恢复保持优化器状态一致，但不保证多 worker
数据顺序与一次不中断的运行逐图片相同。
像 Fizgig 一样，自适应回退快照驻留内存，恢复后的第一轮暂时没有上一轮的回退快照。

## 显存和训练时长

`memory_plan: auto` 根据启动时可用 CUDA 显存、分辨率和 batch size 选择 BF16 或原生
ConvRot8；显存不足时增加层卸载。使用容量估算，不把另一项目的显存/速度测量直接当作
当前项目的保证。保留原生 INT8 后端，避免自动换成未经此路径验证的 4-bit 训练精度。
`manual` 保留用户的量化和卸载设置。MPS 使用手动设置。

`compile: auto` 只在较长、无需卸载的 CUDA 任务启用按层编译，保持现有梯度检查点。
`on` 显式启用，`off` 沿用用户的 `model.compile` 设置。

`epochs` 根据实际 dataloader 长度换算总步数，默认每轮保存和预览。
未设置 `epochs` 而显式设置 `train.steps` 时按步数训练。UI 的「训练时长」可以选择任一方式。

## 验证

不下载底模的数值回归：

```bash
python -m unittest testing.test_qwen21_training
QWEN21_TEST_DEVICE=cuda python -m unittest testing.test_qwen21_training
```

使用本地底模的短训练、预览和续训检查（生成临时的合成数据，不改现有任务）：

```bash
python testing/smoke_qwen21_training.py --model /path/to/Qwen-Image-2.1
# 验证 INT8 和跨步梯度累积：
python testing/smoke_qwen21_training.py --model /path/to/Qwen-Image-2.1 --quantize --gradient-accumulation-steps 2
# 验证参考图缓存与编辑预览：
python testing/smoke_qwen21_training.py --model /path/to/Qwen-Image-2.1 --profile edit
```

人物评测工具：`scripts/evaluate_qwen21_identity.py`。使用固定提示词和种子生成各检查点样图，
按图库人脸中心向量的余弦相似度评分，并报告未检出人脸的图片；分数不等同于审美或综合画质。
Fizgig 的桌面评测界面、轮换全参数微调和其他模型专用训练器不在本 LoRA 路径中。

自适应学习率与问题图片诊断源码保留 Apache-2.0 归属，见 `third_party/fizgig/NOTICE`。
