# 更新日志

## 2026-05-19

### 同步上游（ai-toolkit 主仓重要变更）

本次同步包含上游 ai-toolkit 近期多个关键改进，尚未推送至 origin/main 的 6 个 commit（含一次 merge）：

- **新增 HiDream-O1（hidream_o1）微调支持**：
  - 新增：
    - `extensions_built_in/diffusion_models/hidream/hidream_o1_model.py`
    - `hidream_o1/qwen3_vl_transformers.py`
    - `hidream_o1/pipeline.py`
    - `hidream_o1/model_config.py`
  - 特性：
    - 支持 HiDream-O1 系列图像的 LoRA/微调，适配 Comfy 兼容格式加载与保存。
    - AdvancedPromptEmbeds 引入 `frozen_dtype_keys`，保护部分张量不被强制改变 dtype。
    - 修复注意力实现在不同方法下的不一致性（hidream_o1）。
  - UI：
    - `/jobs/new` 增加 HiDream-O1 模型选项及默认配置（分辨率 2048、噪声尺度、ignore 层等）。
  - 影响：
    - 为需要 HiDream-O1 微调的用户提供开箱即用支持，不影响现有其他模型。

- **DFE7 引入 velocity_equiv_weight 损失加权**（`4bfe944`）：
  - 将 `DiffusionFeatureExtractor7` 中 head / depth / normals / segmentation 四项 loss 乘以：
    - `1 / clamp(t, min=0.1)^2`
  - 影响：
    - 高时间步误差被适度放大，更合理反映不同时间步的重要性。
    - 对使用 DFE7 的训练任务（如 Wan 等）可能改善特征对齐和收敛行为，无需修改配置。

- **恢复训练时清理未来 step 的 Loss 日志**（`b42acb9`）：
  - 在 `UILogger._prune_future_steps()` 中，当从较早 step 恢复时删除 `steps` 和关联 `metrics` 中大于当前 step 的记录，并修正 `metric_keys.last_seen_step`。
  - 影响：
    - 更准确反映真实训练历程，避免“幻影 future 曲线”误导。
    - 不改变训练行为，不影响正常从头训练。

- **DFE 相关模型梯度检查点增强与损失异常值抑制**（`ec58dcd`）：
  - SDTrainer 中为 DFE 模型梯度检查点新增对更多接口命名（如 `gradient_checkpointing_enable`）的兼容。
  - 引入两个新配置项：
    - `max_loss`：对单步 loss 做上限裁剪，抑制极端异常值。
    - `max_loss_debug`：开启后在超过 max_loss 时打印详细信息。
  - 影响：
    - 显存使用可能进一步降低（尤其是 DFE 相关任务）。
    - 使用 `max_loss` / `max_loss_debug` 可提升大模型训练稳定性。

### 影响概览

- 对现有训练：
  - 主要变化在训练日志一致性、DFE7 损失分布、梯度检查点兼容性和损失裁剪上，均为稳定性与可解释性改进，基本向后兼容。
- 对新特性：
  - HiDream-O1 提供全新的微调能力，不影响其他模型的默认行为。
  - 如不需要 HiDream-O1，可继续使用原有模型配置。

## 2026-05-12

### 新增特性

- **ModelScope 模型下载源**：设置页新增「模型下载源」切换（HuggingFace / ModelScope），训练和打标任务的模型下载自动通过 `modelscope` 包重定向到 modelscope.cn。需要在环境中 `pip install modelscope`。
- **自定义 API 打标器模型列表**：测试 API 成功后自动调用 `GET /v1/models` 拉取可用模型列表，以下拉框代替手动输入。
- **打标任务草稿持久化**：打标弹窗中的配置（API URL、Key、提示词等）自动保存到数据库，下次打开时恢复，无需重复输入。
- **默认打标器改为自定义 API**：新建打标任务默认选择"自定义 API"而非之前的音频打标器。

### 改进

- **连通性测试改用 `/health`**：自定义 API 测试先尝试 `GET /health` 端点，失败再回退 `POST /chat/completions`，不再强依赖模型名称。
- **ModelScope `patch_hub()` 正确集成**：使用 `modelscope.utils.hf_util.patcher.patch_hub()` 正确替换 `huggingface_hub.hf_hub_download`，同时添加诊断日志。
- **修复 `CreatableSelectInput` 状态循环 Bug**：选择 Custom 时不再调用 `onChange('')`，避免 `isCustom` 被 useEffect 重置导致无法输入。
- **同步上游合并**：合并 `ostris/ai-toolkit` 上游更新，解决 4 个文件冲突。
- **清除废弃配置**：移除 `next.config.ts` 中废弃的 `devIndicators.buildActivity` 和与 Turbopack 冲突的 `webpack` 配置。
- **添加 AGENTS.md**：项目级 AI 辅助指令文件。
