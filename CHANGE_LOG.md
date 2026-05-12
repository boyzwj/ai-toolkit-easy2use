# 更新日志

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
