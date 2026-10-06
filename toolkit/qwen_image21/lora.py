"""Independent gate/up adapters over a fused, possibly quantized, Qwen MLP."""

import hashlib
import torch


class QwenImage21LoRAProjection(torch.nn.Module):
    """A zero residual branch; it owns no base weights.

    The toolkit's LoRA module supplies a separate A/B pair on each branch.
    The base gate_up GEMM and its quantization buffers stay byte-identical.
    Split adapter keys match Fizgig/diffusers and are also accepted by ComfyUI.
    """
    lora_can_merge = False

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = None

    @property
    def enabled(self):
        return bool(self._forward_hooks) or getattr(self.forward, "__self__", self) is not self

    def forward(self, x):
        return x.new_zeros((*x.shape[:-1], self.out_features))


def split_fused_lora(state_dict):
    """Losslessly split a *LoRA* on gate_up, including its rank/alpha scale.

    Sharing A in the input file is legitimate; copy it to both independent
    branches. Do not concatenate independent A matrices (that changes rank).
    Fused LoKr/diff checkpoints keep their fused module and need no conversion.
    """
    result = {}
    for key, value in state_dict.items():
        if ".img_mlp.gate_up." not in key:
            result[key] = value
            continue
        suffix = key.split(".img_mlp.gate_up.", 1)[1]
        if suffix not in ("lora_A.weight", "lora_down.weight", "lora_B.weight", "lora_up.weight", "alpha"):
            result[key] = value
            continue
        values = value.chunk(2, dim=0) if suffix in ("lora_B.weight", "lora_up.weight") else (value, value)
        for branch, tensor in zip(("gate_layer", "proj"), values):
            result[key.replace(".img_mlp.gate_up.", f".img_mlp.{branch}.")] = tensor.contiguous()
    return result


class FrozenTrainingLoRA:
    """Compatibility with BaseModel's assistant lifecycle, without merging.

    All tensors remain frozen and outside the training network's state_dict.
    Removing hooks also releases their device caches during previews/offload.
    """
    def __init__(self, adapter):
        self.adapter = adapter
        digest = hashlib.sha256()
        with open(adapter.local_path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.sha256 = digest.hexdigest()
        self._is_active = False
        self.is_active = True

    @property
    def is_active(self):
        return self._is_active

    @is_active.setter
    def is_active(self, active):
        self.adapter.detach()
        self._is_active = bool(active)
        if self._is_active:
            self.adapter.attach()

    def force_to(self, device, dtype=None):
        # InferenceLoRA lazily caches per-device copies on the first forward.
        # Re-attach so stale CUDA caches are not held across an offload.
        self.is_active = self._is_active


def load_frozen_lora(holder, path, strength=1.0):
    from toolkit.inference_lora import InferenceLoRA
    adapter = InferenceLoRA(path, strength=strength).load(holder, status_fn=holder.print_and_status_update)
    incomplete = [e.name for e in adapter.entries if (e.A is None) != (e.B is None)]
    if not adapter.entries or adapter.unmatched or incomplete or adapter.convert_error:
        details = adapter.unmatched[:8] + incomplete[:8]
        raise ValueError(f"Qwen frozen LoRA did not load completely: {path}: {details}, {adapter.convert_error}")
    return FrozenTrainingLoRA(adapter)
