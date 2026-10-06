"""Qwen-Image 2.1 for ai-toolkit.

One checkpoint does text-to-image and editing, so there is one arch: give it
reference images and it edits, give it none and it is plain T2I.

The pieces, all pulled from the Comfy-Org repack into the toolkit's ComfyUI
models folder:

  - transformer: 32-layer single-stream block-causal DiT (`src/transformer.py`),
  - text encoder: Qwen3-VL-8B (`src/text_encoder.py`), read at the last decoder
    layer BEFORE its final RMSNorm,
  - VAE: RGBA, 16x spatial, 64 latent channels (`src/vae.py`).

Text and images share one sequence: the Qwen3-VL encoder reserves a
`<|image_pad|>` slot per reference image and the DiT drops that image's VAE
latents into the slot, four latent tokens per slot. Reference tokens and text
are modulated from t=0 (`causal_condition`), so only the target image's tokens
see the sampled timestep.

Flow-matching convention matches ai-toolkit (t=1 noise -> t=0 clean, target =
noise - clean), so `get_noise_prediction` does no time flip or negation.
"""

import os
import re
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
from PIL import Image

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.train_tools import apply_noise_offset
from toolkit.models.base_model import BaseModel
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.qwen_image21.lora import load_frozen_lora, split_fused_lora

from .src.pipeline import (
    QwenImage21Pipeline,
    QwenImage21PromptEncoder,
    VAE_SCALE_FACTOR,
    VISION_TOKEN_PIXELS,
    pack_latents,
    pad_prompt_batch,
    prepare_condition_image,
    run_transformer,
    tensor_to_pil,
)
from .src.text_encoder import QwenImage21TextEncoder
from .src.transformer import QwenImage21Transformer2DModel
from .src.vae import AutoencoderKLQwenImage21

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


# matches scheduler/scheduler_config.json in the base repo
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# the Comfy-Org repack is the weight source; the original repo supplies the
# configs and the processor, which the repack does not carry
COMFY_REPO = "Comfy-Org/Qwen-Image-2.1"
BASE_REPO = "Qwen/Qwen-Image-2.1"

# decode above this many output pixels goes through the VAE's tiled path
TILE_DECODE_ABOVE_PIXELS = 1024 * 1024


def _drop_repeats(images):
    """Drop a reference that repeats the one before it.

    `GenerateImageConfig.ctrl_img_1` defaults to `ctrl_img`, so the sampler
    hands the same reference over twice when only one was configured. Here that
    is not merely wasteful: the prompt would reserve slots for two images while
    only one set of latents arrives.
    """
    kept = []
    for image in images:
        previous = kept[-1] if kept else None
        if (
            previous is not None
            and previous.shape == image.shape
            and torch.equal(previous, image)
        ):
            continue
        kept.append(image)
    return kept


class QwenImage2Model(BaseModel):
    arch = "qwen_image_2"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.use_old_lokr_format = False
        self.target_lora_modules = ["QwenImage21Transformer2DModel"]
        self.vae_scale_factor = VAE_SCALE_FACTOR
        self.prompt_encoder: Optional[QwenImage21PromptEncoder] = None

        # Editing is not a separate model here, it is what happens when the
        # dataset has a control path: reference images ride into the text
        # embeddings as vision tokens and their latents into the sequence.
        self.encode_control_in_text_embeddings = True
        self.has_multiple_control_images = True
        # References keep their own size/aspect; prepare_condition_image budgets them
        # identically in the cache and training paths so slot counts always agree.
        # Training runs heterogeneous reference layouts individually within a batch.
        self.use_raw_control_images = True
        self.qwen_training_options = model_config.model_kwargs.get("qwen_training", {})
        self.context_lora = None
        self.preview_lora = None
        blocks = model_config.model_kwargs.get("train_blocks")
        self.train_blocks = None if blocks is None else frozenset(int(b) for b in blocks)
        if self.train_blocks is not None and (not self.train_blocks or not self.train_blocks <= set(range(32))):
            raise ValueError("Qwen train_blocks must be a nonempty subset of 0..31")

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # 16 for the VAE, 2 more because the DiT groups target latent tokens
        # into 2x2 blocks, one per vision slot
        return VISION_TOKEN_PIXELS

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Qwen-Image 2.1 model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        if base_model_path == model_path and not os.path.isdir(base_model_path):
            # extras default to name_or_path, which is the comfy repack (or a
            # single file); neither carries the configs or the processor
            base_model_path = BASE_REPO
        elif os.path.isdir(model_path) and os.path.isdir(
            os.path.join(model_path, "text_encoder")
        ):
            # a local full checkpoint supplies its own text encoder / vae
            base_model_path = model_path

        self.print_and_status_update("Loading transformer")
        transformer = QwenImage21Transformer2DModel.load(
            model_path,
            config_path=base_model_path,
            **self.component_load_kwargs("transformer"),
        )
        flush()

        te_path = self.model_config.te_name_or_path or base_model_path
        if not self.model_config.te_name_or_path:
            te_file = os.path.join(base_model_path, "text_encoder", "model.safetensors")
            if os.path.isfile(te_file):
                # Single-file encoders may carry Comfy quantization markers;
                # use the state-dict loader so their layout and buffers are restored.
                te_path = te_file
        self.print_and_status_update(f"Loading text encoder from {te_path}")
        self._text_encoder_load_args = (te_path, base_model_path)
        processor = QwenImage21TextEncoder.load_processor(base_model_path)
        text_encoder = QwenImage21TextEncoder.load_model(
            te_path,
            dtype=dtype,
            config_path=base_model_path,
            subfolder="text_encoder",
        )
        # Conditioning reads the decoder backbone directly, so no LM head is
        # needed. T2I recipes also omit the unused vision tower.
        text_encoder.lm_head = None
        if self.model_config.model_kwargs.get("text_only", False):
            text_encoder.drop_vision_tower()
        # the vision tower stays: any prompt may carry reference images. bf16
        # Conv3d has no fast kernel, the equivalent GEMM does
        text_encoder.patch_vision_patch_embed()
        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        flush()

        vae_path = self.model_config.vae_path or base_model_path
        self.print_and_status_update(f"Loading VAE from {vae_path}")
        vae = AutoencoderKLQwenImage21.load(
            vae_path,
            config_path=base_model_path,
            **self.component_load_kwargs("vae"),
        )
        vae.requires_grad_(False)
        vae.eval()

        self.noise_scheduler = QwenImage2Model.get_train_scheduler()

        self.vae = vae
        self.text_encoder = [text_encoder]
        self.tokenizer = [processor.tokenizer]
        self.processor = processor
        self.model = transformer
        # Residual projections have no base parameters: the fused native
        # weights and quantization buffers are not changed by this operation.
        for block in transformer.transformer_blocks:
            block.img_mlp.enable_split_lora()
        self.prompt_encoder = QwenImage21PromptEncoder(text_encoder, processor)
        self.pipeline = QwenImage21Pipeline(self)
        if self.model_config.assistant_lora_path:
            self.assistant_lora = load_frozen_lora(
                self, self.model_config.assistant_lora_path,
                self.qwen_training_options.get("training_adapter_strength", 1.0),
            )
        for role in ("context", "preview"):
            path = self.qwen_training_options.get(f"{role}_lora_path")
            if path:
                adapter = load_frozen_lora(self, path, self.qwen_training_options.get(f"{role}_lora_strength", 1.0))
                setattr(self, f"{role}_lora", adapter)
                adapter.is_active = role == "context"
        from toolkit.qwen_image21.cache import ReferenceLatentCache
        self.reference_latent_cache = ReferenceLatentCache(self)
        self.print_and_status_update("Model Loaded")

    def before_text_encoder_unload(self):
        # The generic unloader replaces holder/pipeline references. This
        # wrapper otherwise keeps the entire Qwen3-VL alive after caching.
        self.prompt_encoder = None

    def reload_text_encoder(self):
        if self.prompt_encoder is not None:
            return
        path, config_path = self._text_encoder_load_args
        text_encoder = QwenImage21TextEncoder.load_model(path, dtype=self.torch_dtype,
                                                       config_path=config_path, subfolder="text_encoder")
        text_encoder.lm_head = None
        if self.model_config.model_kwargs.get("text_only", False):
            text_encoder.drop_vision_tower()
        text_encoder.patch_vision_patch_embed()
        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))
        text_encoder.eval().requires_grad_(False)
        self.text_encoder = [text_encoder]
        self.prompt_encoder = QwenImage21PromptEncoder(text_encoder, self.processor)

    def get_latent_space_version(self):
        version = super().get_latent_space_version()
        if self.model_config.model_kwargs.get("latent_sampling", "sample") == "mode":
            precision = "fp32-v2" if self.qwen_training_options and self.qwen_training_options.get("profile") != "legacy" else "v1"
            return f"{version}:posterior-mode-{precision}"
        return version

    @property
    def latent_cache_dtype(self):
        return torch.float32 if self.qwen_training_options and self.qwen_training_options.get("profile") != "legacy" else self.torch_dtype

    def should_train_lora_module(self, name, module):
        split = self.model_config.model_kwargs.get("split_mlp_lora", False)
        if name.endswith(".img_mlp.gate_up") and split:
            return False
        if module.__class__.__name__ == "QwenImage21LoRAProjection" and not split:
            return False
        if self.train_blocks is not None:
            match = re.search(r"(?:^|\.)transformer_blocks\.(\d+)\.", name)
            return match is not None and int(match.group(1)) in self.train_blocks
        return True

    def convert_lora_weights_before_save(self, state_dict):
        if any(".img_mlp.gate_layer." in k or ".img_mlp.proj." in k for k in state_dict):
            # Split MLP adapters use the diffusers/Fizgig namespace. ComfyUI
            # supports this layout as well as its own fused gate_up layout.
            return {k.replace("diffusion_model.", "transformer.", 1): v for k, v in state_dict.items()}
        return super().convert_lora_weights_before_save(state_dict)

    def convert_lora_weights_before_load(self, state_dict):
        result = super().convert_lora_weights_before_load(state_dict)
        if self.model_config.model_kwargs.get("split_mlp_lora", False):
            result = split_fused_lora(result)
        return result

    def generate_images(self, *args, **kwargs):
        # BaseModel toggles the training helper. Restore every adapter even if
        # preview decoding or sampling raises (including an OOM).
        adapters = [a for a in (self.assistant_lora, self.context_lora, self.preview_lora) if a is not None]
        active = [a.is_active for a in adapters]
        network = self.network
        training = self.model.training
        network_state = None if network is None else (network.training, network.is_active, network.multiplier)
        if network is not None and self.model_config.model_kwargs.get("split_mlp_lora", False):
            network.can_merge_in = False
        try:
            if self.preview_lora is not None:
                self.preview_lora.is_active = True
            return super().generate_images(*args, **kwargs)
        finally:
            for adapter, was_active in zip(adapters, active):
                adapter.is_active = was_active
            self.model.train(training)
            if network is not None:
                network.train(network_state[0])
                network.is_active, network.multiplier = network_state[1:]

    # ------------------------------------------------------------------
    # VAE. The latents are RGBA. Images without alpha get an opaque one on
    # encode, and decode drops it again unless RGBA output is on.
    # ------------------------------------------------------------------
    @property
    def load_rgba(self) -> bool:
        return bool(self.model_config.model_kwargs.get("rgba", False))

    def _latent_stats(self, device, dtype):
        shape = (1, self.vae.config.z_dim, 1, 1, 1)
        mean = torch.tensor(self.vae.config.latents_mean).view(shape).to(device, dtype)
        std = torch.tensor(self.vae.config.latents_std).view(shape).to(device, dtype)
        return mean, std

    def encode_images(self, image_list, device=None, dtype=None):
        device = device or self.vae_device_torch
        dtype = dtype or self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()

        images = torch.stack(
            [image.to(device, dtype=dtype) for image in image_list]
        ).to(device, dtype=dtype)
        if images.shape[1] == 3:
            # opaque alpha, in the [-1, 1] the VAE reads
            images = torch.cat([images, torch.ones_like(images[:, :1])], dim=1)
        images = images.unsqueeze(2)  # single-frame dim

        posterior = self.vae.encode(images).latent_dist
        sampling = self.model_config.model_kwargs.get("latent_sampling", "sample")
        if sampling not in ("sample", "mode"):
            raise ValueError("Qwen latent_sampling must be sample or mode")
        latents = posterior.mode() if sampling == "mode" else posterior.sample()
        if self.qwen_training_options and self.qwen_training_options.get("profile") != "legacy":
            latents = latents.float()
        mean, std = self._latent_stats(latents.device, latents.dtype)
        latents = (latents - mean) / std
        return latents.squeeze(2).to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        images = self._decode_rgba(latents, device=device, dtype=dtype)
        if not self.load_rgba:
            images = images[:, :3]
        return images

    def _decode_rgba(self, latents: torch.Tensor, device=None, dtype=None):
        device = device or self.vae_device_torch
        dtype = dtype or self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype).unsqueeze(2)
        mean, std = self._latent_stats(latents.device, latents.dtype)
        latents = latents * std + mean

        # A one-shot decode of this VAE's 16x upsample stack is heavy: 2048x2048
        # -- the resolution Qwen recommends -- needs more than a 32 GB card has.
        # Tile above 1 MP, and whenever low_vram is set.
        pixels = latents.shape[-2] * latents.shape[-1] * self.vae_scale_factor**2
        tiled = self.model_config.low_vram or pixels > TILE_DECODE_ABOVE_PIXELS
        if tiled:
            self.vae.enable_tiling(
                tile_sample_min_height=1024,
                tile_sample_min_width=1024,
                tile_sample_stride_height=768,
                tile_sample_stride_width=768,
            )
        try:
            images = self.vae.decode(latents).sample
        finally:
            if tiled:
                self.vae.disable_tiling()
        return images.squeeze(2).to(device, dtype=dtype)

    def decode_to_images(self, latents: torch.Tensor) -> List[Image.Image]:
        """Decode to PIL, keeping the alpha channel when load_rgba is set."""
        return [
            self.image_tensor_to_pil(image) for image in self.decode_latents(latents)
        ]

    @staticmethod
    def image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
        """`(C, H, W)` in [-1, 1] -> PIL (RGB, or RGBA when C is 4)."""
        array = (image.float().clamp(-1, 1) / 2 + 0.5).permute(1, 2, 0)
        array = (array.cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(array, mode="RGBA" if array.shape[2] == 4 else "RGB")

    # ------------------------------------------------------------------
    # Reference images
    # ------------------------------------------------------------------
    @property
    def control_image_max_pixels(self) -> int:
        """Pixel budget a reference image is shrunk to fit when no target size is
        known (blank/static prompts). A smaller reference keeps its size, only
        snapped to the 32 px grid."""
        return int(
            self.model_config.model_kwargs.get("control_image_max_pixels", 1024 * 1024)
        )

    @property
    def match_target_res(self) -> bool:
        """References are scaled to the target's pixel area (own aspect kept).
        model_kwargs.match_target_res: false -> control_image_max_pixels cap only."""
        return bool(self.model_config.model_kwargs.get("match_target_res", True))

    @property
    def text_embedding_uses_target_size(self) -> bool:
        # the dataloader adds the item's bucket size to the text-embedding cache key
        return self.match_target_res

    def get_text_embedding_space_version(self) -> str:
        # reference sizing changes the vision tokens; keep caches from different rules apart
        rule = "match" if self.match_target_res else f"cap{self.control_image_max_pixels}"
        return f"{self.text_embedding_space_version}_ref{rule}_vision-no-resize-v2"

    def _target_pixels(self, target_size) -> Optional[int]:
        """`(width, height)` -> pixel area on the 32 px grid, or None. Floors the
        same way generate_single_image does so the TE and VAE passes agree."""
        if target_size is None:
            return None
        divisor = self.get_bucket_divisibility()
        width, height = target_size
        return int(width // divisor * divisor) * int(height // divisor * divisor)

    def _normalize_control_images(self, control_images, batch_size: int) -> List[List]:
        """Any of the shapes the toolkit hands over -> one list per batch item.

        Control images arrive as a `(B, C, H, W)` batch tensor, a
        `(B, N, C, H, W)` multi-reference tensor, a per-sample list of lists, or
        a flat list for a single prompt (sampling / blank-embed caching).
        """
        if control_images is None:
            return [[] for _ in range(batch_size)]
        if isinstance(control_images, torch.Tensor):
            if control_images.dim() == 5:
                control_images = [list(sample) for sample in control_images]
            else:
                control_images = [[sample] for sample in control_images]
        elif len(control_images) > 0 and not isinstance(control_images[0], list):
            control_images = [list(control_images)]
        control_images = [_drop_repeats(sample) for sample in control_images]
        if len(control_images) == 1 and batch_size > 1:
            control_images = control_images * batch_size
        if len(control_images) != batch_size:
            raise ValueError(
                f"got {len(control_images)} control image sets for {batch_size} prompts"
            )
        return control_images

    def _prepare_control_images(
        self,
        control_images: List[List[torch.Tensor]],
        target_pixels: Optional[int] = None,
    ) -> List[List[torch.Tensor]]:
        """Put every reference on the 32 px grid, as `(1, C, H, W)` in [0, 1].
        With match_target_res and a known target, scale each to the target's area."""
        match = self.match_target_res and target_pixels is not None
        budget = target_pixels if match else self.control_image_max_pixels
        prepared = []
        for sample in control_images:
            images = []
            for image in sample:
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                images.append(
                    prepare_condition_image(
                        image.to(self.device_torch), budget, match=match
                    )
                )
            prepared.append(images)
        return prepared

    def encode_condition_images(self, control_images, cache_paths=None):
        """Reference images -> `(B, N, C)` packed latents plus their latent grids.

        `control_images` is a per-sample list of `(1, C, H, W)` tensors in
        [0, 1], already at their final size.
        """
        if not control_images or not any(len(sample) for sample in control_images):
            return None, []

        sample_latents, shapes = [], []
        for index, sample in enumerate(control_images):
            packed = []
            for image in sample:
                path = cache_paths[index] if cache_paths is not None else None
                latent = self.reference_latent_cache.get(image, dataset_image_path=path)
                if index == 0:
                    shapes.append((latent.shape[2], latent.shape[3]))
                packed.append(pack_latents(latent))
            sample_latents.append(torch.cat(packed, dim=1))

        lengths = {latents.shape[1] for latents in sample_latents}
        if len(lengths) > 1:
            raise ValueError(
                "every sample in a batch must contribute the same reference image "
                f"token count, got {sorted(lengths)}. The dataloader bucket-resizes "
                "reference images, so this means the samples disagree on how many."
            )
        return torch.cat(sample_latents, dim=0), shapes

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------
    def get_prompt_embeds(
        self, prompt, control_images=None, target_size=None
    ) -> AdvancedPromptEmbeds:
        if self.prompt_encoder is None:
            raise RuntimeError("Qwen text encoder has been unloaded; use cached embeddings or disable text caching")
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.text_encoder[0].device != self.device_torch:
            self.text_encoder[0].to(self.device_torch)

        images = None
        if control_images is not None:
            samples = self._prepare_control_images(
                self._normalize_control_images(control_images, len(prompt)),
                target_pixels=self._target_pixels(target_size),
            )
            images = [[tensor_to_pil(image) for image in sample] for sample in samples]

        embeds, masks, slot_masks = self.prompt_encoder.encode(
            prompt, images=images, device=self.device_torch
        )
        pe = AdvancedPromptEmbeds(
            text_embeds=[embed.to(self.torch_dtype) for embed in embeds],
            attention_mask=masks,
            image_slot_mask=[slots.to(torch.bool) for slots in slot_masks],
        )
        # the masks are bookkeeping, not activations: a .to(dtype) must not
        # turn them into bf16
        pe.frozen_dtype_keys = ["attention_mask", "image_slot_mask"]
        return pe

    def pad_prompt_embeds(self, prompt_embeds: AdvancedPromptEmbeds):
        """`AdvancedPromptEmbeds` -> the padded `(embeds, mask, slot_mask)` batch."""
        return pad_prompt_batch(
            prompt_embeds.text_embeds,
            prompt_embeds.attention_mask,
            prompt_embeds.image_slot_mask,
            self.device_torch,
            self.torch_dtype,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def condition_noisy_latents(
        self, latents: torch.Tensor, batch: "DataLoaderBatchDTO"
    ):
        # reference latents join the sequence in get_noise_prediction, clean
        return latents.detach()

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 64, h, w)
        timestep: torch.Tensor,  # 0..1000 scale
        text_embeddings: AdvancedPromptEmbeds,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        batch_size = latent_model_input.shape[0]

        prompt_embeds, prompt_mask, slot_mask = self.pad_prompt_embeds(text_embeddings)

        # The prompt is what decides: it reserved the slots the references go
        # into, so a prompt encoded without them (a plain T2I dataset, a fully
        # dropped caption) takes no references here either.
        samples = None
        if batch is not None and bool(slot_mask.any()):
            with torch.no_grad():
                control = batch.control_tensor_list
                if control is None:
                    control = batch.control_tensor
                # same area the dataloader cached the prompt against (bucket crop)
                target_pixels = self._target_pixels((
                    latent_model_input.shape[3] * VAE_SCALE_FACTOR,
                    latent_model_input.shape[2] * VAE_SCALE_FACTOR,
                ))
                samples = self._prepare_control_images(
                    self._normalize_control_images(control, batch_size),
                    target_pixels=target_pixels,
                )

        # toolkit timestep (0..1000, 1000 = pure noise) -> the model's t in [0, 1];
        # same direction, so a plain divide
        t = timestep.to(self.device_torch, dtype=self.torch_dtype) / 1000
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != batch_size:
            t = t.expand(batch_size)

        # Dropout can remove reference slots from only some rows. References
        # can also have different aspect ratios/counts; their RoPE grids are
        # per image. Run these rows separately rather than reuse row 0's layout.
        paths = [item.path for item in batch.file_items] if batch is not None else None
        def predict(indices):
            condition_latents, condition_shapes = None, []
            if samples is not None and bool(slot_mask[indices].any()):
                with torch.no_grad():
                    condition_latents, condition_shapes = self.encode_condition_images(
                        [samples[i] for i in indices],
                        cache_paths=[paths[i] for i in indices] if paths is not None else None,
                    )
            return run_transformer(
                self.transformer,
                latent_model_input[indices].to(self.device_torch, self.torch_dtype),
                t[indices], prompt_embeds[indices], prompt_mask[indices], slot_mask[indices],
                condition_latents=condition_latents, condition_shapes=condition_shapes, **kwargs,
            )
        if samples is not None and batch_size > 1:
            return torch.cat([predict([i]) for i in range(batch_size)], dim=0)
        return predict(list(range(batch_size)))

    def get_loss_target(self, *args, **kwargs):
        # flow-matching velocity target: noise - clean
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        if self.qwen_training_options and self.qwen_training_options.get("profile") != "legacy":
            return (noise.float() - batch.latents.float()).detach()
        return (noise - batch.latents).detach()

    def get_latent_noise_from_latents(self, latents, noise_offset=0.0):
        if self.qwen_training_options and self.qwen_training_options.get("profile") != "legacy":
            return apply_noise_offset(torch.randn_like(latents, dtype=torch.float32), noise_offset)
        return super().get_latent_noise_from_latents(latents, noise_offset=noise_offset)

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return QwenImage21Pipeline(self)

    def generate_single_image(
        self,
        pipeline: QwenImage21Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        divisor = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // divisor * divisor)
        gen_config.height = int(gen_config.height // divisor * divisor)

        # the same list the sampler built for the prompt embeddings, so the two
        # agree on how many references there are
        paths = [
            path
            for path in (
                gen_config.ctrl_img,
                gen_config.ctrl_img_1,
                gen_config.ctrl_img_2,
                gen_config.ctrl_img_3,
            )
            if path is not None
        ]
        condition_images = None
        if paths:
            # same channels the dataloader gives training references, so a
            # transparent reference behaves the same way in both
            mode = "RGBA" if self.load_rgba else "RGB"
            tensors = [
                torch.from_numpy(
                    np.array(Image.open(path).convert(mode), dtype=np.float32) / 255.0
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
                for path in paths
            ]
            condition_images = self._prepare_control_images(
                self._normalize_control_images([tensors], 1),
                target_pixels=self._target_pixels((gen_config.width, gen_config.height)),
            )

        return pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            condition_images=condition_images,
        )[0]

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def save_model(self, output_path, meta, save_dtype):
        # comfy-format single-file save; prequantized layers keep their storage
        transformer: QwenImage21Transformer2DModel = unwrap_model(self.model)
        if not output_path.endswith(".safetensors"):
            output_path += ".safetensors"
        transformer.save_model(
            output_path,
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name=self.arch),
        )

    def get_base_model_version(self):
        return self.arch

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

    def get_quantization_exclude_modules(self):
        return QwenImage21Transformer2DModel.get_quantization_exclude_modules()

    lora_keys_use_comfy_prefix = True
