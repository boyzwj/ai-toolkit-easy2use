"""Qwen 2.1 recipes adapted from Fizgig 1c8ec88 (Apache-2.0).

Explicit toolkit settings win over recipe defaults. ``legacy`` opts out, which
is useful when resuming a job created before these recipes were available.
"""

import copy
import math

TRAINING_ADAPTER = (
    "ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter/"
    "fizgig_qwen_image_2.1_training_adapter.safetensors"
)
IDENTITY_BLOCKS = [10, 11, 12, 13, 14]
PROFILES = {
    "fast": {"rank": 8, "min_lr": 2e-4, "max_lr": 4e-4, "resolution": 704},
    "standard": {"rank": 16, "min_lr": 1e-4, "max_lr": 2e-4, "resolution": 704},
    "identity": {"rank": 8, "min_lr": 2e-4, "max_lr": 4e-4, "resolution": 512,
                 "train_blocks": IDENTITY_BLOCKS},
    "style": {"rank": 16, "lr": 1.5e-4, "resolution": 704},
    "edit": {"rank": 8, "min_lr": 2e-4, "max_lr": 4e-4, "resolution": 704},
    "edit_standard": {"rank": 16, "min_lr": 1e-4, "max_lr": 2e-4, "resolution": 704},
}


def configure_qwen21_training(config):
    """Apply defaults before the trainer constructs its typed config objects."""
    model = config.get("model", {})
    if model.get("arch") != "qwen_image_2":
        return
    train = config.get("train", {})
    # Older saved jobs retain their original architecture and optimizer state.
    # New UI jobs and the new YAML example explicitly enable a recipe.
    if "qwen_image_21" not in train:
        return
    options = train["qwen_image_21"]
    if not isinstance(options, dict):
        raise ValueError("train.qwen_image_21 must be a configuration mapping")
    profile = options.setdefault("profile", "fast")
    if profile == "legacy":
        return
    if profile not in PROFILES:
        raise ValueError(f"Unknown Qwen 2.1 profile: {profile}")
    recipe = PROFILES[profile]
    if config.get("network") is None:
        raise ValueError("Qwen 2.1 recipes train adapters; use profile: legacy for full-model training")
    network = config["network"]
    network.setdefault("linear", network.get("rank", recipe["rank"]))
    network.setdefault("linear_alpha", network.get("alpha", network["linear"]))
    network.setdefault("transformer_only", True)
    train.setdefault("noise_scheduler", "flowmatch")
    train.setdefault("timestep_type", "shifted_logit_normal")
    train.setdefault("dtype", "bf16")
    train.setdefault("lr", recipe.get("lr", math.sqrt(recipe.get("min_lr", 1e-4) * recipe.get("max_lr", 2e-4))))
    train.setdefault("optimizer", "adamw8bit")
    train.setdefault("optimizer_params", {}).setdefault("weight_decay", 0.0)
    train.setdefault("max_grad_norm", 1.0)
    train.setdefault("gradient_checkpointing", True)
    train.setdefault("ema_config", {"use_ema": True, "ema_decay": 0.98})
    train.setdefault("cache_text_embeddings", True)
    train.setdefault("unload_text_encoder", train["cache_text_embeddings"])
    options.setdefault("adaptive_lr", "min_lr" in recipe)
    options.setdefault("adaptive_lr_min", recipe.get("min_lr", train["lr"]))
    options.setdefault("adaptive_lr_max", recipe.get("max_lr", train["lr"]))
    options.setdefault("loss_watch", True)
    options.setdefault("per_image_lr", False)
    options.setdefault("auto_recaption", False)
    options.setdefault("training_adapter", True)
    options.setdefault("training_adapter_strength", 1.0)
    options.setdefault("min_timestep", 0.0)
    options.setdefault("max_timestep", 1.0)
    options.setdefault("memory_plan", "manual")
    options.setdefault("compile", "off")
    if "steps" not in train:
        options.setdefault("epochs", 12 if profile.startswith("edit") else 30)
    if options["adaptive_lr"]:
        lo, hi = float(options["adaptive_lr_min"]), float(options["adaptive_lr_max"])
        if not (0 < lo <= hi):
            raise ValueError("Qwen adaptive LR requires 0 < min <= max")
        if train.get("lr_scheduler", "constant") != "constant":
            raise ValueError("Qwen adaptive LR requires lr_scheduler: constant")
        if train["optimizer"].lower() not in ("adamw", "adamw8bit"):
            raise ValueError("Qwen adaptive LR supports adamw and adamw8bit")
        # Adaptive mode, like Fizgig, starts at the geometric midpoint.
        train["lr"] = math.sqrt(lo * hi)
        train["unet_lr"] = train["lr"]
    if not 0 <= float(options["min_timestep"]) < float(options["max_timestep"]) <= 1:
        raise ValueError("Qwen timestep range must satisfy 0 <= min < max <= 1")
    if train.get("train_text_encoder", False):
        raise ValueError("Qwen 2.1 recipes require a frozen text encoder")
    if options["auto_recaption"]:
        if int(train.get("gradient_accumulation", 1)) != 1 or int(train.get("gradient_accumulation_steps", 1)) != 1:
            raise ValueError("Qwen automatic recaption requires gradient accumulation = 1")
        if not train["cache_text_embeddings"]:
            raise ValueError("Qwen automatic recaption requires text embedding caching")
        options["loss_watch"] = True
    if options.get("epochs") is not None and int(options["epochs"]) < 1:
        raise ValueError("Qwen training epochs must be positive")
    kwargs = model.setdefault("model_kwargs", {})
    if network.get("type", "lora").lower() != "lora":
        options.setdefault("split_mlp_lora", False)
        if options.get("split_mlp_lora"):
            raise ValueError("Independent Qwen MLP adapters require network type: lora")
    kwargs.setdefault("latent_sampling", "mode")
    kwargs.setdefault("split_mlp_lora", options.get("split_mlp_lora", True))
    if train.get("merge_network_on_save", False) and kwargs["split_mlp_lora"]:
        raise ValueError("Independent Qwen MLP adapters must be saved as a LoRA")
    kwargs.setdefault("cache_reference_latents", True)
    kwargs.setdefault("kv_cache", True)
    kwargs.setdefault("train_blocks", options.get("train_blocks", recipe.get("train_blocks")))
    datasets = config.get("datasets") or []
    samples = ((config.get("sample") or {}).get("samples") or []) + ((config.get("first_sample") or {}).get("samples") or [])
    uses_vision = any(d.get("control_path") or d.get("control_from_same_folder") or d.get("control_path_1") or d.get("control_path_2")
                      or d.get("control_path_3") or d.get("control_images") for d in datasets)
    uses_vision |= any(any(s.get(k) for k in ("ctrl_img", "ctrl_img_1", "ctrl_img_2", "ctrl_img_3"))
                       for s in samples if isinstance(s, dict))
    kwargs.setdefault("text_only", not uses_vision)
    # Both sides need the same options; take a copy so runtime planning cannot
    # mutate the user's saved training configuration.
    kwargs["qwen_training"] = copy.deepcopy(options)
    if options["training_adapter"]:
        model.setdefault("assistant_lora_path", TRAINING_ADAPTER)
    elif model.get("assistant_lora_path") == TRAINING_ADAPTER:
        model["assistant_lora_path"] = None
    for dataset in datasets:
        dataset.setdefault("resolution", recipe["resolution"])
        dataset.setdefault("cache_latents_to_disk", True)
    config.setdefault("sample", {}).setdefault("guidance_scale", 1.0)
    model.setdefault("quantize", True)
    model.setdefault("qtype", "convrot8")
    model.setdefault("quantize_te", True)
    model.setdefault("qtype_te", "convrot8")
    model.setdefault("low_vram", True)


def plan_memory(free_gb, *, megapixels=0.5, batch_size=1):
    """Conservative CUDA plan for the toolkit's native quantization backends.

    These are capacity estimates, not speed claims. INT8 convrot remains the
    low-memory default; swapping is preferable to silently changing to a
    different 4-bit training backend.
    """
    activation_gb = 2.0 * max(megapixels / 0.5, 0.5) * batch_size
    if free_gb >= 19.0 + activation_gb + 3.0:
        return {"quantize": False, "qtype": "convrot8", "offload_fraction": 0.0}
    resident_gb = 8.0 + activation_gb + 2.0
    fraction = min(0.85, max(0.0, (resident_gb - free_gb) / 7.0))
    return {"quantize": True, "qtype": "convrot8", "offload_fraction": fraction}
