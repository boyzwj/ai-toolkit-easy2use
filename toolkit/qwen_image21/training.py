"""Qwen-specific training lifecycle using the toolkit's optimizer and EMA.

The loss and adaptive LR implementations are attributed in third_party/fizgig.
No training helper, context LoRA or preview LoRA is part of a user's checkpoint.
"""

import copy
import glob
import hashlib
import json
import math
import os
import random
import tempfile

import torch
from toolkit.print import print_acc

from .adaptive_lr import AdaptiveLR
from .cache import warm_reference_cache
from .config import plan_memory
from .loss_logger import PerImageLossWatch


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().cpu()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def configure_runtime(trainer):
    if trainer.model_config.arch != "qwen_image_2":
        return
    options = trainer.train_config.qwen_image_21
    if options.get("profile") == "legacy":
        return
    if trainer.accelerator.num_processes != 1:
        raise ValueError("Qwen 2.1 recipes currently require single-process training; use legacy for other launch modes")
    if options.get("memory_plan", "manual") not in ("auto", "manual"):
        raise ValueError("Qwen memory_plan must be auto or manual")
    if options.get("memory_plan") == "auto" and trainer.device_torch.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(trainer.device_torch)
        resolutions = [r for ds in trainer.dataset_configs
                       for r in (ds.resolution if isinstance(ds.resolution, list) else [ds.resolution])]
        plan = plan_memory(free_bytes / 1024 ** 3,
                           megapixels=max(resolutions or [704]) ** 2 / 1e6,
                           batch_size=trainer.train_config.batch_size)
        cfg = trainer.model_config
        cfg.quantize, cfg.qtype = plan["quantize"], plan["qtype"]
        cfg.quantize_te, cfg.qtype_te = True, "convrot8"
        cfg.low_vram = True
        cfg.layer_offloading = plan["offload_fraction"] > 0
        cfg.layer_offloading_transformer_percent = plan["offload_fraction"]
        if cfg.layer_offloading:
            cfg.model_kwargs["kv_cache"] = False
        print_acc(f"Qwen 2.1 VRAM plan: {free_bytes / 1024**3:.1f} GiB free, {plan}")
    compile_policy = options.get("compile", "off")
    if compile_policy not in ("off", "auto", "on"):
        raise ValueError("Qwen compile must be off, auto or on")
    # Keep explicit model.compile settings. Auto pays compilation cost only on
    # longer CUDA jobs that do not swap blocks; toolkit handles compile failures.
    if compile_policy != "off":
        trainer.model_config.compile = compile_policy == "on" or (
            trainer.device_torch.type == "cuda" and trainer.train_config.steps >= 500
            and not trainer.model_config.layer_offloading)
        trainer.model_config.block_compile = trainer.model_config.compile
        trainer.model_config.compile_fullgraph = False


class Qwen21TrainingController:
    def __init__(self, trainer):
        from toolkit.accelerator import unwrap_model
        self.trainer = trainer
        self.options = trainer.train_config.qwen_image_21
        self.network = unwrap_model(trainer.network)
        self.adaptive = None
        if self.options.get("adaptive_lr", False):
            self.adaptive = AdaptiveLR(self.options["adaptive_lr_min"], self.options["adaptive_lr_max"])
        self.watch = None
        if self.options.get("loss_watch", True):
            # Absolute file paths avoid collisions between multiple datasets.
            # Passive diagnostics by default: dataset captions remain untouched.
            self.watch = PerImageLossWatch(trainer.save_root, apply_lr=self.options.get("per_image_lr", False),
                                          write_jsonl=True, family="qwen_image_2", dataset_dir=trainer.save_root)
        self.epoch = trainer.epoch_num
        self.loss_sum = 0.0
        self.loss_count = 0
        self.previous_ema = None
        self.has_training_signal = True
        self.epoch_finished = False
        self.accumulation_pending = False
        self.recaption_attempts = {}
        self.datasets = []

    def fingerprint(self):
        cfg = self.trainer.model_config
        return {
            "parameters": {n: list(p.shape) for n, p in self.network.named_parameters() if p.requires_grad},
            "latent_sampling": cfg.model_kwargs.get("latent_sampling", "sample"),
            "train_blocks": cfg.model_kwargs.get("train_blocks"),
            "timestep_type": self.trainer.train_config.timestep_type,
            "timestep_range": [self.options.get("min_timestep", 0), self.options.get("max_timestep", 1)],
            "helper": [cfg.assistant_lora_path, self.options.get("training_adapter_strength", 1)],
            "helper_sha256": getattr(getattr(self.trainer.sd, "assistant_lora", None), "sha256", None),
            "context": [self.options.get("context_lora_path"), self.options.get("context_lora_strength", 1)],
            "context_sha256": getattr(getattr(self.trainer.sd, "context_lora", None), "sha256", None),
            "adaptive": [self.options.get("adaptive_lr", False), self.options.get("adaptive_lr_min"),
                         self.options.get("adaptive_lr_max")],
            "accumulation": [getattr(self.trainer.train_config, "gradient_accumulation", 1),
                             getattr(self.trainer.train_config, "gradient_accumulation_steps", 1)],
            "network_alpha": [getattr(getattr(self.trainer, "network_config", None), key, None)
                              for key in ("linear_alpha", "conv_alpha")],
        }

    def prepare(self):
        from toolkit.data_loader import get_dataloader_datasets
        trainer = self.trainer
        loaders = [loader for loader in (trainer.data_loader, trainer.data_loader_reg) if loader is not None]
        datasets = [ds for loader in loaders for ds in get_dataloader_datasets(loader)]
        self.datasets = datasets
        keys = [item.path for ds in datasets for item in ds.file_list]
        if self.watch is not None:
            self.watch.preflight(keys)
            warmup = self.options.get("warmup_image_paths", [])
            if warmup:
                self.watch.set_warmup_keys(warmup)
        if self.options.get("epochs") is not None:
            if trainer.train_config.gradient_accumulation_steps == -1:
                raise ValueError("Qwen epochs cannot be combined with whole-epoch gradient accumulation")
            steps_per_epoch = max(1, math.ceil(len(trainer.data_loader) / trainer.train_config.gradient_accumulation))
            trainer.train_config.steps = int(self.options["epochs"]) * steps_per_epoch
            if self.options.get("save_every_epoch", True):
                trainer.save_config.save_every = steps_per_epoch
            if self.options.get("sample_every_epoch", True):
                trainer.sample_config.sample_every = steps_per_epoch
            trainer.sd.print_and_status_update(f"Qwen 2.1: {self.options['epochs']} epochs, {trainer.train_config.steps} steps")
        if trainer.model_config.model_kwargs.get("split_mlp_lora", False):
            self.network.can_merge_in = False
        warm_reference_cache(trainer.sd, datasets)
        self.restore_checkpoint()

    def begin_step(self):
        if self.trainer.epoch_num != self.epoch:
            self.finish_epoch()
            self.epoch = self.trainer.epoch_num
            self.loss_sum = 0.0
            self.loss_count = 0
            self.epoch_finished = False
        if not self.accumulation_pending:
            self.has_training_signal = False

    def dataloader_epoch_boundary(self):
        # Called before workers/iterator for the next epoch are created, so
        # caption changes cannot leave that epoch using stale embedding paths.
        if self.options.get("auto_recaption", False):
            self.finish_epoch()
            from .recaption import repair_captions
            repair_captions(self)

    def observe_loss(self, loss, batch, timesteps):
        """Observe real per-image losses before reduction/gradient scaling."""
        scales = []
        values = loss.detach().float().cpu().tolist()
        ts = (timesteps.detach().float().cpu() / 1000).reshape(-1).tolist()
        for item, value, t in zip(batch.file_items, values, ts):
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite Qwen training loss for {item.path}")
            scale = 1.0
            # Regularization images do not define the training plateau/watch.
            if not item.is_reg:
                self.loss_sum += value
                self.loss_count += 1
                if self.watch is not None:
                    scale = 0.0 if self.watch.is_excluded([item.path]) else self.watch.multiplier([item.path])
                    self.watch.observe(epoch=self.epoch, step=self.trainer.step_num,
                                       item_keys=[item.path], timestep=t, loss=value)
            scales.append(scale)
        self.has_training_signal |= any(s > 0 for s in scales)
        return loss * loss.new_tensor(scales)

    def finish_epoch(self):
        if self.epoch_finished or not self.loss_count:
            return
        trainer = self.trainer
        if self.watch is not None:
            self.watch.epoch_boundary(self.epoch)
        if self.adaptive is not None:
            self.adaptive.epoch_boundary(self.epoch, self.loss_sum / self.loss_count, self.network, trainer.optimizer)
            if self.adaptive.last_rollback and trainer.ema is not None and self.previous_ema is not None:
                with torch.no_grad():
                    for shadow, previous in zip(trainer.ema.shadow_params, self.previous_ema):
                        shadow.mul_(1 - AdaptiveLR.BLEND).add_(previous.to(shadow), alpha=AdaptiveLR.BLEND)
            trainer.lr_scheduler.base_lrs = [pg["lr"] for pg in trainer.optimizer.param_groups]
        if trainer.ema is not None:
            self.previous_ema = cpu_copy(trainer.ema.shadow_params)
        self.epoch_finished = True

    @staticmethod
    def checkpoint_hash(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def watch_state(self):
        if self.watch is None:
            return None
        values, sets = {}, []
        for key, value in vars(self.watch).items():
            if key in ("output_dir", "dataset_dir", "_excl_file") or key == "_jsonl":
                continue
            if isinstance(value, set):
                values[key] = sorted(value)
                sets.append(key)
            else:
                values[key] = copy.deepcopy(value)
        return {"values": values, "sets": sets}

    def save_checkpoint(self, checkpoint_path, *, next_step):
        trainer = self.trainer
        if not trainer.accelerator.is_main_process:
            return
        if self.accumulation_pending:
            from toolkit.memory_management import sync_grad_transfers
            sync_grad_transfers()
        state = {
            "version": 1, "checkpoint_sha256": self.checkpoint_hash(checkpoint_path),
            "fingerprint": self.fingerprint(), "next_step": next_step,
            "epoch": self.epoch, "epoch_finished": self.epoch_finished,
            "loss_sum": self.loss_sum, "loss_count": self.loss_count,
            # The exported LoRA contains EMA weights. Resume must use the raw
            # weights that correspond to the saved optimizer momentum.
            "raw_weights": {n: p.detach().clone().cpu() for n, p in self.network.named_parameters() if p.requires_grad},
            "optimizer": cpu_copy(trainer.optimizer.state_dict()),
            "ema": cpu_copy(trainer.ema.state_dict()) if trainer.ema is not None else None,
            "previous_ema": self.previous_ema,
            "adaptive": self.adaptive.state_dict() if self.adaptive is not None else None,
            "watch": self.watch_state(), "rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "recaption_attempts": self.recaption_attempts,
            "accumulation_pending": self.accumulation_pending,
            "accumulation_step": getattr(trainer, "grad_accumulation_step", 1) + int(next_step > trainer.step_num),
            "has_training_signal": self.has_training_signal,
            "gradients": {n: p.grad.detach().clone().cpu() for n, p in self.network.named_parameters()
                          if self.accumulation_pending and p.grad is not None},
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        destination = checkpoint_path + ".training-state"
        fd, temporary = tempfile.mkstemp(dir=os.path.dirname(destination), suffix=".tmp")
        os.close(fd)
        try:
            torch.save(state, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        for stale in glob.glob(os.path.join(trainer.save_root, "*.safetensors.training-state")):
            if not os.path.isfile(stale.removesuffix(".training-state")):
                os.unlink(stale)

    def restore_checkpoint(self):
        trainer = self.trainer
        checkpoint = trainer.get_latest_save_path(include_pretrained_lora=False)
        path = checkpoint + ".training-state" if checkpoint else None
        if not path or not os.path.isfile(path):
            if trainer.start_step > 0:
                trainer.sd.print_and_status_update("Qwen legacy checkpoint has no raw/EMA training state; adaptive rollback starts fresh")
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("version") != 1 or state.get("fingerprint") != self.fingerprint():
            raise ValueError("Qwen checkpoint training configuration differs; resume with the original recipe or use a new job name")
        if state["checkpoint_sha256"] != self.checkpoint_hash(checkpoint):
            raise ValueError("Qwen training state does not match its LoRA checkpoint")
        current = dict(self.network.named_parameters())
        with torch.no_grad():
            for name, value in state["raw_weights"].items():
                current[name].copy_(value.to(current[name]))
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.lr_scheduler.base_lrs = [pg["lr"] for pg in trainer.optimizer.param_groups]
        if trainer.ema is not None and state["ema"] is not None:
            trainer.ema.load_state_dict(state["ema"])
        if self.adaptive is not None:
            self.adaptive.load_state_dict(state["adaptive"])
        if self.watch is not None and state["watch"] is not None:
            self.watch.__dict__.update(state["watch"]["values"])
            for key in state["watch"]["sets"]:
                setattr(self.watch, key, set(getattr(self.watch, key)))
        self.previous_ema = state["previous_ema"]
        self.recaption_attempts = state.get("recaption_attempts", {})
        self.accumulation_pending = state.get("accumulation_pending", False)
        trainer.grad_accumulation_step = state.get("accumulation_step", 1)
        self.has_training_signal = state.get("has_training_signal", True)
        for name, gradient in state.get("gradients", {}).items():
            current[name].grad = gradient.to(current[name])
        self.epoch = trainer.epoch_num = state["epoch"]
        self.epoch_finished = state["epoch_finished"]
        self.loss_sum, self.loss_count = state["loss_sum"], state["loss_count"]
        trainer.step_num = trainer.start_step = state["next_step"]
        torch.set_rng_state(state["rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        trainer.sd.print_and_status_update(f"Restored Qwen raw weights, EMA, optimizer and LR at step {trainer.start_step}")

    def close(self):
        self.finish_epoch()
        if self.watch is not None:
            self.watch.close()
