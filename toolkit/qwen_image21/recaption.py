"""Optional between-epoch caption repair; original captions are backed up.

Unlike the passive loss watch, this feature explicitly changes dataset captions.
It is disabled by default and only runs at an optimizer/epoch boundary.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from toolkit.basic import flush


def atomic_text(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class LocalQwenCaptioner:
    def __init__(self, controller):
        import torch
        from transformers import AutoProcessor
        from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextEncoder
        trainer = controller.trainer
        path = controller.options.get("captioner_model", "Qwen/Qwen3-VL-4B-Instruct")
        self.model = Qwen3VLTextEncoder.load_model(path, dtype=torch.bfloat16, config_path=path, subfolder="")
        self.model.patch_vision_patch_embed()
        self.model.aitk_post_load(qtype="convrot8", dtype=torch.bfloat16, device=trainer.device_torch, low_vram=True)
        self.model.eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(path)
        self.device = trainer.device_torch
        self.options = controller.options

    def caption(self, image_path, attempt):
        import torch
        from PIL import Image, ImageOps
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            image = Image.alpha_composite(background, image).convert("RGB")
            image.thumbnail((768, 768))
        instruction = self.options.get("recaption_instruction", (
            "Describe this image accurately in concise natural language. Include the subject's pose, expression, "
            "clothing, hairstyle, lighting and background. Do not invent names or hidden details."
        ))
        if attempt >= 2:
            instruction = self.options.get("recaption_instruction_detailed", instruction + " Include all visible visual details.")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=384)
        caption = self.processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        caption = caption.split("</think>")[-1].strip()
        if not caption:
            raise ValueError(f"Captioner returned no caption for {image_path}")
        return caption

    def close(self):
        from toolkit.memory_management import MemoryManager
        MemoryManager.free(self.model)
        self.model = None
        self.processor = None
        flush()


def reset_caption_cache(item):
    for name in ("raw_caption", "raw_caption_short", "caption", "caption_short", "caption_dop", "caption_dopsd",
                 "prompt_embeds", "dop_prompt_embeds", "dopsd_prompt_embeds",
                 "_text_embedding_path", "_blank_text_embedding_path", "_dop_text_embedding_path",
                 "_dop_blank_text_embedding_path", "_dopsd_text_embedding_path", "_dopsd_blank_text_embedding_path"):
        if hasattr(item, name):
            setattr(item, name, None)
    item.load_caption()


def repair_captions(controller, captioner_factory=LocalQwenCaptioner):
    if not controller.options.get("auto_recaption", False) or controller.watch is None:
        return
    trainer, model = controller.trainer, controller.trainer.sd
    # Multiple resolution datasets contain the same physical caption. Repair
    # it once, and refresh every corresponding item/cache.
    items = {item.path: item for dataset in controller.datasets for item in dataset.file_list if not item.is_reg}
    candidates = [key for key in controller.watch.verdicts
                  if controller.watch.verdicts[key] == "stuck" and key in items
                  and controller.recaption_attempts.get(key, 0) < 2]
    if not candidates:
        return
    folder = Path(trainer.save_root) / "caption_repairs"
    folder.mkdir(parents=True, exist_ok=True)
    changed, captions, captioner = {}, {}, None
    previous_device = model.model.device
    adapters = [a for a in (model.assistant_lora, model.context_lora) if a is not None]
    try:
        model.print_and_status_update(f"Repairing {len(candidates)} Qwen captions (originals backed up)")
        model.model.to("cpu")
        model.text_encoder_to("cpu")
        for adapter in adapters:
            adapter.force_to("cpu")
        flush()
        captioner = captioner_factory(controller)
        for key in candidates:
            captions[key] = captioner.caption(key, controller.recaption_attempts.get(key, 0) + 1)
        captioner.close()
        captioner = None
        model.reload_text_encoder()
        for key, caption in captions.items():
            item = items[key]
            extension = "." + item.dataset_config.caption_ext.lstrip(".")
            path = Path(key).with_suffix(extension)
            original = path.read_text(encoding="utf-8") if path.is_file() else None
            identifier = hashlib.sha256(key.encode()).hexdigest()[:20]
            backup = folder / f"{identifier}.original.txt"
            if not backup.exists():
                atomic_text(backup, original or "")
            changed[path] = original
            atomic_text(path, caption + "\n")
        for dataset in controller.datasets:
            touched = False
            for item in dataset.file_list:
                if item.path in captions:
                    reset_caption_cache(item)
                    touched = True
            if touched and dataset.dataset_config.cache_text_embeddings:
                dataset.cache_text_embeddings()
        attempts = dict(controller.recaption_attempts)
        for key in captions:
            attempts[key] = attempts.get(key, 0) + 1
        atomic_text(folder / "ledger.json", json.dumps({
            "epoch": controller.epoch, "attempts": attempts, "captions": captions,
        }, ensure_ascii=False, indent=2))
        controller.recaption_attempts = attempts
        for key in captions:
            controller.watch.reset_key(key)
            if controller.recaption_attempts[key] >= 2:
                controller.watch.mark_incorrigible(key)
    except Exception as error:
        # Caption generation/cache refresh must not leave half-written data.
        for path, original in changed.items():
            if original is None:
                path.unlink(missing_ok=True)
            else:
                atomic_text(path, original)
        for dataset in controller.datasets:
            for item in dataset.file_list:
                if item.path in captions:
                    reset_caption_cache(item)
        model.print_and_status_update(f"Qwen caption repair skipped: {error}")
    finally:
        if captioner is not None:
            try:
                captioner.close()
            except Exception as error:
                model.print_and_status_update(f"Qwen captioner cleanup: {error}")
        try:
            if trainer.train_config.cache_text_embeddings:
                from toolkit.unloader import unload_text_encoder
                unload_text_encoder(model)
        finally:
            model.model.to(previous_device)
            for adapter in adapters:
                adapter.force_to(previous_device)
            flush()
