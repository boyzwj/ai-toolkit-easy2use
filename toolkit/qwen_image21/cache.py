"""Content-addressed reference latents, independent of text embedding dropout."""

from collections import OrderedDict
import hashlib
import json
import os
import tempfile

import torch
from safetensors.torch import load_file, save_file


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReferenceLatentCache:
    def __init__(self, model, max_memory_entries=4):
        self.model = model
        self.memory = OrderedDict()
        self.max_memory_entries = max_memory_entries

    def key(self, image):
        signature = {
            "version": 1,
            "latent_space": self.model.get_latent_space_version(),
            "vae": self.model.model_config.vae_path or self.model.model_config.extras_name_or_path,
            "rgba": self.model.load_rgba,
            "shape": list(image.shape),
        }
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode())
        # Hash final pixels, not just the pathname: replacements, augmentations,
        # alpha and target/reference resizing must never reuse a stale latent.
        digest.update(image.detach().float().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def get(self, image, dataset_image_path=None):
        model = self.model
        enabled = (model.model_config.model_kwargs.get("cache_reference_latents", False)
                   and model.model_config.model_kwargs.get("latent_sampling", "sample") == "mode")
        def encode():
            return model.encode_images([image[0].to(model.device_torch) * 2 - 1],
                                       device=model.device_torch, dtype=model.torch_dtype)
        if not enabled:
            return encode()
        key = self.key(image)
        cache_path = None
        if dataset_image_path:
            folder = os.path.join(os.path.dirname(dataset_image_path), "_qwen21_reference_cache")
            cache_path = os.path.join(folder, f"{key}.safetensors")
        if key in self.memory:
            latent = self.memory.pop(key)
        elif cache_path and os.path.isfile(cache_path):
            latent = load_file(cache_path)["latent"]
        else:
            latent = encode().detach().cpu().contiguous()
            if cache_path:
                os.makedirs(folder, exist_ok=True)
                fd, temporary = tempfile.mkstemp(dir=folder, suffix=".safetensors.tmp")
                os.close(fd)
                try:
                    save_file({"latent": latent}, temporary)
                    os.replace(temporary, cache_path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        self.memory[key] = latent
        while len(self.memory) > self.max_memory_entries:
            self.memory.popitem(last=False)
        return latent.to(model.device_torch, model.torch_dtype)


def warm_reference_cache(model, datasets):
    """Cache edit references before optimizer steps so the VAE can stay on CPU."""
    if not model.model_config.model_kwargs.get("cache_reference_latents", False):
        return
    from tqdm import tqdm
    for dataset in datasets:
        for item in tqdm(dataset.file_list, desc="Caching Qwen 2.1 reference latents"):
            if not item.has_control_image or item.dataset_config.control_from_same_folder:
                continue
            item.load_control_image()
            try:
                control = item.control_tensor_list or item.control_tensor
                samples = model._prepare_control_images(
                    model._normalize_control_images(control, 1),
                    target_pixels=model._target_pixels((item.crop_width, item.crop_height)),
                )
                model.encode_condition_images(samples, cache_paths=[item.path])
            finally:
                item.cleanup_control()
    model.vae.to("cpu")
