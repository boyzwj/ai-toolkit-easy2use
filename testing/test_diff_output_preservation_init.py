from types import SimpleNamespace
from unittest.mock import patch

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from jobs.process import BaseSDTrainProcess


class FakeEmbeds:
    def __init__(self, value: str):
        self.value = value

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self

    def clone(self):
        return self


class FakeModule:
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


class FakeSD:
    def __init__(self):
        self.device_torch = "cpu"
        self.torch_dtype = torch.float32
        self.encode_control_in_text_embeddings = False
        self.noise_scheduler = object()
        self.unet = FakeModule()
        self.vae = FakeModule()
        self.prompts = []

    def encode_prompt(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return FakeEmbeds(prompt)

    def text_encoder_to(self, device):
        return None


def build_trainer():
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.sd = FakeSD()
    trainer.device_torch = "cpu"
    trainer.is_caching_text_embeddings = False
    trainer.is_latents_cached = True
    trainer.adapter = None
    trainer.adapter_config = None
    trainer.datasets = None
    trainer.datasets_reg = None
    trainer.data_loader = None
    trainer.sample_config = None
    trainer.save_root = "."
    trainer.cached_blank_embeds = None
    trainer.cached_trigger_embeds = None
    trainer.diff_output_preservation_embeds = None
    trainer.unconditional_embeds = None
    trainer.negative_prompt_pool = None
    trainer.trigger_word = "style_token"
    trainer.do_long_prompts = False
    trainer.do_prior_prediction = True
    trainer.dfe = None
    trainer.timer = lambda *args, **kwargs: SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: False)
    trainer.train_config = SimpleNamespace(
        unconditional_prompt="",
        do_prior_divergence=False,
        unload_text_encoder=False,
        blank_prompt_preservation=False,
        diff_output_preservation=True,
        diff_output_preservation_class="person",
        diffusion_feature_extractor_path=None,
        negative_prompt=None,
        train_text_encoder=False,
        disable_sampling=True,
        gradient_checkpointing=False,
    )
    return trainer


def main():
    trainer = build_trainer()
    with patch.object(BaseSDTrainProcess, "hook_before_train_loop", lambda self: None), patch(
        "extensions_built_in.sd_trainer.SDTrainer.add_all_snr_to_noise_scheduler", lambda *args, **kwargs: None
    ):
        trainer.hook_before_train_loop()

    if trainer.diff_output_preservation_embeds is None:
        raise AssertionError("diff_output_preservation_embeds should be initialized without unloading the text encoder")

    if trainer.diff_output_preservation_embeds.value != "person":
        raise AssertionError("diff_output_preservation_embeds should use diff_output_preservation_class")


if __name__ == "__main__":
    main()
