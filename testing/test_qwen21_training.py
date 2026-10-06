"""CPU/CUDA numerical regression checks; no pretrained weights required.

Run: python -m unittest testing.test_qwen21_training
Set QWEN21_TEST_DEVICE=cuda to exercise native INT8 training on a GPU.
"""

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import weakref
import gc
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import load_file, save_file

from toolkit.config_modules import ModelConfig, NetworkConfig
from toolkit.ema import ExponentialMovingAverage
from toolkit.inference_lora import InferenceLoRA
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.qwen_image21.adaptive_lr import AdaptiveLR
from toolkit.qwen_image21.cache import ReferenceLatentCache
from toolkit.qwen_image21.config import configure_qwen21_training, IDENTITY_BLOCKS, plan_memory
from toolkit.qwen_image21.lora import load_frozen_lora, split_fused_lora
from toolkit.qwen_image21.training import Qwen21TrainingController
from toolkit.qwen_image21.evaluation import identity_scores
from toolkit.qwen_image21.recaption import repair_captions


# Import this model in isolation from the extension registry: unrelated audio
# and video backends are not needed for these numerical checks.
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "extensions_built_in/diffusion_models/qwen_image_2"
spec = importlib.util.spec_from_file_location("_qwen21_test_model", MODEL_DIR / "qwen_image_2.py",
                                             submodule_search_locations=[str(MODEL_DIR)])
qwen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qwen
spec.loader.exec_module(qwen)
transformer_module = sys.modules["_qwen21_test_model.src.transformer"]
pipeline_module = sys.modules["_qwen21_test_model.src.pipeline"]
DEVICE = os.environ.get("QWEN21_TEST_DEVICE", "cpu")


def tiny_model(*, blocks=None, quantize=False):
    torch.manual_seed(123)
    model = qwen.QwenImage2Model(DEVICE, ModelConfig(arch="qwen_image_2", name_or_path="test",
        model_kwargs={"split_mlp_lora": True, "train_blocks": blocks}), dtype="fp32")
    model.model = transformer_module.QwenImage21Transformer2DModel(
        in_channels=4, out_channels=4, num_layers=16, num_attention_heads=2,
        attention_head_dim=8, context_in_dim=12, axes_dims_rope=(2, 2, 4), mlp_ratio=3,
    ).to(DEVICE)
    if quantize:
        from toolkit.util.quantize import quantize as apply_quantize, get_qtype
        apply_quantize(model.model, weights=get_qtype("convrot8"),
                       exclude=model.model.get_quantization_exclude_modules())
    model.model.requires_grad_(False)
    for block in model.model.transformer_blocks:
        block.img_mlp.enable_split_lora()
    return model


def network_for(model, rank=2):
    network = LoRASpecialNetwork([], model.model, lora_dim=rank, alpha=rank,
        train_text_encoder=False, train_unet=True, is_transformer=True,
        target_lin_modules=model.target_lora_modules, transformer_only=True,
        base_model=model, network_config=NetworkConfig(type="lora", linear=rank))
    network.apply_to([], model.model, False, True)
    network.force_to(DEVICE, dtype=torch.float32)
    network._update_torch_multiplier()
    model.network = network
    return network


def forward(model, *, cache=None, mode=None, t=0.4):
    torch.manual_seed(456)
    args = {} if cache is None else {"kv_cache": cache, "kv_cache_mode": mode}
    return pipeline_module.run_transformer(model.model,
        torch.randn(1, 4, 2, 2, device=DEVICE), torch.tensor([t], device=DEVICE),
        torch.randn(1, 3, 12, device=DEVICE), torch.ones(1, 3, device=DEVICE, dtype=torch.bool),
        torch.zeros(1, 3, device=DEVICE, dtype=torch.bool), **args)


class ConfigTests(unittest.TestCase):
    def test_defaults_and_explicit_overrides(self):
        config = {"model": {"arch": "qwen_image_2"}, "network": {}, "datasets": [{}], "train": {"qwen_image_21": {}}}
        configure_qwen21_training(config)
        self.assertEqual(config["network"]["linear"], 8)
        self.assertEqual(config["train"]["timestep_type"], "shifted_logit_normal")
        self.assertEqual(config["train"]["qwen_image_21"]["epochs"], 30)
        self.assertEqual(config["model"]["model_kwargs"]["latent_sampling"], "mode")
        self.assertAlmostEqual(config["train"]["lr"], math.sqrt(2e-4 * 4e-4))
        config = {"model": {"arch": "qwen_image_2"}, "network": {"linear": 24},
                  "train": {"steps": 77, "cache_text_embeddings": False, "timestep_type": "shift",
                            "qwen_image_21": {"profile": "style"}}, "datasets": [{"resolution": 768}]}
        configure_qwen21_training(config)
        self.assertEqual(config["network"]["linear"], 24)
        self.assertFalse(config["train"]["cache_text_embeddings"])
        self.assertFalse(config["train"]["unload_text_encoder"])
        self.assertEqual(config["train"]["steps"], 77)
        self.assertNotIn("epochs", config["train"]["qwen_image_21"])
        self.assertEqual(config["train"]["timestep_type"], "shift")

    def test_legacy_and_other_models_are_unchanged(self):
        for config in ({"model": {"arch": "flux"}, "train": {}, "network": {}},
                       {"model": {"arch": "qwen_image_2"}, "train": {}, "network": {}},
                       {"model": {"arch": "qwen_image_2"}, "train": {"qwen_image_21": {"profile": "legacy"}}}):
            before = copy.deepcopy(config)
            configure_qwen21_training(config)
            self.assertEqual(config, before)

    def test_reference_datasets_keep_vision(self):
        for dataset in ({"control_path": "refs"}, {"control_from_same_folder": True}):
            config = {"model": {"arch": "qwen_image_2"}, "network": {}, "datasets": [dataset], "train": {"qwen_image_21": {}}}
            configure_qwen21_training(config)
            self.assertFalse(config["model"]["model_kwargs"]["text_only"])

    def test_invalid_ranges_and_scheduler_fail_early(self):
        for options in ({"min_timestep": 1, "max_timestep": 0}, {"adaptive_lr_min": 1, "adaptive_lr_max": .1}):
            config = {"model": {"arch": "qwen_image_2"}, "network": {}, "train": {"qwen_image_21": options}}
            with self.assertRaises(ValueError):
                configure_qwen21_training(config)

    def test_memory_plans(self):
        self.assertFalse(plan_memory(32)["quantize"])
        self.assertEqual(plan_memory(16)["offload_fraction"], 0)
        self.assertGreater(plan_memory(9)["offload_fraction"], 0)
        self.assertGreater(plan_memory(12, megapixels=1, batch_size=2)["offload_fraction"], 0)


class SamplerTests(unittest.TestCase):
    def test_matches_fizgig_formula_and_range(self):
        scheduler = qwen.QwenImage2Model.get_train_scheduler()
        for h, w in ((32, 32), (44, 44), (64, 64)):
            latents = torch.empty(1, 64, h, w)
            z = torch.randn(4096, generator=torch.Generator().manual_seed(42))
            mu = .5 + (h * w - 256) * (.9 - .5) / (8192 - 256)
            t = torch.sigmoid(z)
            shifted = math.exp(mu) / (math.exp(mu) + 1 / t - 1)
            expected = (.2 + .6 * shifted) * 1000
            actual = scheduler.sample_training_timesteps(4096, latents=latents, min_t=.2, max_t=.8,
                                                         generator=torch.Generator().manual_seed(42))
            torch.testing.assert_close(actual, expected, atol=.0002, rtol=1e-6)
            self.assertTrue(bool(((actual > 200) & (actual < 800)).all()))

    def test_no_terminal_stretch_and_resolution_shift(self):
        scheduler = qwen.QwenImage2Model.get_train_scheduler()
        def draw(n):
            return scheduler.sample_training_timesteps(10000, latents=torch.empty(1, 64, n, n),
                                                       generator=torch.Generator().manual_seed(10))
        low, high = draw(32), draw(64)
        self.assertGreater(float(high.mean()), float(low.mean()))
        scheduler.register_to_config(shift_terminal=.7)
        torch.testing.assert_close(low, draw(32), atol=0, rtol=0)


class AdapterTests(unittest.TestCase):
    def test_reference_dropout_in_multi_image_batch(self):
        model = tiny_model()
        ref = torch.randn(1, 4, 2, 2, device=DEVICE)
        encoded = []
        def encode(samples, cache_paths=None):
            encoded.extend(cache_paths)
            return pipeline_module.pack_latents(ref), [(2, 2)]
        model.encode_condition_images = encode
        embeds = qwen.AdvancedPromptEmbeds(
            text_embeds=[torch.randn(3, 12, device=DEVICE) for _ in range(2)],
            attention_mask=[torch.ones(3, device=DEVICE, dtype=torch.bool) for _ in range(2)],
            image_slot_mask=[torch.tensor([False, True, False], device=DEVICE),
                             torch.zeros(3, device=DEVICE, dtype=torch.bool)],
        )
        latents = torch.randn(2, 4, 2, 2, device=DEVICE)
        timesteps = torch.tensor([250., 750.], device=DEVICE)
        batch = SimpleNamespace(control_tensor_list=[[torch.ones(1, 3, 32, 32)], [torch.ones(1, 3, 32, 32)]],
                                file_items=[SimpleNamespace(path="a"), SimpleNamespace(path="b")])
        actual = model.get_noise_prediction(latents, timesteps, embeds, batch=batch)
        self.assertEqual(encoded, ["a"])
        text, mask, slots = model.pad_prompt_embeds(embeds)
        expected_plain = pipeline_module.run_transformer(model.model, latents[1:], timesteps[1:] / 1000,
                                                         text[1:], mask[1:], slots[1:])
        torch.testing.assert_close(actual[1:], expected_plain)
        self.assertEqual(actual.shape, latents.shape)

    def test_independent_parameters_and_identity_blocks(self):
        model = tiny_model(blocks=IDENTITY_BLOCKS)
        base = {k: v.clone() for k, v in model.model.state_dict().items()}
        network = network_for(model)
        self.assertEqual(len(network.unet_loras), 5 * 7)
        names = [m.lora_name.replace("$$", ".") for m in network.unet_loras]
        self.assertFalse(any("gate_up" in name for name in names))
        self.assertTrue(any("gate_layer" in name for name in names))
        optimizer = torch.optim.AdamW(network.parameters(), lr=.01)
        with network:
            loss = (forward(model) - 1).square().mean()
            loss.backward()
            optimizer.step()
        for key, value in model.model.state_dict().items():
            torch.testing.assert_close(value, base[key], atol=0, rtol=0)
        self.assertTrue(any(m.lora_up.weight.detach().abs().sum() > 0 for m in network.unet_loras))
        exported = network.get_state_dict(dtype=torch.float32)
        self.assertTrue(all(k.startswith("transformer.") for k in exported))
        self.assertFalse(any("gate_up" in k for k in exported))

    def test_gradient_checkpointing_reaches_independent_adapters(self):
        model = tiny_model(blocks=[10])
        model.model.enable_gradient_checkpointing()
        network = network_for(model)
        with network:
            forward(model).square().mean().backward()
        for layer in network.unet_loras:
            self.assertIsNotNone(layer.lora_up.weight.grad)
        self.assertGreater(sum(float(layer.lora_up.weight.grad.abs().sum()) for layer in network.unet_loras), 0)

    def test_old_fused_lora_conversion_preserves_delta(self):
        A, B = torch.randn(3, 8), torch.randn(24, 3)
        converted = split_fused_lora({"transformer.transformer_blocks.0.img_mlp.gate_up.lora_A.weight": A,
                                     "transformer.transformer_blocks.0.img_mlp.gate_up.lora_B.weight": B})
        prefix = "transformer.transformer_blocks.0.img_mlp."
        gate = converted[prefix + "gate_layer.lora_B.weight"] @ converted[prefix + "gate_layer.lora_A.weight"]
        up = converted[prefix + "proj.lora_B.weight"] @ converted[prefix + "proj.lora_A.weight"]
        torch.testing.assert_close(torch.cat([gate, up]), B @ A)

    def test_frozen_helper_switching_and_export(self):
        model = tiny_model(blocks=[10])
        network = network_for(model)
        prefix = "transformer.transformer_blocks.0.img_mlp.gate_layer."
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "helper.safetensors")
            save_file({prefix + "lora_A.weight": torch.randn(2, 16), prefix + "lora_B.weight": torch.randn(48, 2)}, path)
            before = forward(model).detach()
            helper = load_frozen_lora(model, path, .1)
            after = forward(model).detach()
            self.assertGreater(float((before - after).abs().max()), 0)
            helper.is_active = False
            torch.testing.assert_close(forward(model), before)
            helper.is_active = True
            torch.testing.assert_close(forward(model), after)
            exported = network.get_state_dict(dtype=torch.float32)
            self.assertFalse(any("transformer_blocks.0." in key for key in exported))
            self.assertFalse(any(t.requires_grad for entry in helper.adapter.entries for t in (entry.A, entry.B)))
            helper.is_active = False

    def test_frozen_shape_mismatch_is_rejected(self):
        model = tiny_model()
        prefix = "transformer.transformer_blocks.0.img_mlp.gate_layer."
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "wrong.safetensors")
            save_file({prefix + "lora_A.weight": torch.randn(2, 999), prefix + "lora_B.weight": torch.randn(48, 2)}, path)
            with self.assertRaises(ValueError):
                load_frozen_lora(model, path)

    def test_export_load_roundtrip(self):
        first = tiny_model(blocks=[10])
        network = network_for(first)
        with torch.no_grad():
            for layer in network.unet_loras:
                layer.lora_up.weight.normal_(std=.05)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "user.safetensors")
            network.save_weights(path, dtype=torch.float32)
            second = tiny_model(blocks=[10])
            loaded = network_for(second)
            loaded.load_weights(path)
            with network, loaded:
                torch.testing.assert_close(forward(first), forward(second))

    def test_cached_prefix_matches_full_forward(self):
        model = tiny_model()
        cache = transformer_module.QwenImage21KVCache(len(model.model.transformer_blocks))
        with torch.no_grad():
            forward(model, cache=cache, mode="extract", t=.8)
            cached = forward(model, cache=cache, mode="cached", t=.3)
            torch.testing.assert_close(cached, forward(model, t=.3), atol=1e-5, rtol=1e-5)

    def test_preview_failure_restores_helper_and_network(self):
        from unittest.mock import patch
        from toolkit.models.base_model import BaseModel
        model = tiny_model(blocks=[10])
        network = network_for(model)
        helper = SimpleNamespace(is_active=True)
        preview = SimpleNamespace(is_active=False)
        model.assistant_lora, model.preview_lora = helper, preview
        network.is_active = True
        def fail(*args, **kwargs):
            helper.is_active = False
            network.is_active = False
            model.model.eval()
            raise torch.OutOfMemoryError("preview")
        with patch.object(BaseModel, "generate_images", fail):
            with self.assertRaises(torch.OutOfMemoryError):
                model.generate_images([])
        self.assertTrue(helper.is_active)
        self.assertFalse(preview.is_active)
        self.assertTrue(network.is_active)
        self.assertTrue(model.model.training)

    def test_training_noise_and_target_keep_fp32(self):
        model = tiny_model()
        model.qwen_training_options = {"profile": "fast"}
        latents = torch.zeros(1, 4, 2, 2, device=DEVICE, dtype=torch.bfloat16)
        noise = model.get_latent_noise_from_latents(latents)
        target = model.get_loss_target(noise=noise, batch=SimpleNamespace(latents=latents))
        self.assertEqual(noise.dtype, torch.float32)
        self.assertEqual(target.dtype, torch.float32)

    @unittest.skipUnless(DEVICE.startswith("cuda"), "native ConvRot8 training check requires CUDA")
    def test_quantized_base_is_unchanged(self):
        model = tiny_model(blocks=[10], quantize=True)
        base = {k: v.clone() for k, v in model.model.state_dict().items()}
        network = network_for(model)
        with network:
            forward(model).square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in network.parameters()))
        for key, value in model.model.state_dict().items():
            torch.testing.assert_close(value, base[key], atol=0, rtol=0)


class CacheTests(unittest.TestCase):
    def test_control_replacement_invalidates_text_cache(self):
        from toolkit.dataloader_mixins import TextEmbeddingFileItemDTOMixin
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ref.png"
            path.write_bytes(b"first")
            item = TextEmbeddingFileItemDTOMixin()
            item.caption = "person"
            item.text_embedding_space_version = "qwen_image_2_ref704_vision-no-resize-v2"
            item.encode_control_in_text_embeddings = True
            item.control_path = [str(path)]
            item.load_rgba = False
            first = item.get_text_embedding_info_dict()
            path.write_bytes(b"other")
            second = item.get_text_embedding_info_dict()
            self.assertNotEqual(first, second)
            item.load_rgba = True
            self.assertNotEqual(second, item.get_text_embedding_info_dict())
            self.assertNotIn("control_contents", item.get_text_embedding_info_dict(text_only=True))

    def test_reference_cache_tracks_pixels_alpha_and_sampling(self):
        class FakeModel:
            device_torch = torch.device("cpu")
            torch_dtype = torch.float32
            load_rgba = True
            model_config = SimpleNamespace(vae_path="test-vae", extras_name_or_path="test", model_kwargs={
                "latent_sampling": "mode", "cache_reference_latents": True})
            calls = 0
            def get_latent_space_version(self):
                return self.model_config.model_kwargs["latent_sampling"]
            def encode_images(self, images, **kwargs):
                self.calls += 1
                return images[0].unsqueeze(0)
        model = FakeModel()
        with tempfile.TemporaryDirectory() as folder:
            image = torch.rand(1, 4, 32, 32)
            path = str(Path(folder) / "target.png")
            first = ReferenceLatentCache(model).get(image, path)
            second = ReferenceLatentCache(model).get(image, path)
            torch.testing.assert_close(first, second)
            self.assertEqual(model.calls, 1)
            edited = image.clone()
            edited[:, 3] = 0
            ReferenceLatentCache(model).get(edited, path)
            self.assertEqual(model.calls, 2)
            model.model_config.model_kwargs["latent_sampling"] = "sample"
            cache = ReferenceLatentCache(model)
            cache.get(image, path)
            cache.get(image, path)
            self.assertEqual(model.calls, 4)

    def test_text_encoder_wrapper_releases_reference(self):
        from toolkit.unloader import unload_text_encoder
        model = tiny_model()
        encoder = torch.nn.Linear(2, 2)
        reference = weakref.ref(encoder)
        model.text_encoder = [encoder]
        model.pipeline = SimpleNamespace()
        model.prompt_encoder = SimpleNamespace(text_encoder=encoder)
        del encoder
        unload_text_encoder(model)
        gc.collect()
        self.assertIsNone(model.prompt_encoder)
        self.assertIsNone(reference())


class PromptTests(unittest.TestCase):
    def test_backbone_pre_norm_without_logits_or_reference_resize(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = SimpleNamespace(norm=torch.nn.LayerNorm(3))
            def forward(self, input_ids, **kwargs):
                raw = input_ids.float().unsqueeze(-1) * torch.tensor([1., 2., 3.])
                normalized = self.language_model.norm(raw)
                return SimpleNamespace(hidden_states=(raw, normalized))
        class TextEncoder(torch.nn.Module):
            device = torch.device("cpu")
            def __init__(self):
                super().__init__()
                self.model = Backbone()
            def forward(self, **kwargs):
                raise AssertionError("LM head must not run")
        class Inputs(SimpleNamespace):
            def to(self, device):
                return self
        class Processor:
            tokenizer = SimpleNamespace(encode=lambda text: [9])
            def apply_chat_template(self, *args, **kwargs):
                return [[1]]
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return Inputs(input_ids=torch.tensor([[0, 1, 3, 4], [1, 3, 4, 5]]),
                              attention_mask=torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]))
        encoder, processor = TextEncoder(), Processor()
        prompt = pipeline_module.QwenImage21PromptEncoder(encoder, processor)
        embeds, _, _ = prompt.encode(["portrait", "outdoors"], images=[[object()], [object()]])
        torch.testing.assert_close(embeds[0], torch.tensor([[3., 6., 9.], [4., 8., 12.]]))
        self.assertFalse(processor.kwargs["images_kwargs"]["do_resize"])
        self.assertEqual(len(encoder.model.language_model.norm._forward_hooks), 0)


class RepairAndEvaluationTests(unittest.TestCase):
    def test_identity_scores_keep_missing_faces(self):
        report = identity_scores([[1, 0]], [("same", [2, 0]), ("different", [0, 1]), ("no-face", None)])
        self.assertEqual(report["samples_total"], 3)
        self.assertEqual(report["faces_detected"], 2)
        self.assertAlmostEqual(report["mean_cosine"], .5)
        self.assertIsNone(report["samples"][2]["cosine"])

    def test_caption_repair_backup_and_cache_refresh(self):
        self.run_repair(fail_cache=False)

    def test_caption_repair_failure_restores_original(self):
        self.run_repair(fail_cache=True)

    def run_repair(self, fail_cache):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "person.png"
            caption_path = path.with_suffix(".txt")
            caption_path.write_text("original caption")
            item = SimpleNamespace(path=str(path), is_reg=False, caption="original caption",
                _text_embedding_path="stale", dataset_config=SimpleNamespace(caption_ext="txt"))
            def load_caption():
                item.caption = caption_path.read_text()
            item.load_caption = load_caption
            calls = []
            def refresh():
                calls.append(item.caption)
                if fail_cache:
                    raise RuntimeError("cache failed")
            dataset = SimpleNamespace(file_list=[item], dataset_config=SimpleNamespace(cache_text_embeddings=True),
                                      cache_text_embeddings=refresh)
            class DiT(torch.nn.Linear):
                @property
                def device(self):
                    return self.weight.device
            model = SimpleNamespace(model=DiT(2, 2), assistant_lora=None, context_lora=None,
                text_encoder_to=lambda *args: None, reload_text_encoder=lambda: None,
                print_and_status_update=lambda *args: None)
            attempts, resets = {}, []
            controller = SimpleNamespace(options={"auto_recaption": True},
                trainer=SimpleNamespace(sd=model, save_root=folder, train_config=SimpleNamespace(cache_text_embeddings=False)),
                datasets=[dataset], recaption_attempts=attempts, epoch=4,
                watch=SimpleNamespace(verdicts={str(path): "stuck"}, reset_key=lambda key: resets.append(key), mark_incorrigible=lambda key: None))
            class Captioner:
                def __init__(self, controller):
                    pass
                def caption(self, path, attempt):
                    return "repaired caption"
                def close(self):
                    pass
            repair_captions(controller, captioner_factory=Captioner)
            self.assertEqual(calls, ["repaired caption\n"])
            self.assertEqual(next((Path(folder) / "caption_repairs").glob("*.original.txt")).read_text(), "original caption")
            if fail_cache:
                self.assertEqual(caption_path.read_text(), "original caption")
                self.assertEqual(resets, [])
                self.assertEqual(controller.recaption_attempts, {})
            else:
                self.assertEqual(caption_path.read_text(), "repaired caption\n")
                self.assertEqual(controller.recaption_attempts[str(path)], 1)
                self.assertEqual(resets, [str(path)])
            self.assertIsNone(item._text_embedding_path)


class TrainingStateTests(unittest.TestCase):
    def test_partial_accumulation_restores_gradients_and_counter(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.make_trainer(folder)
            trainer.grad_accumulation_step = 1
            controller = Qwen21TrainingController(trainer)
            trainer.network(torch.ones(1, 2)).sum().backward()
            controller.accumulation_pending = True
            gradients = {n: p.grad.clone() for n, p in trainer.network.named_parameters()}
            checkpoint = str(Path(folder) / "test.safetensors")
            save_file({"test": torch.ones(1)}, checkpoint)
            controller.save_checkpoint(checkpoint, next_step=4)
            resumed = self.make_trainer(folder)
            loaded = Qwen21TrainingController(resumed)
            loaded.restore_checkpoint()
            self.assertTrue(loaded.accumulation_pending)
            self.assertEqual(resumed.grad_accumulation_step, 2)
            for name, parameter in resumed.network.named_parameters():
                torch.testing.assert_close(parameter.grad, gradients[name])
            controller.watch.close()
            loaded.watch.close()

    def test_adaptive_lr_probe_plateau_and_stability_rollback(self):
        network = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            network.weight.fill_(1)
        optimizer = torch.optim.AdamW(network.parameters(), lr=2e-4)
        adaptive = AdaptiveLR(1e-4, 4e-4)
        adaptive.epoch_boundary(0, 3., network, optimizer)
        adaptive.epoch_boundary(1, 2., network, optimizer)
        adaptive.epoch_boundary(2, 1., network, optimizer)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2.5e-4)
        with torch.no_grad():
            network.weight.fill_(2)
        adaptive.epoch_boundary(3, 1.1, network, optimizer)
        self.assertTrue(adaptive.last_rollback)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.25e-4)
        torch.testing.assert_close(network.weight, torch.full_like(network.weight, 1.3))

    def make_trainer(self, folder):
        network = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(network.parameters(), lr=2e-4)
        return SimpleNamespace(network=network, optimizer=optimizer,
            ema=ExponentialMovingAverage(network.parameters(), decay=.98),
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1),
            train_config=SimpleNamespace(timestep_type="shifted_logit_normal", qwen_image_21={
                "adaptive_lr": True, "adaptive_lr_min": 1e-4, "adaptive_lr_max": 4e-4,
                "per_image_lr": True, "loss_watch": True}),
            model_config=ModelConfig(arch="qwen_image_2", name_or_path="test"),
            save_root=folder, epoch_num=0, step_num=3, start_step=0,
            sd=SimpleNamespace(print_and_status_update=lambda text: None),
            accelerator=SimpleNamespace(is_main_process=True),
            get_latest_save_path=lambda **kwargs: str(Path(folder) / "test.safetensors"))

    def test_true_per_image_loss_and_raw_ema_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.make_trainer(folder)
            controller = Qwen21TrainingController(trainer)
            controller.watch._mult["a"] = .5
            batch = SimpleNamespace(file_items=[SimpleNamespace(path="a", is_reg=False),
                                               SimpleNamespace(path="b", is_reg=False)])
            loss = torch.tensor([1., 3.], requires_grad=True)
            scaled = controller.observe_loss(loss, batch, torch.tensor([250., 750.]))
            torch.testing.assert_close(scaled, torch.tensor([.5, 3.]))
            with open(Path(folder) / "loss_log/per_image_loss.jsonl") as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual([r["loss"] for r in records], [1, 3])
            self.assertEqual([r["batch"] for r in records], [1, 1])
            trainer.network(torch.ones(1, 2)).sum().backward()
            trainer.optimizer.step()
            trainer.ema.update()
            raw = {n: p.clone() for n, p in trainer.network.named_parameters()}
            checkpoint = str(Path(folder) / "test.safetensors")
            save_file({"exported_ema": torch.ones(1)}, checkpoint)
            controller.finish_epoch()
            controller.save_checkpoint(checkpoint, next_step=4)
            saved_ema = [p.clone() for p in trainer.ema.shadow_params]
            controller.watch.close()
            resumed = self.make_trainer(folder)
            loaded = Qwen21TrainingController(resumed)
            loaded.restore_checkpoint()
            self.assertEqual(resumed.start_step, 4)
            for name, parameter in resumed.network.named_parameters():
                torch.testing.assert_close(parameter, raw[name])
            for shadow, expected in zip(resumed.ema.shadow_params, saved_ema):
                torch.testing.assert_close(shadow, expected)
            self.assertEqual(loaded.loss_count, 2)
            self.assertAlmostEqual(loaded.adaptive.best_loss, 2)
            self.assertEqual(loaded.watch._mult["a"], controller.watch._mult["a"])
            loaded.watch.close()

    def test_resume_rejects_mismatched_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.make_trainer(folder)
            controller = Qwen21TrainingController(trainer)
            path = str(Path(folder) / "test.safetensors")
            save_file({"test": torch.ones(1)}, path)
            controller.save_checkpoint(path, next_step=4)
            save_file({"test": torch.zeros(1)}, path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                controller.restore_checkpoint()
            controller.watch.close()


if __name__ == "__main__":
    unittest.main()
