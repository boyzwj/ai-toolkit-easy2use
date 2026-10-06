"""Opt-in real Qwen 2.1 training/preview/resume smoke test using synthetic data.

Requires a locally available pretrained model and a free CUDA device. The
Fizgig training adapter is downloaded by the usual toolkit model resolver.
Writes only to a new temporary output directory; it does not modify user jobs.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", default="identity", choices=("fast", "identity", "edit"))
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(tempfile.mkdtemp(prefix="qwen21-smoke-"))
    dataset = output / "data"
    dataset.mkdir()
    refs = output / "references"
    refs.mkdir()
    x, y = np.meshgrid(np.arange(args.resolution), np.arange(args.resolution))
    for i in range(2):
        rgb = np.stack(((x + 50 * i) % 256, y % 256, (x // 2 + y // 2) % 256), axis=-1).astype(np.uint8)
        Image.fromarray(rgb).save(dataset / f"{i}.png")
        Image.fromarray(rgb[:, ::-1].copy()).save(refs / f"{i}.png")
        (dataset / f"{i}.txt").write_text("a colorful abstract gradient, smoke_tok", encoding="utf-8")
    sample = {"prompt": "a colorful abstract gradient, smoke_tok"}
    ds = {"folder_path": str(dataset), "resolution": args.resolution, "caption_ext": "txt",
          "cache_latents_to_disk": True, "caption_dropout_rate": 0.0}
    if args.profile == "edit":
        ds["control_path"] = str(refs)
        sample["ctrl_img"] = str(refs / "0.png")
    config = {"job": "extension", "config": {"name": "qwen21_smoke", "process": [{
        "type": "sd_trainer", "device": args.device, "training_seed": 42,
        "training_folder": str(output / "output"), "trigger_word": "smoke_tok",
        "model": {"arch": "qwen_image_2", "name_or_path": args.model,
                  "quantize": args.quantize, "qtype": "convrot8", "quantize_te": True, "qtype_te": "convrot8"},
        "network": {"type": "lora"}, "datasets": [ds],
        "train": {"steps": 3, "batch_size": 1, "optimizer": "adamw",
                  "gradient_accumulation_steps": args.gradient_accumulation_steps,
                  "skip_first_sample": True, "cache_text_embeddings": True, "unload_text_encoder": True,
                  "qwen_image_21": {"profile": args.profile, "memory_plan": "manual", "compile": "off"}},
        "save": {"save_every": 100, "dtype": "float16"},
        "sample": {"sample_every": 100, "width": args.resolution, "height": args.resolution,
                   "sample_steps": 2, "guidance_scale": 1.0, "seed": 42, "samples": [sample]},
    }]}}
    config_path = output / "config.yaml"
    print(f"Qwen real-model smoke output: {output}", flush=True)
    checkpoint = output / "output/qwen21_smoke/qwen21_smoke.safetensors"
    for steps in (3, 4):
        config["config"]["process"][0]["train"]["steps"] = steps
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        subprocess.run([sys.executable, "run.py", str(config_path)], cwd=root, check=True,
                       env={**os.environ, "SEED": "42"})
        state = torch.load(str(checkpoint) + ".training-state", map_location="cpu", weights_only=True)
        assert state["next_step"] == steps, state["next_step"]
        assert state["optimizer"]["state"], "Optimizer did not take a step"
        assert all(torch.isfinite(p).all() for p in state["raw_weights"].values()), "Non-finite adapter weights"
        print(f"Validated checkpoint and resume state at step {steps}", flush=True)
    report = {"output": str(output), "profile": args.profile, "quantize": args.quantize,
              "steps": 4, "resumed_from": 3, "preview_files": [str(p) for p in checkpoint.parent.rglob("*.jpg")]}
    assert report["preview_files"], "No preview was generated"
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
