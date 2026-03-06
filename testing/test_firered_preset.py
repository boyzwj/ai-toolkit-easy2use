from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPTIONS_TS = ROOT / "ui" / "src" / "app" / "jobs" / "new" / "options.ts"
README_MD = ROOT / "README.md"
EXAMPLE_YAML = ROOT / "config" / "examples" / "train_lora_firered_image_edit_1_1_32gb.yaml"


def assert_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"Missing {label}: {needle}")


def main() -> None:
    options_text = OPTIONS_TS.read_text(encoding="utf-8")
    readme_text = README_MD.read_text(encoding="utf-8")

    assert_contains(options_text, "name: 'firered_image_edit_1_1'", "FireRed model option name")
    assert_contains(options_text, "label: 'FireRed-Image-Edit-1.1'", "FireRed model option label")
    assert_contains(
        options_text,
        "'config.process[0].model.name_or_path': ['FireRedTeam/FireRed-Image-Edit-1.1', defaultNameOrPath]",
        "FireRed model path default",
    )
    assert_contains(options_text, "'config.process[0].network.linear': [128, 32]", "FireRed LoRA rank default")
    assert_contains(
        options_text,
        "'config.process[0].network.linear_alpha': [128, 32]",
        "FireRed LoRA alpha default",
    )
    assert_contains(options_text, "'config.process[0].train.lr': [0.00002, 0.0001]", "FireRed LR default")
    assert_contains(options_text, "'config.process[0].sample.width': [512, 1024]", "FireRed width default")
    assert_contains(options_text, "'config.process[0].sample.height': [512, 1024]", "FireRed height default")

    if not EXAMPLE_YAML.exists():
        raise AssertionError(f"Missing example config: {EXAMPLE_YAML}")

    example_text = EXAMPLE_YAML.read_text(encoding="utf-8")
    assert_contains(example_text, 'name_or_path: "FireRedTeam/FireRed-Image-Edit-1.1"', "example model path")
    assert_contains(example_text, 'arch: "qwen_image_edit_plus"', "example arch")
    assert_contains(example_text, "linear: 128", "example LoRA rank")
    assert_contains(example_text, "linear_alpha: 128", "example LoRA alpha")
    assert_contains(example_text, "lr: 2e-5", "example LR")
    assert_contains(example_text, "width: 512", "example width")
    assert_contains(example_text, "height: 512", "example height")

    assert_contains(readme_text, "FireRed-Image-Edit-1.1", "README FireRed section")
    assert_contains(readme_text, "QwenImageEditPlusPipeline", "README compatibility note")


if __name__ == "__main__":
    main()
