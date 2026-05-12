# AGENTS.md — ai-toolkit-easy2use

Chinese-localized fork of [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) for diffusion model training/inference (FLUX, SD, Wan, audio, etc.). Version in `version.py`.

## Entrypoints

- **CLI training**: `python run.py <config-path>` (looks up relative paths under `config/`, or absolute). Supports `-r` (recover on fail), `-n` (name replacement), `-l` (log file).
- **Gradio UI** (legacy): `python flux_train_ui.py` — FLUX LoRA trainer.
- **Modal cloud**: `modal run run_modal.py --config-file-list-str=<path>`.
- **Next.js Web UI**: see commands below.

## UI commands (in `ui/`)

```bash
npm run dev          # dev mode (port 3000)
npm run build_and_start  # prod build + start (port 8675)
npm run update_db    # prisma generate + db push
npm run start        # start prod server (port 8675, after build)
npm run lint         # next lint
```

After `git pull` from an older version, **always** run in order:
```bash
cd ui
npm install
npm run update_db   # syncs SQLite schema (aitk_db.db)
npm run build
```

Production defaults to port **8675**, not 3000.

## Auth

Set `AI_TOOLKIT_AUTH=your_token` env var to protect API routes. Pass via Bearer token header. Without it, all routes are open.

## Job system

Config files use `job: extension` with `config.process[].type` (e.g. `sd_trainer`). Five job types: `train`, `generate`, `extract`, `mod`, `extension`. Dispatched in `toolkit/job.py:get_job()`. Training defaults in `config/examples/train_lora_flux_24gb.yaml`.

## Python setup

- Python >= 3.10, recommended 3.12. Node.js >= 20.
- `pip install -r requirements.txt` (includes torch 2.9.1 + cu118 via index-url, but you should install torch first per your CUDA version).
- `run.py` auto-sets `HF_HUB_ENABLE_HF_TRANSFER=1`, `NO_ALBUMENTATIONS_UPDATE=1`, `DISABLE_TELEMETRY=YES`.
- `SEED` env var for reproducibility; `DEBUG_TOOLKIT=1` enables `torch.autograd.set_detect_anomaly(True)`.

## Docker

```bash
# build
docker build --pull -t ai-toolkit-easy2use:<version> -f docker/Dockerfile .
# or use the script (reads version from version.py, pushes to Docker Hub)
./build_and_push_docker
```

Prebuilt image: `coco1006/ai-toolkit-easy2use:latest`. Container runs UI on port 8675 via `docker/start.sh`. See `docker-compose.yml` for volume mounts.

## Database

SQLite via Prisma ORM (`ui/prisma/schema.prisma`). DB file at `aitk_db.db` in project root. After schema changes, always run `npm run update_db`.

## Important paths

| Path | Purpose |
|------|---------|
| `config/examples/` | Training config YAML samples |
| `toolkit/` | Core Python training/inference engine |
| `jobs/` | Job type implementations |
| `extensions_built_in/` | Built-in trainer extensions |
| `ui/src/` | Next.js app + API routes |
| `output/` | Default training output |
| `datasets/` | Default dataset folder |
| `scripts/` | Utility scripts (model conversion, captioning, etc.) |

## Testing

No test framework. Ad-hoc test scripts live in `testing/`. No lint/typecheck CI. No pre-commit hooks. Only CI is a stale-issue closer.

## Style & conventions

- Configs are YAML, go under `config/examples/`.
- Job configs use `job: extension` + `config.process[].type:` as the top-level pattern.
- Training config keys follow ai-toolkit conventions (not FireRed native format).
- Caption files: `.txt` with same base name as image.
- Trigger word: `[trigger]` placeholder in captions replaced by `trigger_word` config value.
- No existing lint/format rules beyond Next.js defaults (prettier for UI).
