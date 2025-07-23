# Neural Fields

This repo provides differentiable neural rendering implementations:
- **NeRF** (train_volume.py)

Core logic lives under `core/`. Configure experiments via `configs/`.

## Quickstart

```bash
pip install -r requirements.txt
python train_volume.py --config-name nerf_lego
python train_volume.py --config-name nerf_fern
```
