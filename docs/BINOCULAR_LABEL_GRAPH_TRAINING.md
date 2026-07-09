# Binocular Label-Graph Training

This is the new research training path. It does not replace the deployed
competition system by default.

## Experiments

The training script supports three presets:

```text
effnet_b4_linear
convnext_tiny_linear
dual_branch_label_gcn_asl
```

The first two are single-model multi-label baselines using merged fundus images.
The third model uses left/right fundus branches, gated feature fusion, a
label-correlation graph head, and ASL for long-tailed multi-label learning.

All models output eight sigmoid logits in this fixed order:

```text
N, D, G, C, A, H, M, O
```

## Server Training

Formal training should run on `retinascope-server` under:

```text
/root/wangchen/tiansukai
```

Recommended experiment workspace:

```text
/root/wangchen/tiansukai/retinascope_exp
```

Expected data layout:

```text
/root/wangchen/tiansukai/retinascope_exp/data/train.csv
/root/wangchen/tiansukai/retinascope_exp/data/val.csv
/root/wangchen/tiansukai/retinascope_exp/train/images/<ID>_left.jpg
/root/wangchen/tiansukai/retinascope_exp/train/images/<ID>_right.jpg
/root/wangchen/tiansukai/retinascope_exp/train/images/merged/<ID>_merge.jpg
```

Run the full experiment bundle:

```bash
ssh retinascope-server
cd /root/wangchen/tiansukai/retinascope_exp
bash scripts/run_server_binocular_experiments.sh
```

Useful overrides:

```bash
EPOCHS=30 BATCH=12 NUM_WORKERS=8 bash scripts/run_server_binocular_experiments.sh
```

## Fetch Results

From this local Windows workspace:

```powershell
.\scripts\fetch_binocular_results.ps1
```

## Deployment Protection

The deployed app still uses `utils/model.py` and the legacy eight binary
classifiers unless a future promotion step explicitly changes it. These
research checkpoints are not loaded by the app automatically.
