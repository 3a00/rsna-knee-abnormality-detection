"""scripts/make_training_notebook.py

Generates the complete rsna_efficiency_training.ipynb notebook for Kaggle.
Configured with:
  - Fail-fast preflight (Cell 3b) asserting all paths, splits, labels, scripts, and disk space before caching.
  - Unbuffered live subprocess streaming (run_live) so cells never run blind.
  - One-time 336px physical FOV pre-caching (~97 min).
  - Multi-Fold sequential DDP training across all 3 folds (--folds 0,1,2).
  - Full 3-fold OOF evaluation (scripts/evaluate_3fold_oof.py) with ddof=1 CV statistics.
  - Clean weights inventory in outputs/weights/.
"""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# RSNA Knee Abnormality Detection: Dual-T4 DDP Multi-Fold Training (Efficiency Track)\n",
                "\n",
                "Trains **DINOv2-Small SlotHead** cross-attention architecture using PyTorch **DistributedDataParallel (DDP)** across Kaggle Dual Tesla T4 GPUs (`GPU T4 x2`) across all 3 folds (`--folds 0,1,2`).\n",
                "\n",
                "### Kaggle Session Settings:\n",
                "- **Accelerator:** `GPU T4 x2`\n",
                "- **Internet:** `On`\n",
                "- **Attached Data Sources:**\n",
                "  1. Competition Data: `rsna-knee-abnormality-detection`\n",
                "  2. Codebase Dataset: `rsna_efficiency_codebase.zip`\n",
                "  3. Historical Fold 0 Reference (Optional): `dinov2_slothead_fold0_best.pth`\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 1: Environment & Live Subprocess Streaming Setup\n",
                "Configures NCCL flags for PCIe bridge stability and provides unbuffered subprocess streaming so cell execution never runs blind."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys, shutil, zipfile, subprocess\n",
                "\n",
                "os.environ['NCCL_P2P_DISABLE'] = '1'\n",
                "os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1800'\n",
                "os.environ['PYTHONUNBUFFERED'] = '1'\n",
                "\n",
                "def run_live(cmd):\n",
                "    print(f'>>> Executing: {\" \".join(cmd)}\\n')\n",
                "    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}\n",
                "    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)\n",
                "    for line in p.stdout:\n",
                "        print(line, end='', flush=True)\n",
                "    if p.wait():\n",
                "        raise subprocess.CalledProcessError(p.returncode, cmd)\n",
                "\n",
                "print('Installing dependencies (timm, pydicom, albumentations)...')\n",
                "run_live([sys.executable, '-m', 'pip', 'install', '-q', 'timm', 'pydicom', 'albumentations'])\n",
                "\n",
                "import torch\n",
                "print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Count: {torch.cuda.device_count()}')\n",
                "assert torch.cuda.device_count() >= 2, 'WARNING: Set accelerator to GPU T4 x2!'\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 2: Codebase Discovery & Historical Reference Mount\n",
                "Extracts codebase archive and mounts historical reference weights if attached."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import glob\n",
                "from pathlib import Path\n",
                "\n",
                "WORKING_DIR = Path('/kaggle/working')\n",
                "os.chdir(WORKING_DIR)\n",
                "if str(WORKING_DIR) not in sys.path:\n",
                "    sys.path.insert(0, str(WORKING_DIR))\n",
                "\n",
                "codebase_found = False\n",
                "for item in glob.glob('/kaggle/input/**/rsna_efficiency_codebase.zip', recursive=True):\n",
                "    print(f'Extracting codebase: {item}')\n",
                "    with zipfile.ZipFile(item, 'r') as zf:\n",
                "        zf.extractall(WORKING_DIR)\n",
                "    codebase_found = True\n",
                "    break\n",
                "\n",
                "if not codebase_found:\n",
                "    for root, dirs, files in os.walk('/kaggle/input'):\n",
                "        if 'config.yaml' in files and 'src' in dirs and 'scripts' in dirs:\n",
                "            print(f'Copying codebase directory: {root}')\n",
                "            for d in ['src', 'scripts', 'data']:\n",
                "                src_p = Path(root) / d\n",
                "                dst_p = WORKING_DIR / d\n",
                "                if src_p.exists():\n",
                "                    shutil.copytree(src_p, dst_p, dirs_exist_ok=True)\n",
                "            shutil.copy2(Path(root) / 'config.yaml', WORKING_DIR / 'config.yaml')\n",
                "            codebase_found = True\n",
                "            break\n",
                "\n",
                "for rel in ['outputs/weights', 'outputs/tensorboard', 'data/splits', 'data/labels']:\n",
                "    (WORKING_DIR / rel).mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# Mount historical Fold 0 reference weights under non-colliding name\n",
                "fold0_attached = glob.glob('/kaggle/input/**/dinov2_slothead_fold0_best.pth', recursive=True) + \\\n",
                "                 glob.glob('/kaggle/input/**/dinov2_slothead_best_single.pth', recursive=True)\n",
                "if fold0_attached:\n",
                "    dst_f0 = WORKING_DIR / 'outputs/weights/dinov2_slothead_fold0_v6_reference.pth'\n",
                "    if not dst_f0.is_file():\n",
                "        shutil.copy2(fold0_attached[0], dst_f0)\n",
                "        print(f'Mounted historical reference weights: {dst_f0}')\n",
                "\n",
                "print('Codebase extraction finished.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 3: DINOv2 Backbone Acquisition\n",
                "Loads or downloads `vit_small_patch14_dinov2.lvd142m` weights."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import timm\n",
                "\n",
                "BACKBONE_PATH = WORKING_DIR / 'dinov2_small_backbone.pth'\n",
                "attached = glob.glob('/kaggle/input/**/dinov2_small_backbone.pth', recursive=True) + \\\n",
                "           glob.glob('/kaggle/input/**/vit_small_patch14_dinov2*.pth', recursive=True)\n",
                "\n",
                "if attached:\n",
                "    print(f'Using attached backbone: {attached[0]}')\n",
                "    BACKBONE_PATH = Path(attached[0])\n",
                "elif not BACKBONE_PATH.is_file():\n",
                "    print('Downloading DINOv2-Small backbone via timm...')\n",
                "    backbone_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)\n",
                "    torch.save(backbone_model.state_dict(), BACKBONE_PATH)\n",
                "    del backbone_model\n",
                "    torch.cuda.empty_cache()\n",
                "    print(f'Backbone saved: {BACKBONE_PATH}')\n"
            ]
        },
        # Blocker B7 Fix: Fail-Fast Preflight Cell 3b before the 97-minute cache build
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 3b: Fail-Fast Pipeline Preflight (<5 seconds)\n",
                "Asserts existence of all required splits, labels, scripts, backbone weights, columns, and `/kaggle/tmp` free disk space *before* the ~97-minute caching phase starts."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd, shutil\n",
                "\n",
                "req_files = {\n",
                "    'splits': WORKING_DIR / 'data/splits/cv_folds_3fold.csv',\n",
                "    'labels': WORKING_DIR / 'data/labels/pseudo_labels.csv',\n",
                "    'config': WORKING_DIR / 'config.yaml',\n",
                "    'preprocess script': WORKING_DIR / 'scripts/preprocess_336px_cache.py',\n",
                "    'train script': WORKING_DIR / 'scripts/train_ddp_efficiency.py',\n",
                "    'eval script': WORKING_DIR / 'scripts/evaluate_3fold_oof.py',\n",
                "    'backbone weights': Path(BACKBONE_PATH),\n",
                "}\n",
                "missing_files = [k for k, p in req_files.items() if not p.is_file()]\n",
                "assert not missing_files, f'Preflight FAILED: Missing essential files: {missing_files}'\n",
                "\n",
                "spl_df = pd.read_csv(req_files['splits'])\n",
                "lab_df = pd.read_csv(req_files['labels'], nrows=50)\n",
                "f_col = 'fold_id' if 'fold_id' in spl_df.columns else 'fold'\n",
                "assert {0, 1, 2} <= set(spl_df[f_col].unique()), f'Splits missing required folds 0, 1, 2: {spl_df[f_col].unique()}'\n",
                "assert 'synovitis_soft' in lab_df.columns, 'Labels missing required synovitis_soft column'\n",
                "assert 'is_gold' in lab_df.columns or 'is_gold' in spl_df.columns, 'Missing required is_gold column'\n",
                "\n",
                "free_gb = shutil.disk_usage('/kaggle/tmp').free / 1e9\n",
                "print(f'Preflight OK: All scripts, splits, and backbone verified.')\n",
                "print(f'Free disk on /kaggle/tmp: {free_gb:.1f} GB (336px 5-slice cache requires ~17 GB)')\n",
                "assert free_gb >= 25.0, f'WARNING: Insufficient disk space on /kaggle/tmp ({free_gb:.1f} GB < 25 GB)'\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 4: Idempotent 5-Slice Volume Pre-Caching (~97 min)\n",
                "Pre-extracts 5 centered physical slices per present slot into `/kaggle/tmp/rsna_cache_336_140mm`. Idempotent on restart."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dicom_root = None\n",
                "for root, dirs, files in os.walk('/kaggle/input'):\n",
                "    if 'train_series.csv' in files:\n",
                "        dicom_root = Path(root)\n",
                "        break\n",
                "if dicom_root is None:\n",
                "    dicom_root = Path('/kaggle/input/rsna-knee-abnormality-detection')\n",
                "train_series_csv = dicom_root / 'train_series.csv'\n",
                "cache_dir = Path('/kaggle/tmp/rsna_cache_336_140mm')\n",
                "cache_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "preprocess_cmd = [\n",
                "    sys.executable, 'scripts/preprocess_336px_cache.py',\n",
                "    '--dicom-root', str(dicom_root),\n",
                "    '--cache-dir', str(cache_dir),\n",
                "    '--series-csv', str(train_series_csv),\n",
                "    '--splits-csv', '/kaggle/working/data/splits/cv_folds_3fold.csv',\n",
                "    '--target-size', '336',\n",
                "    '--crop-mm', '140.0',\n",
                "    '--num-workers', '4',\n",
                "]\n",
                "run_live(preprocess_cmd)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 5: Dual-T4 DDP Homogeneous Multi-Fold Training (Folds 0, 1 & 2)\n",
                "Trains Folds 0, 1, and 2 sequentially on the authoritative 14-epoch 4-stage recipe with isolated weight re-initialization and fold resumption."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_cmd = [\n",
                "    sys.executable, 'scripts/train_ddp_efficiency.py',\n",
                "    '--folds', '0,1,2',\n",
                "    '--epochs', '14',\n",
                "    '--batch-size', '8',\n",
                "    '--grad-accum-steps', '1',\n",
                "    '--img-size', '336',\n",
                "    '--crop-mm', '140.0',\n",
                "    '--weight-decay', '0.05',\n",
                "    '--gold-weight', '2.0',\n",
                "    '--cache-dir', str(cache_dir),\n",
                "    '--require-cache',\n",
                "    '--weights-path', str(BACKBONE_PATH),\n",
                "    '--weights-dir', '/kaggle/working/outputs/weights',\n",
                "    '--tb-dir', '/kaggle/working/outputs/tensorboard',\n",
                "    '--dicom-root', str(dicom_root),\n",
                "    '--series-csv', str(train_series_csv),\n",
                "    '--labels-path', '/kaggle/working/data/labels/pseudo_labels.csv',\n",
                "    '--splits-path', '/kaggle/working/data/splits/cv_folds_3fold.csv',\n",
                "    '--config-path', '/kaggle/working/config.yaml',\n",
                "    '--num-workers', '2',\n",
                "]\n",
                "run_live(train_cmd)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 6: Full 3-Fold Out-of-Fold (OOF) Evaluation & Weights Verification\n",
                "Executes `scripts/evaluate_3fold_oof.py` across all 3 folds, computing honest `mean ± std (ddof=1)` CV benchmark statistics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "eval_cmd = [\n",
                "    sys.executable, 'scripts/evaluate_3fold_oof.py',\n",
                "    '--weights-dir', '/kaggle/working/outputs/weights',\n",
                "    '--splits-path', '/kaggle/working/data/splits/cv_folds_3fold.csv',\n",
                "    '--labels-path', '/kaggle/working/data/labels/pseudo_labels.csv',\n",
                "    '--config-path', '/kaggle/working/config.yaml',\n",
                "    '--cache-dir', str(cache_dir),\n",
                "    '--output-csv', '/kaggle/working/outputs/weights/oof_predictions.csv',\n",
                "    '--output-json', '/kaggle/working/outputs/weights/oof_results.json',\n",
                "]\n",
                "run_live(eval_cmd)\n",
                "\n",
                "print('\\n=== Verifying Required Fold Checkpoints ===')\n",
                "weights_dir = WORKING_DIR / 'outputs/weights'\n",
                "for f_id in [0, 1, 2]:\n",
                "    best_f = weights_dir / f'dinov2_slothead_fold{f_id}_best.pth'\n",
                "    assert best_f.is_file(), f'MISSING: {best_f.name}'\n",
                "    print(f'Verified: {best_f.name} ({best_f.stat().st_size / 1e6:.1f} MB)')\n",
                "\n",
                "print('\\n=== Checkpoint Inventory in outputs/weights/ ===')\n",
                "for p in sorted(weights_dir.glob('*.pth')):\n",
                "    print(f'  {p.name} ({p.stat().st_size / 1e6:.1f} MB)')\n",
                "for f in ['oof_predictions.csv', 'oof_results.json', 'multifold_results.json']:\n",
                "    if (weights_dir / f).is_file():\n",
                "        print(f'  {f} ({ (weights_dir / f).stat().st_size / 1e3:.1f} KB)')\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = Path("notebooks/rsna_efficiency_training.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated {output_path} ({output_path.stat().st_size} bytes)")
