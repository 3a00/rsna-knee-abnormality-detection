import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# RSNA Knee Abnormality Detection: Dual-T4 DDP Training (Efficiency Track)\n",
                "\n",
                "This notebook trains the **DINOv2-Small SlotHead** cross-attention architecture using PyTorch **DistributedDataParallel (DDP)** across Kaggle's dual Tesla T4 GPUs (`GPU T4 x2`).\n",
                "\n",
                "### Kaggle Session Settings:\n",
                "- **Accelerator:** `GPU T4 x2`\n",
                "- **Internet:** `On` (to install dependencies and download DINOv2 backbone)\n",
                "- **Attached Data Sources:**\n",
                "  1. Competition Data: `rsna-knee-abnormality-detection`\n",
                "  2. Codebase Dataset: `rsna_efficiency_codebase.zip` (attached Kaggle dataset containing `src/`, `scripts/`, `config.yaml`, `data/splits/`, `data/labels/`)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 1: Environment & Dependency Setup\n",
                "Installs `timm`, `pydicom`, and `albumentations`, configures NCCL environment flags to prevent PCIe bridge stalls on Kaggle Dual-T4, and checks GPU availability."
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
                "# Prevent Kaggle PCIe bridge stall during inter-GPU P2P handshake\n",
                "os.environ['NCCL_P2P_DISABLE'] = '1'\n",
                "os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1800'\n",
                "\n",
                "print('Installing dependencies (timm, pydicom, albumentations)...')\n",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm', 'pydicom', 'albumentations'], check=True)\n",
                "\n",
                "import torch\n",
                "print(f'PyTorch version: {torch.__version__}')\n",
                "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                "num_gpus = torch.cuda.device_count()\n",
                "print(f'GPU count: {num_gpus}')\n",
                "for i in range(num_gpus):\n",
                "    print(f'  Device {i}: {torch.cuda.get_device_name(i)}')\n",
                "\n",
                "if num_gpus < 2:\n",
                "    print('WARNING: Set Kaggle accelerator to \"GPU T4 x2\" to run distributed training!')\n",
                "else:\n",
                "    print('Dual T4 GPUs ready for DDP.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 2: Codebase Discovery & Extraction\n",
                "Discovers the codebase from `/kaggle/input`, extracts or copies all required modules to `/kaggle/working`, and verifies file integrity."
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
                "# 1. Search for uploaded zip archive\n",
                "for item in glob.glob('/kaggle/input/**/rsna_efficiency_codebase.zip', recursive=True):\n",
                "    print(f'Found codebase archive: {item}')\n",
                "    with zipfile.ZipFile(item, 'r') as zf:\n",
                "        zf.extractall(WORKING_DIR)\n",
                "    codebase_found = True\n",
                "    break\n",
                "\n",
                "# 2. Fallback: Search for uncompressed dataset folder\n",
                "if not codebase_found:\n",
                "    for root, dirs, files in os.walk('/kaggle/input'):\n",
                "        if 'config.yaml' in files and 'src' in dirs and 'scripts' in dirs:\n",
                "            print(f'Found codebase directory: {root}')\n",
                "            for d in ['src', 'scripts', 'data']:\n",
                "                src_p = Path(root) / d\n",
                "                dst_p = WORKING_DIR / d\n",
                "                if src_p.exists():\n",
                "                    shutil.copytree(src_p, dst_p, dirs_exist_ok=True)\n",
                "            shutil.copy2(Path(root) / 'config.yaml', WORKING_DIR / 'config.yaml')\n",
                "            codebase_found = True\n",
                "            break\n",
                "\n",
                "# Ensure working subdirectories exist\n",
                "for rel in ['outputs/weights', 'outputs/tensorboard', 'data/splits', 'data/labels']:\n",
                "    (WORKING_DIR / rel).mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# Assert core modules are ready\n",
                "required_files = [\n",
                "    'scripts/train_ddp_efficiency.py',\n",
                "    'src/models/dinov2_slothead.py',\n",
                "    'src/datasets/efficiency_pipeline.py',\n",
                "    'src/training/train_efficiency.py',\n",
                "    'config.yaml',\n",
                "    'data/splits/cv_folds_3fold.csv',\n",
                "    'data/labels/pseudo_labels.csv',\n",
                "]\n",
                "for rf in required_files:\n",
                "    assert (WORKING_DIR / rf).is_file(), f'MISSING: {rf}'\n",
                "    print(f'OK: {rf}')\n",
                "\n",
                "print('Codebase verified successfully!')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 3: DINOv2 Backbone Acquisition\n",
                "Obtains the official `vit_small_patch14_dinov2.lvd142m` backbone weights. If attached in `/kaggle/input`, uses local file; otherwise downloads via `timm` and saves to `/kaggle/working/dinov2_small_backbone.pth`."
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
                "\n",
                "attached = glob.glob('/kaggle/input/**/dinov2_small_backbone.pth', recursive=True) + \\\n",
                "           glob.glob('/kaggle/input/**/vit_small_patch14_dinov2*.pth', recursive=True)\n",
                "\n",
                "if attached:\n",
                "    print(f'Using attached backbone weights: {attached[0]}')\n",
                "    BACKBONE_PATH = Path(attached[0])\n",
                "elif not BACKBONE_PATH.is_file():\n",
                "    print('Downloading DINOv2-Small backbone via timm...')\n",
                "    backbone_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)\n",
                "    torch.save(backbone_model.state_dict(), BACKBONE_PATH)\n",
                "    del backbone_model\n",
                "    torch.cuda.empty_cache()\n",
                "    print(f'Backbone weights saved to {BACKBONE_PATH} ({BACKBONE_PATH.stat().st_size / 1e6:.1f} MB)')\n",
                "else:\n",
                "    print(f'Backbone weights already present at: {BACKBONE_PATH}')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 4: Smoke Test (Optional Dual-GPU Verification)\n",
                "Executes a quick 3-epoch synthetic smoke test across both GPUs to confirm NCCL process group communication, balanced VRAM, and zero host RAM leaks before running full training."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Optional sanity check: run synthetic smoke test across both GPUs\n",
                "!python scripts/smoke_ddp.py --img-size 336\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 5: Dual-T4 DistributedDataParallel Full Training\n",
                "Launches the full distributed training runner (`scripts/train_ddp_efficiency.py`) across both T4 GPUs.\n",
                "- **Fold:** 0 (of the 3-fold scanner-fingerprint StratifiedGroupKFold split)\n",
                "- **Resolution:** 336x336 px\n",
                "- **Batch size:** 4 studies per GPU (effective batch size 8)\n",
                "- **Checkpoints:** Saves Top-3 validation checkpoints and performs arithmetic parameter averaging to produce `dinov2_slothead_top3.pth`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python scripts/train_ddp_efficiency.py \\\n",
                "    --fold 0 \\\n",
                "    --epochs 12 \\\n",
                "    --batch-size 4 \\\n",
                "    --grad-accum-steps 1 \\\n",
                "    --img-size 336 \\\n",
                "    --lr-backbone 3e-5 \\\n",
                "    --lr-head 1e-3 \\\n",
                "    --weight-decay 0.02 \\\n",
                "    --gold-weight 5.0 \\\n",
                "    --weights-path {str(BACKBONE_PATH)} \\\n",
                "    --weights-dir /kaggle/working/outputs/weights \\\n",
                "    --tb-dir /kaggle/working/outputs/tensorboard \\\n",
                "    --dicom-root /kaggle/input/rsna-knee-abnormality-detection \\\n",
                "    --series-csv /kaggle/input/rsna-knee-abnormality-detection/train_series.csv \\\n",
                "    --labels-path /kaggle/working/data/labels/pseudo_labels.csv \\\n",
                "    --splits-path /kaggle/working/data/splits/cv_folds_3fold.csv \\\n",
                "    --config-path /kaggle/working/config.yaml \\\n",
                "    --num-workers 2\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 6: Inspect Training Results & Export Trained Checkpoints\n",
                "Reads `ddp_results.json` and copies the final weights (`dinov2_slothead_top3.pth` and `dinov2_slothead_best_single.pth`) directly to `/kaggle/working/` for Kaggle dataset creation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "\n",
                "results_file = WORKING_DIR / 'outputs/weights/ddp_results.json'\n",
                "if results_file.is_file():\n",
                "    with open(results_file) as f:\n",
                "        res = json.load(f)\n",
                "    print('=== DDP Training Summary ===')\n",
                "    print(f\"Best Score (Macro AUC 11): {res.get('best_score', 0):.4f} (Epoch {res.get('best_epoch', 0)})\")\n",
                "    print(f\"Averaged Macro AUC 11:     {res.get('averaged_macro_auc_11')}\")\n",
                "    print(f\"Averaged Macro AUC 12:     {res.get('averaged_macro_auc_12')}\")\n",
                "    print(f\"Recommended Weights:       {res.get('recommended_weights_path')}\")\n",
                "\n",
                "print('\\n=== Saved Checkpoints ===')\n",
                "for p in (WORKING_DIR / 'outputs/weights').glob('*.pth'):\n",
                "    print(f'  {p.name}: {p.stat().st_size / 1e6:.1f} MB')\n",
                "\n",
                "# Copy recommended weights to root of /kaggle/working for direct dataset publishing\n",
                "top3_pth = WORKING_DIR / 'outputs/weights/dinov2_slothead_top3.pth'\n",
                "if top3_pth.is_file():\n",
                "    shutil.copy2(top3_pth, WORKING_DIR / 'dinov2_slothead_top3.pth')\n",
                "    print(f'Copied {top3_pth.name} -> /kaggle/working/')\n",
                "\n",
                "best_single = WORKING_DIR / 'outputs/weights/dinov2_slothead_best_single.pth'\n",
                "if best_single.is_file():\n",
                "    shutil.copy2(best_single, WORKING_DIR / 'dinov2_slothead_best_single.pth')\n",
                "    print(f'Copied {best_single.name} -> /kaggle/working/')\n"
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
