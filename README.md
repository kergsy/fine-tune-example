## Setup (ONLY TESTED ON 5070)

```bash
# 1. Create virtual environment
python -m venv env
source env/bin/activate

# 2. Install PyTorch first (required for building)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Set environment variables (for RTX 50-series compatibility)
export TORCH_CUDA_ARCH_LIST="8.9"
export XFORMERS_DISABLED=1
export HF_NO_FLASH_ATTENTION=1
```

## Run

```bash
python -m fine_tune_relu2
```

## What it does

- Fine-tunes `ridger/MMfreeLM-370M` with ReLU² activation for sparsity (SPARSITY NOT WORKING)
- Uses LoRA (r=64) on `gate_proj` and `down_proj` layers  
- Measures activation sparsity during training
- Logs to W&B

## Key files

- `fine_tune_relu2.py` - Main training script
- `requirements.txt` - Dependencies
- `mmfreelm/` - Custom model architecture
