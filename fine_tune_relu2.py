## 5070 issues - disable some of the kernels (~20% slower)
"""
# activate the same venv
source ~/Documents/repos/fine-tune-example/env/bin/activate

# wipe the conflicting wheels
pip uninstall -y xformers flash-attn triton torch torchvision torchaudio

#  install the SM-120-aware PyTorch stack
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# install matching Triton nightly (>=3.3.1)
pip install --pre triton \
  --extra-index-url https://huggingface.github.io/triton-nightly/cu128

# hard-disable any attempt to pull flash-attn/xformers at runtime
export XFORMERS_DISABLED=1
export HF_NO_FLASH_ATTENTION=1
export FLASH_ATTENTION_SKIP_CUDA_BUILD=1   # covers model-level imports

# verify the tool-chain
python - <<'PY'
import torch, triton, os
print("torch  :", torch.__version__)
print("triton :", triton.__version__)
print("GPU CC :", torch.cuda.get_device_capability(0))
PY
# Expect:  torch 2.8.0.dev…  • triton 3.3.1 • GPU CC (12, 0)
"""

##!!!! READ: Further 5070 issues, comment out if you are on non 50-series GPU !!!!
import triton, triton.language as tl

if not hasattr(tl.math, "round"):
    @triton.jit
    def _tl_round(x):
        return tl.floor(x + 0.5)          # nearest-integer
    tl.math.round = _tl_round

from mmfreelm.models.hgrn_bit import HGRNBitConfig
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch, torch.nn as nn
import transformers
import importlib

REPO = "ridger/MMfreeLM-370M"   # <- model to test
TOK  = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)

# Fix tokenizer - ensure it has a pad token
if TOK.pad_token is None:
    TOK.pad_token = TOK.eos_token

# monkey-patch FFN → ReLU² (TurboSparse style)
class ReLU2GLU(nn.Module):
    def __init__(self, d_in, d_ff):
        super().__init__()
        self.w_up   = nn.Linear(d_in, d_ff, bias=False)
        self.w_gate = nn.Linear(d_in, d_ff, bias=False)
    def forward(self, x):
        return self.w_up(x) * torch.relu(self.w_gate(x)) ** 2

mflm = importlib.import_module("mmfreelm.layers.hgrn_bit")
mflm.ReLU2GLU = ReLU2GLU        # overrides in all new layers

model = AutoModelForCausalLM.from_pretrained(
    REPO, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# add LoRA-64 to the two linear layers inside ReLU²
peft_cfg = LoraConfig(
    r=64, lora_alpha=16, lora_dropout=0.05,
    target_modules=["gate_proj", "down_proj"],  # Fixed: actual layer names in HGRNBit
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)

# toy dataset (10 MB of OpenWebText)
dataset = load_dataset("openwebtext", split="train[:1%]").map(
    lambda ex: TOK(ex["text"], truncation=True, max_length=256, padding="max_length"),
    batched=True, remove_columns=["text"]
)

# Split dataset for training and evaluation
train_size = int(0.9 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

# Global variable to store activations for sparsity measurement
activation_sparsities = []

def register_activation_hooks(model):
    """Register hooks to capture MLP activations and measure sparsity"""
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            sparsity = (output == 0).float().mean().item() * 100
            activation_sparsities.append(sparsity)
    
    # Register hooks on MLP modules
    for name, module in model.named_modules():
        if 'mlp' in name and hasattr(module, 'forward'):
            module.register_forward_hook(hook_fn)

# Register hooks to measure MLP activation sparsity
register_activation_hooks(model)

# helper to measure sparsity on eval batch - FIXED for PEFT model
@torch.no_grad()
def pct_zero():
    global activation_sparsities
    activation_sparsities.clear()
    
    batch = next(iter(torch.utils.data.DataLoader(
          eval_dataset, batch_size=16, collate_fn=data_collator)))
    inputs = {k: v.to(model.device) for k, v in batch.items()}

    # Forward pass (this will trigger our hooks)
    model.eval()
    _ = model(**inputs)
    
    # Return average sparsity across all captured activations
    if activation_sparsities:
        return sum(activation_sparsities) / len(activation_sparsities)
    return 0.0

# Trainer with sparsity callback - FIXED to work during training
class SparsityCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log sparsity every logging step
        if state.global_step % args.logging_steps == 0 and state.global_step > 0:
            sparsity = pct_zero()
            print(f"Step {state.global_step:>5}: MLP activation sparsity = {sparsity:.1f}%")
            logs["activation_sparsity"] = sparsity
    
    def on_evaluate(self, args, state, control, **kwargs):
        sparsity = pct_zero()
        print(f"Eval at step {state.global_step:>5}: MLP activation sparsity = {sparsity:.1f}%")

# Use proper data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=TOK, mlm=False)

args = TrainingArguments(
    "runs/sparse_lora370m",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True, max_steps=1000,
    learning_rate=2e-4, logging_steps=50,
    eval_steps=200, save_steps=1000,
)
trainer = Trainer(model, args, 
                  train_dataset=train_dataset,  # Use split train dataset
                  eval_dataset=eval_dataset,    # Add eval dataset for evaluation to trigger
                  data_collator=data_collator,
                  callbacks=[SparsityCallback()])
trainer.train()


