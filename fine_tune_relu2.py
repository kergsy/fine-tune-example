## 5070 issues - disable some of the kernels (~20% slower)
"""
# activate the same venv
source /env/bin/activate

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

m_hgrn = importlib.import_module("mmfreelm.models.hgrn_bit.modeling_hgrn_bit")
HGRNBitMLP = m_hgrn.HGRNBitMLP          # shorthand

def relu2_forward(self, x):
    up_and_gate = self.gate_proj(x)     # shape [..., 2*intermediate]
    gate, up = up_and_gate.chunk(2, -1)
    y = torch.relu(gate)**2 * up        # <-- sparse ReLU² gating
    return self.down_proj(y)

# HGRNBitMLP.forward = relu2_forward      # global patch BEFORE model load

def mlp_topk_forward(self, x, k_ratio: float = 0.02):     # keep 2 % activations
    up_and_gate = self.gate_proj(x)                       # [..., 2*intermediate]
    gate, up = up_and_gate.chunk(2, -1)

    k = max(1, int(k_ratio * gate.shape[-1]))
    topk_vals, _ = torch.topk(gate.abs(), k, dim=-1)
    thresh = topk_vals[..., -1:]                          # broadcast min-top-k
    mask   = (gate.abs() >= thresh).float()

    sparsity = (mask == 0).float().mean().item() * 100    # % zeros
    activation_sparsities.append(sparsity)

    y = (gate * mask) * up                                # sparse product
    return self.down_proj(y)                              # densifies afterward


HGRNBitMLP.forward = mlp_topk_forward      # global patch BEFORE model load


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
train_sz = int(0.9 * len(dataset))
train_ds, eval_ds = torch.utils.data.random_split(dataset, [train_sz, len(dataset)-train_sz])

data_collator = DataCollatorForLanguageModeling(tokenizer=TOK, mlm=False)


activation_sparsities = []

def hook_fn(_mod, _inp, out):
    sparsity = (out == 0).float().mean().item() * 100
    activation_sparsities.append(sparsity)

for mod in model.modules():
    if isinstance(mod, HGRNBitMLP):
        mod.register_forward_hook(hook_fn)

@torch.no_grad()
def pct_zero():
    activation_sparsities.clear()
    batch = next(iter(torch.utils.data.DataLoader(
        eval_ds, batch_size=16, collate_fn=data_collator)))
    batch = {k: v.to(model.device) for k, v in batch.items()}
    model.eval(); model(**batch)                 # triggers hooks
    return (sum(activation_sparsities) / len(activation_sparsities)) if activation_sparsities else 0.


class SparsityCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step and state.global_step % args.logging_steps == 0:
            logs = logs or {}
            logs["activation_sparsity"] = pct_zero()
            print(f"step {state.global_step:>5}  sparsity {logs['activation_sparsity']:.1f}%")

    def on_evaluate(self, args, state, control, **kwargs):
        print(f"eval  {state.global_step:>5}  sparsity {pct_zero():.1f}%")


args = TrainingArguments(
    "runs/sparse_lora370m",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
    max_steps=1000,
    learning_rate=2e-4,
    logging_steps=50,
    eval_steps=200,
    save_steps=1000,
)
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    callbacks=[SparsityCallback()]
)
trainer.train()


