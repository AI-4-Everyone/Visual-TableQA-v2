# 🧪 Finetuning VLMs on Visual-TableQA

This guide walks you through the steps to finetune and run inference with two Vision-Language Models (VLMs): **MiniCPM** and **InternVL**, using the [AI-4-Everyone/Visual-TableQA](https://huggingface.co/datasets/AI-4-Everyone/Visual-TableQA) dataset.

---

## 🚀 Setup

### 1. Install Dependencies

Make sure you have Python ≥ 3.10. Then install all required packages:

```bash
pip install -r requirements.txt
```

---

## 🧠 Finetuning & Inference with MiniCPM

### Step 1: Clone MiniCPM Repository

```bash
git clone https://github.com/OpenSQZ/MiniCPM-V-CookBook.git
```

### Step 2: Prepare Dataset

Follow the finetuning instructions in the MiniCPM repo for a custom dataset.
You can use the script `build_minicpm_dataset.py` to preprocess the Visual-TableQA dataset.

### Step 3: Add Finetuning Script

Place your `minicpm_finetune_lora.sh` script inside the `MiniCPM-V-CookBook/finetune/` directory.

### Step 4: Run Finetuning

```bash
cd MiniCPM-V-CookBook/finetune/
sh minicpm_finetune_lora.sh
```

### Step 5: Inference (Example Script)

```python
import os, torch
from PIL import Image
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

# Load Visual-TableQA
ds = load_dataset("AI-4-Everyone/Visual-TableQA")
train = ds["train"]

# Load model + LoRA adapter
model_type = "openbmb/MiniCPM-Llama3-V-2_5"
adapter_path = "MiniCPM-sft-lora"

base_model = AutoModel.from_pretrained(
    model_type,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)

# Prepare inputs
image = Image.open(f"minicpm_ds/images/{train[0]['table_id']}.jpg").convert('RGB')
system_prompt = (
    "You are a Vision Language Model specialized in interpreting visual data from charts and diagrams images.\n"
    "Answer the questions strictly from the image, with clear, rigorous step-by-step justification. Stay concise, but include all reasoning that’s relevant."
)

question = f"{system_prompt}\n{train[0]['question']}"
msgs = [{'role': 'user', 'content': [image, question]}]

# Inference
response = model.chat(
    msgs=msgs,
    image=image,
    tokenizer=tokenizer,
    sampling=False,     # <- switch off sampling (default is True)
    num_beams=1,        # <- greedy decoding for determinism
    max_new_tokens=5000
)

print(response)
```

---

## 🧠 Finetuning & Inference with InternVL

### Step 1: Clone InternVL Repository

```bash
git clone https://github.com/OpenGVLab/InternVL.git
```

### Step 2: Prepare Dataset

Follow the finetuning instructions in the InternVL repo.  
You can reuse `build_internvl_dataset.py` for preprocessing.

- Place `internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora.sh` inside `InternVL/internvl_chat/`
- Place `tableqa.json` inside `InternVL/internvl_chat/shell/data/`

### Step 3: Run Finetuning

```bash
cd InternVL/internvl_chat/
sh internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora.sh
```

### Step 4: Inference (Example Script)

```python
import os, sys, torch
from datasets import load_dataset
from transformers import AutoTokenizer
from internvl_utils import load_image
from internvl.model.internvl_chat import InternVLChatModel

# Paths
DATASET_REPO = "AI-4-Everyone/Visual-TableQA"
adapter_path = "/notebooks/InternVL-sft-lora"
model_path = "/notebooks/InternVL/pretrained/InternVL2-8B"
preprocessed_data_path = "internvl_data"
sys.path.append('/notebooks/InternVL/internvl_chat')

# Load Visual-TableQA
ds = load_dataset(DATASET_REPO)
train = ds["train"]
sample = train[0]

# Load model
model = InternVLChatModel.from_pretrained(
    adapter_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False
)

# Prepare inputs
img_path = f"/notebooks/{preprocessed_data_path}/images/{sample['table_id']}.jpg"
pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
system_prompt = (
    "You are a Vision Language Model specialized in interpreting visual data from charts and diagrams images.\n"
    "Answer the questions strictly from the image, with clear, rigorous step-by-step justification. Stay concise, but include all reasoning that’s relevant."
)

question = f"<image>\n{system_prompt}\n{sample['question']}"

# Inference
response = model.chat(
    tokenizer=tokenizer,
    pixel_values=pixel_values,
    question=question,
    generation_config={"max_new_tokens": 5000, "do_sample": False},
    history=None,
    return_history=False
)

print(response)
```

---

## 🗂️ File Structure Overview

```
.
├── build_minicpm_dataset.py        # Preprocessing script for MiniCPM
├── build_internvl_dataset.py       # Preprocessing script for InternVL
├── minicpm_finetune_lora.sh        # Shell script to finetune MiniCPM
├── internvl2_8b_..._lora.sh        # Shell script to finetune InternVL
├── MiniCPM-V-CookBook/             # MiniCPM repo
├── InternVL/                       # InternVL repo
├── tableqa.json                    # Training data for InternVL
├── internvl_data/                  # Preprocessed images for InternVL
└── requirements.txt
```
