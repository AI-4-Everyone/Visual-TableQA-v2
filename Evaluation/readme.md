# 📊 Locally evaluating Qwen2.5-VL on Visual-TableQA

The notebook provides an evaluation pipeline for **Qwen2.5-VL** on the [Visual-TableQA](https://huggingface.co/datasets/AI-4-Everyone/Visual-TableQA) dataset.

## 🛠️ Model Compatibility

The evaluation code is easily customizable to locally run other VLMs such as:

- **LLaVA**
- **InternVL**
- **MiniCPM**

To do this, simply follow the **inference instructions** corresponding to your model of interest, as outlined in its dedicated finetuning notebook located in:

```
Visual-TableQA/Finetuning/
```

Each model's notebook provides full details on how to prepare inputs, load fine-tuned weights, and run inference on any dataset.

## ⚙️ Environment Setup (Important)

Before running this evaluation notebook:

1. Identify the model you want to evaluate (e.g., Qwen2.5-VL, LLaVA, InternVL, etc.).
2. Follow the **environment setup instructions** provided in the `Visual-TableQA/Finetuning/` directory for that model.

These setup steps may involve:
- Installing model-specific dependencies
- Mounting pretrained or LoRA adapters
- Configuring the inference processor

> ⚠️ Skipping environment setup may result in missing dependencies or incorrect model loading.

---
