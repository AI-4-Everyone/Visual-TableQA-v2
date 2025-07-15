# 🧠 TableQA-Synthetic: Generating QA Datasets from Table Images

Welcome to **TableQA-Synthetic**, a project designed to generate high-quality **synthetic question-answer datasets** associated to **images of tables**. This resource is ideal for training and evaluating models on visually-grounded table understanding tasks such as **document QA**, **table parsing**, and **multimodal reasoning**.

---

## 🚀 What’s Inside

- 📓 **Notebook** for generating synthetic table images and their QA pairs  
- 🖼️ Support for structured and stylized LaTeX tables or custom-designed visuals  
- 🔍 Automatic generation of questions and answers with ground-truth alignment  
- ⚡ Ready-to-use for fine-tuning LLMs, vision-language models, or benchmarking pipelines

---

## 📘 Paper (Coming Soon)

📝 A detailed dataset paper describing the methodology, QA strategy, and dataset statistics is coming soon.  
<!-- Replace the placeholder below with your actual paper link -->
**[📄 Read the Paper (coming soon)](https://arxiv.org/abs/XXXXX)**

---

## 📁 Repository Structure

```text
├── Generation/TableQA.ipynb # Main notebook to generate synthetic QA
├── Examples/ # Example generated images and records
├── Evaluation/
├── README.md
└── LICENSE
```
## 📦 Installation

```bash
pip install -r requirements.txt
```

## 📤 Dataset Access
You can find the generated dataset hosted on 🤗 Hugging Face:
**[TableQA Synthetic Dataset →](https://huggingface.co/datasets/AI-4-Everyone/TableQA-v2)**

## 📄 License

This project is licensed under the [MIT License](LICENSE).

You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided that the original copyright and permission notice are included in all copies or substantial portions of the software.

## 📚 Citation

If you use this code or dataset in your research, please cite:

**Plain-text citation:**  
Marc Haraoui, Boammani Aser Lompo *Table\_QA*. GitHub repository: https://github.com/AI-4-Everyone/TableQA

**BibTeX:**
```bibtex
@misc{haraouilompo2025tableqa,
  author       = {Marc Haraoui and Boammani Aser Lompo},
  title        = {TableQA},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/datasets/AI-4-Everyone/TableQA-v2}},
}
