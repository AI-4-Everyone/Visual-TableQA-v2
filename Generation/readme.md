# 🏗️ Dataset Construction Pipeline

This directory contains the complete pipeline used to build the **Visual-TableQA** dataset from scratch, including:

- Synthetic table generation
- Visual rendering (LaTeX + TikZ)
- QA pair synthesis
- ROSCOE evaluation of QA pairs
---

## 📦 Environment Setup (Docker Recommended)

To ensure a smooth and reproducible setup, we **strongly recommend running this notebook inside a Docker container** built from the provided `Dockerfile`.

Some dependencies — such as **TeX Live** — are extremely large and can take several hours to install manually on a fresh system. Using Docker ensures:

- Preinstalled LaTeX packages (full TeX Live suite)
- Version consistency across environments
- No impact on your local Python or system packages

To run ROSCOE evaluation we recommend to first clone the ParlAI repository
```bash
git clone --depth 1 https://github.com/facebookresearch/ParlAI.git
```
---

## 🚀 Quick Start

1. **Build the Docker Image**
   ```bash
   docker build -t tableqa-image .
   ```

2. **Run the Container**
   ```bash
   docker run --gpus all -it --rm \
     -v $(pwd):/workspace \
     tableqa-image
   ```

> This command mounts the current project directory to the Docker container, giving you full access to the code, generated files, and notebooks inside the containerized environment.

---

## 📂 Output

After successful execution, the notebook will generate:

- `.tex` files for tables and diagrams
- `.png` renderings of visual tables
- `.json` files with structured QA pairs

Feel free to modify the templates, generation logic, or augmentation scripts to adapt to your own dataset needs.
