# 📦 Environment Setup (Docker Recommended)

To ensure a smooth and reproducible setup, we **strongly recommend running these notebooks inside a Docker container** built from the provided `Dockerfile`.

This is especially important because some dependencies — such as **TeX Live** — are extremely large and may take **hours to install manually** on a fresh system.

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

This will mount the current project directory inside the container and allow you to run the notebooks with all necessary dependencies pre-installed.
