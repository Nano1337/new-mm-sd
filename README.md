# Multimodal Speculative Decoding

This is a project by Haoli Yin and Siddharth Shah for the course Advanced Machine Learning at Vanderbilt University, Fall 2024. The aim of this project is to extend speculative decoding to vision-language models and to explore its potential benefits in the multimodal setting as compared to text-only speculative decoding.

## Getting Started

## Setup

We wanted to set this project up using the new `uv` tool for its incredibly fast dependency management.

Before doing anything, please install `uv` via curl using: 
```bash 
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$PATH:/home/ubuntu/.cargo/bin"
source ~/.bashrc
```

This project was created by first running `uv init mm-sd` to create a project template and then setting up a virtual environment by running: 
```bash 
uv venv 
source .venv/bin/activate
```


TODO: update the rest of this description to use `uv` tooling instead:

### Prerequisites

- Python 3.11.10 (We recommend using `pyenv` to manage Python versions)
- CUDA-compatible GPU with at least 24GB VRAM to be safe. 
    - We advise to use CUDA 12.1 or higher.

### Installation

0. Ensure your Ubuntu version is 22.04 and is up to date. 

1. Install Python 3.11.10 using pyenv: 
   ```
   pyenv install 3.11.10
   pyenv local 3.11.10
   ```
   and then verify the installation with:
   ```
   python --version
   ```
   If you run into any issues installing with pyenv, it may be that your apt is outdated. You can update it with the following commands:
   ```
   sudo apt update
   sudo apt upgrade
   ```
   Any other issues can be resolved with ChatGPT. 

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install uv (a faster alternative to pip):
   ```
   pip install uv
   ```

4. Install dependencies:
   ```
   uv pip install autoawq
   uv pip install -r requirements.txt --no-deps
   uv pip install wheel
   pip install flash-attn --no-build-isolation
   ```

   Note: The order of installation is important due to potential conflicts between packages.

### Troubleshooting

- If you encounter issues with CUDA or GPU support, ensure you have the latest NVIDIA drivers installed.
- For `flash-attn` installation problems, refer to its [official documentation](https://github.com/Dao-AILab/flash-attention) for system-specific instructions.


# Assisted Generation Benchmarks

Example command:
```
python benchmark_qwen_open.py
```

See `get_parsed_args()` in `utils.py` for a list of flags.




