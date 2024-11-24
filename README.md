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
uv venv --python 3.11.10
source .venv/bin/activate
```

## Installing dependencies:

To install all dependencies (except for flash-attn), run:
```bash
uv sync
```

To install flash-attn, please cross-check with the prebuilt wheel that matches your system specs. Please read this article for more details: https://til.simonwillison.net/python/installing-flash-attention

For example, here are my system specifications and the corresponding flash-attention wheel I installed:
- Python: 3.11.10
- CUDA: 12.1
- GPU: NVIDIA A10G (24GB VRAM)
- Driver: 535.183.01
- Instance: AWS g6.xlarge
- OS: Ubuntu 22.04
- Architecture: x86_64
- PyTorch: 2.3.1

I downloaded the matching wheel from the [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases).
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

uv pip install --no-deps --upgrade flash_attn-2.7.0.post2+cu12torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```


# Running Benchmark

Example command:
```
python benchmark_qwen_open.py --no_reduce_acceptance_rate --num_samples 10 --num_draft_samples 10
```

See `parse_args()` in `utils.py` for a list of flags.


## TODOs: 
- [ ] Implement BenchmarkMetrics to display different colors for which token is associated with which model that generated it
- [ ] Do this for a visual reasoning dataset?
- [ ] Do this for [WildVision-Bench](https://github.com/WildVision-AI/WildVision-Bench/blob/main/data/vision_bench_0617/model_answers/Qwen_Qwen-VL-Chat.jsonl)
gi