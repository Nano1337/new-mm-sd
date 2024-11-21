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

Install dependencies:
```bash
uv sync
```

To install flash-attn, please cross-check with the prebuilt wheel that matches your system specs. Please read this article for more details: https://til.simonwillison.net/python/installing-flash-attention

As an example, I'm running python 3.11.10, cuda 12.1, ubuntu 22.04, x86_64, so I installed the correct version from the [releases page](https://github.com/Dao-AILab/flash-attention/releases): 
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
uv pip install --no-deps --upgrade flash_attn-2.7.0.post2+cu12torch2.5cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```


# TODO: how to run - Assisted Generation Benchmarks

TODO: this section is outdated, will update soon

Example command:
```
python benchmark_qwen_open.py
```

See `get_parsed_args()` in `utils.py` for a list of flags.




