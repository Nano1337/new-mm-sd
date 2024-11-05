# Multimodal Speculative Decoding

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

TODO: 
- [ ] Fix the bug in `spd_ov.py`
- [ ] Add details on how to install the dependencies
- [ ] Add details on how to run the scripts
- [ ] Add results
- [ ] Add instructions for running the benchmarks
- [ ] Port over the README from the original repo