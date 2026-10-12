# Environments

One section per machine. Each records exactly what was run, so the setup can be repeated.

## Server node (aarch64, NVIDIA GB10)

Set up on 2026-10-11. Model code runs inside NVIDIA's PyTorch container, which already carries a PyTorch build that supports the GB10. LeRobot is installed into a virtual environment in the repo folder that reuses that PyTorch.

- **Container image:** `nvcr.io/nvidia/pytorch:25.11-py3` (Python 3.12.3, PyTorch 2.10.0a0 for CUDA 13.0).
- **LeRobot:** tag `v0.6.0` (the tag the evaluation harness pins), extras `pi`, `smolvla`, `dataset`.
- **Constraints:** `envs/spark-constraints.txt` pins `torch` and `torchvision` to the container's builds so `pip` cannot replace them.
- **Model cache:** the host's Hugging Face cache folder, mounted at `/hf` with `HF_HOME=/hf`.

Start a shell in the environment (run from the repo root on the server node; replace `<hf-cache>` with the host cache folder):

```bash
docker run --rm -it --device nvidia.com/gpu=all --user "$(id -u):$(id -g)" --ipc=host \
  -e HOME=/tmp -e HF_HOME=/hf \
  -v <hf-cache>:/hf -v "$PWD":/work -w /work \
  nvcr.io/nvidia/pytorch:25.11-py3 bash
```

First-time setup, inside that shell:

```bash
python -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -c envs/spark-constraints.txt \
  "lerobot[pi,smolvla,dataset] @ git+https://github.com/huggingface/lerobot.git@v0.6.0"
```

Later sessions only need `. .venv/bin/activate`. The virtual environment points at the container's Python, so it works only inside the container.

Notes:

- The `dataset` extra is required even for inference: importing `lerobot.policies` pulls in the dataset code, which needs the `av` package.
- The π0.5 checkpoint's preprocessor loads its tokenizer from `google/paligemma-3b-pt-224`, which is gated on Hugging Face (accept Google's terms on the model page first). To keep credentials off the server node, download only the tokenizer files on a machine that is logged in, then copy the cache folder across:

  ```bash
  huggingface-cli download google/paligemma-3b-pt-224 \
    --include "tokenizer*" "special_tokens_map.json" "added_tokens.json" "config.json" \
    --cache-dir <temp-dir>
  rsync -a <temp-dir>/models--google--paligemma-3b-pt-224 <server>:<hf-cache>/hub/
  ```

  `config.json` is needed as well: without it the tokenizer loader cannot pick its class. On the server node, run π0.5 with `-e HF_HUB_OFFLINE=1` so the libraries use the local cache instead of asking the Hub.
- `scripts/time_models.py` does not use the tokenizer; it feeds placeholder token ids.
