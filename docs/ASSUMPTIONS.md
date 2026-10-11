# Assumptions, unknowns and evidence

Every claim the project depends on lives here with one of four labels:

- **Verified:** we ran it ourselves. The evidence column gives the command and the result.
- **Stated in source:** a named source says so. We have not tested it.
- **Assumption:** we believe it and have not checked.
- **Unknown:** must be measured or checked.

An item moves to "Verified" only with evidence: the command run, its output, or a file and line.

Last updated: 2026-10-10.

## Verified

| ID | Claim | Evidence |
| --- | --- | --- |
| V1 | The demo server node is aarch64, Ubuntu 24.04.5 LTS, with an NVIDIA GB10, driver 580.178.04, CUDA 13.0. | `uname -m`, `/etc/os-release`, `nvidia-smi` on the server node, 2026-10-10. |
| V2 | The server node has 121 GiB memory (118 GiB available when idle) and 3.4 TB free disk on one NVMe volume. | `free -h`, `df -h ~`, `lsblk`, 2026-10-10. |
| V3 | On the server node, `nvidia-smi` reports GPU memory usage as "Not Supported". Peak memory there must be measured another way. | `nvidia-smi` output, 2026-10-10. |
| V4 | The server node has Docker 29.6.2 and our user can reach the daemon without sudo. | `docker ps` returned an empty container list; `id` shows the `docker` group, 2026-10-10. |
| V5 | The server node has system Python 3.12.3 with no `torch`, and no `uv`, `conda`, `mamba` or `apptainer` on the login PATH. | `python3 --version`, `python3 -c "import torch"`, `command -v ...` in a login shell, 2026-10-10. |
| V6 | The server node has git 2.43.0. | `git --version`, 2026-10-10. |

## Stated in source (not tested by us)

Items marked (summary) were read through a summarizing web fetch on 2026-10-10, not from the raw file. Re-read the raw source before relying on them.

| ID | Claim | Source |
| --- | --- | --- |
| S1 | LIBERO pins `numpy==1.22.4`, `robosuite==1.4.0`, `bddl==1.0.1`, `transformers==4.21.1`, `gym==0.25.2`, `robomimic==0.2.0`. The file has no mujoco pin. | LIBERO `requirements.txt` on the master branch (summary). |
| S2 | A third-party page says LIBERO needs mujoco 3.2.3. This does not come from LIBERO's own requirements file. | RoboVerse wiki. |
| S3 | LeRobot's LIBERO docs set `MUJOCO_GL=egl` on headless servers. | LeRobot docs (read by the authors). |
| S4 | SmolVLA checkpoints fine-tuned on LIBERO exist: `lerobot/smolvla_libero`, `lerobot/smolvla_libero_plus` and `HuggingFaceVLA/smolvla_libero`, all tagged Apache-2.0. Training suites and reported scores not yet read. | Hugging Face Hub search, 2026-10-10. |
| S5 | openpi is Apache-2.0 and also ships a `LICENSE_GEMMA.txt`. Its π0.5 LIBERO checkpoint is at `gs://openpi-assets/checkpoints/pi05_libero`. | openpi README and repo sidebar (summary). |
| S6 | openpi has PyTorch implementations of π0 and π0.5, validated on LIBERO. Inference needs more than 8 GB of GPU memory. | openpi README (summary). |
| S7 | openpi "has been tested with Ubuntu 22.04" and does not support other operating systems. It uses `uv`. | openpi README (summary). |
| S8 | openpi's policy server is `scripts/serve_policy.py` and listens on port 8000. | openpi README (summary). |
| S9 | The vla-evaluation-harness runs a model server (`vla-eval serve`) and a benchmark client (`vla-eval run`). The client runs in Docker by default; `--no-docker` and `--runtime charliecloud` exist. Apptainer is not mentioned. | allenai/vla-evaluation-harness README (summary). |
| S10 | In the harness, π0.5 and GR00T N1.7 are marked reproduced and are served through LeRobot (`configs/model_servers/lerobot/pi05_libero.yaml`). SmolVLA is marked "integrated, awaiting first reproduction". | Harness README badges (summary; badge symbols may be misread). |
| S11 | In the harness, LIBERO and LIBERO-Plus are marked reproduced, RoboCasa365 is integrated but not reproduced, and LIBERO-Para is not listed. | Harness README badges (summary). |
| S12 | The harness is Apache-2.0. | Harness README (summary). |
| S13 | Six recent open policies score within 1.58 points on clean LIBERO (96.52% to 98.10%); LIBERO-Long is the most discriminative clean suite. π0.5: 96.52% clean, 85.31% LIBERO-Plus, 71.33% LIBERO-Para, 58.8 ms, 14.6 GB. GR00T N1.7: 97.88%, 75.12%, 74.26%, 19.1 ms, 7.8 GB. | IndustrialVLA-Bench, arXiv 2609.25562 (read by the authors). |
| S14 | Clean LIBERO is shortcut-solvable on three of four suites; about 20% of LIBERO SOTA claims are statistically significant. | arXiv 2606.04233 (read by the authors). |
| S15 | LIBERO-Para reports drops of 22 to 52 points under paraphrased instructions. | arXiv 2603.28301 (read by the authors). |
| S16 | RSI-Jev v6.0-VL is an open 4B model on Qwen3.5-4B-Base with Apache-2.0 weights, unaffiliated with TypeSafe, with self-reported benchmarks. | OrcaRouter blog (secondary source, read by the authors). |

## Assumptions

| ID | Claim | Cheapest check |
| --- | --- | --- |
| A1 | The RTX 8000s (Turing) lack native BF16, so models need FP16 or FP32. | Read which dtype each policy config requests; run one step on the cluster. |
| A2 | Both VLAs give correct results in FP16 on Turing. | Compare clean-suite success on the cluster to published numbers. |
| A3 | Our router can sit as a proxy between the harness's benchmark client and the model servers. | Read the harness client/server protocol code. |
| A4 | MuJoCo state can be saved and restored exactly. | Save state, apply fixed actions, restore, repeat, compare the max absolute state difference. |
| A5 | SmolVLA's action generation is stochastic enough for sampling spread to mean something. | Read the installed policy source for where noise is sampled; run one observation twice with different seeds. |
| A6 | LIBERO-Plus and LIBERO-Para install alongside LIBERO in one environment. | Read each repo's install instructions, then try it. |
| A7 | The LeRobot π0.5 LIBERO checkpoint used by the harness matches openpi's `pi05_libero` in quality. | Read the harness config and the checkpoint's model card. |
| A8 | `tc` needs root and will not be available on the shared cluster; delay can be injected inside the proxy instead. | Ask the cluster admins; test `tc qdisc show`. |
| A9 | At 100 episodes per model, a success gap must be roughly 14 points to be clear of noise at 95% (worst case at 50% success). | Arithmetic: 1.96 * sqrt(2 * 0.25 / 100). Recheck once real success rates are known. |

## Unknown

| ID | Question | Cheapest check |
| --- | --- | --- |
| U1 | **Most important:** the success gap between SmolVLA and π0.5 on LIBERO-Long, LIBERO-Plus and LIBERO-Para on our hardware. | Both models, same seeds, 10 tasks x 10 episodes on LIBERO-Long first. |
| U2 | Latency and peak memory of each model on the RTX 8000s, the Mac (MPS) and the server node. | Log both during the U1 runs; one inference step on each other machine. |
| U3 | Whether the course cluster allows Docker, Apptainer or Charliecloud; its scheduler; whether compute nodes have internet; disk quota. | `docker info`, `apptainer --version`, ask the admins. |
| U4 | Whether LIBERO and SmolVLA run on macOS with MPS. | Install and run one inference step. |
| U5 | Whether π0.5 runs on the server node (aarch64, CUDA 13), through LeRobot or openpi, and which PyTorch build supports the GB10. | Read NVIDIA's documentation for the device, then one inference step. |
| U6 | Whether a container on the server node can access the GPU. | Run `nvidia-smi` inside a CUDA base container. |
| U7 | What uncertainty signal SmolVLA exposes, if any. | Read the installed policy source. |
| U8 | Whether RSI-Jev loads in FP16 on Turing, and whether it takes images or only structured state. | Read its model card first. |
| U9 | Licenses for the π0.5 weights (Gemma terms?), LIBERO, LIBERO-Plus and LIBERO-Para. | Read each LICENSE file. |
| U10 | The harness's actual wire protocol, message format and port. | Read the harness source. |
| U11 | Whether a LIBERO-Para integration exists anywhere, or we must write one. | Find its repo. |
| U12 | Round-trip time between the edge node and the server node on the demo network. | `ping` once the network is chosen. |
