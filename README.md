# ReMap-DP

This repository contains the official PyTorch implementation for **ReMap-DP**.

## TODO LIST
- [x] Release evaluation code.
- [x] Release checkpoints on huggingface.
- [x] Release training code.
- [ ] Release training Dataset on huggingface.

## Installation
1. Clone the repository:
   ```bash
   git clone -b main https://github.com/ICR-Lab/ReMAP-DP.git
   cd ReMAP-DP
   ```

2. Install the required dependencies:
   ```bash
   conda create -n Remap-DP python=3.10
   conda activate Remap-DP
   pip install -r requirements.txt
   ```

## Checkpoints and Dataset
Ckpts can be downloaded at [Here](https://huggingface.co/ICR-Lab/ReMAP-DP) from Huggingface.
Datasets coming soon.

## Training
1. Review or modify the training configurations in `diffusion_policy/config/dp.yaml` and `diffusion_policy/config/train_dp_robotwin.yaml`.
2. Start training using the `train_policy.sh` script:

   ```bash
   # Example:
   bash ./train_policy.sh train_dp_robotwin beat_block_hammer 0211 0 0
   ```
   
   **Arguments explanation:**
   - **`$1`**: Algorithm configuration name (e.g., `train_dp_robotwin`)
   - **`$2`**: Task name (e.g., `beat_block_hammer`)
   - **`$3`**: Additional information / run tag (e.g., date `0211`)
   - **`$4`**: Random seed (e.g., `0`)
   - **`$5`**: GPU ID (e.g., `0`)

## Evaluation

1. Clone [**RoboTwin 2.0**](https://github.com/RoboTwin-Platform/RoboTwin) from GitHub and Install necessary requirements according to RoboTwin 2.0 Repo.
   ```bash
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git
   ```

2. Follow the [RoboTwin 2.0 Deployment Documentation](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html). You will need to copy the `ReMAP-DP` directory to RoboTwin directory `policy/`.
   
   Your directory structure should look similar to this:
   ```text
   ReMap-DP/
   ├── deploy_policy.py
   ├── deploy_policy.yml
   ├── diffusion_policy/
   ├── eval.sh
   └── __init__.py
   ```

3. Specify your checkpoint path in `deploy_policy.yml`:
   ```yaml
   policy_name: Remap-DP
   # ...
   ckpt_path: /path/to/your/checkpoint.ckpt
   # ...
   ```

4. Run the evaluation script:
   ```bash
   cd RoboTwin/policy/ReMAP-DP
   # Example: 
   bash eval.sh beat_block_hammer demo_clean default 0 0
   ```

## Acknowledgements

This codebase is built upon the foundation of [**Diffusion Policy**](https://github.com/real-stanford/diffusion_policy). We sincerely thank the original authors for their groundbreaking work and excellent open-source contribution to the robotics community.
