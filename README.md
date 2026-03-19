# ReMap-DP

This repository contains the official PyTorch implementation for **ReMap-DP**.

## TODO LIST
- [x] Release evaluation code.
- [ ] Release checkpoints on huggingface.
- [ ] Release training code.
- [ ] Release training Dataset on huggingface.

## Checkpoints and Dataset
coming soon

## Training
coming soon

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
