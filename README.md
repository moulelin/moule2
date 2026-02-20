# VtD: Verification-then-Distillation for Reasoning Models

Train a small reasoning model by distilling from a larger teacher, guided by answer verification.

## Motivation

On-policy RL methods (GRPO, REINFORCE++) for reasoning treat each rollout as a bandit problem: reward correct responses and penalize wrong ones. But the ground-truth labels that *verify* correctness are never used beyond a scalar reward signal. This is wasteful: when we already know *whether* a response is correct, we can do much more than just assign +1/−1.

VtD exploits this verification signal to route each response to a targeted learning objective, turning wasted label information into structured supervision.

## Core Idea

For each prompt, the student generates K responses and verifies them against ground-truth labels:

- **Correct** → **Distillation**: minimize KL divergence between student and teacher logits on the student's own correct chain-of-thought
- **Incorrect** → **Contrastive (DPO)**: use reference CoT as chosen, student's wrong response as rejected

This creates a self-improving loop: the student learns *how to think* from the teacher (not just *what to answer*), while contrastive loss steers it away from wrong reasoning paths.

## Project Structure

```
scripts/
  train_vtd_ray.sh          # VtD training (Ray + vLLM + DeepSpeed)
  eval/
    eval_aime24.py           # AIME 2024 evaluation
    eval_aime25.py           # AIME 2025 evaluation
    eval_hmmt25.py           # HMMT February 2025 evaluation
    eval_amo_bench.py        # AMO-Bench evaluation
    run_baselines.sh         # Run all baselines
    generate_latex_table.py  # Format results to LaTeX
OpenRLHF/                    # Training framework (modified)
  openrlhf/trainer/
    vtd_trainer_ray.py       # VtD training loop
    vtd_trainer.py           # VtD loss computation
    ray/vtd_actor.py         # Student/Teacher Ray actors
```

## Quick Start

### Training

```bash
# Train Qwen3-1.7B with Qwen3-8B as teacher on GSM8K (4× H100)
bash scripts/train_vtd_ray.sh
```

Key hyperparameters:
- `--vtd_distill_alpha 5.0` — distillation loss weight
- `--vtd_contrast_beta 0.1` — contrastive loss weight
- `--n_samples_per_prompt 8` — responses per prompt for verification

### Evaluation

```bash
# Single dataset
python scripts/eval/eval_aime24.py --model Qwen/Qwen3-1.7B --mode greedy
python scripts/eval/eval_aime25.py --model Qwen/Qwen3-1.7B --mode majority --n_samples 16

# All baselines (3 models × 4 datasets × 2 modes)
bash scripts/eval/run_baselines.sh
```

Evaluation modes: `greedy` (pass@1), `majority` (maj@N), `average` (avg@N).
