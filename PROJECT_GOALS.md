# LMChess: GRPO Training for Chess

A learning project implementing DeepSeek-style GRPO (Group Relative Policy Optimization) to teach Qwen3-0.6B to play chess using Reinforcement Learning from Verifiable Rewards (RLVR).

---

## Objectives

### Primary: Learn GRPO/RLVR
- Implement GRPO training loop from scratch using HookedTransformers
- Understand group-relative advantage estimation
- Explore reward shaping with verifiable signals (chess engine evaluation)

### Secondary: Competent Chess Model
- Train a model that plays legal, reasonable chess moves
- Benchmark against Stockfish at various depths
- Success metric: Model consistently plays legal moves and shows improvement over baseline

### Tertiary: Interpretability (Optional)
- Probe what the model learns about board state
- Investigate representations of piece positions, threats, material balance
- Leverage HookedTransformer's activation access for analysis

---

## Technical Approach

### Model
- **Base model:** Qwen3-0.6B-Instruct
- **Framework:** TransformerLens (HookedTransformer) for full training loop control
- **Training:** Custom GRPO implementation (no trl dependency)

### Chess Representation
- **Positions:** FEN strings
- **Moves:** UCI notation (e.g., `e2e4`, `g1f3`, `e7e8q`)
- **Input format:**
  ```
  FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
  Move:
  ```

### Data Pipeline
- **Source:** GM games dataset (`./data/gm_games_original.csv`, ~400k games)
- **Usage:** Sample diverse positions from real games (not for move supervision)
- **Position extraction:** Parse UCI move sequences, replay to random positions
- **No self-play:** All positions come from human GM games

### Reward Structure (Verifiable)

Two reward signals to investigate:

1. **Binary reward (best move match)**
   - Query Stockfish for best move at position
   - Reward = 1 if model's move matches Stockfish top-N, else 0
   - Experiment with thresholds: top-1, top-3, within 50cp of best

2. **Continuous reward (evaluation delta)**
   - Compute Stockfish eval before and after model's move
   - Reward = f(eval_after - eval_before) or normalized score
   - Rewards legal moves that don't blunder

Both rewards are fully verifiable—no learned reward model needed.

### GRPO Algorithm

DeepSeek-style Group Relative Policy Optimization:

```
For each position:
    1. Sample K candidate moves from policy π(a|s)
    2. Compute reward r_i for each candidate
    3. Compute group-relative advantage: A_i = (r_i - mean(r)) / std(r)
    4. Update policy to increase probability of high-advantage moves
```

Key components:
- **Group size K:** Hyperparameter to tune (start with 8-16, experiment up to 64)
- **Reference model:** Frozen copy for KL penalty (periodic sync vs EMA—TBD)
- **KL penalty:** Prevent policy from diverging too far from reference

Loss function:
```
L = -E[A_i * log π(a_i|s)] + β * KL(π || π_ref)
```

---

## Project Phases

### Phase 1: Infrastructure
- [ ] Position sampling from GM games dataset
- [ ] FEN extraction and move replay utilities
- [ ] Stockfish integration (python-chess + stockfish binary)
- [ ] Reward computation functions (binary + continuous)
- [ ] Basic evaluation harness (legal move %, win rate vs Stockfish)

### Phase 2: Model Setup
- [ ] Load Qwen3-0.6B-Instruct via HookedTransformer
- [ ] Tokenization pipeline for FEN → Move format
- [ ] Verify model can generate move-shaped outputs
- [ ] Baseline evaluation (untrained model's chess "ability")

### Phase 3: GRPO Implementation
- [ ] Group sampling from policy
- [ ] Advantage computation (group-relative normalization)
- [ ] Policy gradient with advantages
- [ ] KL penalty implementation
- [ ] Reference model management

### Phase 4: Training Loop
- [ ] Batch position sampling
- [ ] Forward pass + move generation
- [ ] Reward computation (batched Stockfish queries)
- [ ] Backward pass + optimizer step
- [ ] Logging (wandb integration per style guide)
- [ ] Checkpointing

### Phase 5: Experiments
- [ ] Binary vs continuous reward comparison
- [ ] Group size ablation (8, 16, 32, 64)
- [ ] KL penalty coefficient tuning
- [ ] Best move threshold experiments (top-1, top-3, within-Xcp)

### Phase 6: Evaluation & Analysis
- [ ] Legal move rate over training
- [ ] Win/draw/loss rate vs Stockfish (depth 1, 5, 10)
- [ ] Elo estimation (if possible)
- [ ] Interpretability probes (optional): board state representations

---

## Infrastructure

### Environment
- **Package manager:** uv
- **Python:** 3.13+
- **Structure:** Flat (Python files at root)

### Hardware
| Environment | GPU | VRAM | Use Case |
|-------------|-----|------|----------|
| Local | RTX 4070 Ti | 12GB | Development, iteration |
| HPC | A100 | 40GB | Training runs |

### Dependencies (initial)
```toml
[project]
dependencies = [
    "torch",
    "transformer-lens",
    "transformers",
    "python-chess",
    "stockfish",
    "pandas",
    "wandb",
    "tqdm",
    "einops",
    "plotly",
]
```

### External
- Stockfish binary (install separately, path configurable)

---

## File Structure (Planned)

```
lmchess/
├── main.py              # Entry point, training loop
├── utils.py             # Helpers, colors, tec()
├── data.py              # Position sampling, FEN extraction
├── rewards.py           # Stockfish integration, reward functions
├── grpo.py              # GRPO algorithm implementation
├── eval.py              # Evaluation harness
├── config.py            # Dataclass configs
├── data/
│   └── gm_games_original.csv
├── checkpoints/         # Model saves
├── PROJECT_GOALS.md     # This file
├── interp_guide.md      # Style guide
└── pyproject.toml
```

---

## Open Questions

1. **Tokenization:** How does Qwen3 tokenize UCI moves? May need move-specific handling.
2. **Position diversity:** How to balance openings/middlegame/endgame sampling?
3. **Stockfish depth:** Higher depth = better signal but slower. What's the sweet spot?
4. **Batch Stockfish:** Can we parallelize engine queries efficiently?
5. **Reference model sync:** EMA vs periodic hard sync—which works better for chess?
6. **Illegal move handling:** Penalize? Mask logits? Rejection sample?

---

## Success Criteria

### Minimum (Learning Goal Achieved)
- Working GRPO training loop
- Model shows improvement in legal move rate over training
- Clear understanding of GRPO mechanics and reward shaping

### Target (Competent Model)
- >95% legal move rate
- Plays reasonable chess (doesn't immediately blunder pieces)
- Beats Stockfish depth-1 consistently

### Stretch
- Meaningful Elo rating (even if low)
- Interpretability insights into learned chess representations
- Ablation studies documented

---

## References

- [DeepSeek-R1 Paper](https://arxiv.org/abs/2401.02954) - GRPO for reasoning
- [DeepSeekMath Paper](https://arxiv.org/abs/2402.03300) - GRPO details
- [TransformerLens Docs](https://neelnanda-io.github.io/TransformerLens/)
- [python-chess](https://python-chess.readthedocs.io/)
