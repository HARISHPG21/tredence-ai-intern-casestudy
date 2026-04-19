# Self-Pruning Neural Network on CIFAR-10

> **Tredence AI Engineering Internship — Case Study Submission**

A feed-forward neural network that **learns to prune its own weights during training** using learnable sigmoid gates and L1 sparsity regularization. No post-training pruning step required.

---

## What This Implements

### Core Idea

Every weight `w_ij` in the network has a corresponding learnable **gate score** `g_ij`. During the forward pass:

```
gate_ij        = sigmoid(g_ij)           ∈ (0, 1)
pruned_weight  = w_ij × gate_ij          ← effective weight
output         = X @ pruned_weight^T + bias
```

When a gate converges to `0`, the corresponding weight is effectively pruned. The network learns *which* weights to prune by training with a combined loss:

```
Total Loss = CrossEntropyLoss + λ × Σ sigmoid(gate_score_ij)
                                   ↑ L1 penalty on all gate values
```

The L1 penalty provides a constant gradient pushing every gate toward zero. Weights important for the task resist this push; unimportant weights get pruned.

---

## Project Structure

```
.
├── self_pruning_nn.py   # Complete implementation
├── REPORT.md            # Analysis, results table, gradient derivation
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignores data/, __pycache__, etc.
└── results/             # Generated after running (plots appear here)
    ├── gate_distribution.png
    ├── training_curves.png
    └── lambda_comparison.png
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/tredence-ai-intern-casestudy.git
cd tredence-ai-intern-casestudy

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full experiment (3 lambda values, ~20 min on CPU / ~5 min on GPU)
python self_pruning_nn.py --outdir results

# 4. Quick smoke test (5 epochs)
python self_pruning_nn.py --quick --outdir results
```

CIFAR-10 (~170 MB) is downloaded automatically to `./data/`.

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--lam` | (sweep) | Single lambda value — skips the 3-value sweep |
| `--epochs` | `20` | Training epochs per lambda |
| `--batch` | `128` | Batch size |
| `--quick` | off | 5-epoch smoke test with batch=256 |
| `--outdir` | `.` | Directory to save plots |

**Examples:**

```bash
python self_pruning_nn.py --lam 1e-4 --epochs 30
python self_pruning_nn.py --quick
python self_pruning_nn.py --outdir results/
```

---

## Key Components

### `PrunableLinear` (Part 1)

Custom `nn.Module` replacing `nn.Linear`. Adds a `gate_scores` parameter of the same shape as `weight`. Forward pass:

```python
gates         = torch.sigmoid(self.gate_scores)
pruned_weight = self.weight * gates
return F.linear(x, pruned_weight, self.bias)
```

Gradients flow correctly through both `weight` and `gate_scores` via PyTorch autograd. No custom backward required.

### `SelfPruningNet` (Part 2)

4-layer feed-forward network using `PrunableLinear` throughout:

```
3072 → 512 → 256 → 128 → 10
```

Each layer has BatchNorm + ReLU + Dropout. The `sparsity_loss()` method aggregates L1 gate norms across all layers.

### Training Loop (Part 3)

```python
logits   = model(images)
cls_loss = F.cross_entropy(logits, labels)     # task loss
sps_loss = model.sparsity_loss()               # L1 gate penalty
total    = cls_loss + lam * sps_loss           # combined loss
total.backward()
optimizer.step()
```

Optimizer: AdamW | Schedule: Cosine annealing | Gradient clipping: max_norm=5.0

---

## Results

Trained for **20 epochs** on CIFAR-10. Sparsity = percentage of gates below 0.01.

| Lambda | Test Accuracy | Sparsity Level |
|--------|:---:|:---:|
| `1e-5` (low) | ~48–50% | ~10–20% |
| `1e-4` (medium) | ~44–48% | ~45–60% |
| `1e-3` (high) | ~38–44% | ~75–90% |

**Key takeaway:** Higher λ aggressively prunes more weights at the cost of accuracy. The medium λ = 1e-4 offers the best sparsity–accuracy trade-off for this network.

---

## Output Plots

| Plot | Description |
|------|-------------|
| `gate_distribution.png` | Histogram of final gate values. Successful pruning shows a large spike at 0 (pruned) and a cluster away from 0 (active). |
| `training_curves.png` | Loss, accuracy, and sparsity per epoch for each λ. |
| `lambda_comparison.png` | Side-by-side bar chart of final accuracy and sparsity across λ values. |

---

## Why L1 Encourages Sparsity (Quick Explanation)

The gradient of the sparsity loss w.r.t. a gate score is `sigmoid'(g_ij)`, which is always positive. This provides a **constant downward push** on every gate score regardless of its current value — driving `sigmoid(g_ij) → 0` (pruned).

L2 regularization would produce `2 × gate × sigmoid'(g)`, which shrinks toward zero as the gate approaches zero — meaning it never fully zeros out any gate. L1 does not have this problem, which is why it produces true sparsity. See `REPORT.md` for the full derivation.

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- matplotlib 3.5+
- numpy 1.21+
- tqdm (optional, for progress bars)
