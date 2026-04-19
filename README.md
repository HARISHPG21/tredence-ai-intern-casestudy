# Self-Pruning Neural Network – Tredence Case Study

A feed-forward neural network that **learns to prune itself during training** using learnable sigmoid gates and L1 sparsity regularization, trained on CIFAR-10.

## Quick Start

```bash
pip install torch torchvision matplotlib numpy
python self_pruning_nn.py
```

CIFAR-10 is downloaded automatically (~170 MB).

## What's Inside

| File | Description |
|------|-------------|
| `self_pruning_nn.py` | Complete solution – PrunableLinear, network, training loop, evaluation |
| `REPORT.md` | Analysis: why L1 encourages sparsity, results table, gate distribution explanation |
| `gate_distribution.png` | Generated after training – shows pruned vs. active gates |

## Key Concepts

- **PrunableLinear**: Custom `nn.Module` with learnable `gate_scores`. Each weight is multiplied by `sigmoid(gate_score)`. Gates near 0 = pruned.
- **Sparsity Loss**: L1 norm of all gate values, added to CrossEntropy loss.
- **λ (lambda)**: Controls the sparsity–accuracy trade-off. Three values tested: `1e-5`, `1e-4`, `1e-3`.

## Results

| Lambda | Test Accuracy | Sparsity |
|--------|:---:|:---:|
| 1e-5 (low) | ~48% | ~15% |
| 1e-4 (medium) | ~46% | ~50% |
| 1e-3 (high) | ~41% | ~82% |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib, numpy
