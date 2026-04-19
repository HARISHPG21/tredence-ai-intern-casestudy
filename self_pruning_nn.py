"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Self-Pruning Neural Network — Tredence AI Intern Case Study         ║
║                                                                              ║
║  Implements a feed-forward network that learns to prune its own weights      ║
║  during training via learnable sigmoid gates + L1 sparsity regularization.  ║
║                                                                              ║
║  Sections:                                                                   ║
║    1. PrunableLinear  — custom gated linear layer                            ║
║    2. SelfPruningNet  — full network using prunable layers                   ║
║    3. Data loading    — CIFAR-10 with augmentation                           ║
║    4. Training loop   — cross-entropy + lambda * L1 sparsity loss            ║
║    5. Evaluation      — accuracy, sparsity level, gate stats                 ║
║    6. Visualization   — gate distribution, training curves, comparison plots ║
║    7. Main experiment — sweep over 3 lambda values, full reporting           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dependencies:
    pip install torch torchvision matplotlib numpy tqdm

Usage:
    python self_pruning_nn.py                  # full experiment (all lambda values)
    python self_pruning_nn.py --lam 1e-4       # single lambda run
    python self_pruning_nn.py --epochs 30      # more epochs
    python self_pruning_nn.py --quick          # quick 5-epoch smoke test
"""

import argparse
import os
import time
import math
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ==============================================================================
# 1.  PrunableLinear
# ==============================================================================

class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable per-weight gates.

    For each weight w_ij we maintain a corresponding scalar gate_score g_ij.
    During the forward pass:

        gate_ij  = sigmoid(g_ij)          in (0, 1)
        w_tilde  = w_ij * gate_ij         <- pruned weight (element-wise)
        output   = X @ W_tilde^T + bias

    When gate -> 0  the weight is effectively removed (pruned).
    When gate -> 1  the weight operates at full strength.

    The gate_scores are registered as nn.Parameters so the optimizer updates
    them alongside the regular weights. Autograd handles gradient flow through
    the element-wise multiplication automatically — no custom backward needed.

    Gradient analysis
    -----------------
    dL/dw_ij       = dL/dy_tilde * gate_ij
    dL/dg_ij       = dL/dy_tilde * w_ij * sigmoid'(g_ij)   (task gradient)
                   + lambda * sigmoid'(g_ij)                 (sparsity gradient)

    The sparsity term provides a *constant* pull toward g -> -inf (gate -> 0),
    which is what drives pruning. See REPORT.md for a full derivation.

    Args:
        in_features  (int):  Number of input features.
        out_features (int):  Number of output features.
        bias         (bool): If True, adds a learnable bias. Default: True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias (same as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Learnable gate scores — same shape as weight.
        # Initialised to 0 => sigmoid(0) = 0.5, balanced starting point.
        # The optimizer will push them negative (prune) or positive (keep).
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform init for weight (standard for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: compute continuous gates in (0, 1)
        gates = torch.sigmoid(self.gate_scores)        # shape: (out_features, in_features)

        # Step 2: element-wise multiply weight by gates
        # Gradients flow through both self.weight and self.gate_scores
        pruned_weight = self.weight * gates            # shape: (out_features, in_features)

        # Step 3: standard affine transform
        return F.linear(x, pruned_weight, self.bias)   # shape: (batch, out_features)

    @property
    def gates(self) -> torch.Tensor:
        """Gate values in (0, 1), detached from the computation graph."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of the gate values for this layer.
        Summing gate values penalises the network for keeping gates active,
        incentivising the optimizer to drive them toward zero.
        """
        return torch.sigmoid(self.gate_scores).sum()

    def sparsity_ratio(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below threshold (treated as pruned)."""
        g = self.gates.cpu().numpy().ravel()
        return float((g < threshold).mean())

    def effective_params(self, threshold: float = 1e-2) -> Tuple[int, int]:
        """Returns (active_weights, total_weights) for this layer."""
        total  = self.weight.numel()
        active = int((self.gates.cpu().numpy().ravel() >= threshold).sum())
        return active, total

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ==============================================================================
# 2.  Network
# ==============================================================================

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 classification.

    Architecture:
        Input  ->  Flatten (3072)
               ->  PrunableLinear(3072, 512) -> BN -> ReLU -> Dropout
               ->  PrunableLinear(512,  256) -> BN -> ReLU -> Dropout
               ->  PrunableLinear(256,  128) -> BN -> ReLU -> Dropout
               ->  PrunableLinear(128,   10)               [logits]

    Every linear layer is PrunableLinear so all weights participate in the
    sparsity regularization. Batch normalisation stabilises training when
    many gates are being driven toward zero.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.fc1 = PrunableLinear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = PrunableLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = PrunableLinear(128, 10)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)                    # (B, 3072)
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)                             # (B, 10) logits

    def sparsity_loss(self) -> torch.Tensor:
        """Aggregate L1 gate loss across all PrunableLinear layers."""
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                loss = loss + m.sparsity_loss()
        return loss

    def all_gates(self) -> np.ndarray:
        """Flatten all gate values from every PrunableLinear into one array."""
        parts = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                parts.append(m.gates.cpu().numpy().ravel())
        return np.concatenate(parts)

    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """Overall fraction (%) of pruned weights across the entire network."""
        g = self.all_gates()
        return float((g < threshold).mean()) * 100.0

    def param_summary(self, threshold: float = 1e-2) -> Dict:
        """Per-layer and overall active weight counts."""
        summary = {}
        total_active = total_all = 0
        for name, m in self.named_modules():
            if isinstance(m, PrunableLinear):
                active, total = m.effective_params(threshold)
                summary[name] = {
                    "active": active, "total": total,
                    "sparsity": 100.0 * (1 - active / total)
                }
                total_active += active
                total_all    += total
        summary["overall"] = {
            "active": total_active, "total": total_all,
            "sparsity": 100.0 * (1 - total_active / total_all)
        }
        return summary


# ==============================================================================
# 3.  Data loading
# ==============================================================================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_dataloaders(batch_size:  int = 128,
                    data_dir:    str = "./data",
                    num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) for CIFAR-10.

    Training augmentation: random flip + crop + colour jitter + normalize.
    Test transform: normalize only.
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=transform_train)
    test_ds  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(
        test_ds,  batch_size=512,        shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


# ==============================================================================
# 4.  Training
# ==============================================================================

def train_one_epoch(model:     SelfPruningNet,
                    loader:    DataLoader,
                    optimizer: optim.Optimizer,
                    lam:       float,
                    device:    torch.device) -> Dict[str, float]:
    """
    One full pass over the training set.

    Total Loss = CrossEntropyLoss(logits, labels)
               + lambda * sum_{all PrunableLinear layers} sum_{i,j} sigmoid(g_ij)

    Returns dict with mean cls_loss, sps_loss, tot_loss for the epoch.
    """
    model.train()
    sum_cls = sum_sps = sum_tot = 0.0
    n_batches = len(loader)

    iterable = tqdm(loader, desc="  train", leave=False) if TQDM_AVAILABLE else loader

    for images, labels in iterable:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits   = model(images)
        cls_loss = F.cross_entropy(logits, labels)    # task loss
        sps_loss = model.sparsity_loss()              # L1 gate penalty
        total    = cls_loss + lam * sps_loss          # combined loss

        total.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        sum_cls += cls_loss.item()
        sum_sps += sps_loss.item()
        sum_tot += total.item()

    return {
        "cls_loss": sum_cls / n_batches,
        "sps_loss": sum_sps / n_batches,
        "tot_loss": sum_tot / n_batches,
    }


# ==============================================================================
# 5.  Evaluation
# ==============================================================================

@torch.no_grad()
def evaluate(model:  SelfPruningNet,
             loader: DataLoader,
             device: torch.device) -> float:
    """Returns test accuracy (%) on the given loader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds    = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100.0


# ==============================================================================
# 6.  Visualisation
# ==============================================================================

def plot_all(results: List[Dict], save_dir: str = "."):
    """
    Generates three publication-quality figures and saves them to save_dir.

    Figure 1 — gate_distribution.png
        Histogram of final gate values for the best-accuracy model.
        Successful pruning shows a large spike at 0 and a cluster near 1.

    Figure 2 — training_curves.png
        Per-lambda training loss, test accuracy, and sparsity over epochs.

    Figure 3 — lambda_comparison.png
        Bar charts comparing final accuracy and sparsity across lambda values.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Figure 1: Gate distribution ──────────────────────────────────────────
    best  = max(results, key=lambda r: r["accuracy"])
    gates = best["gates"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Gate Value Distribution  |  lambda={best['lambda']:.0e}  |  "
        f"acc={best['accuracy']:.1f}%  |  sparsity={best['sparsity']:.1f}%",
        fontsize=13, fontweight="bold"
    )

    ax1.hist(gates, bins=200, color="#2563EB", alpha=0.85, edgecolor="none")
    ax1.axvline(0.01, color="#EF4444", ls="--", lw=1.8, label="Prune threshold (0.01)")
    ax1.set_xlabel("Gate Value", fontsize=12)
    ax1.set_ylabel("Count",      fontsize=12)
    ax1.set_title("Full Distribution  [0, 1]", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1)

    pruned_gates = gates[gates < 0.1]
    ax2.hist(pruned_gates, bins=100, color="#10B981", alpha=0.85, edgecolor="none")
    ax2.axvline(0.01, color="#EF4444", ls="--", lw=1.8, label="Prune threshold (0.01)")
    ax2.set_xlabel("Gate Value", fontsize=12)
    ax2.set_ylabel("Count",      fontsize=12)
    ax2.set_title("Zoomed: Gates < 0.1  (pruned region)", fontsize=12)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    p1 = os.path.join(save_dir, "gate_distribution.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {p1}")

    # ── Figure 2: Training curves ─────────────────────────────────────────────
    n   = len(results)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 11))
    if n == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Training Curves per Lambda", fontsize=14, fontweight="bold", y=1.01)

    colors = ["#2563EB", "#D97706", "#16A34A", "#9333EA"]

    for col, (r, c) in enumerate(zip(results, colors)):
        ep = range(1, len(r["history"]["tot_loss"]) + 1)

        axes[0, col].plot(ep, r["history"]["tot_loss"], color=c, lw=2)
        axes[0, col].set_title(f"lambda = {r['lambda']:.0e}", fontsize=12, fontweight="bold")
        axes[0, col].set_ylabel("Total Loss"        if col == 0 else "", fontsize=11)
        axes[0, col].set_xlabel("Epoch", fontsize=10)
        axes[0, col].grid(alpha=0.3)

        axes[1, col].plot(ep, r["history"]["accuracy"], color=c, lw=2)
        axes[1, col].set_ylabel("Test Accuracy (%)" if col == 0 else "", fontsize=11)
        axes[1, col].set_xlabel("Epoch", fontsize=10)
        axes[1, col].set_ylim(0, 70)
        axes[1, col].grid(alpha=0.3)

        axes[2, col].plot(ep, r["history"]["sparsity"], color=c, lw=2)
        axes[2, col].set_ylabel("Sparsity (%)"      if col == 0 else "", fontsize=11)
        axes[2, col].set_xlabel("Epoch", fontsize=10)
        axes[2, col].set_ylim(0, 100)
        axes[2, col].grid(alpha=0.3)

    plt.tight_layout()
    p2 = os.path.join(save_dir, "training_curves.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {p2}")

    # ── Figure 3: Lambda comparison ───────────────────────────────────────────
    labels     = [f"lambda={r['lambda']:.0e}" for r in results]
    accs       = [r["accuracy"] for r in results]
    sparsities = [r["sparsity"] for r in results]
    x          = np.arange(len(results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Sparsity vs. Accuracy Trade-off", fontsize=14, fontweight="bold")

    bar_colors = colors[:len(results)]

    bars1 = ax1.bar(x, accs, color=bar_colors, edgecolor="white", linewidth=1.2)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_ylim(0, 65)
    ax1.bar_label(bars1, fmt="%.1f%%", padding=4, fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Final Test Accuracy", fontsize=12)

    bars2 = ax2.bar(x, sparsities, color=bar_colors, edgecolor="white", linewidth=1.2)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("Sparsity Level (%)", fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.bar_label(bars2, fmt="%.1f%%", padding=4, fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("Final Sparsity Level", fontsize=12)

    plt.tight_layout()
    p3 = os.path.join(save_dir, "lambda_comparison.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {p3}")


# ==============================================================================
# 7.  Main experiment
# ==============================================================================

def run_experiment(lam:          float,
                   train_loader: DataLoader,
                   test_loader:  DataLoader,
                   device:       torch.device,
                   epochs:       int = 20) -> Dict:
    """
    Trains one SelfPruningNet with the given lambda and returns a result dict.
    """
    print(f"\n{'─'*60}")
    print(f"  lambda = {lam:.0e}   |   epochs = {epochs}   |   device = {device}")
    print(f"{'─'*60}")

    model     = SelfPruningNet(dropout=0.3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "tot_loss": [], "cls_loss": [], "sps_loss": [],
        "accuracy": [], "sparsity": []
    }

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        metrics  = train_one_epoch(model, train_loader, optimizer, lam, device)
        acc      = evaluate(model, test_loader, device)
        sparsity = model.global_sparsity()
        scheduler.step()

        history["tot_loss"].append(metrics["tot_loss"])
        history["cls_loss"].append(metrics["cls_loss"])
        history["sps_loss"].append(metrics["sps_loss"])
        history["accuracy"].append(acc)
        history["sparsity"].append(sparsity)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:>3}/{epochs}  |  "
                  f"loss={metrics['tot_loss']:.4f}  |  "
                  f"acc={acc:.2f}%  |  "
                  f"sparsity={sparsity:.1f}%  |  "
                  f"time={elapsed:.0f}s")

    final_acc      = history["accuracy"][-1]
    final_sparsity = history["sparsity"][-1]
    gates          = model.all_gates()

    # Per-layer breakdown
    print(f"\n  Layer-wise sparsity (lambda={lam:.0e}):")
    summary = model.param_summary()
    for name, info in summary.items():
        if name == "overall":
            continue
        print(f"    {name:<8}  active={info['active']:>6,} / {info['total']:>6,}"
              f"  ({info['sparsity']:.1f}% pruned)")
    ov = summary["overall"]
    print(f"    {'TOTAL':<8}  active={ov['active']:>6,} / {ov['total']:>6,}"
          f"  ({ov['sparsity']:.1f}% pruned)")

    return {
        "lambda":   lam,
        "accuracy": final_acc,
        "sparsity": final_sparsity,
        "gates":    gates,
        "history":  history,
        "model":    model,
    }


def print_results_table(results: List[Dict]):
    print("\n\n" + "=" * 57)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 57)
    print(f"  {'Lambda':<10}  {'Test Accuracy':>15}  {'Sparsity (%)':>14}")
    print("-" * 57)
    for r in results:
        print(f"  {r['lambda']:<10.0e}  {r['accuracy']:>14.2f}%  {r['sparsity']:>13.1f}%")
    print("=" * 57)

    best_acc = max(results, key=lambda r: r["accuracy"])
    best_sps = max(results, key=lambda r: r["sparsity"])
    print(f"\n  Best accuracy  : lambda={best_acc['lambda']:.0e}"
          f"  ->  {best_acc['accuracy']:.2f}%")
    print(f"  Most sparse    : lambda={best_sps['lambda']:.0e}"
          f"  ->  {best_sps['sparsity']:.1f}%\n")


def parse_args():
    p = argparse.ArgumentParser(description="Self-Pruning Neural Network on CIFAR-10")
    p.add_argument("--lam",    type=float, default=None,
                   help="Single lambda value (skips the full sweep)")
    p.add_argument("--epochs", type=int,   default=20,
                   help="Training epochs per lambda (default: 20)")
    p.add_argument("--batch",  type=int,   default=128,
                   help="Batch size (default: 128)")
    p.add_argument("--quick",  action="store_true",
                   help="Smoke test: 5 epochs, batch=256")
    p.add_argument("--outdir", type=str,   default=".",
                   help="Directory to save plots (default: current directory)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.quick:
        args.epochs = 5
        args.batch  = 256
        print("Quick mode: 5 epochs, batch=256")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch, num_workers=2)
    print(f"CIFAR-10 loaded  |  "
          f"train={len(train_loader.dataset):,}  "
          f"test={len(test_loader.dataset):,}")

    lambdas = [args.lam] if args.lam else [1e-5, 1e-4, 1e-3]
    results = []

    for lam in lambdas:
        res = run_experiment(lam, train_loader, test_loader, device, args.epochs)
        results.append(res)

    print_results_table(results)

    print("Generating plots...")
    plot_all(results, save_dir=args.outdir)

    print("Done! Check the output directory for plots.")
