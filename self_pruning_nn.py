"""
Self-Pruning Neural Network on CIFAR-10
Tredence AI Engineer Intern – Case Study Solution

This script implements:
  1. PrunableLinear – a custom linear layer with learnable sigmoid gates
  2. SparsityLoss   – L1 penalty on gate values to drive pruning
  3. Training loop  – trains on CIFAR-10 with three lambda values
  4. Evaluation     – reports test accuracy and sparsity level per lambda
  5. Plotting       – gate-value distribution for the best model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Part 1 – PrunableLinear
# ──────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that multiplies each weight by a
    learnable sigmoid gate.  When a gate converges to 0 the corresponding
    weight is effectively pruned.

    Forward pass:
        gates        = sigmoid(gate_scores)          # values in (0, 1)
        pruned_w     = weight * gates                # element-wise
        output       = x @ pruned_w.T + bias
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias (same as nn.Linear)
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores – same shape as weight.
        # Initialised near 0 so that sigmoid(gate_scores) ≈ 0.5 at the start,
        # giving the optimiser room to push them toward 0 (pruned) or 1 (kept).
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming initialisation for the weight (good default for ReLU nets)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute continuous gates in (0, 1)
        gates = torch.sigmoid(self.gate_scores)          # (out, in)

        # Mask the weights – gradients flow through both weight and gate_scores
        pruned_weights = self.weight * gates             # (out, in)

        # Standard affine transform
        return F.linear(x, pruned_weights, self.bias)   # (batch, out)

    def get_gates(self) -> torch.Tensor:
        """Return gate values (detached) for analysis."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of gate values for this layer."""
        return torch.sigmoid(self.gate_scores).abs().sum()


# ──────────────────────────────────────────────────────────────────────────────
# Part 2 – Network definition
# ──────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Simple feed-forward network for CIFAR-10 (32×32×3 → 10 classes).
    All linear layers use PrunableLinear so that gates can be learned.
    """

    def __init__(self):
        super().__init__()
        # Input: 3072 (flattened 32×32×3)
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512,  256)
        self.fc3 = PrunableLinear(256,  128)
        self.fc4 = PrunableLinear(128,  10)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)                  # logits (no softmax – use CrossEntropy)

    def sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 gate norms across all PrunableLinear layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                total = total + module.sparsity_loss()
        return total

    def all_gates(self) -> torch.Tensor:
        """Concatenate gate values from every PrunableLinear for analysis."""
        parts = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                parts.append(module.get_gates().flatten())
        return torch.cat(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Part 3 – Data loading
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_ds  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Part 3 – Training & evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lam, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # Total Loss = Classification Loss + λ × Sparsity Loss
        cls_loss     = F.cross_entropy(logits, labels)
        sparse_loss  = model.sparsity_loss()
        loss         = cls_loss + lam * sparse_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100   # accuracy %


@torch.no_grad()
def compute_sparsity(model, threshold: float = 1e-2) -> float:
    """Percentage of gates below the threshold (treated as pruned)."""
    gates = model.all_gates().cpu().numpy()
    pruned = (gates < threshold).sum()
    return pruned / gates.size * 100


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(lam: float, train_loader, test_loader, device,
                   epochs: int = 20) -> dict:
    print(f"\n{'='*55}")
    print(f"  Training with λ = {lam}")
    print(f"{'='*55}")

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            acc      = evaluate(model, test_loader, device)
            sparsity = compute_sparsity(model)
            print(f"  Epoch {epoch:>3} | loss={avg_loss:.4f} | "
                  f"acc={acc:.2f}% | sparsity={sparsity:.1f}%")

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = compute_sparsity(model)
    gates          = model.all_gates().cpu().numpy()

    print(f"\n  ✅  Final → Accuracy: {final_acc:.2f}%  |  "
          f"Sparsity: {final_sparsity:.1f}%")

    return {
        "lambda":   lam,
        "accuracy": final_acc,
        "sparsity": final_sparsity,
        "gates":    gates,
        "model":    model,
    }


def plot_gate_distribution(result: dict, save_path: str = "gate_distribution.png"):
    gates = result["gates"]
    lam   = result["lambda"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(gates, bins=100, color="#4C72B0", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Gate Value", fontsize=13)
    ax.set_ylabel("Count",      fontsize=13)
    ax.set_title(
        f"Gate Value Distribution  (λ={lam})\n"
        f"Accuracy: {result['accuracy']:.2f}%  |  "
        f"Sparsity: {result['sparsity']:.1f}%",
        fontsize=13,
    )
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5,
               label="Prune threshold (0.01)")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 Gate distribution plot saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=128)

    # Three lambda values: low / medium / high
    lambdas = [1e-5, 1e-4, 1e-3]
    results = []

    for lam in lambdas:
        res = run_experiment(lam, train_loader, test_loader, device, epochs=20)
        results.append(res)

    # ── Results table ──────────────────────────────────────────────────────
    print("\n\n" + "="*55)
    print("  RESULTS SUMMARY")
    print("="*55)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>14}")
    print("-"*55)
    for r in results:
        print(f"  {r['lambda']:<12} {r['accuracy']:>14.2f}% {r['sparsity']:>13.1f}%")
    print("="*55)

    # ── Plot gate distribution for best accuracy model ─────────────────────
    best = max(results, key=lambda r: r["accuracy"])
    plot_gate_distribution(best, save_path="gate_distribution.png")

    print("\nDone! 🎉  Check gate_distribution.png for the plot.")
