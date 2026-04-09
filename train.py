import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

from dataset1 import IcopeDataset, ValTestDataset
from model import CNN_LSTM

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR     = "dataset_frames"
SEQUENCE_LENGTH = 16
BATCH_SIZE      = 8
EPOCHS          = 40
LR              = 1e-4
WEIGHT_DECAY    = 1e-5
EARLY_STOP_PAT  = 8
SEED            = 42
NUM_WORKERS     = 0   # set to 0 if you get DataLoader errors on Windows

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ── Stratified 70 / 15 / 15 split ────────────────────────────────────────────
full_dataset = IcopeDataset(DATASET_DIR, sequence_length=SEQUENCE_LENGTH, train=True)

all_indices = list(range(len(full_dataset)))
all_labels  = [full_dataset.samples[i][1] for i in all_indices]

# Split off test first
train_val_idx, test_idx = train_test_split(
    all_indices, test_size=0.15, stratify=all_labels, random_state=SEED
)
# Split remaining into train / val
train_val_labels = [all_labels[i] for i in train_val_idx]
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=0.176,          # 0.176 × 85% ≈ 15% of total
    stratify=train_val_labels, random_state=SEED
)

from torch.utils.data import Subset
train_subset = Subset(full_dataset, train_idx)
val_set      = ValTestDataset(Subset(full_dataset, val_idx),  SEQUENCE_LENGTH)
test_set     = ValTestDataset(Subset(full_dataset, test_idx), SEQUENCE_LENGTH)

print(f"Split — Train: {len(train_subset)} | Val: {len(val_set)} | Test: {len(test_set)}")

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_set,      batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_set,     batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)


# ── Class weights (handles imbalance) ────────────────────────────────────────
train_labels  = [all_labels[i] for i in train_idx]
n_no_pain     = train_labels.count(0)
n_pain        = train_labels.count(1)
total_train   = len(train_labels)
class_weights = torch.tensor([
    total_train / (2.0 * n_no_pain),
    total_train / (2.0 * n_pain),
], dtype=torch.float).to(device)
print(f"Class weights → no_pain: {class_weights[0]:.3f} | pain: {class_weights[1]:.3f}")


# ── Model ─────────────────────────────────────────────────────────────────────
model     = CNN_LSTM(hidden_size=256, num_layers=2, lstm_dropout=0.3).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)
# Cosine annealing: smooth LR decay, no manual plateau tuning needed
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for inputs, labels_batch in loader:
            inputs       = inputs.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss    = criterion(outputs, labels_batch)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            preds_all.extend(torch.argmax(outputs, 1).cpu().numpy())
            labels_all.extend(labels_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(labels_all, preds_all)
    return avg_loss, acc, preds_all, labels_all


# ── Training loop ─────────────────────────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0
train_losses, val_losses, val_accs = [], [], []

print("\n" + "─" * 60)
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc, _, _          = run_epoch(train_loader, train=True)
    val_loss,   val_acc,   _, val_labels = run_epoch(val_loader,   train=False)

    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {lr_now:.2e}")

    # Save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        }, "models/best_model.pth")
        print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PAT:
            print(f"\n⏹  Early stopping triggered at epoch {epoch}")
            break

print("─" * 60)


# ── Test evaluation ───────────────────────────────────────────────────────────
print("\n📊 Loading best model for test evaluation...")
checkpoint = torch.load("models/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model"])

test_loss, test_acc, test_preds, test_labels = run_epoch(test_loader, train=False)

print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=["no_pain", "pain"]))


# ── Plots ─────────────────────────────────────────────────────────────────────
epochs_ran = range(1, len(train_losses) + 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Loss curves
axes[0].plot(epochs_ran, train_losses, label="Train loss")
axes[0].plot(epochs_ran, val_losses,   label="Val loss")
axes[0].set_title("Loss curves")
axes[0].set_xlabel("Epoch")
axes[0].legend()

# Accuracy curve
axes[1].plot(epochs_ran, val_accs, label="Val accuracy", color="green")
axes[1].set_title("Validation accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylim(0, 1)
axes[1].legend()

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True, fmt="d", ax=axes[2],
            xticklabels=["no_pain", "pain"],
            yticklabels=["no_pain", "pain"],
            cmap="Blues")
axes[2].set_title("Confusion matrix (test set)")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("results/training_summary.png", dpi=150)
plt.show()
print("\n✅ Saved results/training_summary.png")