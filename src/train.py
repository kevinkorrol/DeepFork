import datetime
import math
import torch
from torch.utils.data import DataLoader, IterableDataset
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DeepForkNet
import os
from data_preprocessing import get_project_root


class ChessDataset(IterableDataset):
    def __init__(self, processed_dir: str, samples_per_file: int, n_samples: int, files=None):
        if files is not None:
            self.files = files
        else:
            if n_samples is None:
                self.files = sorted(Path(processed_dir).glob("*.pt"))
            else:
                self.files = sorted(Path(processed_dir).glob("*.pt"))[:math.ceil(n_samples / samples_per_file)]
        self.samples_per_file = samples_per_file
        self.count = 0
        print(f"Dataset initialized with {len(self.files)} files")

    def _yield_file(self, path):
        data = torch.load(path)
        for sample in data:
            self.count += 1
            yield (
                torch.tensor(sample["state"], dtype=torch.float32),
                torch.tensor(sample["action"], dtype=torch.long)
            )

    def __len__(self):
        return (len(self.files) - 1) * self.samples_per_file + len(torch.load(self.files[-1]))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            files = self.files
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            files = self.files[worker_id::num_workers]

        for f in files:
            yield from self._yield_file(f)


class PolicyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_loss_fn = nn.CrossEntropyLoss()

    def forward(self, policy_probs, policy_target):
        loss = self.policy_loss_fn(policy_probs, policy_target)
        return loss



def train_model(model, processed_dir, epochs=5, batch_size=32, lr=1e-3, device='cuda', samples_per_file=300,
                n_samples=None, val_split=0.2):
    all_files = sorted(Path(processed_dir).glob("*.pt"))
    if n_samples is not None:
        all_files = all_files[:math.ceil(n_samples / samples_per_file)]

    n_val = max(1, int(len(all_files) * val_split))
    train_files = all_files[:-n_val]
    val_files = all_files[-n_val:]

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    train_dataset = ChessDataset(processed_dir, samples_per_file, n_samples, files=train_files)
    val_dataset = ChessDataset(processed_dir, samples_per_file, n_samples, files=val_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=device == 'cuda',
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=device == 'cuda',
        persistent_workers=True,
        prefetch_factor=2,
    )

    train_history = []
    train_accuracy_history = []
    val_history = []
    val_accuracy_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = PolicyLoss()
    model.to(device)

    total_len = n_samples // batch_size if n_samples is not None else len(train_loader) // batch_size

    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        train_batches = 0
        train_samples = 0
        train_hits = 0
        for state, action in tqdm(train_loader, unit="batch", total=total_len*(1 - val_split)):
            state = state.to(device)
            policy_targets = action.to(device)

            optimizer.zero_grad()
            policy_probs = model(state)
            loss = criterion(policy_probs, policy_targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            train_batches += 1
            train_samples += policy_targets.size(0)

            predicted_action = torch.argmax(policy_probs, dim=1)
            correct_predictions = (predicted_action == policy_targets).sum().item()
            train_hits += correct_predictions

        avg_train_loss = training_loss / train_batches
        avg_train_accuracy = train_hits / train_samples
        train_history.append(avg_train_loss)
        train_accuracy_history.append(avg_train_accuracy)

        model.eval()
        val_loss = 0
        val_batches = 0
        val_samples = 0
        val_hits = 0
        with torch.no_grad():
            for state, action in tqdm(val_loader, unit="batch", total=total_len*val_split):
                state = state.to(device)
                policy_targets = action.to(device)

                policy_probs = model(state)
                loss = criterion(policy_probs, policy_targets)

                val_loss += loss.item()
                val_batches += 1
                val_samples += policy_targets.size(0)

                predicted_action = torch.argmax(policy_probs, dim=1)
                correct_predictions = (predicted_action == policy_targets).sum().item()
                val_hits += correct_predictions

        avg_val_loss = val_loss / val_batches
        avg_val_accuracy = val_hits / val_samples
        val_accuracy_history.append(avg_val_accuracy)
        val_history.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Train loss: {avg_train_loss:.4f} "
            f"Val loss: {avg_val_loss:.4f} "
            f"Train accuracy: {avg_train_accuracy:.4f} "
            f"Val accuracy: {avg_val_accuracy:.4f}"
        )

    return train_history, val_history, train_accuracy_history, val_accuracy_history


if __name__ == "__main__":
    torch.manual_seed(283)
    model = DeepForkNet(depth=5, filter_count=128, history_size=1)
    root = get_project_root()
    processed_dir = root / "data" / "processed"

    epochs = 20
    n_samples = 334_438
    batch_size = 512

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'
    train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history = train_model(model, processed_dir, epochs, batch_size, device=device, n_samples=n_samples)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(train_loss_history, marker='o', label="Train Loss")
    ax1.plot(val_loss_history, marker='s', label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss Over Epochs")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(train_accuracy_history, marker='o', label="Train Accuracy")
    ax2.plot(val_accuracy_history, marker='s', label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy Over Epochs")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    output_dir = root / "model_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"Metrics_vs_Epoch_{datetime.datetime.today().strftime('%Y-%m-%d')}.png"
    plt.savefig(output_dir / filename)

    # Save the accuracy plot
    filename_acc = f"Accuracy_vs_Epoch_{datetime.datetime.today().strftime('%Y-%m-%d')}.png"
    plt.savefig(output_dir / filename_acc)

    save_path = root / "models" / "checkpoints"
    model_name = f"{epochs}epochs_{'all' if n_samples is None else n_samples}samples_{batch_size}batch_size.pt"
    torch.save(model.state_dict(), save_path / model_name)
