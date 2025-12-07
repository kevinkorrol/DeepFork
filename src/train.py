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
                torch.tensor(sample["action"], dtype=torch.float32),
                torch.tensor(sample["result"], dtype=torch.float32),
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


class AZLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, policy_logits, policy_target):
        log_probs = torch.nn.functional.log_softmax(policy_logits, dim=1)
        loss = self.policy_loss_fn(log_probs, policy_target)
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
    val_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = AZLoss()
    model.to(device)

    total_len = n_samples // samples_per_file if n_samples is not None else len(train_loader)

    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        batches = 0
        for _, (states, actions) in tqdm(enumerate(train_loader, 0), unit="batch", total=total_len):
            states = states.to(device)
            policy_targets = actions.to(device)

            optimizer.zero_grad()
            policy_logits = model(states)
            loss, loss_policy_val, loss_value_val = criterion(policy_logits, policy_targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            batches += 1

        avg_train_loss = training_loss / batches
        train_history.append(avg_train_loss)

        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(device)
                policy_targets = actions.to(device)

                policy_logits = model(states)
                loss, _, _ = criterion(policy_logits, policy_targets)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_history.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} — "
            f"Train loss: {avg_train_loss:.4f}, "
            f"Val loss: {avg_val_loss:.4f}"
        )

    return train_history, val_history


if __name__ == "__main__":
    model = DeepForkNet(depth=4, filter_count=64, history_size=1)
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    epochs = 50
    n_samples = None
    batch_size = 512
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'
    train_loss, val_loss = train_model(model, processed_dir, epochs, batch_size, device=device, n_samples=n_samples)

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, marker='o', label="Train Loss")
    plt.plot(val_loss, marker='s', label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training + Validation Loss")
    plt.grid(True)
    plt.legend()

    output_dir = root / "model_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"Loss_vs_Epoch_{datetime.datetime.today().strftime('%Y-%m-%d')}.png"
    plt.savefig(output_dir / filename)

    save_path = root / "models" / "checkpoints"
    model_name = f"{epochs}epochs_{'all' if n_samples is None else n_samples}samples_{batch_size}batch_size.pt"
    torch.save(model.state_dict(), save_path / model_name)
