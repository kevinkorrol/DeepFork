import threading
from abc import ABC

import torch
from torch.utils.data import DataLoader, IterableDataset
import torch.nn as nn
from pathlib import Path

from model import DeepForkNet
import os

from data_preprocessing import get_project_root


class ChessDataset(IterableDataset):
    def __init__(self, processed_dir: str):
        self.files = sorted(Path(processed_dir).glob("*.pt"))
        print("Found files:", self.files)

    def _yield_file(self, path):
        data = torch.load(path)
        for sample in data:
            yield (
                torch.tensor(sample["state"], dtype=torch.float32),
                torch.tensor(sample["action"], dtype=torch.long),
                torch.tensor(sample["result"], dtype=torch.float32),
            )

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
        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.NLLLoss()

    def forward(self, policy_pred, value_pred, policy_target_idx, value_target):
        loss_policy = self.policy_loss(policy_pred, policy_target_idx)
        loss_value = self.value_loss(value_pred.squeeze(), value_target)
        return loss_policy + loss_value


def train_model(model, processed_dir, epochs=5, batch_size=32, lr=1e-3, device='cuda', samples_per_file=300, n_samples=None):
    dataset = ChessDataset(processed_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=device == 'cuda',
        persistent_workers=True,
        prefetch_factor=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = AZLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for states, policy_targets, value_targets in loader:
            states = states.to(device)
            policy_targets = policy_targets.to(device, dtype=torch.long)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            value_pred, policy_pred = model(states)
            loss = criterion(policy_pred, value_pred, policy_targets, value_targets)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")


if __name__ == "__main__":
    model = DeepForkNet(depth=10, history_size=6)
    processed_dir = get_project_root() / "data" / "processed"
    epochs = 7
    n_samples = None
    batch_size = 512
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'
    train_model(model, processed_dir, epochs, batch_size, device=device, n_samples=n_samples)
    save_path = get_project_root() / "models" / "checkpoints"
    model_name = f"{epochs}epochs_{'all' if n_samples is None else n_samples}samples_{batch_size}batch_size.pt"
    torch.save(model.state_dict(), save_path / model_name)
