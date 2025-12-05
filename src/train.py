import datetime
import math
import torch
from torch.utils.data import DataLoader, IterableDataset
import torch.nn as nn
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import DeepForkNet
import os

from data_preprocessing import get_project_root


class ChessDataset(IterableDataset):
    def __init__(self, processed_dir: str, samples_per_file: int, n_samples: int):
        if n_samples is None:
            self.files = sorted(Path(processed_dir).glob("*.pt"))
        else:
            self.files = sorted(Path(processed_dir).glob("*.pt"))[:math.ceil(n_samples / samples_per_file)]
        self.samples_per_file = samples_per_file
        self.count = 0
        print(f"Found {len(self.files)} files")

    def _yield_file(self, path):
        data = torch.load(path)
        for sample in data:
            self.count += 1
            yield (
                torch.tensor(sample["state"], dtype=torch.float32),
                torch.tensor(sample["action"], dtype=torch.long),
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
        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.NLLLoss()

    def forward(self, policy_pred, value_pred, policy_target_idx, value_target):
        loss_policy = self.policy_loss(policy_pred, policy_target_idx)
        loss_value = self.value_loss(value_pred.squeeze(), value_target)
        return loss_policy + loss_value


def train_model(model, processed_dir, epochs=5, batch_size=32, lr=1e-3, device='cuda', samples_per_file=300, n_samples=None):
    dataset = ChessDataset(processed_dir, samples_per_file, n_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=device == 'cuda',
        persistent_workers=True,
        prefetch_factor=4
    )

    loss_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = AZLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for _, (states, action, value_targets) in tqdm(enumerate(loader, 0), unit="batch", total=len(loader)):
            states = states.to(device)
            policy_targets = action.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            value_pred, policy_pred = model(states)
            loss = criterion(policy_pred, value_pred, policy_targets, value_targets)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss

        avg_loss = total_loss/len(loader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return loss_history


if __name__ == "__main__":
    model = DeepForkNet(depth=10, filter_count=128, history_size=6)
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    epochs = 5
    n_samples = None
    batch_size = 512
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'
    loss_history = train_model(model, processed_dir, epochs, batch_size, device=device, n_samples=n_samples)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    output_dir = root / "model_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"Loss_vs_Epoch_{datetime.datetime.today().strftime('%Y-%m-%d')}.png"
    plt.savefig(output_dir / filename)
    save_path = root / "models" / "checkpoints"
    model_name = f"{epochs}epochs_{'all' if n_samples is None else n_samples}samples_{batch_size}batch_size.pt"
    torch.save(model.state_dict(), save_path / model_name)
