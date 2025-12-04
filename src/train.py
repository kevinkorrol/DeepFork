import threading
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pathlib import Path

from tqdm import tqdm

from model import DeepForkNet
import os

from data_preprocessing import get_project_root

file_cache = {}
current_file_path = None
cache_lock = threading.Lock()

MAX_CACHE_FILES = 2000

class ChessDataset(Dataset):
    def __init__(self, processed_dir: str, samples_per_file: int, n_samples: int = None):
        self.files = sorted(Path(processed_dir).glob("*.pt"))
        print("Found files:", self.files)
        self.samples_per_file = samples_per_file

        self.total_samples = (len(self.files) - 1) * samples_per_file + len(torch.load(self.files[-1], weights_only=False))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        sample_idx_in_file = idx % self.samples_per_file

        file_path = str(self.files[file_idx])

        global file_cache, current_file_path, cache_lock

        if file_path != current_file_path:
            # Cache Miss: Load the new file and update the cache
            with cache_lock: # Only one worker loads the file at a time
                # Re-check the path after acquiring the lock in case another thread/process
                # loaded it while we were waiting.
                if file_path != current_file_path:
                    data = torch.load(file_path, weights_only=False)
                    file_cache[file_path] = data
                    current_file_path = file_path  # Update the path marker

                    if len(file_cache) > MAX_CACHE_FILES:
                        # Find and remove an arbitrary "old" entry
                        oldest_file = next(iter(file_cache))
                        if oldest_file != file_path:  # Don't delete the file we just loaded!
                            del file_cache[oldest_file]

        data = file_cache[file_path]
        sample = data[sample_idx_in_file]

        state = torch.tensor(sample["state"], dtype=torch.float32)
        action = torch.tensor(sample["action"], dtype=torch.long)
        value_target = torch.tensor(sample["result"], dtype=torch.float32)

        return state, action, value_target


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
        shuffle=True,
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
    model = DeepForkNet(depth=10)
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
