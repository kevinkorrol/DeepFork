import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pathlib import Path
import chess
from model import DeepForkNet
from utils.chess_utils import move_to_action


class ChessDataset(Dataset):
    def __init__(self, processed_dir: str, n_samples: int = None):
        self.files = sorted(Path(processed_dir).glob("*.pt"))
        print("Found files:", self.files)
        self.samples = []

        for f in self.files:
            with torch.serialization.safe_globals([np.ndarray, np._core.multiarray._reconstruct]):
                data = torch.load(f, weights_only=False)
            for sample in data:
                move_obj = chess.Move.from_uci(sample["move"])
                idx = move_to_action(move_obj)
                if 0 <= idx < 4672:
                    self.samples.append(sample)
                else:
                    print(f"WARNING: Skipping move {sample['move']} with out-of-bounds index {idx}")

        if n_samples is not None:
            self.samples = self.samples[:n_samples]

        print(f"Loaded {len(self.samples)} samples from {len(self.files)} files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        state = torch.tensor(sample["state"], dtype=torch.float32)

        move_obj = chess.Move.from_uci(sample["move"])
        move_idx = move_to_action(move_obj)

        value_target = torch.tensor(sample["result"], dtype=torch.float32)

        return state, move_idx, value_target


class AZLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.NLLLoss()

    def forward(self, policy_pred, value_pred, policy_target_idx, value_target):
        loss_policy = self.policy_loss(policy_pred, policy_target_idx)
        loss_value = self.value_loss(value_pred.squeeze(), value_target)
        return loss_policy + loss_value


def train_model(model, processed_dir, epochs=5, batch_size=32, lr=1e-3, device='cuda', n_samples=None):
    dataset = ChessDataset(processed_dir, n_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = AZLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for states, policy_targets, value_targets in loader:
            states, policy_targets, value_targets = states.to(device), policy_targets.to(device), value_targets.to(device)

            optimizer.zero_grad()
            value_pred, policy_pred = model(states)
            loss = criterion(policy_pred, value_pred, policy_targets, value_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")



if __name__ == "__main__":
    model = DeepForkNet(depth=5)
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    train_model(model, processed_dir, epochs=100, batch_size=8, device='cpu', n_samples=100)
