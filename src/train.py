import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import chess


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

        policy_target = torch.zeros(4672, dtype=torch.float32)
        policy_target[move_idx] = 1.0
        value_target = torch.tensor(sample["result"], dtype=torch.float32)

        return state, policy_target, value_target




def move_to_action(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    square_offset = from_sq * 73

    if move.promotion and move.promotion != chess.QUEEN:
        promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        file_diff = chess.square_file(to_sq) - chess.square_file(from_sq)
        direction = 0 if file_diff == 0 else (1 if file_diff > 0 else 2)
        promotion_idx = promotion_pieces.index(move.promotion)
        return square_offset + 64 + direction * 3 + promotion_idx

    rank_diff = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    file_diff = chess.square_file(to_sq) - chess.square_file(from_sq)
    rooklike = (rank_diff == 0 or file_diff == 0)
    bishoplike = (abs(rank_diff) == abs(file_diff))
    queenlike = rooklike or bishoplike
    if queenlike:
        direction = (int(rooklike) << 2) | (int(rank_diff > 0) << 1) | int(file_diff > 0)
        distance = max(abs(rank_diff), abs(file_diff)) - 1
        return square_offset + direction * 7  + distance

    knight_rank_bit = int(abs(rank_diff) == 2)
    rank_pos_bit = int(rank_diff > 0)
    file_pos_bit = int(file_diff > 0)
    direction_encoding = (knight_rank_bit << 2) | (rank_pos_bit << 1) | file_pos_bit
    return square_offset + 56 + direction_encoding




class ConvBlock(nn.Module):
    def __init__(self, history_size=8, filter_count=256):
        super().__init__()
        self.conv = nn.Conv2d(14*history_size + 7, filter_count, 3, padding=1)
        self.bn = nn.BatchNorm2d(filter_count)

    def forward(self, x):
        x = x.view(-1, x.shape[1], 8, 8)
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, filter_count=256):
        super().__init__()
        self.conv1 = nn.Conv2d(filter_count, filter_count, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_count)
        self.conv2 = nn.Conv2d(filter_count, filter_count, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_count)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += res
        return F.relu(x)



class OutBlock(nn.Module):
    def __init__(self, filter_count=256):
        super().__init__()
        # Value head
        self.convV = nn.Conv2d(filter_count, 1, 1)
        self.bnV = nn.BatchNorm2d(1)
        self.lnV1 = nn.Linear(8*8, 256)
        self.lnV2 = nn.Linear(256, 1)

        # Policy head
        self.convP = nn.Conv2d(filter_count, 73, 1)
        self.bnP = nn.BatchNorm2d(73)

    def forward(self, x):
        # Value head
        v = F.relu(self.bnV(self.convV(x)))
        v = F.relu(self.lnV1(v.view(v.size(0), -1)))
        v = torch.tanh(self.lnV2(v))

        # Policy head
        p = F.relu(self.bnP(self.convP(x)))
        p = p.permute(0, 2, 3, 1).contiguous()
        p = p.view(p.size(0), -1)

        return v, p



class DeepForkNet(nn.Module):
    def __init__(self, depth=10, filter_count=256, history_size=8):
        super().__init__()
        self.conv_block = ConvBlock(history_size=history_size, filter_count=filter_count)
        self.res_blocks = nn.ModuleList([ResBlock(filter_count) for _ in range(depth)])
        self.out_block = OutBlock(filter_count=filter_count)

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        v, p = self.out_block(x)
        return v, p



class AZLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()

    def forward(self, policy_logits, value_pred, policy_target, value_target):
        policy_idx = torch.argmax(policy_target, dim=1)  # Shape: (B,)
        loss_policy = self.policy_loss(policy_logits, policy_idx)
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
            policy_logits, value_pred = model(states)
            loss = criterion(policy_logits, value_pred, policy_targets, value_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")



if __name__ == "__main__":
    model = DeepForkNet()
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    train_model(model, processed_dir, epochs=1, batch_size=8, device='cpu', n_samples=100)
