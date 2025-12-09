import math
import os
import random
from pathlib import Path

import torch
from torch import nn as nn
from torch.utils.data import IterableDataset, DataLoader


def get_random_states(game_len: int, min_diff: int, num_states: int = 20) -> list:
    num_states = min(num_states, game_len // (min_diff * 2))
    idxs = []
    prev_idx = -min_diff

    part = game_len // num_states
    for i in range(num_states):
        min_idx = prev_idx + min_diff
        if min_idx >= game_len:
            return idxs
        max_idx = min(min_idx + part, game_len - 1)
        rand_idx = random.randrange(min_idx, max_idx)
        idxs.append(rand_idx)
        prev_idx = rand_idx

    return idxs


class ChessDataset(IterableDataset):
    """Iterable dataset over saved tensor shards produced by preprocessing.

    :param processed_dir: Directory containing .pt shard files
    :param samples_per_file: Number of samples in each shard file
    :param n_samples: Optional cap on total samples (used to limit files)
    :param files: Optional explicit list of files to iterate over
    """
    def __init__(self, processed_dir: str, samples_per_file: int, n_samples: int, min_diff: int, model_head: str, files=None):
        if files is not None:
            self.files = files
        else:
            if n_samples is None:
                self.files = sorted(Path(processed_dir).glob("*.pt"))
            else:
                self.files = sorted(Path(processed_dir).glob("*.pt"))[:math.ceil(n_samples / samples_per_file)]
        self.samples_per_file = samples_per_file
        self.min_diff = min_diff
        self.model_head = model_head
        print(f"Dataset initialized with {len(self.files)} files")

    def _yield_file(self, path):
        """Yield samples (state, action) from a single shard file.

        :param path: Path to a .pt file saved during preprocessing
        :yield: Tuples (state: FloatTensor, action: LongTensor)
        """
        data = torch.load(path)
        if self.model_head == "value":
            current_game_idx = -1
            game_state_idx = 0
            idxs = []
            for sample in data:
                if current_game_idx != sample["game_id"]:
                    idxs = get_random_states(sample["game_length"], min_diff=self.min_diff)
                    game_state_idx = 0
                if game_state_idx in idxs:
                    yield (
                        torch.tensor(sample["state"], dtype=torch.float32),
                        torch.tensor(sample["game_result"], dtype=torch.float32)
                    )
                game_state_idx += 1
        elif self.model_head == "policy":
            for sample in data:
                yield (
                    torch.tensor(sample["state"], dtype=torch.float32),
                    torch.tensor(sample["action"], dtype=torch.long)
                )

    def __len__(self):
        """Approximate dataset length across all shard files."""
        return (len(self.files) - 1) * self.samples_per_file + len(torch.load(self.files[-1]))

    def __iter__(self):
        """Iterate over samples, sharded across DataLoader workers if any."""
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
    """Cross-entropy loss wrapper for policy head outputs."""
    def __init__(self):
        super().__init__()
        self.policy_loss_fn = nn.CrossEntropyLoss()

    def forward(self, policy_probs, policy_target):
        """Compute cross-entropy between logits and target action indices."""
        loss = self.policy_loss_fn(policy_probs, policy_target)
        return loss


class ValueLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_loss_fn = nn.MSELoss()

    def forward(self, value_est, value_target):
        loss = self.value_loss_fn(value_est, value_target)
        return loss




def get_data_loaders(samples_per_file: int, n_samples: int, test_split: float, processed_dir,
                     model_head: str, batch_size: int, device: str, min_diff: int =1):
    all_files = sorted(Path(processed_dir).glob("*.pt"))
    if n_samples is not None:
        all_files = all_files[:math.ceil(n_samples / samples_per_file)]

    n_test = max(1, int(len(all_files) * test_split))
    train_files = all_files[:-n_test]
    test_files = all_files[-n_test:]

    print(f"Train files: {len(train_files)}, test files: {len(test_files)}")

    train_dataset = ChessDataset(processed_dir, samples_per_file, n_samples, files=train_files,
                                 min_diff=min_diff, model_head=model_head)
    test_dataset = ChessDataset(processed_dir, samples_per_file, n_samples, files=test_files,
                                min_diff=min_diff, model_head="policy")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=device == 'cuda',
        persistent_workers=True,
        prefetch_factor=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=device == 'cuda',
        persistent_workers=True,
        prefetch_factor=2,
    )
    return train_loader, test_loader

if __name__ == "__main__":
    print(get_random_states(100, 15))
