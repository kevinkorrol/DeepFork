import math
import os
import random
from pathlib import Path
import torch
from torch import nn as nn
from torch.utils.data import IterableDataset, DataLoader


class ChessDataset(IterableDataset):
    """Iterable dataset over saved tensor shards produced by preprocessing.

    :param processed_dir: Directory containing .pt shard files
    :param samples_per_file: Number of samples in each shard file
    :param n_samples: Optional cap on total samples (used to limit files)
    :param files: Optional explicit list of files to iterate over
    """
    def __init__(self, processed_dir: str, samples_per_file: int, n_samples: int,
                 model_head: str, files=None, buffer_size=10_000):
        if files is not None:
            self.files = files
        else:
            if n_samples is None:
                self.files = sorted(Path(processed_dir).glob("*.pt"))
            else:
                self.files = sorted(Path(processed_dir).glob("*.pt"))[:math.ceil(n_samples / samples_per_file)]
        self.samples_per_file = samples_per_file
        self.model_head = model_head
        self.buffer_size = buffer_size
        self.buffer_size_per_class = buffer_size // 3

    def _get_random_indices(self, game_length, num_samples=10):
        """Pick N random unique indices from the game."""
        if game_length <= 0: return []
        # Ensure we don't try to pick more samples than exist
        k = min(num_samples, game_length)
        return set(random.sample(range(game_length), k))

    def _get_samples_from_file(self, data):
        selected_samples = []
        samples_by_game = {} # game_id: [sample_1, sample_2, ...]

        for sample in data:
            game_id = sample["game_id"]
            if game_id not in samples_by_game:
                samples_by_game[game_id] = []
            samples_by_game[game_id].append(sample)

        for game_id, samples in samples_by_game.items():
            indices = self._get_random_indices(len(samples), num_samples=10)
            for idx in indices:
                selected_samples.append(samples[idx])

        return selected_samples

    def _yield_file(self, path):
        """Yield samples (state, action) from a single shard file.

        :param path: Path to a .pt file saved during preprocessing
        :yield: Tuples (state: FloatTensor, action: LongTensor)
        """
        data = torch.load(path)
        if self.model_head == "policy":
            for sample in data:
                yield sample
        elif self.model_head == "value":
            for sample in self._get_samples_from_file(data):
                yield sample

    def __len__(self):
        """Approximate dataset length across all shard files."""
        return (len(self.files) - 1) * self.samples_per_file + len(torch.load(self.files[-1]))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            files = self.files
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            files = self.files[worker_id::num_workers]

        buffers = {
            1: [],
            0: [],
            -1: []
        }
        policy_buffer = []

        sample_stream = (sample for f in files for sample in self._yield_file(f))

        for sample in sample_stream:
            state = torch.tensor(sample["state"], dtype=torch.float32)
            result = sample["result"]

            if self.model_head == "policy":
                action = torch.tensor(sample["action"], dtype=torch.long)
                processed = (state, action)

                if len(policy_buffer) < self.buffer_size:
                    policy_buffer.append(processed)
                else:
                    idx = random.randint(0, len(policy_buffer) - 1)
                    yield policy_buffer[idx]
                    policy_buffer[idx] = processed

                continue

            processed = (state, torch.tensor(float(result), dtype=torch.float32))
            target_buffer = buffers[result]

            # Fill Buffer
            if len(target_buffer) < self.buffer_size_per_class:
                buffers[result].append(processed)
            else:
                idx = random.randint(0, len(target_buffer) - 1)
                target_buffer[idx] = processed

            if all(len(b) > 0 for b in buffers.values()):
                # Pick a class uniformly at random (33% chance each)
                choice = random.choice([1, 0, -1])

                chosen_buffer = buffers[choice]
                rand_idx = random.randint(0, len(chosen_buffer) - 1)

                yield chosen_buffer[rand_idx]
                chosen_buffer[rand_idx] = chosen_buffer[-1]
                chosen_buffer.pop()

        if self.model_head == "policy":
            random.shuffle(policy_buffer)
            for item in policy_buffer:
                yield item
        elif self.model_head == "value":
            leftover = buffers[1] + buffers[0] + buffers[-1]
            random.shuffle(leftover)
            for item in leftover:
                yield item


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
        self.mse = nn.MSELoss()

    def forward(self, value_est, value_target):
        return self.mse(value_est.view(-1), value_target)


def get_data_loaders(samples_per_file: int, n_samples: int, test_split: float, processed_dir,
                     model_head: str, batch_size: int, device: str, buffer_size=10_000):
    all_files = sorted(Path(processed_dir).glob("*.pt"))
    if n_samples is not None:
        all_files = all_files[:math.ceil(n_samples / samples_per_file)]

    n_test = max(1, int(len(all_files) * test_split))
    train_files = all_files[:-n_test]
    test_files = all_files[-n_test:]

    print(f"Train files: {len(train_files)}, test files: {len(test_files)}")

    train_dataset = ChessDataset(processed_dir, samples_per_file, n_samples, files=train_files,
                                 model_head=model_head, buffer_size=buffer_size)
    test_dataset = ChessDataset(processed_dir, samples_per_file, n_samples, files=test_files,
                                model_head=model_head, buffer_size=buffer_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=device == 'cuda',
        persistent_workers=True,
        prefetch_factor=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=device == 'cuda',
        persistent_workers=True,
        prefetch_factor=2
    )
    return train_loader, test_loader

if __name__ == "__main__":
    pass
