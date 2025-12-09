import datetime
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DeepForkNet
from data_preprocessing import get_project_root
from utils.model_utils import PolicyLoss, ValueLoss, get_data_loaders


def train_policy_model(model, processed_dir, epochs=5, batch_size=32, lr=1e-3, device='cuda', samples_per_file=300,
                       n_samples=None, test_split=0.2):
    """Train the policy network on preprocessed chess move data.

    Splits shard files into train/testidation, streams batches with DataLoader,
    and returns loss/accuracy histories for both splits.

    :param model: Instance of `DeepForkNet`
    :param processed_dir: Directory with processed .pt shard files
    :param epochs: Number of epochs to train
    :param batch_size: Global batch size
    :param lr: Learning rate for Adam optimizer
    :param device: 'cuda' or 'cpu'
    :param samples_per_file: Number of samples stored per shard file
    :param n_samples: Optional cap on number of samples used from the dataset
    :param test_split: Fraction of shard files used for testidation (0..1)
    :return: (train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist)
    """
    
    train_loader, test_loader = get_data_loaders(samples_per_file, n_samples, test_split, processed_dir, model_head)

    train_history = []
    train_accuracy_history = []
    test_history = []
    test_accuracy_history = []

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
        for state, action in tqdm(train_loader, unit="batch", total=total_len*(1 - test_split)):
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

        model.etest()
        test_loss = 0
        test_batches = 0
        test_samples = 0
        test_hits = 0
        with torch.no_grad():
            for state, action in tqdm(test_loader, unit="batch", total=total_len*test_split):
                state = state.to(device)
                policy_targets = action.to(device)

                policy_probs = model(state)
                loss = criterion(policy_probs, policy_targets)

                test_loss += loss.item()
                test_batches += 1
                test_samples += policy_targets.size(0)

                predicted_action = torch.argmax(policy_probs, dim=1)
                correct_predictions = (predicted_action == policy_targets).sum().item()
                test_hits += correct_predictions

        avg_test_loss = test_loss / test_batches
        avg_test_accuracy = test_hits / test_samples
        test_accuracy_history.append(avg_test_accuracy)
        test_history.append(avg_test_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Train loss: {avg_train_loss:.4f} "
            f"test loss: {avg_test_loss:.4f} "
            f"Train accuracy: {avg_train_accuracy:.4f} "
            f"test accuracy: {avg_test_accuracy:.4f}"
        )

    return train_history, test_history, train_accuracy_history, test_accuracy_history


def train_value_model(model, processed_dir, epochs=5, batch_size=32, lr=1e-3, device='cuda', samples_per_file=300,
                       n_samples=None, test_split=0.05, min_diff=15):
    train_loader, test_loader = get_data_loaders(samples_per_file, n_samples, test_split,
                                                 processed_dir, model_head, min_diff)

    train_history = []
    test_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ValueLoss()
    model.to(device)

    total_len = n_samples // batch_size if n_samples is not None else len(train_loader) // batch_size

    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        train_batches = 0
        for state, game_result in tqdm(train_loader, unit="batch", total=total_len * (1 - test_split)):
            state = state.to(device)
            game_result = game_result.to(device)

            optimizer.zero_grad()
            value_est = model(state)
            loss = criterion(value_est, game_result)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            train_batches += 1

        avg_train_loss = training_loss / train_batches
        train_history.append(avg_train_loss)

        model.etest()
        test_loss = 0
        test_batches = 0
        with torch.no_grad():
            for state, game_result in tqdm(test_loader, unit="batch", total=total_len * test_split):
                state = state.to(device)
                game_result = game_result.to(device)

                value_est = model(state)
                loss = criterion(value_est, game_result)

                test_loss += loss.item()
                test_batches += 1
        avg_test_loss = test_loss / test_batches
        test_history.append(avg_test_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Train loss: {avg_train_loss:.4f} "
            f"test loss: {avg_test_loss:.4f} "
        )

        return train_history, test_history


if __name__ == "__main__":

    epochs = 10
    n_samples = None
    batch_size = 512
    depth = 5
    filter_count = 256
    history_size = 1
    model_head = "policy"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'
    torch.manual_seed(283)
    model = DeepForkNet(model_head, depth, filter_count, history_size)
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    if model_head == "policy":
        train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history = train_policy_model(
            model,
            processed_dir,
            epochs,
            batch_size,
            device=device,
            n_samples=n_samples
        )
    elif model_head == "value":
        train_loss_history, test_loss_history = train_value_model(
            model,
            processed_dir,
            epochs,
            batch_size,
            device=device,
            n_samples=n_samples,
            min_diff=15
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(train_loss_history, marker='o', label="Train Loss")
    ax1.plot(test_loss_history, marker='s', label="testidation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and testidation Loss Over Epochs")
    ax1.grid(True)
    ax1.legend()

    if model_head == "policy":
        ax2.plot(train_accuracy_history, marker='o', label="Train Accuracy")
        ax2.plot(test_accuracy_history, marker='s', label="testidation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training and testidation Accuracy Over Epochs")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()

    output_dir = root / "model_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"Metrics_vs_Epoch_{datetime.datetime.today().strftime('%Y-%m-%d-%h-%m')}.png"
    plt.savefig(output_dir / filename)

    save_path = root / "models" / "checkpoints"
    model_name = f"{model_head}__{'all' if n_samples is None else n_samples}_samples__{depth}_depth__{filter_count}_filters__{history_size}_history_size.pt"
    torch.save(model.state_dict(), save_path / model_name)
