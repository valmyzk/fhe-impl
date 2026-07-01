import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from src.datasets.shanghai import FEATURE_COLS, load_dataset

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

CGM_MIN = 40
CGM_MAX = 400

def normalize(values: torch.Tensor):
    return (values - CGM_MIN) / (CGM_MAX - CGM_MIN)

def unnormalize(values: torch.Tensor):
    return values * (CGM_MAX - CGM_MIN) + CGM_MIN

def make_tensors(df):
    X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
    y = torch.tensor(df["cgm"].values, dtype=torch.float32).unsqueeze(1)
    return normalize(X), normalize(y)


def train(
    model: nn.Module, epochs: int = 30, lr: float = 1e-3, batch_size: int = 64,
    patience: int | None = 5,
) -> None:
    df = load_dataset()
    X, y = make_tensors(df)
    dataset = TensorDataset(X, y)
    n_train = int(0.8 * len(dataset))
    train_ds, _ = random_split(dataset, [n_train, len(dataset) - n_train])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    with tqdm(range(epochs), desc="Training", unit="epoch") as pbar:
        for _ in pbar:
            model.train()
            epoch_loss = sum(
                _train_step(model, xb, yb, optimizer, criterion) for xb, yb in loader
            ) / len(loader)

            if patience is not None and no_improve == patience:
                break

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improve = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1

            pbar.set_postfix(loss=f"{epoch_loss:.4f}", best=f"{best_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()


def _train_step(model, xb, yb, optimizer, criterion) -> float:
    optimizer.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward()
    optimizer.step()
    return loss.item()


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path) -> bool:
    if not path.exists():
        return False
    try:
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return True
    except RuntimeError:
        path.unlink(missing_ok=True)
        return False


__all__ = ["CHECKPOINT_DIR", "train", "save_checkpoint", "load_checkpoint"]
