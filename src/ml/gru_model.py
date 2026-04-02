"""GRU sequence model for temporal pattern detection in trading.

Captures sequential dependencies that tree-based models cannot:
- Multi-bar momentum sequences
- Volatility regime transitions
- Volume accumulation patterns

Architecture: GRU → Dropout → Linear → 3-class (SELL/HOLD/BUY)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """Converts tabular features into overlapping sequences for GRU input."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 20):
        self.seq_len = seq_len
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_len]
        y_target = self.y[idx + self.seq_len - 1]  # Label for the last bar in sequence
        return x_seq, y_target


class TradingGRU(nn.Module):
    """GRU network for 3-class trading signal prediction."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.3, num_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]  # Use last timestep
        out = self.dropout(last_hidden)
        return self.fc(out)


class GRUModelTrainer:
    """Train and evaluate GRU models for trading signals."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.seq_len = cfg.get("seq_len", 20)
        self.hidden_size = cfg.get("hidden_size", 64)
        self.num_layers = cfg.get("num_layers", 1)
        self.dropout = cfg.get("dropout", 0.3)
        self.lr = cfg.get("learning_rate", 0.001)
        self.epochs = cfg.get("epochs", 100)
        self.batch_size = cfg.get("batch_size", 32)
        self.patience = cfg.get("patience", 15)
        self.weight_decay = cfg.get("weight_decay", 1e-4)

        self.device = torch.device("mps" if torch.backends.mps.is_available()
                                   else "cuda" if torch.cuda.is_available()
                                   else "cpu")
        logger.info("GRU device: %s", self.device)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        class_weights: dict | None = None,
    ) -> tuple[TradingGRU, dict]:
        """
        Train a GRU model.

        X_train/y_train: features and labels (y mapped to 0,1,2)
        Returns: (trained_model, training_history)
        """
        input_size = X_train.shape[1]

        # Normalize features (fit on train only)
        train_mean = X_train.mean()
        train_std = X_train.std().replace(0, 1)
        X_train_norm = ((X_train - train_mean) / train_std).values
        self._norm_params = (train_mean, train_std)

        # Create datasets
        train_ds = SequenceDataset(X_train_norm, y_train.values, self.seq_len)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_norm = ((X_val - train_mean) / train_std).values
            val_ds = SequenceDataset(X_val_norm, y_val.values, self.seq_len)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Build model
        model = TradingGRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Class-weighted loss
        if class_weights:
            weights = torch.FloatTensor([class_weights.get(i, 1.0) for i in range(3)])
            criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5)

        # Training loop with early stopping
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            model.train()
            train_loss = 0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # Validate
            if val_loader:
                val_loss, val_acc = self._evaluate(model, val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d (val_loss=%.4f)", epoch, best_val_loss)
                    break

                if (epoch + 1) % 20 == 0:
                    logger.info("Epoch %d: train_loss=%.4f, val_loss=%.4f, val_acc=%.3f",
                                epoch + 1, avg_train_loss, val_loss, val_acc)
            else:
                if (epoch + 1) % 20 == 0:
                    logger.info("Epoch %d: train_loss=%.4f", epoch + 1, avg_train_loss)

        # Load best weights
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        logger.info("GRU training complete: %d epochs, best_val_loss=%.4f",
                     epoch + 1, best_val_loss)
        return model, history

    def predict(self, model: TradingGRU, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict using trained GRU.

        Returns (predictions, probabilities) with length == len(X).
        The first seq_len-1 predictions use uniform probability (no sequence context).
        """
        train_mean, train_std = self._norm_params
        # Align features to training feature set
        common_cols = [c for c in train_mean.index if c in X.columns]
        X_aligned = X[common_cols]
        X_norm = ((X_aligned - train_mean[common_cols]) / train_std[common_cols].replace(0, 1)).values
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

        n_samples = len(X)
        n_classes = 3

        # Pad start so we get predictions for ALL rows
        pad = np.zeros((self.seq_len - 1, X_norm.shape[1]))
        X_padded = np.vstack([pad, X_norm])

        ds = SequenceDataset(X_padded, np.zeros(len(X_padded), dtype=int), self.seq_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        preds = np.array(all_preds)[:n_samples]
        probs = np.array(all_probs)[:n_samples]

        return preds, probs

    def save(self, model: TradingGRU, path: str | Path):
        """Save model + normalization params."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "config": {
                "input_size": model.gru.input_size,
                "hidden_size": model.gru.hidden_size,
                "num_layers": model.gru.num_layers,
                "dropout": self.dropout,
                "seq_len": self.seq_len,
            },
            "norm_params": {
                "mean": self._norm_params[0].to_dict(),
                "std": self._norm_params[1].to_dict(),
            },
        }, path)
        logger.info("GRU model saved: %s", path)

    def load(self, path: str | Path) -> TradingGRU:
        """Load a saved GRU model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cfg = checkpoint["config"]
        model = TradingGRU(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        norm = checkpoint["norm_params"]
        self._norm_params = (pd.Series(norm["mean"]), pd.Series(norm["std"]))
        self.seq_len = cfg["seq_len"]

        return model

    def _evaluate(self, model, loader, criterion):
        """Evaluate on validation set."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy
