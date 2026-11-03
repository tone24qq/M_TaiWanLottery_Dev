"""Small CPU neural benchmarks; loaded only when neural models are requested."""

from __future__ import annotations
import numpy as np
import torch
from torch import nn

from taiwan_lottery.ml.models.dqn_model import DuelingDQNNet


class BingoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(80, 32)
        self.position = nn.Parameter(torch.zeros(1, 16, 32))
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, dropout=0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.output = nn.Linear(32, 80)

    def forward(self, x):
        return self.output(self.encoder(self.input(x) + self.position)[:, -1])


def _sequence(matrix):
    result = np.zeros((16, 80), dtype=np.float32)
    if len(matrix):
        result[-min(16, len(matrix)) :] = matrix[-16:]
    return result


def training_sequences(matrix, min_context=20):
    """Each sequence at target t ends at t-1; target values are separate."""
    if len(matrix) <= min_context:
        return np.empty((0, 16, 80), dtype=np.float32)
    return np.stack([_sequence(matrix[:t]) for t in range(min_context, len(matrix))])


def fit_neural(name, matrix, x, y, threads, progress):
    torch.set_num_threads(threads)
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    if name == "transformer":
        model = BingoTransformer()
        # Target t sees exactly the 16 draws ending at t-1.
        sequences = training_sequences(matrix)
        inputs, labels = torch.from_numpy(sequences), torch.from_numpy(y)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        for epoch in range(6):
            model.train()
            for ids in np.array_split(
                rng.permutation(len(inputs)), max(1, int(np.ceil(len(inputs) / 32)))
            ):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model(inputs[ids]), labels[ids]
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            progress(epoch + 1, 6)
    else:
        # Offline Double DQN. Action is a number; reward is whether that number
        # appeared in the next draw. Historical labels reveal all 80 action
        # rewards, allowing bounded full-action replay without Monte Carlo data.
        model = DuelingDQNNet(800, 80, hidden_dim=64)
        target = DuelingDQNNet(800, 80, hidden_dim=64)
        target.load_state_dict(model.state_dict())
        target.eval()
        inputs = torch.from_numpy(x.reshape(len(x), -1))
        rewards = torch.from_numpy(y)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        for epoch in range(12):
            model.train()
            # A transition uses x[t], y[t], x[t+1]; there is no fabricated
            # terminal transition or bootstrapping through the dataset boundary.
            for ids in np.array_split(
                rng.permutation(len(x) - 1), max(1, int(np.ceil((len(x) - 1) / 64)))
            ):
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    model.eval()
                    next_actions = model(inputs[ids + 1]).argmax(dim=1, keepdim=True)
                    next_values = target(inputs[ids + 1]).gather(1, next_actions)
                    bellman = rewards[ids] + 0.2 * next_values
                model.train()
                loss = nn.functional.smooth_l1_loss(model(inputs[ids]), bellman)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            if epoch % 2:
                target.load_state_dict(model.state_dict())
            progress(epoch + 1, 12)
    model.eval()
    return model.cpu()


def neural_scores(name, model, matrix, row):
    model.eval()
    with torch.inference_mode():
        inputs = (
            torch.from_numpy(_sequence(matrix)).unsqueeze(0)
            if name == "transformer"
            else torch.from_numpy(row.reshape(1, -1))
        )
        values = model(inputs)[0]
        if name == "transformer":
            values = values.sigmoid()
    return values.numpy()
