## NOT complete use Attension for built model


class VolatilityPredictor_LSTM(nn.Module):
    """
    LSTM-based approach for multi-task forecasting.
    """

    def __init__(self, input_dim=8, hidden_dim=16, n_layers=1, n_tasks=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=n_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, n_tasks)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        out, (h, c) = self.lstm(x)  # out: [batch_size, seq_len, hidden_dim]
        last_step = out[:, -1, :]
        return self.fc(last_step)