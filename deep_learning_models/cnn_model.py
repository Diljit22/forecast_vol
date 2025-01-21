## NOT complete use Attension for built model

class VolatilityPredictor_CNN(nn.Module):
    """
    1D CNN approach:
        - Conv over time dimension
        - Global pool, then FC.
    """

    def __init__(self, input_dim=8, n_tasks=2, num_filters=16, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # global pool => [batch, num_filters, 1]
        self.fc = nn.Linear(num_filters, n_tasks)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # Rearrange to [batch_size, input_dim, seq_len] for conv1d
        x = x.permute(0, 2, 1)  # => [batch, input_dim, seq_len]
        out = self.conv1(x)  # => [batch, num_filters, new_len]
        out = self.relu(out)
        out = self.pool(out)  # => [batch, num_filters, 1]
        out = out.squeeze(-1)  # => [batch, num_filters]
        out = self.fc(out)  # => [batch, n_tasks]
        return out