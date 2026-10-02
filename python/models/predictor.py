"""The original 3D-convolution / ConvLSTM / RGB decoder architecture."""

from torch import nn

from .convlstm import ConvLSTM


class PaperModel(nn.Module):
    """Predict one RGB field image from a (B, T, 9, H, W) sequence.

    Each frame contains the RGB field, RGB grain-orientation map, and RGB
    C-rate map. The default model preserves the original state-dictionary keys.
    """

    def __init__(self, input_channels=9, hidden_dim=32, kernel_size=(5, 5), num_layers=3):
        super().__init__()
        self.conv3d = nn.Conv3d(input_channels, hidden_dim, (3, 5, 5), padding=(1, 2, 2))
        self.conv_lstm = ConvLSTM(
            hidden_dim, [hidden_dim] * num_layers, [kernel_size] * num_layers,
            num_layers, batch_first=True, return_all_layers=False,
        )
        self.final_conv = nn.Conv2d(hidden_dim, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] != self.conv3d.in_channels:
            raise ValueError("Expected input shape (batch, time, 9, height, width).")
        features = self.conv3d(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        layer_outputs, _ = self.conv_lstm(features)
        return self.sigmoid(self.final_conv(layer_outputs[0][:, -1]))
