"""ConvLSTM cells retained from the original project implementation.

Parameter names and gate order are preserved for legacy checkpoint compatibility.
Adapted from ndrplz/ConvLSTM_pytorch (MIT); see
third_party/ConvLSTM_pytorch-LICENSE for the upstream copyright notice.
"""

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = self.conv(torch.cat([input_tensor, h_cur], dim=1))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_dim, dim=1)
        c_next = torch.sigmoid(cc_f) * c_cur + torch.sigmoid(cc_i) * torch.tanh(cc_g)
        h_next = torch.sigmoid(cc_o) * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        shape = (batch_size, self.hidden_dim, *image_size)
        return self.conv.weight.new_zeros(shape), self.conv.weight.new_zeros(shape)


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers
        if isinstance(kernel_size, tuple):
            kernel_size = [kernel_size] * num_layers
        if len(hidden_dim) != num_layers or len(kernel_size) != num_layers:
            raise ValueError("Provide one hidden dimension and kernel per layer.")
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        self.cell_list = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim[i - 1],
                         hidden_dim[i], kernel_size[i], bias)
            for i in range(num_layers)
        ])

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        batch, _, _, height, width = input_tensor.shape
        if hidden_state is None:
            hidden_state = [cell.init_hidden(batch, (height, width)) for cell in self.cell_list]
        layer_outputs, last_states = [], []
        current = input_tensor
        for cell, (h, c) in zip(self.cell_list, hidden_state):
            output = []
            for frame in current.unbind(dim=1):
                h, c = cell(frame, (h, c))
                output.append(h)
            current = torch.stack(output, dim=1)
            layer_outputs.append(current)
            last_states.append([h, c])
        if not self.return_all_layers:
            layer_outputs, last_states = layer_outputs[-1:], last_states[-1:]
        return layer_outputs, last_states
