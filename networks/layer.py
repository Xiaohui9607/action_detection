import torch
from torch import nn

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, forget_bias=1.0, padding=0):
        super(ConvLSTM, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=out_channels + in_channels, out_channels=4 * out_channels,
                              kernel_size=kernel_size, stride=1, padding=padding)
        self.forget_bias = forget_bias

    def forward(self, inputs, states):
        if states is None:
            states = (torch.zeros([inputs.shape[0], self.out_channels, inputs.shape[2], inputs.shape[3]], device=inputs.device),
                      torch.zeros([inputs.shape[0], self.out_channels, inputs.shape[2], inputs.shape[3]], device=inputs.device))
        if not isinstance(states, tuple):
            raise TypeError('states type is not right')

        c, h = states
        if not (len(c.shape) == 4 and len(h.shape) == 4 and len(inputs.shape) == 4):
            raise TypeError('')

        inputs_h = torch.cat((inputs, h), dim=1)
        i_j_f_o = self.conv(inputs_h)
        i, j, f, o = torch.split(i_j_f_o,  self.out_channels, dim=1)

        new_c = c * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(i) * torch.tanh(j)
        new_h = torch.tanh(new_c) * torch.sigmoid(o)

        return new_h, (new_c, new_h)



