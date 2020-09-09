import math
import torch
from torch import nn
from torch.nn import functional as F
from .layer import ConvLSTM

DNA_KERN_SIZE = 5
NUM_BEHAVRIORS = 16


class network(nn.Module):
    def __init__(self, opt, channels=3,
                 height=64,
                 width=64):
        super(network, self).__init__()

        lstm_size = [32, 32, 64, 64, 128, 64, 32]
        self.channels = channels
        self.opt = opt
        self.height = height
        self.width = width

        # N * 3 * H * W -> N * 32 * H/2 * W/2
        self.enc0 = nn.Conv2d(in_channels=channels, out_channels=lstm_size[0], kernel_size=5, stride=2, padding=2)
        self.enc0_norm = nn.LayerNorm([lstm_size[0], self.height//2, self.width//2])
        # N * 32 * H/2 * W/2 -> N * 32 * H/2 * W/2
        self.lstm1 = ConvLSTM(in_channels=lstm_size[0], out_channels=lstm_size[0], kernel_size=5, padding=2)
        self.lstm1_norm = nn.LayerNorm([lstm_size[0], self.height//2, self.width//2])
        # N * 32 * H/2 * W/2 -> N * 32 * H/2 * W/2
        self.lstm2 = ConvLSTM(in_channels=lstm_size[0], out_channels=lstm_size[1], kernel_size=5, padding=2)
        self.lstm2_norm = nn.LayerNorm([lstm_size[1], self.height//2, self.width//2])

        # N * 32 * H/4 * W/4 -> N * 32 * H/4 * W/4
        self.enc1 = nn.Conv2d(in_channels=lstm_size[1], out_channels=lstm_size[1], kernel_size=3, stride=2, padding=1)
        # N * 32 * H/4 * W/4 -> N * 64 * H/4 * W/4
        self.lstm3 = ConvLSTM(in_channels=lstm_size[1], out_channels=lstm_size[2], kernel_size=5, padding=2)
        self.lstm3_norm = nn.LayerNorm([lstm_size[2], self.height//4, self.width//4])
        # N * 64 * H/4 * W/4 -> N * 64 * H/4 * W/4
        self.lstm4 = ConvLSTM(in_channels=lstm_size[2], out_channels=lstm_size[3], kernel_size=5, padding=2)
        self.lstm4_norm = nn.LayerNorm([lstm_size[3], self.height//4, self.width//4])

        # N * 64 * H/4 * W/4 -> N * 64 * H/8 * W/8
        self.enc2 = nn.Conv2d(in_channels=lstm_size[3], out_channels=lstm_size[3], kernel_size=3, stride=2, padding=1)

        # N * (10+64) * H/8 * W/8 -> N * 64 * H/8 * W/8
        self.enc3 = nn.Conv2d(in_channels=lstm_size[3], out_channels=lstm_size[3], kernel_size=1, stride=1)
        # N * 64 * H/8 * W/8 -> N * 128 * H/8 * W/8
        self.lstm5 = ConvLSTM(in_channels=lstm_size[3], out_channels=lstm_size[4], kernel_size=5, padding=2)
        self.lstm5_norm = nn.LayerNorm([lstm_size[4], self.height//8, self.width//8])
        # N * 128 * H/8 * W/8 -> N * 128 * H/4 * W/4
        self.enc4 = nn.ConvTranspose2d(in_channels=lstm_size[4], out_channels=lstm_size[4], kernel_size=3, stride=2, output_padding=1, padding=1)
        # N * 128 * H/4 * W/4 -> N * 64 * H/4 * W/4
        self.lstm6 = ConvLSTM(in_channels=lstm_size[4], out_channels=lstm_size[5], kernel_size=5, padding=2)
        self.lstm6_norm = nn.LayerNorm([lstm_size[5], self.height//4, self.width//4])

        # N * 64 * H/4 * W/4 -> N *64  * H/2 * W/2
        self.enc5 = nn.ConvTranspose2d(in_channels=lstm_size[5]+lstm_size[1], out_channels=lstm_size[5]+lstm_size[1], kernel_size=3, stride=2, output_padding=1, padding=1)
        # N * 64 * H/2 * W/2 -> N * 32 * H/2 * W/2
        self.lstm7 = ConvLSTM(in_channels=lstm_size[5]+lstm_size[1], out_channels=lstm_size[6], kernel_size=5, padding=2)
        self.lstm7_norm = nn.LayerNorm([lstm_size[6], self.height//2, self.width//2])
        # N * 32 * H/2 * W/2 -> N * 32 * H * W
        self.enc6 = nn.ConvTranspose2d(in_channels=lstm_size[6]+lstm_size[0], out_channels=lstm_size[6], kernel_size=3, stride=2, output_padding=1, padding=1)
        self.enc6_norm = nn.LayerNorm([lstm_size[6], self.height, self.width])

        # N * 32 * H * W -> N * 3 * H * W
        self.enc7 = nn.ConvTranspose2d(in_channels=lstm_size[6], out_channels=channels, kernel_size=1, stride=1)

        in_dim = int(self.channels * self.height * self.width)
        self.fc1 = nn.Linear(in_dim, 20)
        self.fc2 = nn.Linear(20, NUM_BEHAVRIORS)


    def forward(self, images, train=True):
        """
        :param inputs: T * N * C * H * W
        :param state: T * N * C
        :param action: T * N * C
        :return:
        """

        lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
        lstm_state5, lstm_state6, lstm_state7 = None, None, None

        labels_list_prob = []

        for image in images:
            enc0 = self.enc0_norm(torch.relu(self.enc0(image)))

            lstm1, lstm_state1 = self.lstm1(enc0, lstm_state1)
            lstm1 = self.lstm1_norm(lstm1)

            lstm2, lstm_state2 = self.lstm2(lstm1, lstm_state2)
            lstm2 = self.lstm2_norm(lstm2)

            enc1 = torch.relu(self.enc1(lstm2))

            lstm3, lstm_state3 = self.lstm3(enc1, lstm_state3)
            lstm3 = self.lstm3_norm(lstm3)

            lstm4, lstm_state4 = self.lstm4(lstm3, lstm_state4)
            lstm4 = self.lstm4_norm(lstm4)

            enc2 = torch.relu(self.enc2(lstm4))

            enc3 = torch.relu(self.enc3(enc2))

            lstm5, lstm_state5 = self.lstm5(enc3, lstm_state5)
            lstm5 = self.lstm5_norm(lstm5)
            enc4 = torch.relu(self.enc4(lstm5))

            lstm6, lstm_state6 = self.lstm6(enc4, lstm_state6)
            lstm6 = self.lstm6_norm(lstm6)
            # skip connection
            lstm6 = torch.cat([lstm6, enc1], dim=1)

            enc5 = torch.relu(self.enc5(lstm6))

            lstm7, lstm_state7 = self.lstm7(enc5, lstm_state7)
            lstm7 = self.lstm7_norm(lstm7)
            # skip connection
            lstm7 = torch.cat([lstm7, enc0], dim=1)

            enc6 = self.enc6_norm(torch.relu(self.enc6(lstm7)))

            enc7 = torch.relu(self.enc7(enc6))
            enc7 = enc7.view(self.opt.sequence_length, -1)

            fc1 = torch.relu(self.fc1(enc7))
            fc2 = torch.relu(self.fc2(fc1))
            fc2 = torch.softmax(fc2, dim=1)

            labels_list_prob.append(fc2)

        labels_list_prob = torch.stack(labels_list_prob)

        return labels_list_prob



if __name__ == '__main__':

    from options import Options
    import cv2
    opt = Options().parse()
    opt.batch_size = 2

    a_network = network(opt)
