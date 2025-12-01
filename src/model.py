"""
Neural network architecture for the DeepFork chess agent.

This module defines an AlphaZero-style convolutional residual network with
separate value and policy heads operating on an 8x8 board representation
constructed elsewhere in the project.
"""

import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Initial convolutional block processing the input state tensor."""

    def __init__(self, history_size=8, filter_count=256):
        super(ConvBlock, self).__init__()
        self.history_size = history_size
        self.conv = nn.Conv2d(14 * history_size + 7, filter_count, 3, padding=1)
        self.bn = nn.BatchNorm2d(filter_count)

    def forward(self, data):
        """
        :param data: Input tensor of shape (batch, (14*h + 7)*8*8) or already (batch, channels, 8, 8)
        :return: Feature map after a conv + BN + ReLU
        """
        data = data.view(-1, 14 * self.history_size + 7, 8, 8)  # batch-size, channels, board_w, board_h
        return F.relu(self.bn(self.conv(data)))


class ResBlock(nn.Module):
    """A standard residual block with two 3x3 convolutions."""

    def __init__(self, filter_count=256):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_count, filter_count, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_count)
        self.conv2 = nn.Conv2d(filter_count, filter_count, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_count)

    def forward(self, data):
        """
        :param data: Feature map tensor
        :return: Feature map after residual addition and ReLU
        """
        res = data
        data = F.relu(self.bn1(self.conv1(data)))
        data = self.bn2(self.conv2(data))
        data += res
        return F.relu(data)


class OutBlock(nn.Module):
    """Output heads: scalar value and flattened policy logits over 73x8x8."""

    def __init__(self, filter_count=256):
        super(OutBlock, self).__init__()
        # Value head
        self.convV = nn.Conv2d(filter_count, 1, 1)
        self.bnV = nn.BatchNorm2d(1)
        self.lnV1 = nn.Linear(8 * 8, 256)
        self.lnV2 = nn.Linear(256, 1)

        # Policy head
        self.convP = nn.Conv2d(filter_count, 73, 1)
        self.bnP = nn.BatchNorm2d(73)
        self.lsmP = nn.LogSoftmax(dim=1)

    def forward(self, data):
        """
        :param data: Feature map tensor
        :return: Tuple (value, policy_log_probs)
                 value shape: (batch, 1), policy shape: (batch, 73*8*8)
        """
        # Value head
        v = F.relu(self.bnV(self.convV(data)))
        v = F.relu(self.lnV1(v.view(-1, 8 * 8)))
        v = self.lnV2(v).tanh()

        # Policy head
        p = F.relu(self.bnP(self.convP(data)))
        p = self.lsmP(p.view(-1, 73 * 8 * 8))

        return v, p


class DeepForkNet(nn.Module):
    """
    Residual convolutional network with AlphaZero-style heads for policy and value.

    :param depth: Number of residual blocks
    :param filter_count: Channel width for feature maps
    :param history_size: Number of historical board states encoded in input
    """

    def __init__(self, depth=10, filter_count=256, history_size=8):
        super(DeepForkNet, self).__init__()
        self.filter_count = filter_count
        self.depth = depth
        self.conv_block = ConvBlock(history_size=history_size, filter_count=filter_count)
        self.res_blocks = nn.ModuleList([ResBlock(filter_count) for _ in range(depth)])
        self.out_block = OutBlock(filter_count=filter_count)

    def forward(self, data):
        """
        :param data: Input board tensor of shape (batch, (14*h + 7)*8*8)
        :return: Tuple (value, policy_log_probs)
        """
        data = self.conv_block(data)
        for block in self.res_blocks:
            data = block(data)
        v, p = self.out_block(data)
        return v, p